"""Supervised fine-tuning dataset utilities.

Implements the Stanford Alpaca plain-text prompt format and a dataset registry
so additional SFT corpora (Dolly, OASST1, …) can be added without touching the
trainer.  A sample is represented on disk as::

    {"input_ids": [int, ...], "prompt_len": int}

where ``prompt_len`` is the number of tokens up to *and including* the
``### Response:\\n`` marker.  The trainer masks ``target_ids[:prompt_len - 1]``
with ``-100`` so gradients only flow through response-token predictions.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import Dataset

from pikogpt.tokenizer import get_gpt2_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alpaca prompt template (Stanford, plain text — no special tokens added)
# ---------------------------------------------------------------------------

_PREAMBLE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
_PREAMBLE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request."
)


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Return the prompt string *up to and including* ``### Response:\\n``.

    The response body is appended separately so we can precisely count the
    prompt-token boundary used for loss masking.
    """
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return (
            f"{_PREAMBLE_WITH_INPUT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return (
        f"{_PREAMBLE_NO_INPUT}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )


# ---------------------------------------------------------------------------
# Tokeniser access (GPT-2 via transformers.GPT2TokenizerFast)
# ---------------------------------------------------------------------------


def _get_tokenizer():
    return get_gpt2_tokenizer()


def _encode(enc, text: str) -> list[int]:
    return enc.encode(text, add_special_tokens=False)


EOS_TOKEN_ID = 50256  # GPT-2 "<|endoftext|>" — also serves as pad for SFT


# ---------------------------------------------------------------------------
# Tokenisation of a single example
# ---------------------------------------------------------------------------


@dataclass
class SFTExample:
    """A tokenised SFT example.

    Attributes:
        input_ids: Full token sequence [prompt tokens || response tokens || eos].
        prompt_len: Number of tokens that belong to the prompt (mask boundary).
        source: SFT dataset key that produced the example.
        truncated: Whether the response was shortened to fit ``max_seq_len``.
    """

    input_ids: list[int]
    prompt_len: int
    source: str = "unknown"
    truncated: bool = False


def tokenise_pair(
    enc,
    instruction: str,
    input_text: str,
    output: str,
    max_seq_len: int,
) -> SFTExample | None:
    """Tokenise one instruction/response pair.

    Returns ``None`` if the prompt alone is longer than ``max_seq_len`` (we
    cannot fit any response) or if the response is empty.  If prompt+response
    exceeds ``max_seq_len``, the response is truncated.
    """
    output = (output or "").strip()
    if not output:
        return None

    prompt_str = format_alpaca_prompt(instruction, input_text)
    prompt_ids = _encode(enc, prompt_str)
    response_body_ids = _encode(enc, output)

    if len(prompt_ids) >= max_seq_len:
        # No budget left for the response — dropping.
        return None

    budget = max_seq_len - len(prompt_ids)
    truncated = len(response_body_ids) + 1 > budget
    if truncated:
        # Always keep EOS so generation learns a stop boundary even when the
        # response body has to be shortened.
        response_body_ids = response_body_ids[: max(0, budget - 1)]
    response_ids = response_body_ids + [EOS_TOKEN_ID]

    input_ids = prompt_ids + response_ids
    return SFTExample(
        input_ids=input_ids,
        prompt_len=len(prompt_ids),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Dataset loaders — registry
# ---------------------------------------------------------------------------


def _load_alpaca(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield raw ``{instruction, input, output}`` dicts from tatsu-lab/alpaca."""
    from datasets import load_dataset

    logger.info("Downloading tatsu-lab/alpaca from HuggingFace…")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("Alpaca: capped to %d samples (max_samples)", max_samples)
    for row in ds:
        yield {
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "output": row.get("output", ""),
        }


def _load_dolly(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield English Dolly rows mapped into Alpaca ``instruction/input/output`` triples."""
    from datasets import load_dataset

    logger.info("Downloading databricks/databricks-dolly-15k from HuggingFace…")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("Dolly: capped to %d samples (max_samples)", max_samples)

    kept = 0
    for row in ds:
        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("response") or "").strip()
        if not instruction or not output:
            continue
        kept += 1
        yield {
            "instruction": instruction,
            "input": str(row.get("context") or "").strip(),
            "output": output,
        }

    logger.info("Dolly: kept %d English prompt/response pairs", kept)


def _oasst1_row_is_usable(row: dict[str, Any]) -> bool:
    """Return whether an OASST1 message is safe to turn into SFT data."""
    text = str(row.get("text") or "").strip()
    role = str(row.get("role") or "")
    return bool(
        text
        and row.get("lang") == "en"
        and role in {"prompter", "assistant"}
        and bool(row.get("review_result", False))
        and not bool(row.get("deleted", False))
        and not bool(row.get("synthetic", False))
        and row.get("tree_state") == "ready_for_export"
    )


def _oasst1_reply_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    """Prefer the best-ranked reviewed assistant reply for each prompt."""
    rank = row.get("rank")
    rank_key = int(rank) if isinstance(rank, int) else 1_000_000
    review_count = int(row.get("review_count") or 0)
    created = str(row.get("created_date") or "")
    message_id = str(row.get("message_id") or "")
    return (rank_key, -review_count, created, message_id)


def _oasst1_conversation_chain(
    message_id: str,
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Return the root→leaf chain for one assistant message if it alternates cleanly."""
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = rows_by_id.get(message_id)
    while current is not None:
        current_id = str(current.get("message_id") or "")
        if not current_id or current_id in seen:
            return None
        seen.add(current_id)
        chain.append(current)

        parent_id = str(current.get("parent_id") or "")
        if not parent_id:
            break
        current = rows_by_id.get(parent_id)
        if current is None:
            return None

    chain.reverse()
    if not chain or chain[0]["role"] != "prompter" or chain[-1]["role"] != "assistant":
        return None

    for idx, row in enumerate(chain):
        expected_role = "prompter" if idx % 2 == 0 else "assistant"
        if row["role"] != expected_role:
            return None
    return chain


def _format_oasst1_history(history_rows: list[dict[str, Any]]) -> str:
    """Serialise earlier chat turns into the Alpaca ``### Input:`` block."""
    if not history_rows:
        return ""

    blocks = ["Previous conversation:"]
    for row in history_rows:
        speaker = "User" if row["role"] == "prompter" else "Assistant"
        blocks.append(f"{speaker}:\n{row['text']}")
    return "\n\n".join(blocks)


def _load_oasst1(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield English OASST1 chats mapped into Alpaca ``instruction/input/output`` triples."""
    from datasets import load_dataset

    logger.info("Downloading OpenAssistant/oasst1 from HuggingFace…")
    dataset_dict = load_dataset("OpenAssistant/oasst1")

    usable_rows: list[dict[str, Any]] = []
    for split_name in sorted(dataset_dict.keys()):
        for raw_row in dataset_dict[split_name]:
            if not _oasst1_row_is_usable(raw_row):
                continue
            row = dict(raw_row)
            row["message_id"] = str(row.get("message_id") or "")
            row["parent_id"] = str(row.get("parent_id") or "")
            row["text"] = str(row.get("text") or "").strip()
            usable_rows.append(row)

    rows_by_id = {
        row["message_id"]: row for row in usable_rows if row["message_id"]
    }

    assistant_children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in usable_rows:
        if row["role"] != "assistant" or not row["parent_id"]:
            continue
        parent = rows_by_id.get(row["parent_id"])
        if parent is None or parent["role"] != "prompter":
            continue
        assistant_children_by_parent.setdefault(row["parent_id"], []).append(row)

    examples: list[dict[str, str]] = []
    for children in assistant_children_by_parent.values():
        assistant_row = min(children, key=_oasst1_reply_sort_key)
        chain = _oasst1_conversation_chain(assistant_row["message_id"], rows_by_id)
        if chain is None or len(chain) < 2:
            continue

        prompt_row = chain[-2]
        examples.append(
            {
                "instruction": prompt_row["text"],
                "input": _format_oasst1_history(chain[:-2]),
                "output": assistant_row["text"],
            }
        )

    if max_samples is not None and max_samples > 0 and max_samples < len(examples):
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_samples]
        logger.info("OASST1: capped to %d samples (max_samples)", max_samples)

    logger.info(
        "OASST1: kept %d English reviewed prompt/response pairs",
        len(examples),
    )
    yield from examples


# Mapping name → loader.  To add another dataset, drop in a loader with the
# same yield-shape and register it here.
DATASET_LOADERS: dict[str, Callable[[int | None, int], Iterable[dict[str, str]]]] = {
    "alpaca": _load_alpaca,
    "dolly": _load_dolly,
    "oasst1": _load_oasst1,
}


# ---------------------------------------------------------------------------
# Build + cache the tokenised SFT dataset
# ---------------------------------------------------------------------------


def _cache_paths(cache_dir: Path, dataset_key: str) -> tuple[Path, Path]:
    base = cache_dir / dataset_key
    return base / "train.pt", base / "val.pt"


def _normalise_dataset_weights(
    datasets: list[str],
    dataset_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Return validated per-dataset weights with defaults filled to 1.0."""
    weights = {name: 1.0 for name in datasets}
    if not dataset_weights:
        return weights

    unknown = sorted(set(dataset_weights) - set(datasets))
    if unknown:
        raise ValueError(
            f"dataset_weights contains unknown dataset keys: {unknown}. "
            f"Configured datasets: {datasets}"
        )

    for name, value in dataset_weights.items():
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"dataset_weights[{name!r}] must be a positive finite number, got {value!r}"
            )
        weights[name] = weight

    return weights


def _dataset_weights_cache_suffix(weights: dict[str, float]) -> str:
    """Return a deterministic cache suffix for dataset weights."""
    return "_W" + "+".join(
        f"{name}x{weights[name]:g}" for name in sorted(weights)
    )


def _split_dataset_examples(
    examples: list[SFTExample],
    *,
    val_fraction: float,
    seed: int,
    dataset_name: str,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Split one dataset into train/val before any train-time weighting."""
    shuffled = list(examples)
    random.Random(f"{seed}:{dataset_name}:split").shuffle(shuffled)

    if not shuffled or val_fraction <= 0:
        return shuffled, []

    n_val = max(1, int(len(shuffled) * val_fraction))
    if len(shuffled) > 1:
        n_val = min(n_val, len(shuffled) - 1)

    return shuffled[n_val:], shuffled[:n_val]


def _apply_dataset_weight(
    examples: list[SFTExample],
    *,
    weight: float,
    seed: int,
    dataset_name: str,
) -> list[SFTExample]:
    """Scale one dataset's train split via deterministic over/under-sampling."""
    if not examples:
        return []

    target_count = max(1, int(round(len(examples) * weight)))
    full_copies, remainder = divmod(target_count, len(examples))

    weighted: list[SFTExample] = list(examples) * full_copies
    if remainder:
        sampled = list(examples)
        random.Random(f"{seed}:{dataset_name}:weight").shuffle(sampled)
        weighted.extend(sampled[:remainder])

    return weighted


def prepare_sft_dataset(
    datasets: list[str],
    cache_dir: str | Path,
    *,
    dataset_weights: dict[str, float] | None = None,
    max_seq_len: int = 1024,
    val_fraction: float = 0.02,
    max_samples: int | None = None,
    seed: int = 42,
    force: bool = False,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Prepare (or load cached) tokenised SFT train/val splits.

    The cache key encodes the dataset list, weights, max_seq_len and
    max_samples so that changing any of these invalidates the cache.
    """
    cache_dir = Path(cache_dir)
    weights = _normalise_dataset_weights(datasets, dataset_weights)
    dataset_key = (
        f"v2_{'+'.join(sorted(datasets))}"
        f"{_dataset_weights_cache_suffix(weights)}"
        f"_L{max_seq_len}"
        f"_N{max_samples if max_samples else 'all'}"
    )
    train_path, val_path = _cache_paths(cache_dir, dataset_key)

    if not force and train_path.exists() and val_path.exists():
        logger.info(
            "[sft] Using cached tokenised SFT data at %s",
            train_path.parent,
        )
        train = [
            SFTExample(**d) for d in torch.load(train_path, weights_only=False)
        ]
        val = [
            SFTExample(**d) for d in torch.load(val_path, weights_only=False)
        ]
        logger.info(
            "[sft] Loaded %d train + %d val examples from cache",
            len(train),
            len(val),
        )
        return train, val

    enc = _get_tokenizer()
    train_examples: list[SFTExample] = []
    val_examples: list[SFTExample] = []
    dropped = 0

    for name in datasets:
        if name not in DATASET_LOADERS:
            raise ValueError(
                f"Unknown SFT dataset: {name!r}. "
                f"Registered: {sorted(DATASET_LOADERS)}"
            )
        logger.info("[sft] Tokenising %s …", name)
        raw_examples: list[SFTExample] = []
        source_dropped = 0
        source_truncated = 0
        for row in DATASET_LOADERS[name](max_samples, seed):
            ex = tokenise_pair(
                enc,
                row["instruction"],
                row.get("input", ""),
                row["output"],
                max_seq_len=max_seq_len,
            )
            if ex is None:
                dropped += 1
                source_dropped += 1
                continue
            ex.source = name
            if ex.truncated:
                source_truncated += 1
            raw_examples.append(ex)

        base_train, base_val = _split_dataset_examples(
            raw_examples,
            val_fraction=val_fraction,
            seed=seed,
            dataset_name=name,
        )
        weighted_train = _apply_dataset_weight(
            base_train,
            weight=weights[name],
            seed=seed,
            dataset_name=name,
        )
        train_examples.extend(weighted_train)
        val_examples.extend(base_val)
        logger.info(
            "[sft] %s: %d tokenised -> %d train / %d val before weighting -> %d train after weight %.3g",
            name,
            len(raw_examples),
            len(base_train),
            len(base_val),
            len(weighted_train),
            weights[name],
        )
        total_seen = len(raw_examples) + source_dropped
        trunc_rate = source_truncated / max(len(raw_examples), 1)
        drop_rate = source_dropped / max(total_seen, 1)
        logger.info(
            "[sft] %s quality stats: truncated=%d/%d (%.2f%%), dropped=%d/%d (%.2f%%)",
            name,
            source_truncated,
            len(raw_examples),
            trunc_rate * 100,
            source_dropped,
            total_seen,
            drop_rate * 100,
        )

    if dropped:
        logger.warning(
            "[sft] Dropped %d examples that didn't fit in max_seq_len=%d or had empty output",
            dropped,
            max_seq_len,
        )

    rng = random.Random(seed)
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    logger.info(
        "[sft] Built %d train + %d val examples (val_fraction=%.2f%%; val is unweighted)",
        len(train_examples),
        len(val_examples),
        val_fraction * 100,
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([ex.__dict__ for ex in train_examples], train_path)
    torch.save([ex.__dict__ for ex in val_examples], val_path)
    logger.info("[sft] Cached tokenised data to %s", train_path.parent)

    return train_examples, val_examples


# ---------------------------------------------------------------------------
# torch Dataset + collator
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    """In-memory dataset of tokenised SFT examples.

    ``__getitem__`` returns the raw ``SFTExample``; padding and target-mask
    construction are done by :func:`sft_collate_fn` so batches can have
    varying prompt lengths without wasteful pre-padding to ``max_seq_len``.
    """

    def __init__(self, examples: list[SFTExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SFTExample:
        return self.examples[idx]


def sft_collate_fn(
    batch: list[SFTExample],
    pad_token_id: int = EOS_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch and build input/target tensors.

    The loss is computed via ``F.cross_entropy(..., ignore_index=-100)`` so
    both prompt tokens and pad tokens carry ``-100`` in the target.

    Given a tokenised sequence ``tokens`` of length ``L`` with prompt length
    ``k``:
      - ``input  = tokens[:-1]`` (length ``L-1``)
      - ``target = tokens[1:]``  (length ``L-1``) with ``target[:k-1] = -100``

    Batches are right-padded to the max ``L-1`` in the batch; pad positions in
    target are ``-100``.
    """
    # We predict ``L-1`` positions for a sequence of length ``L``.
    lengths = [len(ex.input_ids) - 1 for ex in batch]
    max_len = max(lengths)

    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        ids = ex.input_ids
        L = len(ids)
        k = ex.prompt_len  # prompt length in tokens

        inp = ids[:-1]  # length L-1
        tgt = ids[1:]   # length L-1

        input_ids[i, : L - 1] = torch.tensor(inp, dtype=torch.long)
        # Mask prompt positions in target (first k-1 predictions are of prompt tokens).
        # The position at index ``k-1`` predicts the first *response* token — keep it.
        tgt_tensor = torch.tensor(tgt, dtype=torch.long)
        if k - 1 > 0:
            tgt_tensor[: k - 1] = -100
        targets[i, : L - 1] = tgt_tensor
        # Positions L-1 … max_len-1 remain ``-100`` (pad).

    return input_ids, targets


__all__ = [
    "DATASET_LOADERS",
    "EOS_TOKEN_ID",
    "SFTDataset",
    "SFTExample",
    "format_alpaca_prompt",
    "prepare_sft_dataset",
    "sft_collate_fn",
    "tokenise_pair",
]
