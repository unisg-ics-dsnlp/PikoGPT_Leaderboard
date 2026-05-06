#!/usr/bin/env python3
"""Inference-only leaderboard entrypoint for the run013 PikoGPT_HSG checkpoint."""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import re
import sys
import warnings

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from transformers.utils import logging as hf_logging

ROOT = pathlib.Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if not SRC_ROOT.exists():
    # Allows testing this file from the repository's lecture/ folder before
    # copying it into the leaderboard submission folder.
    SRC_ROOT = ROOT.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from gpt_arch.model import GPTTiny

VOCAB_SIZE = 50257
BLOCK_SIZE = 1024
N_LAYER = 11
N_HEAD = 8
N_KV_HEAD = 4
N_EMBD = 384
DROPOUT = 0.05


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_deterministic(seed: int, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _build_model(device: str) -> GPTTiny:
    model = GPTTiny(
        vocab_size=VOCAB_SIZE,
        context_length=BLOCK_SIZE,
        emb_dim=N_EMBD,
        mlp_ratio=4,
        n_heads=N_HEAD,
        n_kv_heads=N_KV_HEAD,
        n_decoders=N_LAYER,
        drop_rate=DROPOUT,
        qkv_bias=False,
        use_gradient_checkpointing=False,
    )
    return model.to(device)


def _load_model(checkpoint_path: pathlib.Path, device: str) -> GPTTiny:
    model = _build_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    for token_id in set(generated[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


def _banned_ngram_tokens(generated: torch.Tensor, ngram_size: int) -> set[int]:
    if ngram_size <= 0 or generated.size(1) < ngram_size - 1:
        return set()
    ids = generated[0].tolist()
    prefix = tuple(ids[-(ngram_size - 1):])
    banned: set[int] = set()
    for idx in range(len(ids) - ngram_size + 1):
        ngram = tuple(ids[idx:idx + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def _is_mcq_prompt(prompt: str) -> bool:
    return bool(re.search(r"Answer:\s*$", prompt.rstrip()))


@torch.no_grad()
def _score_mcq(model: GPTTiny, tokenizer: GPT2TokenizerFast, prompt: str, device: str) -> str:
    """Calibrated letter scoring: log P(letter|context) - log P(letter|'Answer:').

    Subtracting the prior removes the GPT-2 pretraining A-token bias so that
    the ranking reflects contextual evidence rather than the raw letter frequency.
    """
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[:, -BLOCK_SIZE:]
    logits = model(ids, causal=True, use_cache=True)[0][:, -1, :].float()
    log_probs = F.log_softmax(logits[0], dim=-1)

    bias_ids = tokenizer.encode("Answer:", return_tensors="pt").to(device)
    bias_logits = model(bias_ids, causal=True, use_cache=True)[0][:, -1, :].float()
    bias_log_probs = F.log_softmax(bias_logits[0], dim=-1)

    letters = "ABCD" if re.search(r"\n[CD]\)", prompt) else "AB"
    letter_ids = {c: tokenizer.encode(c, add_special_tokens=False)[0] for c in letters}
    return max(letter_ids, key=lambda c: (log_probs[letter_ids[c]] - bias_log_probs[letter_ids[c]]).item())


def _parse_choices(prompt: str) -> dict[str, str]:
    pattern = re.compile(r"\n([A-D])\)\s*(.*?)(?=\n[A-D]\)|\nAnswer:|$)", re.DOTALL)
    return {m.group(1): m.group(2).strip() for m in pattern.finditer(prompt)}


@torch.no_grad()
def _choice_score(
    model: GPTTiny,
    tokenizer: GPT2TokenizerFast,
    prefix: str,
    choice_text: str,
    device: str,
    *,
    normalize: bool,
) -> float:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + " " + choice_text.strip(), add_special_tokens=False)
    n_choice = len(full_ids) - len(prefix_ids)
    if n_choice <= 0:
        return float("-inf")

    if len(full_ids) > BLOCK_SIZE:
        full_ids = full_ids[-BLOCK_SIZE:]
        prefix_len = max(0, BLOCK_SIZE - n_choice)
    else:
        prefix_len = len(prefix_ids)

    input_t = torch.tensor([full_ids], dtype=torch.long).to(device)
    logits = model(input_t, causal=True, use_cache=True)[0].float()
    log_probs = torch.log_softmax(logits[0], dim=-1)

    total, n = 0.0, 0
    for idx in range(max(0, prefix_len - 1), len(full_ids) - 1):
        total += log_probs[idx, full_ids[idx + 1]].item()
        n += 1
    if n <= 0:
        return float("-inf")
    return total / n if normalize else total


@torch.no_grad()
def _score_binary_swapped(model: GPTTiny, tokenizer: GPT2TokenizerFast, prompt: str, device: str) -> str | None:
    choices = _parse_choices(prompt)
    if set(choices) != {"A", "B"}:
        return None
    first_choice = re.search(r"\n[A-B]\)", prompt)
    answer = re.search(r"\nAnswer:\s*$", prompt)
    if first_choice is None or answer is None:
        return None

    context = prompt[:first_choice.start()]
    swapped_context = context + f"\nA) {choices['B']}\nB) {choices['A']}"
    alpha = 4.0

    score_a = _choice_score(model, tokenizer, context + "\nA)", choices["A"], device, normalize=False)
    score_a += _choice_score(model, tokenizer, swapped_context + "\nB)", choices["A"], device, normalize=False)
    score_b = _choice_score(model, tokenizer, context + "\nB)", choices["B"], device, normalize=False)
    score_b += _choice_score(model, tokenizer, swapped_context + "\nA)", choices["B"], device, normalize=False)

    base_a = _choice_score(model, tokenizer, "Answer:", choices["A"], device, normalize=False)
    base_b = _choice_score(model, tokenizer, "Answer:", choices["B"], device, normalize=False)
    if base_a != float("-inf"):
        score_a -= 2.0 * alpha * base_a
    if base_b != float("-inf"):
        score_b -= 2.0 * alpha * base_b

    return "A" if score_a >= score_b else "B"


@torch.no_grad()
def _score_winogrande_cloze(model: GPTTiny, tokenizer: GPT2TokenizerFast, prompt: str, device: str) -> str | None:
    choices = _parse_choices(prompt)
    if set(choices) != {"A", "B"}:
        return None
    first_choice = re.search(r"\n[A-B]\)", prompt)
    if first_choice is None:
        return None

    context = prompt[:first_choice.start()]
    if context.startswith("Context:"):
        context = context[len("Context:"):].strip()
    if "_" not in context:
        return None

    scores: dict[str, float] = {}
    for letter, choice_text in choices.items():
        filled = context.replace("_", choice_text.strip(), 1)
        score = _choice_score(
            model,
            tokenizer,
            "",
            filled,
            device,
            normalize=True,
        )
        scores[letter] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


@torch.no_grad()
def _score_hellaswag_continuation(model: GPTTiny, tokenizer: GPT2TokenizerFast, prompt: str, device: str) -> str | None:
    choices = _parse_choices(prompt)
    if len(choices) != 4 or not prompt.lstrip().startswith("Context:"):
        return None
    first_choice = re.search(r"\n[A-D]\)", prompt)
    if first_choice is None:
        return None

    context = prompt[:first_choice.start()]
    if context.startswith("Context:"):
        context = context[len("Context:"):].strip()

    scores: dict[str, float] = {}
    alpha = 0.5
    for letter, choice_text in choices.items():
        choice_text = choice_text.strip()
        if not choice_text:
            continue
        score = _choice_score(model, tokenizer, context, choice_text, device, normalize=True)
        base = _choice_score(model, tokenizer, "The", choice_text, device, normalize=True)
        if base != float("-inf"):
            score -= alpha * base
        scores[letter] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


@torch.no_grad()
def _score_openbookqa_answer(model: GPTTiny, tokenizer: GPT2TokenizerFast, prompt: str, device: str) -> str | None:
    choices = _parse_choices(prompt)
    if len(choices) != 4 or not prompt.lstrip().startswith("Question:"):
        return None
    first_choice = re.search(r"\n[A-D]\)", prompt)
    if first_choice is None:
        return None

    question = prompt[:first_choice.start()]
    scores: dict[str, float] = {}
    alpha = 2.0
    for letter, choice_text in choices.items():
        choice_text = choice_text.strip()
        if not choice_text:
            continue
        score = _choice_score(model, tokenizer, question + "\nAnswer:", choice_text, device, normalize=False)
        base = _choice_score(model, tokenizer, "Answer:", choice_text, device, normalize=False)
        if base != float("-inf"):
            score -= alpha * base
        scores[letter] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


@torch.no_grad()
def _score_mcq_perplexity(model: GPTTiny, tokenizer: GPT2TokenizerFast, prompt: str, device: str) -> str:
    """Score each MCQ choice by average log-prob of choice tokens given context."""
    choices = _parse_choices(prompt)
    first_choice = re.search(r"\n[A-D]\)", prompt)
    if set(choices) == {"A", "B"}:
        cloze = _score_winogrande_cloze(model, tokenizer, prompt, device)
        if cloze is not None:
            return cloze
        swapped = _score_binary_swapped(model, tokenizer, prompt, device)
        if swapped is not None:
            return swapped
    if len(choices) == 4:
        hella = _score_hellaswag_continuation(model, tokenizer, prompt, device)
        if hella is not None:
            return hella
        obqa = _score_openbookqa_answer(model, tokenizer, prompt, device)
        if obqa is not None:
            return obqa
    if not choices or not first_choice:
        # Fallback: letter logit scoring
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[:, -BLOCK_SIZE:]
        logits = model(ids, causal=True, use_cache=True)[0][:, -1, :].float()
        letters = "".join(choices.keys()) or ("ABCD" if re.search(r"\n[CD]\)", prompt) else "AB")
        letter_ids = {c: tokenizer.encode(c, add_special_tokens=False)[0] for c in letters}
        return max(letter_ids, key=lambda c: logits[0, letter_ids[c]].item())

    context = prompt[:first_choice.start()]
    best_letter, best_score = None, float("-inf")

    for letter, choice_text in choices.items():
        # Do NOT include the trailing space in prefix — avoids BPE boundary merge
        # e.g. tokenize("A)") + tokenize(" bank") != tokenize("A) ") + tokenize("bank")
        prefix = context + f"\n{letter})"
        full_text = context + f"\n{letter}) {choice_text}"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        n_choice = len(full_ids) - len(prefix_ids)

        if n_choice <= 0:
            continue

        if len(full_ids) > BLOCK_SIZE:
            full_ids = full_ids[-BLOCK_SIZE:]
            prefix_len = max(0, BLOCK_SIZE - n_choice)
        else:
            prefix_len = len(prefix_ids)

        input_t = torch.tensor([full_ids], dtype=torch.long).to(device)
        logits = model(input_t, causal=True, use_cache=True)[0].float()
        log_probs = torch.log_softmax(logits[0], dim=-1)

        total, n = 0.0, 0
        for i in range(max(0, prefix_len - 1), len(full_ids) - 1):
            total += log_probs[i, full_ids[i + 1]].item()
            n += 1

        score = total / n if n > 0 else float("-inf")
        if score > best_score:
            best_score, best_letter = score, letter

    return best_letter or next(iter(choices))


@torch.no_grad()
def generate(
    model: GPTTiny,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: str,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = ids.clone()
    past_key_values = None

    for _ in range(max_tokens):
        if past_key_values is None:
            input_ids = generated[:, -BLOCK_SIZE:]
        else:
            input_ids = generated[:, -1:]

        out = model(input_ids, causal=True, past_key_values=past_key_values, use_cache=True)
        logits, past_key_values = out
        next_logits = logits[:, -1, :].float()
        next_logits = _apply_repetition_penalty(next_logits, generated, repetition_penalty)

        banned = _banned_ngram_tokens(generated, no_repeat_ngram_size)
        if banned:
            next_logits[:, list(banned)] = -float("inf")

        if temperature == 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break

    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PikoGPT_HSG leaderboard inference")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=50, dest="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stage != "inference":
        raise SystemExit("This submission entrypoint supports only --stage inference")

    device = _resolve_device(args.device)
    _set_deterministic(args.seed, device)

    real_stdout = sys.stdout
    real_stderr = sys.stderr
    if args.leaderboard:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf_logging.set_verbosity_error()
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model = _load_model(pathlib.Path(args.checkpoint), device)
    finally:
        if args.leaderboard:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = real_stdout
            sys.stderr = real_stderr

    if _is_mcq_prompt(args.prompt):
        text = _score_mcq_perplexity(model, tokenizer, args.prompt, device)
    else:
        text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt.rstrip(),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    if args.leaderboard:
        print(text, end="")
    else:
        print(f"Device     : {device}")
        print(f"Checkpoint : {args.checkpoint}")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
