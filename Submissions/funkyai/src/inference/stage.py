from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.inference.model import CausalTransformerLM, resolve_device, set_seed


REMOTE_CHECKPOINT_URL = "https://pub-331934def20c42cea813b6434c78a240.r2.dev/model_final_sftv3.pt"


def _is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _read_checkpoint_locator(checkpoint_path: Path) -> str | None:
    if checkpoint_path.suffix.lower() not in {".url", ".txt"}:
        return None
    content = checkpoint_path.read_text(encoding="utf-8").strip()
    return content or None


def _download_to_cache(url: str, cache_root: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name or "checkpoint.pt"
    cache_root.mkdir(parents=True, exist_ok=True)
    target_path = cache_root / filename
    if target_path.exists():
        return target_path

    fd, temp_path_str = tempfile.mkstemp(
        prefix=f"{filename}.",
        suffix=".tmp",
        dir=str(cache_root),
    )
    os.close(fd)
    temp_path = Path(temp_path_str)
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(request) as response, temp_path.open("wb") as dst:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        temp_path.replace(target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target_path


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    raw_value = str(checkpoint_path).strip()
    if _is_url(raw_value):
        return _download_to_cache(raw_value, Path(".model_cache"))

    path = Path(raw_value)
    locator = _read_checkpoint_locator(path) if path.exists() else None
    if locator:
        if not _is_url(locator):
            raise ValueError(f"Checkpoint locator file must contain an HTTP(S) URL: {path}")
        return _download_to_cache(locator, path.parent / ".model_cache")

    return path


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # `weights_only` exists in newer torch versions only.
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict")

    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            "Checkpoint is missing model weights. Expected key 'state_dict' "
            "or 'model_state_dict'."
        )

    if "model" not in payload or not isinstance(payload["model"], dict):
        raise ValueError("Checkpoint is missing model metadata under key 'model'")
    if "tokenizer" not in payload or not isinstance(payload["tokenizer"], dict):
        raise ValueError("Checkpoint is missing tokenizer metadata under key 'tokenizer'")

    return {
        "state_dict": state_dict,
        "model": payload["model"],
        "tokenizer": payload["tokenizer"],
    }


def _build_model(model_cfg: dict[str, Any]) -> CausalTransformerLM:
    required = [
        "vocab_size",
        "max_seq_len",
        "n_embd",
        "n_layer",
        "n_head",
        "dropout",
        "layer_norm_epsilon",
    ]
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise ValueError(f"Checkpoint model metadata missing keys: {missing}")

    return CausalTransformerLM(
        vocab_size=int(model_cfg["vocab_size"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        n_embd=int(model_cfg["n_embd"]),
        n_layer=int(model_cfg["n_layer"]),
        n_head=int(model_cfg["n_head"]),
        dropout=float(model_cfg["dropout"]),
        layer_norm_epsilon=float(model_cfg["layer_norm_epsilon"]),
        activation=str(model_cfg.get("activation", "gelu")),
    )


def _load_tokenizer(tokenizer_name: str, quiet: bool = False):
    if not quiet:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Leaderboard mode must emit only generated text.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    with contextlib.redirect_stderr(io.StringIO()):
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _generate(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    if input_ids.shape[1] == 0:
        if tokenizer.eos_token_id is None:
            raise ValueError("Prompt tokenized to empty input and tokenizer has no eos_token_id")
        input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)

    generated_ids: list[int] = []
    model.eval()

    with torch.no_grad():
        for _ in range(max_tokens):
            model_input = input_ids[:, -model.max_seq_len :]
            outputs = model(input_ids=model_input)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_ids.append(int(next_token.item()))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = prompt + generated_text
    return {
        "generated_text": generated_text,
        "full_text": full_text,
        "generated_token_ids": generated_ids,
    }


def _parse_mc_options(prompt: str) -> list[tuple[str, str]]:
    matches = re.findall(r"(?m)^([A-E])\)\s*(.+)$", prompt)
    return [(letter, text.strip()) for letter, text in matches]


def _score_continuation(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not continuation_ids:
        return float("-inf")

    full_ids = prompt_ids + continuation_ids
    if len(full_ids) < 2:
        return float("-inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    model_input = input_ids[:, -model.max_seq_len :]
    with torch.no_grad():
        logits = model(input_ids=model_input)["logits"]

    # Align labels with the truncated model input.
    truncated_ids = model_input[0].tolist()
    truncated_prompt_len = min(len(prompt_ids), len(truncated_ids))
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = model_input[:, 1:]

    score = 0.0
    for pos in range(target_ids.shape[1]):
        target_index = pos + 1
        if target_index < truncated_prompt_len:
            continue
        token_id = target_ids[0, pos].item()
        score += float(log_probs[0, pos, token_id].item())
    return score


def _predict_mc_letter(
    model: CausalTransformerLM,
    tokenizer,
    prompt: str,
    device: torch.device,
) -> str | None:
    options = _parse_mc_options(prompt)
    if len(options) < 2:
        return None

    best_letter: str | None = None
    best_score = float("-inf")
    for letter, text in options:
        # Score the full answer span rather than just the bare letter so the model
        # can use the option content when it maps question -> answer choice.
        continuation = f" {letter}) {text}"
        score = _score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=continuation,
            device=device,
        )
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def main(
    checkpoint_path: str | Path,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    device: str = "auto",
    leaderboard: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    set_seed(seed)
    resolved_device = resolve_device(device)

    payload = _load_checkpoint_payload(checkpoint_path)
    tokenizer_name = payload["tokenizer"].get("name")
    if not tokenizer_name:
        raise ValueError("Checkpoint tokenizer metadata must include 'name'")

    tokenizer = _load_tokenizer(tokenizer_name=tokenizer_name, quiet=leaderboard)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(payload["model"])
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device)
    model.eval()

    if leaderboard:
        mc_letter = _predict_mc_letter(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=resolved_device,
        )
        if mc_letter is not None:
            result = {
                "generated_text": mc_letter,
                "full_text": prompt + mc_letter,
                "generated_token_ids": [],
                "device": str(resolved_device),
            }
            print(result["generated_text"])
            return result

    result = _generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        device=resolved_device,
    )
    result["device"] = str(resolved_device)

    if leaderboard:
        print(result["generated_text"])
    else:
        print(f"Device: {resolved_device}")
        print("Prompt:")
        print(prompt)
        print("\nGenerated text:")
        print(result["generated_text"])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PikoGPT inference stage")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 = greedy decoding)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, mps, or cpu",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Output only generated continuation text",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling",
    )
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        leaderboard=args.leaderboard,
        seed=args.seed,
    )
