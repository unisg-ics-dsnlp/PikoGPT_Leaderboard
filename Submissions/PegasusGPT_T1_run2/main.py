#!/usr/bin/env python3
"""Leaderboard adapter for PegasusGPT_T1.

Implements the CLI signature expected by leaderboard/run_benchmarks.py and
delegates inference to the project's GPTInferenceService. In --leaderboard
mode, only the generated continuation is written to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent

SOURCE_ROOT = next(
    (
        p
        for p in (HERE / "src", HERE, *HERE.parents)
        if (p / "domain" / "inference" / "inference_service.py").is_file()
    ),
    HERE / "src",
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


# Fallback model hyperparameters used only if the checkpoint does not embed a
# model_config. Mirrors configs/inference.toml for the current run.
DEFAULT_MODEL_CONFIG: dict = dict(
    architecture="llama3",
    vocab_size=50257,
    max_position_embeddings=1024,
    hidden_size=320,
    num_layers=22,
    num_attention_heads=16,
    tie_word_embeddings=True,
    mlp_hidden_size=None,
    qkv_bias=False,
    dropout=0.0,
    n_kv_heads=1,
    intermediate_size=853,
    rope_theta=10000.0,
)

# Default-on for leaderboard runs. Pass --disable-multiple-choice-scoring to
# force plain generation even when the prompt matches a registered benchmark.
ENABLE_MULTIPLE_CHOICE_SCORING = True


def _silence_third_party_logging() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.getLogger().setLevel(logging.ERROR)
    # Windows console defaults to cp1252; tolerate any chars the model emits.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PegasusGPT_T1 leaderboard adapter")
    p.add_argument("--stage", default="inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--leaderboard", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", dest="top_k", type=int, default=50)
    p.add_argument("--disable-multiple-choice-scoring", action="store_true")
    p.add_argument("--return-scored-text", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage != "inference":
        sys.stderr.write(f"unsupported stage: {args.stage}\n")
        return 2

    _silence_third_party_logging()
    _seed_all(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()

    prompt = args.prompt.rstrip()
    model_config = SimpleNamespace(**DEFAULT_MODEL_CONFIG)
    device_name = _resolve_device(args.device)

    if ENABLE_MULTIPLE_CHOICE_SCORING and not args.disable_multiple_choice_scoring:
        from domain.scoring import MultipleChoiceScoringService

        scoring_service = MultipleChoiceScoringService()
        # ----- Debug Tool -----
        if args.return_scored_text:
            scored_texts = scoring_service.scored_texts_for_prompt(prompt)
            if scored_texts is not None:
                benchmark, choices = scored_texts
                sys.stdout.write(
                    json.dumps(
                        {
                            "benchmark": benchmark,
                            "choices": [
                                {"letter": letter, "scored_text": scored_text}
                                for letter, scored_text in choices
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                sys.stdout.flush()
                return 0
        # ----- End Debug -----

        scoring_result = scoring_service.run(
            checkpoint_path=str(checkpoint_path),
            model_config=model_config,
            input_text=prompt,
            device_name=device_name,
            vocab_size=DEFAULT_MODEL_CONFIG["vocab_size"],
        )
        if scoring_result is not None:
            sys.stdout.write(scoring_result.reply)
            sys.stdout.flush()
            return 0

    from domain.inference.inference_service import GPTInferenceService

    service = GPTInferenceService()
    result = service.run(
        checkpoint_path=str(checkpoint_path),
        model_config=model_config,
        input_text=prompt,
        max_new_tokens=args.max_tokens,
        device_name=device_name,
        vocab_size=DEFAULT_MODEL_CONFIG["vocab_size"],
        temperature=args.temperature,
        top_k=args.top_k,
    )

    from domain.tokenization import load_gpt2_tokenizer

    tokenizer = load_gpt2_tokenizer()
    continuation_ids = result.generated_token_ids[len(result.input_token_ids):]
    continuation = tokenizer.decode(continuation_ids)

    sys.stdout.write(continuation)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
