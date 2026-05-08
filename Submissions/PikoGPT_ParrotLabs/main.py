"""Inference-only entry point for the PikoGPT leaderboard submission."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from src.eval.inference import run_inference
from src.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="PikoGPT ParrotLabs inference")
    parser.add_argument("--stage", required=True, choices=["inference"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    inference_cfg = SimpleNamespace(
        device=args.device,
        max_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
    )
    project_config = SimpleNamespace(inference=inference_cfg)

    run_inference(
        project_config,
        checkpoint=args.checkpoint,
        device=get_device(args.device),
        prompt=args.prompt,
        max_tokens_override=args.max_tokens,
        temperature_override=args.temperature,
        leaderboard=args.leaderboard,
    )


if __name__ == "__main__":
    main()
