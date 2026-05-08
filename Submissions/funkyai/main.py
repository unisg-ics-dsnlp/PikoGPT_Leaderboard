import argparse


def main():
    parser = argparse.ArgumentParser(description="PikoGPT inference entry point")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["inference"],
        help="Only inference is supported in this trimmed submission",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic inference (default: 42)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
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
        help="Leaderboard mode: output only generated text",
    )

    args = parser.parse_args()
    from src.inference.stage import main as inference_main

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --stage inference")
    if args.prompt is None:
        raise ValueError("--prompt is required for --stage inference")
    if args.max_tokens < 0:
        raise ValueError("--max-tokens must be >= 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")

    inference_main(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        leaderboard=args.leaderboard,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
