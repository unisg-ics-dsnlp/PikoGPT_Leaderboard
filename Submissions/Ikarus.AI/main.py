#!/usr/bin/env python3
"""PikoGPT — single entry point with stage-based CLI routing.

Usage:
    uv run python main.py --stage <stage> [stage-specific flags]

Stages:
    data          Download, preprocess, and tokenize training data
    train         Pre-train the model
    posttraining  Supervised fine-tuning (SFT) / DPO
    evaluate      Run perplexity and benchmark evaluations
    inference     Generate text from a checkpoint (mandatory contract)
    demo          Launch Gradio chat interface
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


# Ensure local `src/` package imports work when running `python main.py`.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists():
    src_str = str(_SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from src.pikogpt.utils.device import resolve_device


# ---------------------------------------------------------------------------
# Cache-metadata helpers — detect stale caches when parameters change
# ---------------------------------------------------------------------------


def _write_cache_meta(meta_path: Path, **params) -> None:
    """Write a JSON file recording the parameters used to produce a cached artifact."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)


def _cache_is_valid(meta_path: Path, **params) -> bool:
    """Check whether cached data was produced with the same parameters.

    Returns True  if meta exists and parameters match (fresh cache).
    Returns True  if meta is missing — backwards-compat for legacy caches (warns).
    Returns False if parameters differ (stale cache, will be re-generated).
    """
    if not meta_path.exists():
        print(
            f"[cache] No metadata at {meta_path.name} — assuming cache is valid (legacy data).\n"
            f"[cache] Future runs will track parameters automatically."
        )
        return True

    with open(meta_path) as f:
        stored = json.load(f)

    if stored == params:
        return True

    # Report which parameters changed
    print(f"[cache] Stale cache detected ({meta_path.parent.name}):")
    all_keys = sorted(set(stored) | set(params))
    for key in all_keys:
        old_val = stored.get(key, "<missing>")
        new_val = params.get(key, "<missing>")
        if old_val != new_val:
            print(f"[cache]   {key}: {old_val!r} → {new_val!r}")
    print("[cache] Re-generating this stage.")
    return False


logger = logging.getLogger("pikogpt.cli")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pikogpt",
        description="PikoGPT — a 40M-parameter decoder-only LLM",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "data",
            "train",
            "train_ddp",
            "posttraining",
            "evaluate",
            "inference",
            "demo",
        ],
        help="Pipeline stage to run",
    )

    # --- Shared flags ---
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (YAML or TOML)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cpu, cuda, mps, auto"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Inference contract (mandatory flags for --stage inference) ---
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Text prompt for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (0=greedy)"
    )
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Print raw output only (no banners/logs)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        choices=["raw", "alpaca", "simple"],
        help=(
            "Prompt template for inference / demo. Stage-dependent default: "
            "'alpaca' for --stage demo (matches the SFT training format of "
            "### Instruction / ### Response + EOS, required for instruction "
            "following), 'raw' for --stage inference. Pass --chat-template raw "
            "to disable templating; 'simple' is the pre-SFT 'User: / Assistant:' "
            "layout (demo only)."
        ),
    )

    # --- Training flags ---
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[], help="Config overrides as key=value"
    )

    # --- Data pipeline flags ---
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Max documents to download (single-source default: 100k; "
        "multi-source: unlimited unless set, split proportionally by ratio)",
    )
    parser.add_argument(
        "--eval-test-dir",
        type=str,
        default="data/NLP26_OWT_eval/test",
        help="Path to eval test set (HF dataset on disk) for overlap removal",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run, ignoring cached intermediates",
    )
    parser.add_argument(
        "--single-source",
        action="store_true",
        help="Use OWT-only pipeline even when a multi-source config is provided "
        "(for ablation experiments comparing single- vs multi-source training)",
    )

    # --- Demo flags ---
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio share link"
    )

    return parser


# ---------------------------------------------------------------------------
# Stage handlers — each uses lazy imports to avoid loading unused deps
# ---------------------------------------------------------------------------


def stage_data(args: argparse.Namespace) -> None:
    from src.pikogpt.utils.seed import set_seed

    set_seed(args.seed)

    # Check if a config with multi-source data is provided
    if args.config and not getattr(args, "single_source", False):
        from src.pikogpt.utils.config import load_config

        config = load_config(args.config)
        if config.data.sources:
            _run_multi_source_pipeline(args, config)
            return

    # Single-source pipeline (OpenWebText only)
    _run_single_source_pipeline(args)


def _run_single_source_pipeline(args: argparse.Namespace) -> None:
    """Original single-source data pipeline (OpenWebText only)."""
    from datasets import load_from_disk

    from src.pikogpt.data.download import download_dataset
    from src.pikogpt.data.preprocess import preprocess
    from src.pikogpt.data.tokenize_data import tokenize_eval_test_set, tokenize_to_memmap

    force = getattr(args, "force", False)

    raw_cache = Path("data/raw/openwebtext_subset")
    processed_cache = Path("data/processed/openwebtext_cleaned")
    tokenized_dir = Path("data/tokenized")

    logger.info("=== Stage: Data Pipeline (single-source) ===")

    # Step 1 — Download (with caching)
    if not force and raw_cache.exists():
        logger.info(
            "[data] Found cached raw dataset at %s, skipping download.", raw_cache
        )
        dataset = load_from_disk(str(raw_cache))
    else:
        num_samples = args.num_samples if args.num_samples is not None else 100_000
        dataset = download_dataset(num_samples=num_samples, seed=args.seed)
        dataset.save_to_disk(str(raw_cache))
        logger.info("[data] Saved raw dataset to %s", raw_cache)

    # Step 2 — Preprocess (with caching)
    if not force and processed_cache.exists():
        logger.info(
            "[data] Found cached processed dataset at %s, skipping preprocessing.",
            processed_cache,
        )
        dataset = load_from_disk(str(processed_cache))
    else:
        dataset = preprocess(dataset, eval_test_dir=args.eval_test_dir, seed=args.seed)
        dataset.save_to_disk(str(processed_cache))
        logger.info("[data] Saved processed dataset to %s", processed_cache)

    # Step 3 — Tokenize train/val (with parameter-aware caching)
    tok_params = dict(seed=args.seed, val_fraction=0.05)
    tok_meta = tokenized_dir / ".meta_trainval.json"
    if (
        not force
        and (tokenized_dir / "train.bin").exists()
        and (tokenized_dir / "val.bin").exists()
        and _cache_is_valid(tok_meta, **tok_params)
    ):
        logger.info(
            "[data] Found cached tokenized data at %s, skipping tokenization.",
            tokenized_dir,
        )
    else:
        tokenize_to_memmap(dataset, output_dir=tokenized_dir, seed=args.seed)
        _write_cache_meta(tok_meta, **tok_params)

    # Step 4 — Tokenize eval test set → test.bin
    test_params = dict(eval_test_dir=args.eval_test_dir)
    test_meta = tokenized_dir / ".meta_test.json"
    if (
        not force
        and (tokenized_dir / "test.bin").exists()
        and _cache_is_valid(test_meta, **test_params)
    ):
        logger.info(
            "[data] Found cached test.bin at %s, skipping test tokenization.",
            tokenized_dir,
        )
    else:
        tokenize_eval_test_set(args.eval_test_dir, output_dir=tokenized_dir)
        _write_cache_meta(test_meta, **test_params)

    logger.info("Data pipeline complete.")


def _run_multi_source_pipeline(args: argparse.Namespace, config) -> None:
    """Multi-source data pipeline: download → preprocess → decontaminate → tokenize.

    Uses per-source caching so that only stale stages are re-run.  Cache layout::

        data/raw/<source>/              save_to_disk() output
        data/raw/<source>/.meta_raw.json
        data/processed/<source>/        save_to_disk() output
        data/processed/<source>/.meta_processed.json
        data/tokenized/.meta_trainval.json
    """
    from datasets import load_from_disk

    from src.pikogpt.data.decontaminate import build_ngram_index
    from src.pikogpt.data.preprocess import cross_source_dedup, preprocess
    from src.pikogpt.data.sources import download_source
    from src.pikogpt.data.tokenize_data import (
        tokenize_eval_test_set,
        tokenize_multi_source_to_memmap,
    )

    force = getattr(args, "force", False)
    num_samples = getattr(args, "num_samples", None)
    tokenized_dir = Path("data/tokenized")
    total_tokens = config.data.total_tokens
    decontam_enabled = config.data.decontaminate

    if num_samples is not None:
        logger.info(
            "=== Stage: Data Pipeline (multi-source, --num-samples %s, capping downloads) ===",
            f"{num_samples:,}",
        )
    else:
        logger.info(
            "=== Stage: Data Pipeline (multi-source, %s total tokens) ===",
            f"{total_tokens:,}",
        )

    import time as _time

    # Step 1 — Build decontamination index (once, before per-source loop)
    ngram_index = None
    decontam_config_key = "none"
    if decontam_enabled:
        logger.info("[data] Step 1/5: Building comprehensive n-gram decontamination index ...")
        t0 = _time.monotonic()
        ngram_index = build_ngram_index(owt_eval_test_dir=args.eval_test_dir)
        decontam_config_key = f"ngram13_threshold0_owt={args.eval_test_dir}"
        logger.info("[data] Decontamination index built in %.1fs.", _time.monotonic() - t0)

    # Step 2 — Per-source download + preprocess with caching
    processed_datasets: dict[str, object] = {}
    source_names = [s.name for s in config.data.sources]
    n_sources = len(source_names)

    for src_idx, src_cfg in enumerate(config.data.sources, 1):
        name = src_cfg.name
        target_tokens = int(src_cfg.ratio * total_tokens)
        # --num-samples caps total docs, split proportionally by ratio
        source_max_docs = (
            max(1, int(num_samples * src_cfg.ratio)) if num_samples else None
        )

        logger.info(
            "[data] ── Source %d/%d: %s (ratio=%.0f%%, ~%s docs) ──",
            src_idx, n_sources, name, src_cfg.ratio * 100,
            f"{source_max_docs:,}" if source_max_docs else "auto",
        )

        # --- Raw cache ---
        raw_dir = Path(f"data/raw/{name}")
        raw_meta = raw_dir / ".meta_raw.json"
        raw_params = dict(
            hf_id=src_cfg.hf_dataset_id,
            hf_subset=getattr(src_cfg, "hf_subset", None),
            hf_split=src_cfg.hf_split,
            target_tokens=target_tokens,
            max_docs=source_max_docs,
            seed=args.seed,
        )

        if not force and raw_dir.exists() and _cache_is_valid(raw_meta, **raw_params):
            logger.info("[data]   Using cached raw %s from %s", name, raw_dir)
            raw_ds = load_from_disk(str(raw_dir))
        else:
            t_dl = _time.monotonic()
            raw_ds = download_source(
                name=name,
                hf_dataset_id=src_cfg.hf_dataset_id,
                hf_subset=getattr(src_cfg, "hf_subset", None),
                hf_split=src_cfg.hf_split,
                text_column=src_cfg.text_column,
                target_tokens=target_tokens,
                seed=args.seed,
                max_docs=source_max_docs,
            )
            logger.info(
                "[data]   Download complete: %s docs in %.1fs",
                f"{len(raw_ds):,}",
                _time.monotonic() - t_dl,
            )
            raw_ds.save_to_disk(str(raw_dir))
            _write_cache_meta(raw_meta, **raw_params)
            logger.info("[data]   Saved raw %s to %s", name, raw_dir)

        # --- Processed cache ---
        proc_dir = Path(f"data/processed/{name}")
        proc_meta = proc_dir / ".meta_processed.json"
        proc_params = dict(
            seed=args.seed,
            source_name=name,
            decontam=decontam_config_key,
        )

        if (
            not force
            and proc_dir.exists()
            and _cache_is_valid(proc_meta, **proc_params)
        ):
            logger.info("[data]   Using cached processed %s from %s", name, proc_dir)
            processed_ds = load_from_disk(str(proc_dir))
        else:
            t_pp = _time.monotonic()
            logger.info(
                "[data]   Preprocessing %s (%s docs) ...",
                name,
                f"{len(raw_ds):,}",
            )
            processed_ds = preprocess(
                raw_ds,
                eval_test_dir=args.eval_test_dir,
                seed=args.seed,
                source_name=name,
                ngram_index=ngram_index,
            )
            logger.info(
                "[data]   Preprocessing complete: %s → %s docs in %.1fs",
                f"{len(raw_ds):,}",
                f"{len(processed_ds):,}",
                _time.monotonic() - t_pp,
            )
            processed_ds.save_to_disk(str(proc_dir))
            _write_cache_meta(proc_meta, **proc_params)

        processed_datasets[name] = processed_ds

    # Step 3 — Cross-source deduplication (highest eval-value sources kept first)
    logger.info("[data] Step 3/5: Cross-source deduplication ...")
    t_dedup = _time.monotonic()
    priority_order = [
        n
        for n in [
            "openwebtext",
            "fineweb_edu",
            "wikipedia",
            "fineweb",
            "c4",
        ]
        if n in processed_datasets
    ]
    # Include any sources not in the default priority list at the end
    priority_order += [n for n in source_names if n not in priority_order]
    processed_datasets = cross_source_dedup(processed_datasets, priority_order)
    total_docs = sum(len(ds) for ds in processed_datasets.values())
    logger.info(
        "[data] Cross-dedup done in %.1fs — %s total docs across %d sources.",
        _time.monotonic() - t_dedup,
        f"{total_docs:,}",
        len(processed_datasets),
    )

    # Step 4 — Tokenize into unified train.bin / val.bin
    tok_params = dict(
        seed=args.seed,
        val_fraction=0.05,
        sources=source_names,
        total_tokens=total_tokens,
    )
    tok_meta = tokenized_dir / ".meta_trainval.json"
    if (
        not force
        and (tokenized_dir / "train.bin").exists()
        and (tokenized_dir / "val.bin").exists()
        and _cache_is_valid(tok_meta, **tok_params)
    ):
        logger.info(
            "[data] Found cached tokenized data at %s, skipping tokenization.",
            tokenized_dir,
        )
    else:
        logger.info("[data] Step 4/5: Tokenizing all sources into unified memmap files ...")
        t_tok = _time.monotonic()
        tokenize_multi_source_to_memmap(
            processed_datasets, output_dir=tokenized_dir, seed=args.seed
        )
        logger.info("[data] Tokenization done in %.1fs.", _time.monotonic() - t_tok)
        _write_cache_meta(tok_meta, **tok_params)

    # Step 5 — Tokenize eval test set → test.bin
    test_params = dict(eval_test_dir=args.eval_test_dir)
    test_meta = tokenized_dir / ".meta_test.json"
    if (
        not force
        and (tokenized_dir / "test.bin").exists()
        and _cache_is_valid(test_meta, **test_params)
    ):
        logger.info(
            "[data] Found cached test.bin at %s, skipping test tokenization.",
            tokenized_dir,
        )
    else:
        tokenize_eval_test_set(args.eval_test_dir, output_dir=tokenized_dir)
        _write_cache_meta(test_meta, **test_params)

    logger.info("Multi-source data pipeline complete.")


def _run_train_stage(args: argparse.Namespace, require_ddp: bool = False) -> None:
    from src.pikogpt.utils.config import load_config, merge_cli_overrides
    from src.pikogpt.utils.seed import set_seed
    from src.pikogpt.training.distributed import cleanup_distributed, setup_distributed

    if not args.config:
        logger.error("--config is required for training.")
        sys.exit(1)

    config = load_config(args.config)
    config = merge_cli_overrides(config, args.overrides)
    train_cfg = config.training.model_dump()

    dist_ctx = setup_distributed(
        requested_device=args.device,
        backend=train_cfg.get("ddp_backend"),
    )
    if require_ddp and not dist_ctx.is_distributed:
        logger.error(
            "--stage train_ddp must be launched with torchrun (WORLD_SIZE > 1)."
        )
        cleanup_distributed(dist_ctx)
        sys.exit(1)

    set_seed(args.seed + dist_ctx.rank)

    from src.pikogpt.data.dataset import MemmapDataset
    from src.pikogpt.model.transformer import GPTModel
    from src.pikogpt.training.trainer import Trainer

    logger.info("=== Stage: Training on %s ===", dist_ctx.device)
    model = GPTModel(config.model.model_dump())
    ctx_len = config.model.context_length
    train_ds = MemmapDataset(config.data.train_path, ctx_len)

    val_ds = None
    if config.data.val_path:
        val_ds = MemmapDataset(config.data.val_path, ctx_len)

    trainer = Trainer(
        model,
        train_ds,
        config.training.model_dump(),
        dist_ctx.device,
        val_dataset=val_ds,
        model_config=config.model.model_dump(),
        resume_from=args.resume,
        distributed_context=dist_ctx,
        azure_config=config.azure,
    )
    trainer.train()


def stage_train(args: argparse.Namespace) -> None:
    _run_train_stage(args, require_ddp=False)


def stage_train_ddp(args: argparse.Namespace) -> None:
    _run_train_stage(args, require_ddp=True)


def stage_posttraining(args: argparse.Namespace) -> None:
    from src.pikogpt.utils.config import load_config, merge_cli_overrides
    from src.pikogpt.utils.seed import set_seed

    if not args.config:
        logger.error("--config is required for post-training.")
        sys.exit(1)
    if not args.checkpoint:
        logger.error("--checkpoint is required for post-training.")
        sys.exit(1)

    config = load_config(args.config)
    config = merge_cli_overrides(config, args.overrides)
    if config.posttraining is None:
        logger.error(
            "Config %s has no [posttraining] section.", args.config
        )
        sys.exit(1)

    device = resolve_device(args.device)
    set_seed(args.seed)

    from src.pikogpt.model.transformer import GPTModel
    from src.pikogpt.posttraining.sft import run_sft
    from src.pikogpt.training.checkpoint import load_checkpoint

    logger.info("=== Stage: Post-Training on %s ===", device)
    checkpoint_meta = load_checkpoint(args.checkpoint, model=None, device=device)
    config_model = config.model.model_dump()
    resolved_model_config = checkpoint_meta.get("model_config") or config_model
    if checkpoint_meta.get("model_config") is None:
        logger.warning(
            "Checkpoint %s has no embedded model_config; falling back to [model] from %s.",
            args.checkpoint,
            args.config,
        )
    elif checkpoint_meta["model_config"] != config_model:
        logger.warning(
            "Checkpoint model_config differs from [model] in %s; using the checkpoint architecture.",
            args.config,
        )

    model = GPTModel(resolved_model_config)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    method = (config.posttraining.method or "sft").lower()
    if method != "sft":
        logger.error(
            "Post-training method %r not implemented yet "
            "(supported: 'sft').",
            method,
        )
        sys.exit(1)

    run_sft(
        model=model,
        config=config.posttraining.model_dump(),
        device=device,
        model_config=resolved_model_config,
        seed=args.seed,
    )


def stage_evaluate(args: argparse.Namespace) -> None:
    from src.pikogpt.utils.seed import set_seed

    if not args.checkpoint:
        logger.error("--checkpoint is required for evaluation.")
        sys.exit(1)

    device = resolve_device(args.device)
    set_seed(args.seed)

    from src.pikogpt.eval.benchmarks import run_all_benchmarks
    from src.pikogpt.eval.perplexity import evaluate_perplexity
    from src.pikogpt.model.transformer import GPTModel
    from src.pikogpt.training.checkpoint import load_checkpoint

    logger.info("=== Stage: Evaluation on %s ===", device)
    # Peek at checkpoint metadata to get model architecture config
    checkpoint_meta = load_checkpoint(args.checkpoint, model=None, device=device)
    model_config = checkpoint_meta["model_config"]
    if model_config is None:
        if not args.config:
            logger.error(
                "Checkpoint missing model_config and no --config provided. "
                "Pass --config to specify the model architecture."
            )
            sys.exit(1)
        from src.pikogpt.utils.config import load_config, merge_cli_overrides

        cfg = load_config(args.config)
        cfg = merge_cli_overrides(cfg, args.overrides)
        model_config = cfg.model.model_dump()
        logger.info("Using model config from %s (older checkpoint format)", args.config)
    model = GPTModel(model_config)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    model.eval()

    ppl_owt = evaluate_perplexity(
        model, device=device, data_path="data/tokenized/test.bin"
    )
    logger.info("OpenWebText test PPL: %.2f", ppl_owt)

    ppl_wiki = evaluate_perplexity(model, device=device, data_path="wikitext-103")
    logger.info("Wikitext-103 test PPL: %.2f", ppl_wiki)

    benchmark_results = run_all_benchmarks(model, device)
    for name, metrics in benchmark_results.items():
        logger.info("%s: %s", name, metrics)

    # Save results to eval_outputs/
    from src.pikogpt.eval.results import save_eval_results

    save_eval_results(
        checkpoint_path=args.checkpoint,
        meta=checkpoint_meta,
        model_config=model_config,
        n_params=sum(p.numel() for p in model.parameters()),
        perplexity={"openwebtext_test": ppl_owt, "wikitext_103": ppl_wiki},
        benchmarks=benchmark_results,
    )


def stage_inference(args: argparse.Namespace) -> None:
    """Mandatory inference contract entry point."""
    if not args.checkpoint:
        logger.error("--checkpoint is required for inference.")
        sys.exit(1)
    if not args.prompt:
        logger.error("--prompt is required for inference.")
        sys.exit(1)

    from src.pikogpt.inference.generate import generate

    device = resolve_device(args.device)
    chat_template = args.chat_template or "raw"

    output = generate(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=str(device),
        seed=args.seed,
        config_path=args.config,
        chat_template=chat_template,
    )

    if args.leaderboard:
        # Raw output only — no banners, no logging
        print(output)
        sys.exit(0)

    logger.info("=== Generated Text ===")
    print(output)


def stage_demo(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        logger.error("--checkpoint is required for demo.")
        sys.exit(1)

    from src.pikogpt.demo.chat import launch_demo

    # Demo defaults to the Alpaca template since the main use case post-SFT is
    # instruction following. Pass --chat-template raw to disable templating, or
    # --chat-template simple for a pre-SFT model.
    demo_template = args.chat_template or "alpaca"
    launch_demo(
        checkpoint_path=args.checkpoint,
        device=args.device or "cpu",
        share=args.share,
        chat_template=demo_template,
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

STAGE_HANDLERS = {
    "data": stage_data,
    "train": stage_train,
    "train_ddp": stage_train_ddp,
    "posttraining": stage_posttraining,
    "evaluate": stage_evaluate,
    "inference": stage_inference,
    "demo": stage_demo,
}


def main():
    import logging as _logging

    from src.pikogpt.utils.logging import setup_logging

    parser = build_parser()
    args = parser.parse_args()

    # Initialise logging — use config levels if a config file is provided
    logging_kwargs: dict = {}
    if args.config:
        try:
            from src.pikogpt.utils.config import load_config

            _cfg = load_config(args.config)
            logging_kwargs = {
                "root_level": _cfg.logging.root_level,
                "component_levels": _cfg.logging.levels,
            }
        except Exception:
            pass  # fall back to defaults
    setup_logging(**logging_kwargs)

    # Leaderboard mode: redirect all log output to stderr so stdout contains
    # only the generated completion (as required by the evaluation harness).
    if getattr(args, "leaderboard", False):
        piko_root = _logging.getLogger("pikogpt")
        for _h in piko_root.handlers:
            if isinstance(_h, _logging.StreamHandler):
                _h.stream = sys.stderr

    handler = STAGE_HANDLERS[args.stage]
    handler(args)


if __name__ == "__main__":
    main()
