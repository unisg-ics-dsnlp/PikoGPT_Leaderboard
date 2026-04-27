"""Save and load model checkpoints with training metadata.

Checkpoints are PyTorch .pt files (serialised dicts) that store everything
needed to resume training or run inference:

  - model_state_dict:     all model weights (~150 MB for a 37M-param model)
  - optimizer_state_dict: AdamW momentum buffers (~300 MB — 2× model size
                          because AdamW stores two moving averages per param)
  - step, config, val_loss, tokens_seen: metadata for reproducibility

What's NOT saved (and why):
  - LR scheduler: the cosine schedule is deterministic — given the step number
    you can recompute exactly where you are.  On resume, the Trainer replays
    scheduler.step() to the correct position.
  - GradScaler: it auto-calibrates within a few steps of resuming.

Three types of checkpoints:
  - Periodic (step_003000.pt): full state (model + optimiser) for resuming.
  - Best (best_step_003000.pt): top-K models by val_loss (weights only).
  - best.pt: symlink (or copy) pointing to the overall best checkpoint.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_save(checkpoint: dict, path: Path) -> None:
    """Write checkpoint atomically: tmp file + os.replace (single rename syscall)."""
    tmp_path = path.with_suffix(".pt.tmp")
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Save / load / prune
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    step: int,
    config: dict,
    path: str | Path,
    val_loss: float | None = None,
    tokens_seen: int = 0,
    model_config: dict | None = None,
    uploader=None,
) -> Path:
    """Save a training checkpoint to disk.

    Args:
        model: The model to save.
        optimizer: The optimiser state (None for inference-only / best-model checkpoints).
        step: Current training step.
        config: Full training config dict (saved for reproducibility).
        path: Output file path (.pt).
        val_loss: Current best validation loss (optional).
        tokens_seen: Total tokens processed so far.
        model_config: Model architecture config dict (needed to reconstruct the model).

    Returns:
        Path to the saved checkpoint file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        # optimizer_state_dict is None for best-model checkpoints (lighter files,
        # and post-training creates its own optimiser anyway)
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "step": step,
        "config": config,
        "model_config": model_config,
        "val_loss": val_loss,
        "tokens_seen": tokens_seen,
    }

    _atomic_save(checkpoint, path)
    logger.info("Checkpoint saved: %s (step=%d, val_loss=%s)", path, step, val_loss)
    if uploader is not None:
        uploader.upload(path, is_best=False)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Load a training checkpoint and restore model/optimiser state.

    Args:
        path: Path to .pt checkpoint file.
        model: Model to load weights into (must have matching architecture).
            Pass None to peek at metadata without loading weights.
        optimizer: Optimiser to restore state into (pass None to skip, e.g. for inference).
        device: Device to map tensors to (e.g. "cpu" or "cuda").

    Returns:
        Metadata dict with keys: step, config, model_config, val_loss, tokens_seen.
    """
    path = Path(path)
    # map_location ensures tensors are loaded onto the right device
    # (e.g. if checkpoint was saved on GPU but we're loading on CPU)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model weights loaded from %s", path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        opt_state = checkpoint["optimizer_state_dict"]
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            logger.info("Optimizer state restored")

    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config", {}),
        "model_config": checkpoint.get("model_config"),
        "val_loss": checkpoint.get("val_loss"),
        "tokens_seen": checkpoint.get("tokens_seen", 0),
    }


def prune_checkpoints(checkpoint_dir: str | Path, max_keep: int = 3) -> None:
    """Keep only the most recent periodic checkpoints, deleting older ones.

    Only prunes files matching ``step_*.pt``.  The ``best.pt`` file is
    never touched — it's the best model for evaluation/post-training.

    This prevents disk usage from growing unboundedly during long training
    runs.  At our model size, each full checkpoint is ~500 MB, so keeping
    3 means ~1.5 GB of disk.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        max_keep: Maximum number of periodic checkpoints to retain.
    """
    ckpt_dir = Path(checkpoint_dir)
    # Sort by modification time so we keep the most recent ones
    periodic = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)

    if len(periodic) <= max_keep:
        return

    to_remove = periodic[: len(periodic) - max_keep]
    for p in to_remove:
        p.unlink()
        logger.info("Pruned old checkpoint: %s", p.name)


# ---------------------------------------------------------------------------
# Top-K best checkpoints
# ---------------------------------------------------------------------------


def _update_best_link(checkpoint_dir: Path, target_path: Path) -> None:
    """Make best.pt point to the given target file (symlink with copy fallback)."""
    link = checkpoint_dir / "best.pt"
    try:
        tmp_link = link.with_suffix(".pt.lnk")
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(target_path.name)  # relative symlink
        os.replace(tmp_link, link)
    except OSError:
        import shutil

        shutil.copy2(target_path, link)


def save_best_checkpoint(
    model: nn.Module,
    step: int,
    config: dict,
    checkpoint_dir: str | Path,
    val_loss: float,
    tokens_seen: int = 0,
    model_config: dict | None = None,
    max_best: int = 3,
    uploader=None,
) -> Path | None:
    """Save a best checkpoint if val_loss is in the top-K.

    Stateless/resumable: rediscovers existing best_step_*.pt files from disk
    on every call — no in-memory manifest needed.

    Args:
        model: The model to save.
        step: Current training step.
        config: Full training config dict.
        checkpoint_dir: Directory for checkpoint files.
        val_loss: Validation loss for this checkpoint.
        tokens_seen: Total tokens processed so far.
        model_config: Model architecture config dict.
        max_best: Maximum number of best checkpoints to keep.

    Returns:
        Path to saved checkpoint, or None if val_loss is not in top-K.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Discover existing best checkpoints and read their val_loss metadata
    existing: list[tuple[float, Path]] = []
    for p in ckpt_dir.glob("best_step_*.pt"):
        try:
            meta = torch.load(p, map_location="cpu", weights_only=False)
            loss = meta.get("val_loss")
            if loss is not None:
                existing.append((loss, p))
        except Exception:
            logger.warning("Could not read best checkpoint %s, skipping", p.name)

    existing.sort(key=lambda t: t[0])  # ascending = best first

    # Check if new val_loss qualifies for top-K
    if len(existing) >= max_best and val_loss >= existing[-1][0]:
        return None  # not in top-K

    # Save new best checkpoint (model weights only, no optimizer)
    new_path = ckpt_dir / f"best_step_{step:06d}.pt"
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "step": step,
        "config": config,
        "model_config": model_config,
        "val_loss": val_loss,
        "tokens_seen": tokens_seen,
    }
    _atomic_save(checkpoint, new_path)
    if uploader is not None:
        uploader.upload(new_path, is_best=True)

    # Add to list, re-sort, prune if over max_best
    existing.append((val_loss, new_path))
    existing.sort(key=lambda t: t[0])

    while len(existing) > max_best:
        _loss, worst_path = existing.pop()
        worst_path.unlink(missing_ok=True)
        logger.info("Pruned best checkpoint: %s (val_loss=%.4f)", worst_path.name, _loss)

    # Update best.pt to point to the overall best
    _update_best_link(ckpt_dir, existing[0][1])
    if uploader is not None:
        uploader.upload(ckpt_dir / "best.pt", is_best=True)

    return new_path
