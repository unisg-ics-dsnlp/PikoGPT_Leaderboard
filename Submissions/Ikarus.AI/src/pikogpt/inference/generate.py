"""Text generation for inference — the canonical entry point for the leaderboard."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from pikogpt.model.transformer import GPTModel
from pikogpt.tokenizer import decode_tokens, encode_text
from pikogpt.training.checkpoint import load_checkpoint
from pikogpt.utils.seed import set_seed

logger = logging.getLogger(__name__)


def generate(
    checkpoint_path: str | Path,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "cpu",
    seed: int = 42,
    config_path: str | Path | None = None,
    chat_template: str = "raw",
) -> str:
    """Generate text from a prompt using a saved checkpoint.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        prompt: Text prompt to continue from.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: If set, restrict sampling to top-k tokens.
        device: Device string ('cpu', 'cuda', 'mps').
        seed: Random seed for reproducibility.
        config_path: Fallback config file for older checkpoints lacking model_config.
        chat_template: ``"raw"`` prepends no template; ``"alpaca"`` wraps the
            prompt with the Stanford-Alpaca ``### Instruction / ### Response``
            markers used during SFT and stops at ``<|endoftext|>``.

    Returns:
        When ``chat_template="alpaca"``: just the generated response text
        (EOS stripped). Otherwise: the full prompt + continuation string.
    """
    set_seed(seed)
    dev = torch.device(device)

    # 1) Peek at checkpoint to get model architecture config
    meta = load_checkpoint(checkpoint_path, model=None, device=dev)
    model_config = meta["model_config"]
    if model_config is None:
        if config_path is None:
            raise ValueError(
                "Checkpoint missing model_config and no config_path provided. "
                "Pass config_path to specify the model architecture."
            )
        from pikogpt.utils.config import load_config

        cfg = load_config(str(config_path))
        model_config = cfg.model.model_dump()
        logger.info("Using model config from %s (older checkpoint format)", config_path)

    # 2) Build model and load weights
    model = GPTModel(model_config)
    load_checkpoint(checkpoint_path, model, device=dev)
    model.to(dev)
    model.eval()

    # 3) Resolve chat template — auto-detect from checkpoint when caller passes "raw"
    template = (chat_template or "raw").lower()
    if template == "raw":
        saved_template = (model_config or {}).get("chat_template", "")
        if saved_template:
            template = saved_template.lower()
            logger.debug("Auto-detected chat_template=%r from checkpoint", template)

    eos_id: int | None = None
    if template == "alpaca":
        from pikogpt.posttraining.sft_dataset import (
            EOS_TOKEN_ID,
            format_alpaca_prompt,
        )

        prompt_text = format_alpaca_prompt(prompt, "")
        eos_id = EOS_TOKEN_ID
    else:
        prompt_text = prompt

    # 4) Tokenize
    prompt_ids = encode_text(prompt_text)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    # 5) Generate (stops early on EOS when using the Alpaca template)
    output_ids = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_id,
    )

    # 6) Decode — return only the newly generated tokens
    output_list = output_ids[0].tolist()
    new_ids = output_list[len(prompt_ids):]
    if eos_id is not None and new_ids and new_ids[-1] == eos_id:
        new_ids = new_ids[:-1]

    if template == "alpaca":
        return decode_tokens(new_ids).strip()

    return decode_tokens(output_list)
