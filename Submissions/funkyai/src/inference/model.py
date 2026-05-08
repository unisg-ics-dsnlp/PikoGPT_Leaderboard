from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_ACTIVATIONS: dict[str, Any] = {
    "gelu": "gelu",
    "relu": "relu",
    "silu": F.silu,
    "tanh": torch.tanh,
}


def _resolve_activation(name: str) -> Any:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{name}'. Choose one of: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        dropout: float,
        layer_norm_epsilon: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation=_resolve_activation(activation),
            layer_norm_eps=layer_norm_epsilon,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_seq_len {self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = self._build_causal_mask(seq_len=seq_len, device=input_ids.device)
        x = self.transformer(x, mask=causal_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss}
