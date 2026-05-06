from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .block import TransformerBlock
from .embedding import EmbeddingLayer


class GPTTiny(nn.Module):
    """A small, general GPT-style language model.

    Flow:
      token ids -> token embeddings -> dropout -> transformer blocks (RoPE inside attn) -> final LN -> vocab logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int,
        mlp_ratio: int,
        n_heads: int,
        n_decoders: int,
        drop_rate: float,
        qkv_bias: bool,
        n_kv_heads: int | None = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.embedding = EmbeddingLayer(vocab_size, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

        # SwiGLU uses 3 projections vs 2 for GELU. To keep total FFN params equal:
        # 2 * (emb_dim * 4*emb_dim) = 3 * (emb_dim * ff_hidden_size)  →  ff = 8/3 * emb_dim
        # Round to nearest multiple of 64 for memory alignment.
        ff_hidden_size = round(mlp_ratio * emb_dim * 2 / 3 / 64) * 64
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_size=emb_dim,
                num_heads=n_heads,
                ff_hidden_size=ff_hidden_size,
                n_kv_heads=n_kv_heads,
                dropout=drop_rate,
                qkv_bias=qkv_bias,
                max_seq_length=context_length,
            )
            for _ in range(n_decoders)
        ])

        self.ln_f = nn.RMSNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.embedding.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = True,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.embedding(input_ids)
        x = self.dropout(x)

        new_key_values = [] if use_cache else None
        if past_key_values is None:
            past_key_values = [None] * len(self.transformer_blocks)

        for block, past_kv in zip(self.transformer_blocks, past_key_values):
            position_offset = past_kv[0].size(2) if past_kv is not None else 0
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, padding_mask, causal, use_reentrant=False)
            else:
                block_out = block(
                    x,
                    padding_mask=padding_mask,
                    causal=causal,
                    past_kv=past_kv,
                    use_cache=use_cache,
                    position_offset=position_offset,
                )
                if use_cache:
                    x, present_kv = block_out
                    new_key_values.append(present_kv)
                else:
                    x = block_out

        x = self.ln_f(x)
        logits = self.fc_out(x)
        if use_cache:
            return logits, new_key_values
        return logits
