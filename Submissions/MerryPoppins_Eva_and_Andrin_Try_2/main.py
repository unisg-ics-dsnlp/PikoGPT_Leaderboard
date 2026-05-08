#!/usr/bin/env python3
"""
PikoGPT leaderboard inference entry point.

Usage (leaderboard):
    python main.py --stage inference \
        --checkpoint runs/best_ckpt_dto.pt \
        --prompt "..." \
        --max-tokens 3 \
        --temperature 0 \
        --device auto \
        --leaderboard \
        --seed 0
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    n_embd: int = 384
    embedding_dim: Optional[int] = None
    tie_embeddings: bool = True
    context_len: int = 1024
    dropout: float = 0.0
    bias: bool = False
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    positional_embedding: str = "learned"
    rope_theta: float = 10_000.0
    rope_fraction: float = 1.0
    mlp_type: str = "gelu"
    mlp_hidden_mult: float = 4.0
    mlp_hidden_dim: Optional[int] = None
    qk_norm: bool = False
    block_style: str = "sequential"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight.float()).to(dtype=x.dtype)


def _build_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.n_embd, eps=config.norm_eps, bias=config.bias)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    raise ValueError(f"Unsupported norm_type: {config.norm_type}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        cos = angles.cos().to(dtype=dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self(q.size(-2), device=q.device, dtype=q.dtype)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads else config.n_heads
        self.n_groups = self.n_heads // self.n_kv_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_heads
        self.qk_norm = config.qk_norm
        self.rotary = None
        self.rotary_dim = 0
        if config.positional_embedding == "rope":
            rotary_dim = min(self.head_dim, int(round(self.head_dim * config.rope_fraction)))
            if rotary_dim % 2 != 0:
                rotary_dim -= 1
            if rotary_dim >= 2:
                self.rotary_dim = rotary_dim
                self.rotary = RotaryEmbedding(self.rotary_dim, base=config.rope_theta)

        if self.n_kv_heads == self.n_heads:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        else:
            self.q_proj = nn.Linear(config.n_embd, self.n_heads * self.head_dim, bias=config.bias)
            self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
            self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        if self.n_kv_heads == self.n_heads:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
        if self.rotary is not None:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot, k_rot = self.rotary.apply_rotary(q_rot, k_rot)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        if self.qk_norm:
            scale = math.sqrt(self.head_dim)
            q = F.normalize(q, dim=-1, eps=1e-6) * scale
            k = F.normalize(k, dim=-1, eps=1e-6) * scale

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_hidden_dim if config.mlp_hidden_dim else max(1, int(round(config.mlp_hidden_mult * config.n_embd)))
        self.mlp_type = config.mlp_type
        if config.mlp_type == "gelu":
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        elif config.mlp_type == "swiglu":
            self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)
        else:
            raise ValueError(f"Unsupported mlp_type: {config.mlp_type}")
        self.hidden_dim = hidden_dim
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "gelu":
            x = F.gelu(self.c_fc(x))
        else:
            gate, value = self.c_fc(x).chunk(2, dim=-1)
            x = F.silu(gate) * value
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.block_style = config.block_style
        self.ln_1 = _build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _build_norm(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_style == "parallel":
            return x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        transformer_layers: dict[str, nn.Module] = {
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            "ln_f": _build_norm(config),
        }
        if config.positional_embedding == "learned":
            transformer_layers["wpe"] = nn.Embedding(config.context_len, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_layers)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.context_len, f"Sequence length {T} exceeds context_len {self.config.context_len}"
        x = self.transformer["wte"](idx)
        if self.config.positional_embedding == "learned":
            pos = torch.arange(T, dtype=torch.long, device=idx.device)
            x = x + self.transformer["wpe"](pos)
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        eos_token_id = 50256
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == eos_token_id:
                break
        return idx


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(model_cfg).to(device)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    if args.temperature == 0.0:
        with torch.no_grad():
            if args.leaderboard and args.max_tokens <= 3:
                # MC leaderboard: pick the answer letter with the highest logit.
                # Restricted logit scoring over A/B/C/D is more reliable than
                # free-form generation for a pretrained (non-instruction-tuned) model.
                mc_letters = "ABCD"
                mc_ids = [tokenizer.encode(f" {c}")[0] for c in mc_letters]
                idx_cond = input_ids[:, -model_cfg.context_len:]
                logits, _ = model(idx_cond)
                best = int(logits[0, -1, mc_ids].argmax())
                output_text = mc_letters[best]
            else:
                eos_token_id = 50256
                for _ in range(args.max_tokens):
                    idx_cond = input_ids[:, -model_cfg.context_len:]
                    logits, _ = model(idx_cond)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat((input_ids, next_token), dim=1)
                    if next_token.item() == eos_token_id:
                        break
                prompt_len = len(tokenizer.encode(args.prompt))
                generated_ids = input_ids[0][prompt_len:].tolist()
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        input_ids = model.generate(input_ids, max_new_tokens=args.max_tokens, temperature=args.temperature)
        prompt_len = len(tokenizer.encode(args.prompt))
        generated_ids = input_ids[0][prompt_len:].tolist()
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if args.leaderboard:
        sys.stdout.write(output_text)
        sys.stdout.flush()
    else:
        print(output_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pikogpt", description="PikoGPT inference")
    parser.add_argument("--stage", required=True, choices=["inference"], help="Pipeline stage")
    parser.add_argument("--checkpoint", required=True, type=Path, metavar="CKPT.pt")
    parser.add_argument("--prompt", default="", metavar="TEXT")
    parser.add_argument("--max-tokens", type=int, default=100, dest="max_tokens", metavar="N")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_inference(args)
    except Exception as exc:
        if args.leaderboard:
            sys.exit(1)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
