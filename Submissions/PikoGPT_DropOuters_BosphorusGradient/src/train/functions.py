"""Training utilities extracted from notebook 03."""

from __future__ import annotations

import math
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config.app_config import load_app_config
from src.gpt_arch import GPTTiny


def load_config(root: pathlib.Path) -> Dict[str, Any]:
    return load_app_config(root)


@dataclass
class TrainingConfig:
    # Data
    data_dir: pathlib.Path
    train_bin: pathlib.Path
    val_bin: pathlib.Path
    test_bin: pathlib.Path

    # Runtime
    device: str
    num_workers: int
    num_gpus: int
    seed: int

    # Model
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    block_size: int
    dropout: float
    use_gradient_checkpointing: bool

    # Training
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: float
    max_iters: int
    learning_rate: float
    min_lr_ratio: float
    min_lr: float
    warmup_ratio: float
    warmup_iters: int
    lr_decay_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float

    # Eval / logging
    eval_interval: int
    eval_iters: int
    sample_interval: int
    sample_prompt: str

    # Checkpointing
    checkpoint_dir: pathlib.Path
    save_interval: int
    auto_resume: bool
    early_stop: bool
    min_delta: float


MAX_ALLOWED_LAYERS = 24
MAX_ALLOWED_PARAMS = 40_000_000
MAX_CONTEXT_SIZE = 1024


def build_config(cfg: Dict[str, Any]) -> TrainingConfig:
    train_cfg = cfg.get("train", {})
    data_dir = pathlib.Path(train_cfg.get("data_dir", "data/cleaned"))
    n_head = int(train_cfg.get("n_head", 8))
    default_n_kv_head = max(1, n_head // 2)
    if n_head % default_n_kv_head != 0:
        default_n_kv_head = 1
    n_kv_head = int(train_cfg.get("n_kv_head", default_n_kv_head))
    if n_head % n_kv_head != 0:
        raise ValueError(f"train.n_head ({n_head}) must be divisible by train.n_kv_head ({n_kv_head})")

    config = TrainingConfig(
        # ---- Data ----
        data_dir=data_dir,
        train_bin=data_dir / train_cfg.get("train_bin", "train.bin"),
        val_bin=data_dir / train_cfg.get("val_bin", "val.bin"),
        test_bin=data_dir / train_cfg.get("test_bin", "test.bin"),

        # ---- Runtime ----
        device=train_cfg.get("device", "cuda"),
        num_workers=int(train_cfg.get("num_workers", 0)),
        num_gpus=int(train_cfg.get("num_gpus", 1)),
        seed=int(train_cfg.get("seed", 42)),

        # ---- Model ----
        vocab_size=int(train_cfg.get("vocab_size", 50257)),
        n_layer=int(train_cfg.get("n_layer", 6)),
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=int(train_cfg.get("n_embd", 320)),
        block_size=int(train_cfg.get("block_size", 256)),
        dropout=float(train_cfg.get("dropout", 0.0)),
        use_gradient_checkpointing=bool(train_cfg.get("use_gradient_checkpointing", False)),

        # ---- Training ----
        batch_size=int(train_cfg.get("batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        num_epochs=float(train_cfg.get("num_epochs", 1)),
        max_iters=int(train_cfg.get("max_iters", 82000)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
        min_lr=float(train_cfg.get("learning_rate", 3e-4)) * float(train_cfg.get("min_lr_ratio", 0.1)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.02)),
        warmup_iters=int(train_cfg.get("warmup_iters", 1000)),
        lr_decay_iters=int(train_cfg.get("lr_decay_iters", 82000)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        beta1=float(train_cfg.get("beta1", 0.9)),
        beta2=float(train_cfg.get("beta2", 0.95)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),

        # ---- Eval / logging ----
        eval_interval=int(train_cfg.get("eval_interval", 500)),
        eval_iters=int(train_cfg.get("eval_iters", 20)),
        sample_interval=int(train_cfg.get("sample_interval", 5000)),
        sample_prompt=str(train_cfg.get("sample_prompt", "The future of")),

        # ---- Checkpointing ----
        checkpoint_dir=pathlib.Path(train_cfg.get("checkpoint_dir", "runs/checkpoints")),
        save_interval=int(train_cfg.get("save_interval", 2000)),
        auto_resume=bool(train_cfg.get("auto_resume", True)),
        early_stop=bool(train_cfg.get("early_stop", True)),
        min_delta=float(train_cfg.get("min_delta", 0.0)),
    )
    _validate_architecture_constraints(config)
    return config


def _validate_architecture_constraints(config: TrainingConfig) -> None:
    if config.block_size > MAX_CONTEXT_SIZE:
        raise ValueError(
            f"train.block_size must be <= {MAX_CONTEXT_SIZE} (got {config.block_size})"
        )
    if config.n_layer > MAX_ALLOWED_LAYERS:
        raise ValueError(
            f"train.n_layer must be <= {MAX_ALLOWED_LAYERS} (got {config.n_layer})"
        )


class TokenDataset(Dataset):
    """
    Memory-mapped dataset over a pre-tokenised flat uint16 binary file.
    No RAM loading, no tokenization. __getitem__ is a single numpy slice.
    """

    def __init__(self, bin_path: pathlib.Path, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_chunks = (len(self.data) - 1) // block_size
        print(f"  {bin_path.name:<12}  {len(self.data):>12,} tokens  {self.n_chunks:>10,} chunks")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = torch.from_numpy(self.data[start : start + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def worker_init_fn(worker_id: int) -> None:
    """Unique seed per worker to avoid identical sampling."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    for p in (config.train_bin, config.val_bin):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run the pre-tokenisation step from notebook 01 first."
            )

    print("Mapping binary files:")
    train_dataset = TokenDataset(config.train_bin, config.block_size)
    val_dataset = TokenDataset(config.val_bin, config.block_size)

    pf = 2 if config.num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        worker_init_fn=worker_init_fn,
        prefetch_factor=pf,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        worker_init_fn=worker_init_fn,
        prefetch_factor=pf,
    )

    return train_loader, val_loader


def build_model(config: TrainingConfig) -> torch.nn.Module:
    _validate_architecture_constraints(config)
    model = GPTTiny(
        vocab_size=config.vocab_size,
        context_length=config.block_size,
        emb_dim=config.n_embd,
        mlp_ratio=4,
        n_heads=config.n_head,
        n_kv_heads=config.n_kv_head,
        n_decoders=config.n_layer,
        drop_rate=config.dropout,
        qkv_bias=False,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
    ).to(config.device)

    if config.num_gpus > 1:
        model = torch.nn.DataParallel(model)
        gpu_ids = list(range(config.num_gpus))
        print(f"DataParallel enabled - GPUs: {gpu_ids}")

    n_params = sum(p.numel() for p in model.parameters())
    if n_params > MAX_ALLOWED_PARAMS:
        raise ValueError(
            f"Model has {n_params:,} parameters, exceeding max allowed {MAX_ALLOWED_PARAMS:,}"
        )
    print(f"Parameters : {n_params/1e6:.2f}M")
    print(f"Device     : {config.device}")
    return model


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    print("Optimizer: AdamW")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    return optimizer


def get_lr(iter_num: int, config: TrainingConfig) -> float:
    """Cosine decay with linear warmup."""
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def forward_logits_and_loss(model, x, y, tokenizer, config: TrainingConfig):
    # Packed memmap LM training uses contiguous token streams without padding.
    # GPT-2 reuses EOS as pad_token, so masking pad_token_id would incorrectly
    # hide valid EOS tokens from attention and loss.
    padding_mask = None
    logits = model(x, padding_mask=padding_mask, causal=True)
    loss = None
    if y is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
    return logits, loss


@torch.no_grad()
def estimate_metrics(model, data_loader: Iterable, tokenizer, config: TrainingConfig) -> Dict[str, float]:
    """Returns {"loss": float, "acc": float} averaged over eval_iters batches."""
    model.eval()
    losses, accs = [], []
    for i, (x, y) in enumerate(data_loader):
        if i >= config.eval_iters:
            break
        x, y = x.to(config.device), y.to(config.device)
        logits, loss = forward_logits_and_loss(model, x, y, tokenizer, config)
        losses.append(loss.item())
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        accs.append(acc)
    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
    }


@torch.no_grad()
def quick_sample(model, tokenizer, config: TrainingConfig, prompt: str | None = None, max_new_tokens: int = 50):
    prompt = prompt or config.sample_prompt
    m = _raw_model(model)
    m.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -config.block_size :]
        logits = m(ids_cond, causal=True)
        logits = logits[:, -1, :] / 0.8
        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = -float("inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    m.train()
    return tokenizer.decode(ids[0].tolist())


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
    block_size: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    top_k: int = 0,
    top_p: float = 1.0,
    use_kv_cache: bool = False,
) -> str:
    """Generate text with configurable decoding controls."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    m = _raw_model(model)
    m.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = None
    next_input = None
    for _ in range(max_new_tokens):
        if use_kv_cache and past_key_values is not None and next_input is not None:
            logits, past_key_values = m(
                next_input,
                causal=False,
                past_key_values=past_key_values,
                use_cache=True,
            )
        elif use_kv_cache:
            logits, past_key_values = m(ids[:, -block_size:], causal=True, use_cache=True)
        else:
            logits = m(ids[:, -block_size:], causal=True)
        logits = logits[:, -1, :]
        generated_ids = ids[0].tolist()

        if repetition_penalty != 1.0:
            for token_id in set(generated_ids):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty
        if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size - 1:
            prefix = tuple(generated_ids[-(no_repeat_ngram_size - 1) :])
            banned = set()
            for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
                gram = tuple(generated_ids[i : i + no_repeat_ngram_size])
                if gram[:-1] == prefix:
                    banned.add(gram[-1])
            if banned:
                logits[:, list(banned)] = -float("inf")
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                keep = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, keep, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                remove = torch.zeros_like(logits, dtype=torch.bool)
                remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
                logits = logits.masked_fill(remove, -float("inf"))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
        next_input = next_id
        if use_kv_cache and past_key_values and past_key_values[0][0].size(2) >= block_size:
            past_key_values = None
            next_input = None
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def save_checkpoint(model, optimizer, iteration: int, loss: float, path: pathlib.Path, verbose: bool = True) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": _raw_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "loss": loss,
        },
        path,
    )
    if verbose:
        print(f"Checkpoint saved -> {path}")


def load_checkpoint(model, optimizer, path: pathlib.Path, device: str, load_optimizer: bool = True):
    checkpoint = torch.load(path, map_location=device)
    _raw_model(model).load_state_dict(checkpoint["model_state_dict"])
    if load_optimizer and optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("iteration", 0), checkpoint.get("loss", float("nan"))


def find_latest_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path | None:
    if not checkpoint_dir.exists():
        return None

    preferred = [
        checkpoint_dir / "last_model.pt",
        checkpoint_dir / "final_model.pt",  # legacy fallback
        checkpoint_dir / "best_model.pt",
    ]
    for path in preferred:
        if path.exists():
            return path

    ckpts = sorted(checkpoint_dir.glob("checkpoint_iter_*.pt"))
    return ckpts[-1] if ckpts else None


def train_loop(
    model,
    optimizer,
    train_loader,
    val_loader,
    tokenizer,
    config: TrainingConfig,
    start_iter: int = 0,
    partial_results_path: pathlib.Path | None = None,
):
    use_amp = config.device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp, growth_interval=1000)

    tokens_per_iter = config.batch_size * config.block_size * config.gradient_accumulation_steps

    # FLOPs approximation: 6 × N_params × tokens per step (forward ≈ 2N, backward ≈ 4N)
    n_params = sum(p.numel() for p in _raw_model(model).parameters())
    flops_per_step = 6 * n_params * tokens_per_iter
    tflops_per_step = flops_per_step / 1e12

    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []
    train_accs, val_accs = [], []
    grad_norms = []
    iterations = []

    iter_num = start_iter
    micro_step = 0
    running_loss = 0.0
    total_flops = 0.0
    last_grad_norm = float("nan")
    last_step_time = float("nan")
    last_tokens_per_sec = float("nan")
    last_tflops_per_sec = float("nan")
    raw_loss = float("nan")
    nan_detected = False
    early_stop_triggered = False
    best_val_loss = float("inf")
    improved_since_last_save = False
    eval_since_last_save = False

    print("=" * 52)
    print("  Starting Training")
    print(f"  max_iters  : {config.max_iters:,}")
    print(
        f"  batch      : {config.batch_size} x block {config.block_size} x accum {config.gradient_accumulation_steps} = {tokens_per_iter:,} tok/step"
    )
    print(f"  LR         : warmup {config.warmup_iters} -> cosine -> {config.min_lr}")
    print(f"  eval every : {config.eval_interval} iters  ({config.eval_iters} val batches)")
    print(f"  sample every {config.sample_interval:,} iters")
    print(f"  TFLOPs/step: {tflops_per_step:.3f}  ({n_params/1e6:.1f}M params)")
    print("=" * 52)

    from tqdm.auto import tqdm

    progress_bar = tqdm(total=max(0, config.max_iters - start_iter), desc="Training")
    step_start = None

    while iter_num < config.max_iters:
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)

            if micro_step % config.gradient_accumulation_steps == 0:
                import time

                step_start = time.time()

            with ctx:
                _, loss = forward_logits_and_loss(model, x, y, tokenizer, config)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps == 0:
                lr = get_lr(iter_num, config)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                raw_loss = loss.item() * config.gradient_accumulation_steps

                if math.isnan(raw_loss) or math.isinf(raw_loss):
                    print("!" * 52)
                    print(f"  NaN/Inf loss at iter {iter_num}  gnorm={last_grad_norm:.2f}")
                    print("  Likely cause: LR too high or gradient explosion")
                    print("  Resume from last_model.pt if one was already saved")
                    print("!" * 52)
                    nan_detected = True
                    break

                running_loss += raw_loss

                if step_start is not None:
                    import time

                    last_step_time = time.time() - step_start
                last_tokens_per_sec = tokens_per_iter / last_step_time if last_step_time > 0 else float("nan")
                total_flops += flops_per_step
                last_tflops_per_sec = tflops_per_step / last_step_time if last_step_time > 0 else float("nan")

                if iter_num % config.eval_interval == 0 and iter_num > 0:
                    train_m = estimate_metrics(model, train_loader, tokenizer, config)
                    val_m = estimate_metrics(model, val_loader, tokenizer, config)
                    avg_train = train_m["loss"]
                    avg_val = val_m["loss"]
                    if avg_val < best_val_loss - config.min_delta:
                        best_val_loss = avg_val
                        improved_since_last_save = True
                        save_checkpoint(
                            model,
                            optimizer,
                            iter_num,
                            raw_loss,
                            config.checkpoint_dir / "best_model.pt",
                        )
                    eval_since_last_save = True
                    train_losses.append(avg_train)
                    val_losses.append(avg_val)
                    train_ppls.append(math.exp(avg_train))
                    val_ppls.append(math.exp(avg_val))
                    train_accs.append(train_m["acc"])
                    val_accs.append(val_m["acc"])
                    grad_norms.append(last_grad_norm)
                    iterations.append(iter_num)
                    running_loss = 0.0

                    postfix = {
                        "train": f"{avg_train:.4f}",
                        "val": f"{avg_val:.4f}",
                        "acc": f"{val_m['acc']:.3f}",
                        "ppl": f"{math.exp(avg_train):.1f}",
                        "lr": f"{lr:.2e}",
                        "gnorm": f"{last_grad_norm:.2f}",
                        "tflop/s": f"{last_tflops_per_sec:.2f}",
                    }
                    progress_bar.set_postfix(postfix)

                if iter_num % config.sample_interval == 0 and iter_num > 0:
                    sample = quick_sample(model, tokenizer, config)
                    print(f"\n[iter {iter_num}] {sample[:200]}\n")

                if iter_num % config.save_interval == 0 and iter_num > 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        iter_num,
                        raw_loss,
                        config.checkpoint_dir / "last_model.pt",
                    )
                    if partial_results_path is not None:
                        import json as _json
                        partial = {
                            "iterations": iterations,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "train_ppls": train_ppls,
                            "val_ppls": val_ppls,
                            "train_accs": train_accs,
                            "val_accs": val_accs,
                            "grad_norms": grad_norms,
                            "total_tflops": total_flops / 1e12,
                        }
                        partial_results_path.write_text(_json.dumps(partial, indent=2), encoding="utf-8")
                    if config.early_stop:
                        if eval_since_last_save and not improved_since_last_save:
                            print("Early stop: no validation loss improvement since the previous save window")
                            early_stop_triggered = True
                            break
                        improved_since_last_save = False
                        eval_since_last_save = False

                iter_num += 1
                progress_bar.update(1)

                if iter_num >= config.max_iters:
                    break

        if nan_detected or early_stop_triggered or iter_num >= config.max_iters:
            break

    progress_bar.close()

    if nan_detected:
        status = "nan"
        print("!" * 60)
        print("  STOPPED — NaN/Inf loss detected")
        print("  Lower the learning rate and resume from last_model.pt")
        print("!" * 60)
    elif early_stop_triggered:
        status = "early_stop"
        save_checkpoint(model, optimizer, iter_num, raw_loss, config.checkpoint_dir / "last_model.pt")
    else:
        status = "complete"
        save_checkpoint(model, optimizer, iter_num, raw_loss, config.checkpoint_dir / "last_model.pt")

    return {
        "status": status,
        "final_iter": iter_num,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else float("nan"),
        "iterations": iterations,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ppls": train_ppls,
        "val_ppls": val_ppls,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "grad_norms": grad_norms,
        "total_tflops": total_flops / 1e12,
    }


def build_tokenizer() -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    return tokenizer
