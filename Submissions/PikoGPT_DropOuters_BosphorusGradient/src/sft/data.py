from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


def format_alpaca_prompt(instruction: str, input_text: str | None = None) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def normalize_instruction_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalised = []
    for rec in records:
        # Chat / messages format: [{"role": "user", ...}, {"role": "assistant", ...}]
        if "messages" in rec:
            msgs = rec["messages"]
            user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user"]
            asst_msgs = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
            if not user_msgs or not asst_msgs:
                continue
            instruction = user_msgs[0].strip()
            response = asst_msgs[0].strip()
            input_text = ""
        else:
            instruction = (rec.get("instruction") or rec.get("prompt") or "").strip()
            input_text = (rec.get("input") or rec.get("context") or "").strip()
            response = (
                rec.get("output")
                or rec.get("response")
                or rec.get("answer")
                or rec.get("completion")
                or ""
            )
            response = response.strip()

        if instruction and response:
            item = {
                "instruction": instruction,
                "input": input_text,
                "response": response,
            }
            for key in ("category", "answer_letter"):
                value = (rec.get(key) or "").strip()
                if value:
                    item[key] = value
            # Auto-detect answer_letter and category from MCQ responses like "A) text"
            if "answer_letter" not in item:
                import re as _re
                m = _re.match(r"^([A-D])\b", response)
                if m:
                    item["answer_letter"] = m.group(1).upper()
            # Auto-detect MCQ category when not explicitly set
            if "category" not in item and "answer_letter" in item:
                item["category"] = "multiple_choice"
            normalised.append(item)

    return normalised


def load_instruction_records(path: pathlib.Path, seed: int = 42, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SFT dataset file: {path}")

    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data["data"] if isinstance(data, dict) and "data" in data else data
    else:
        raise ValueError(f"Unsupported SFT dataset format: {path.suffix}")

    normalised = normalize_instruction_records(records)

    if limit and limit > 0 and len(normalised) > limit:
        rng = random.Random(seed)
        rng.shuffle(normalised)
        normalised = normalised[:limit]

    if not normalised:
        raise RuntimeError(f"No valid instruction-response examples found in {path}")
    return normalised


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


class AlpacaSFTDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], tokenizer, block_size: int):
        self.records = records
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        prompt = format_alpaca_prompt(rec["instruction"], rec.get("input"))
        response = rec["response"]

        # We manually trim to block_size below. Avoid tokenizer max-length warnings
        # by using the raw tokenization path instead of encode().
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, verbose=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False, verbose=False)["input_ids"]
        response_ids = response_ids + [self.tokenizer.eos_token_id]

        # Preserve as much of the response as possible; trim prompt first.
        if len(response_ids) >= self.block_size:
            response_ids = response_ids[: self.block_size - 1] + [self.tokenizer.eos_token_id]
            prompt_ids = []
        else:
            prompt_budget = self.block_size - len(response_ids)
            prompt_ids = prompt_ids[-prompt_budget:]

        full_ids = prompt_ids + response_ids
        if len(full_ids) < 2:
            raise RuntimeError("SFT example is too short after tokenization")

        # Autoregressive SFT:
        # - inputs are the sequence shifted right by one token
        # - targets are the next tokens
        # - loss is computed only when the *target* token belongs to the response
        input_ids = full_ids[:-1]
        next_tokens = full_ids[1:]

        # If prompt has length p, then target positions 0..p-2 still predict prompt
        # tokens and must be masked. Position p-1 predicts the first response token
        # and should contribute to the loss.
        prompt_prefix = max(0, len(prompt_ids) - 1)
        labels = ([-100] * prompt_prefix) + next_tokens[prompt_prefix:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class AlpacaCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            ids = item["input_ids"]
            lab = item["labels"]
            valid_len = ids.size(0)
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            input_ids.append(ids)
            labels.append(lab)
            mask = torch.zeros(max_len, dtype=torch.long)
            mask[:valid_len] = 1
            attention_mask.append(mask)
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }
