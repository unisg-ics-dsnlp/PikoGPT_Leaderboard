from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2TokenizerFast

from domain.scoring.base import ChoiceCandidate, build_scored_text, prepend_to_prompt


@dataclass(frozen=True)
class ChoiceScore:
    letter: str
    text: str
    log_likelihood: float
    token_count: int


class MultipleChoiceScorer:
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: GPT2TokenizerFast,
        device_name: str,
        max_position_embeddings: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device_name)
        self.max_position_embeddings = max_position_embeddings

    def score(self, candidate: ChoiceCandidate, pre_prompt: str = "") -> ChoiceScore:
        full_prefix = prepend_to_prompt(pre_prompt, candidate.scoring_prefix)
        prefix_ids = self.tokenizer.encode(full_prefix, add_special_tokens=False)
        continuation_ids = self.tokenizer.encode(candidate.scoring_continuation, add_special_tokens=False)
        if not prefix_ids:
            raise ValueError("scoring prefix does not contain tokenizable content")
        if not continuation_ids:
            raise ValueError("choice does not contain tokenizable content")
        if len(continuation_ids) + 1 > self.max_position_embeddings:
            raise ValueError("choice is too long to score within max_position_embeddings")

        token_ids = prefix_ids + continuation_ids
        if len(token_ids) > self.max_position_embeddings:
            token_ids = token_ids[-self.max_position_embeddings:]
            retained_prefix_len = len(token_ids) - len(continuation_ids)
        else:
            retained_prefix_len = len(prefix_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        logits = self.model(input_ids)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]

        continuation_start = max(0, retained_prefix_len - 1)
        continuation_log_probs = log_probs[:, continuation_start:, :].gather(
            dim=-1,
            index=target_ids[:, continuation_start:].unsqueeze(-1),
        )
        log_likelihood = float(continuation_log_probs.sum().item())

        return ChoiceScore(
            letter=candidate.letter,
            text=build_scored_text(candidate, pre_prompt),
            log_likelihood=log_likelihood,
            token_count=len(continuation_ids),
        )
