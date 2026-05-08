from __future__ import annotations

from dataclasses import dataclass

import torch

from domain.inference.inference_service import GPTInferenceService
from domain.scoring.base import ChoiceCandidate, NormalizedMultipleChoicePrompt, build_scored_text
from domain.scoring.multiple_choice_scorer import ChoiceScore, MultipleChoiceScorer
from domain.scoring.registry import BenchmarkRegistry, default_registry
from domain.tokenization import load_gpt2_tokenizer


@dataclass(frozen=True)
class MultipleChoiceScoringResult:
    benchmark: str
    reply: str
    scores: tuple[ChoiceScore, ...]


class MultipleChoiceScoringService:
    def __init__(self, registry: BenchmarkRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def detect_and_normalize(self, prompt: str) -> NormalizedMultipleChoicePrompt | None:
        adapter = self.registry.detect(prompt)
        if adapter is None:
            return None
        return adapter.normalize(prompt)

    def scored_texts_for_prompt(self, prompt: str) -> tuple[str, tuple[tuple[str, str], ...]] | None:
        normalized = self.detect_and_normalize(prompt)
        if normalized is None:
            return None
        return (
            normalized.benchmark,
            tuple(
                (
                    candidate.letter,
                    build_scored_text(candidate, normalized.scoring_pre_prompt),
                )
                for candidate in normalized.candidates
            ),
        )

    def run(
        self,
        *,
        checkpoint_path: str,
        model_config,
        input_text: str,
        device_name: str,
        vocab_size: int | None = None,
    ) -> MultipleChoiceScoringResult | None:
        normalized = self.detect_and_normalize(input_text)
        if normalized is None:
            return None

        inference_service = GPTInferenceService()
        model, built_config = inference_service.load_model(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            vocab_size=vocab_size,
        )

        model.to(device_name)
        was_training = model.training
        model.eval()
        tokenizer = load_gpt2_tokenizer()
        scorer = MultipleChoiceScorer(
            model=model,
            tokenizer=tokenizer,
            device_name=device_name,
            max_position_embeddings=built_config.max_position_embeddings,
        )

        try:
            with torch.no_grad():
                scores = tuple(
                    scorer.score(
                        candidate,
                        pre_prompt=normalized.scoring_pre_prompt,
                    )
                    for candidate in normalized.candidates
                )
                best = self._best_score(
                    normalized=normalized,
                    scores=scores,
                    scorer=scorer,
                )
        finally:
            if was_training:
                model.train()

        return MultipleChoiceScoringResult(
            benchmark=normalized.benchmark,
            reply=best.letter,
            scores=scores,
        )

    def _best_score(
        self,
        *,
        normalized: NormalizedMultipleChoicePrompt,
        scores: tuple[ChoiceScore, ...],
        scorer: MultipleChoiceScorer,
    ) -> ChoiceScore:
        if normalized.benchmark in {"openbookqa", "winogrande"}:
            calibrated_scores: list[tuple[float, ChoiceScore]] = []
            for candidate, score in zip(normalized.candidates, scores):
                if candidate.calibration_prefix is None:
                    calibrated_scores.append((score.log_likelihood / score.token_count, score))
                    continue

                calibration = scorer.score(
                    ChoiceCandidate(
                        letter=candidate.letter,
                        text=candidate.text,
                        scoring_prefix=candidate.calibration_prefix,
                        scoring_continuation=(
                            candidate.calibration_continuation
                            if candidate.calibration_continuation is not None
                            else candidate.scoring_continuation
                        ),
                    )
                )
                calibrated_scores.append(
                    (
                        score.log_likelihood / score.token_count
                        - calibration.log_likelihood / calibration.token_count,
                        score,
                    )
                )
            return max(calibrated_scores, key=lambda item: item[0])[1]
        return max(scores, key=lambda score: score.log_likelihood)
