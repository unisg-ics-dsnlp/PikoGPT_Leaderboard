from __future__ import annotations

import re

from domain.scoring.base import ChoiceCandidate, NormalizedMultipleChoicePrompt


_CHOICE_RE = re.compile(r"^([A-D])\)\s*(.+?)\s*$")


class OpenBookQAAdapter:
    name = "openbookqa"
    scoring_pre_prompt = "Choose the scientifically correct answer.\n\n"

    def detect(self, prompt: str) -> bool:
        try:
            normalized = self.normalize(prompt)
        except ValueError:
            return False
        return len(normalized.candidates) == 4

    def normalize(self, prompt: str) -> NormalizedMultipleChoicePrompt:
        lines = [line.rstrip() for line in prompt.strip().splitlines()]
        if len(lines) < 6 or not lines[0].startswith("Question:"):
            raise ValueError("not an OpenBookQA-style prompt")

        answer_idx = self._find_answer_line(lines)
        choice_lines = lines[1:answer_idx]

        choices: list[tuple[str, str]] = []
        first_choice_idx = None
        for idx, line in enumerate(choice_lines):
            match = _CHOICE_RE.match(line)
            if match is None:
                continue
            if first_choice_idx is None:
                first_choice_idx = idx
            choices.append((match.group(1), match.group(2)))

        if first_choice_idx is None:
            raise ValueError("OpenBookQA prompt has no choices")
        expected_letters = ["A", "B", "C", "D"]
        if [letter for letter, _ in choices] != expected_letters:
            raise ValueError("OpenBookQA prompt choices must be A-D")

        question_lines = [lines[0][len("Question:"):].strip(), *choice_lines[:first_choice_idx]]
        question = "\n".join(line for line in question_lines if line).strip()
        if not question:
            raise ValueError("OpenBookQA prompt has empty question")

        answer_prompt = "\n".join(lines[: answer_idx + 1]).strip()

        candidates = tuple(
            ChoiceCandidate(
                letter=letter,
                text=text,
                scoring_prefix=answer_prompt,
                scoring_continuation=f" {text}",
                calibration_prefix="Answer:",
                calibration_continuation=f" {text}",
            )
            for letter, text in choices
        )

        return NormalizedMultipleChoicePrompt(
            benchmark=self.name,
            original_prompt=prompt,
            candidates=candidates,
            scoring_pre_prompt=self.scoring_pre_prompt,
        )

    def _find_answer_line(self, lines: list[str]) -> int:
        for idx, line in enumerate(lines):
            if line.strip() == "Answer:":
                return idx
        raise ValueError("OpenBookQA prompt has no Answer line")
