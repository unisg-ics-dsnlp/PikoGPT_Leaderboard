from __future__ import annotations

import re

from domain.scoring.base import ChoiceCandidate, NormalizedMultipleChoicePrompt
from domain.sft.chat_template import build_prompt

_CHOICE_RE = re.compile(r"^([A-D])\)\s*(.+?)\s*$")


class HellaSwagAdapter:
    name = "hellaswag"
    scoring_pre_prompt = ""

    def detect(self, prompt: str) -> bool:
        try:
            normalized = self.normalize(prompt)
        except ValueError:
            return False
        return len(normalized.candidates) == 4

    def normalize(self, prompt: str) -> NormalizedMultipleChoicePrompt:
        lines = [line.rstrip() for line in prompt.strip().splitlines()]
        if len(lines) < 6 or not lines[0].startswith("Context:"):
            raise ValueError("not a HellaSwag-style prompt")

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
            raise ValueError("HellaSwag prompt has no choices")

        expected_letters = ["A", "B", "C", "D"]
        if [letter for letter, _ in choices] != expected_letters:
            raise ValueError("HellaSwag prompt choices must be A-D")

        context_lines = [lines[0][len("Context:"):].strip(), *choice_lines[:first_choice_idx]]
        context = "\n".join(line for line in context_lines if line).strip()
        if not context:
            raise ValueError("HellaSwag prompt has empty context")

        candidates = tuple(
            ChoiceCandidate(
                letter=letter,
                text=text,
                scoring_prefix=context,
                scoring_continuation=f" {text}",
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
        raise ValueError("HellaSwag prompt has no Answer line")
