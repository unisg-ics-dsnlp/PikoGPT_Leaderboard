from __future__ import annotations

import re

from domain.scoring.base import ChoiceCandidate, NormalizedMultipleChoicePrompt

_CHOICE_RE = re.compile(r"^([A-B])\)\s*(.+?)\s*$")


class WinoGrandeAdapter:

    name = "winogrande"
    scoring_pre_prompt = ""

    def detect(self, prompt: str) -> bool:
        try:
            normalized = self.normalize(prompt)
        except ValueError:
            return False
        return len(normalized.candidates) == 2

    def normalize(self, prompt: str) -> NormalizedMultipleChoicePrompt:
        lines = [line.rstrip() for line in prompt.strip().splitlines()]
        if len(lines) < 4 or not lines[0].startswith("Context:"):
            raise ValueError("not a WinoGrande-style prompt")

        answer_idx = self._find_answer_line(lines)
        body_lines = lines[1:answer_idx]

        choices: list[tuple[str, str]] = []
        first_choice_idx = None
        for idx, line in enumerate(body_lines):
            match = _CHOICE_RE.match(line)
            if match is None:
                continue
            if first_choice_idx is None:
                first_choice_idx = idx
            choices.append((match.group(1), match.group(2)))

        if first_choice_idx is None:
            raise ValueError("WinoGrande prompt has no choices")

        expected_letters = ["A", "B"]
        if [letter for letter, _ in choices] != expected_letters:
            raise ValueError("WinoGrande prompt choices must be A-B")

        context_lines = [lines[0][len("Context:"):].strip(), *body_lines[:first_choice_idx]]
        context = "\n".join(line for line in context_lines if line).strip()
        if not context:
            raise ValueError("WinoGrande prompt has empty context")

        if "_" not in context:
            raise ValueError("WinoGrande context must contain a '_' placeholder")

        # Split the context at the first blank
        blank_pos = context.index("_")
        head_template = context[:blank_pos].rstrip()
        tail = context[blank_pos + 1:]

        candidates = []
        for letter, option_text in choices:
            substituted_head = (
                f"{head_template} {option_text}" if head_template else option_text
            )
            candidates.append(
                ChoiceCandidate(
                    letter=letter,
                    text=option_text,
                    scoring_prefix=substituted_head,
                    scoring_continuation=tail,
                    calibration_prefix=option_text,
                    calibration_continuation=tail,
                )
            )

        return NormalizedMultipleChoicePrompt(
            benchmark=self.name,
            original_prompt=prompt,
            candidates=tuple(candidates),
            scoring_pre_prompt=self.scoring_pre_prompt,
        )

    def _find_answer_line(self, lines: list[str]) -> int:
        for idx, line in enumerate(lines):
            if line.strip() == "Answer:":
                return idx
        raise ValueError("WinoGrande prompt has no Answer line")
