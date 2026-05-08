"""Standard Alpaca-style chat template for SFT.

Two variants depending on whether the example has a non-empty input field:
- with_input: instruction + context/input + response
- without_input: instruction + response only

The template is deterministic and used identically at training and inference.
"""

PROMPT_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_WITHOUT_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def build_prompt(instruction: str, input_text: str) -> str:
    """Build the prompt portion (everything before the response)."""
    if input_text and input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_WITHOUT_INPUT.format(instruction=instruction)


def build_full_text(instruction: str, input_text: str, output: str) -> tuple[str, str]:
    """Build (prompt, full_text) where full_text = prompt + output.

    Returns both so the tokenizer can compute the prompt length for loss masking.
    """
    prompt = build_prompt(instruction, input_text)
    full_text = prompt + output
    return prompt, full_text