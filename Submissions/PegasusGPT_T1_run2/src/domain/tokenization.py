from __future__ import annotations

import time
from functools import lru_cache

from transformers import GPT2TokenizerFast


@lru_cache(maxsize=1)
def load_gpt2_tokenizer() -> GPT2TokenizerFast:
    """Load GPT-2 tokenizer from cache first, with a short online fallback."""
    try:
        return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    except Exception as local_error:
        last_error: Exception = local_error

    for attempt in range(3):
        try:
            return GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)

    raise RuntimeError(
        "Unable to load the GPT-2 tokenizer from the local Hugging Face cache "
        "or from huggingface.co. Run once with network access to populate the "
        "cache, then subsequent leaderboard runs will use the cached tokenizer."
    ) from last_error
