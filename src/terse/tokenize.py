"""Token counting: cl100k (headroom-eval parity) + Anthropic count_tokens (truth).

Both are optional at import time so the spike can run on whatever is installed:
each counter returns None when its backend is unavailable, and the report shows
the gap explicitly rather than silently substituting one for the other.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def _cl100k():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_cl100k(text: str) -> Optional[int]:
    """cl100k_base token count, or None if tiktoken / the encoding is unavailable.

    Kept for apples-to-apples comparison with the headroom eval. NOT the consumer
    tokenizer — see count_anthropic for ground truth.
    """
    enc = _cl100k()
    return len(enc.encode(text)) if enc is not None else None


def encode_cl100k(text: str) -> Optional[list[int]]:
    """cl100k_base token ids, or None if unavailable. Used by the probes to reason
    about token-level overlap/redundancy, not just counts."""
    enc = _cl100k()
    return enc.encode(text) if enc is not None else None


def count_anthropic(text: str, model: str = "claude-opus-4-8") -> Optional[int]:
    """Ground-truth token count for the real consumer, or None if unavailable.

    Requires the `anthropic` extra and credentials. Network call; the spike
    batches these. Left lazy + best-effort so a no-key environment still runs.
    """
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return resp.input_tokens
    except Exception:
        return None
