"""Token counting: cl100k (headroom-eval parity) + o200k (cross-tokenizer invariance).

Counters are optional at import time so terse can run on whatever is installed:
each returns None when tiktoken is unavailable, and the report shows the gap
explicitly rather than silently substituting one for the other.
"""

from __future__ import annotations

from functools import lru_cache

CL100K = "cl100k_base"   # GPT-3.5/4 — headroom-eval parity
O200K = "o200k_base"     # GPT-4o — second, very different vocab for invariance checks


@lru_cache(maxsize=4)
def _enc(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:
        return None


def count(text: str, encoding: str = CL100K) -> int | None:
    """Token count under a named tiktoken encoding, or None if unavailable."""
    enc = _enc(encoding)
    return len(enc.encode(text)) if enc is not None else None


def count_cl100k(text: str) -> int | None:
    """cl100k_base token count. NOT the consumer tokenizer (no public Claude
    tokenizer exists); see the cross-tokenizer invariance check for robustness."""
    return count(text, CL100K)


def encode_cl100k(text: str) -> list[int] | None:
    """cl100k_base token ids, or None if unavailable. Used by the probes to reason
    about token-level overlap/redundancy, not just counts."""
    enc = _enc(CL100K)
    return enc.encode(text) if enc is not None else None
