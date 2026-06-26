"""Lossless-ceiling probes (build order B).

These do NOT compress anything. They measure whether the higher-ceiling lossless
levers (Tier 0.5) are worth building, at near-zero cost on the corpus the spike
already captures:

  - value_redundancy: how many tokens are repeated VALUES across rows (beyond the
    repeated KEYS that tabularize already collapses). High -> dictionary coding
    has real headroom above tabularize. Low -> skip Tier 0.5.
  - cross_call_overlap: token overlap between successive payloads of the same tool
    in an agent loop. High -> structural diffing (delta + reference) compounds in
    the loop. Low -> skip diffing.

Both are reported as findings; neither gates the lossless pipeline.
"""

from __future__ import annotations

from typing import Any


# TODO(spike): implement against the captured array-of-records bucket using the
# tokenizer from tokenize.py. Count value-token occurrences across rows; report
# the fraction attributable to values that repeat >=2x. Plan Section 7.
def value_redundancy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of value-tokens that are repeated across rows. (TODO: implement.)"""
    raise NotImplementedError("value_redundancy probe — implement during the spike")


# TODO(spike): implement token-set / sequence overlap between two successive
# payloads from the same tool. Plan Section 7 "cross-call-overlap probe".
def cross_call_overlap(prev_raw: str, curr_raw: str) -> dict[str, Any]:
    """Token overlap between successive same-tool payloads. (TODO: implement.)"""
    raise NotImplementedError("cross_call_overlap probe — implement during the spike")
