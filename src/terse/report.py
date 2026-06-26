"""Spike report: token delta PER TIER PER SHAPE BUCKET, coverage made explicit.

Honesty requirements (plan Section 7, principle #24):
  - near-zero buckets (compact-JSON, single-object) are shown, never averaged away
  - corpus coverage is a first-class field: which tools were captured, how many
    payloads each, so a thin sample can't read as "nothing to compress"
"""

from __future__ import annotations

from typing import Any


# TODO(spike): aggregate per (tier, shape_bucket): baseline tokens, transformed
# tokens, delta, % saved — for both cl100k and Anthropic counts. Include a
# coverage block (tool -> payload count) and the round-trip gate result (must be
# 100% pass or the report is invalid). Emit markdown to reports/.
def build_report(rows: list[dict[str, Any]]) -> str:
    """Render the per-tier per-bucket savings report. (TODO: implement.)"""
    raise NotImplementedError("build_report — implement during the spike")
