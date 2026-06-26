"""terse — lossless-first compression layer for AI-agent tool outputs.

Phase-0 spike. The measurable spine lives in `transforms` (lossless Tier-0:
minify + tabularize) and `tokenize` (cl100k + Anthropic count). Everything else
(`capture`, `probes`, `report`, `cli`) is stubbed to the plan at
~/.claude/plans/terse-lossless-tool-output-compression.md and filled in as the
spike runs.

Design invariant: any field NOT matched by a policy is treated as critical and
only ever sees lossless tiers. Lossy is strictly opt-in.
"""

__version__ = "0.0.1"
