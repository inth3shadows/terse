"""terse — lossless-first compression layer for AI-agent tool outputs.

The measurable spine lives in `transforms` (lossless Tier-0: minify +
tabularize, Tier-0.5 dictionary coding, Tier-0.7 cross-call diffing) and
`tokenize` (cl100k + o200k counts). `capture`, `probes`, `measure`, `report`,
and `cli` build measurement tooling around that spine; `policy`, `proxy`, and
`install_mcp` are the operational shell; `fluency` and `dropeval` are the
behavioral evals that gate every diff/lossy tier.

Design invariant: any field NOT matched by a policy is treated as critical and
only ever sees lossless tiers. Lossy is strictly opt-in.
"""

__version__ = "0.0.1"
