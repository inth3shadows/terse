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

# Version is derived from the git tag by hatch-vcs (see pyproject `[tool.hatch.version]`).
# A build writes the resolved value to `_version.py`; prefer it. In a raw source tree
# that was never built, fall back to the installed dist metadata, then to a sentinel —
# never a hand-edited literal (that is the drift bug this whole scheme removes).
try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        __version__ = _dist_version("terse")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
