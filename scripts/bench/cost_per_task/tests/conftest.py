"""Make the flat cost_per_task scripts importable by their bare module names
(`arms`, `checkers`, `analysis`, `runner`, `mcp_share`) -- this directory is
a standalone script bundle, not an installable package."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
