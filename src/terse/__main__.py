"""Enable `python -m terse` so a wrapped MCP entry can launch terse by an
absolute interpreter path (no reliance on `terse` being on the launcher's PATH)."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
