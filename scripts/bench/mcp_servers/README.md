# Third-party MCP server benchmarks

Measures what terse does **automatically** (zero-config) to the output of widely-used,
credential-free MCP servers — and how that moves with **repo size**. Produces
BENCHMARKS.md §6.

Unlike §1–3 (a fixed corpus committed to this repo), these numbers come from running real
servers on your machine. Everything here is reproducible: every server is credential-free,
every repo fixture is pinned to a tag, and the web fixture is a static local file.

## Prerequisites

- `terse` on PATH (or set `TERSE_BIN=/path/to/terse`)
- `npx` (Node) and `uvx` (uv) — the reference servers ship on both
- For playwright only: a matching Chromium build (see below)

## Repo fixtures (pinned)

```bash
mkdir -p /tmp/mcp-fixtures && cd /tmp/mcp-fixtures
git clone https://github.com/expressjs/express.git && git -C express checkout v5.2.1
git clone https://github.com/fastapi/fastapi.git   && git -C fastapi checkout 0.139.2
git clone https://github.com/django/django.git     && git -C django  checkout 5.2.16
```

Small → medium → large: 218 / 3,131 / 6,926 tracked files.

## Run

```bash
REPO=/tmp/mcp-fixtures/django
CORPUS=$(mktemp -d) && LEDGER="$CORPUS.jsonl"

# filesystem — the JSON size-axis tool
python mcp_probe.py filesystem "$CORPUS" "$LEDGER" \
  "[{\"name\":\"directory_tree\",\"arguments\":{\"path\":\"$REPO/django/db\"}}]" \
  -- npx -y @modelcontextprotocol/server-filesystem "$REPO"

terse measure --corpus "$CORPUS"   # per-tool codec %, shape bucket, tier attribution
terse stats   --log    "$LEDGER"   # decisions + diff-reason breakdown
terse policy generate --corpus "$CORPUS"   # what terse auto-authors for these tools
```

Swap the server command for the others:

| server | launch |
|---|---|
| filesystem | `npx -y @modelcontextprotocol/server-filesystem <repo>` |
| git | `uvx --from mcp-server-git mcp-server-git --repository <repo>` |
| memory | `npx -y @modelcontextprotocol/server-memory` |
| fetch | `uvx mcp-server-fetch --ignore-robots-txt` |
| serena | `uvx --from git+https://github.com/oraios/serena serena start-mcp-server --project <repo> --context ide-assistant --transport stdio --enable-web-dashboard false --enable-gui-log-window false` |
| playwright | `npx -y @playwright/mcp@latest --browser chromium --headless --isolated --no-sandbox` |

## Static web fixture (fetch + playwright)

So the target is deterministic instead of a live page:

```bash
python3 -m http.server 8919 --directory ./webfix   # serves webfix/index.html
```

Then call `fetch` / `browser_navigate` against `http://127.0.0.1:8919/index.html`.

## Gotchas worth knowing

- **Hold stdin open.** `mcp_probe.py` keeps the pipe open until every response arrives.
  Closing it right after writing the requests tears the child down mid-call: fast servers
  still answer, slow ones (browser launch, HTTP fetch) return *nothing* — which looks like
  "that server is broken" when it is the harness. Raise `PROBE_DEADLINE` (default 300s)
  for slow servers.
- **playwright** needs `--no-sandbox` where user namespaces are restricted (WSL,
  containers), and its pinned playwright version may want a newer Chromium than you have
  cached — install it with
  `npx -y playwright@<version> install chromium` (the version is in
  `npm view @playwright/mcp dependencies`).
- **serena** indexes the project with a language server on first call; allow a generous
  deadline on a large repo.
- Tool argument names are worth reading off `tools/list` rather than guessing — e.g.
  serena's `find_symbol` takes `name_path_pattern`, not `name_path`.
