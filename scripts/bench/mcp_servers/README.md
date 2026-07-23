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

## Transports beyond stdio

terse also proxies an MCP **Streamable-HTTP** endpoint, and can front several servers from
one process. Both are covered in §6:

```bash
# HTTP downstream — a single target containing "://" selects the URL transport
npx -y @modelcontextprotocol/server-everything streamableHttp     # listens on :3001
python mcp_probe.py everything-http "$CORPUS" "$LEDGER" \
  '[{"name":"echo","arguments":{"message":"hi"}}]' -- http://127.0.0.1:3001/mcp

# multi-peer fan-out, mixed transports (2 stdio + 1 HTTP) in one process
cat > multi.json <<'JSON'
{"downstreams":[
  {"name":"fs","command":["npx","-y","@modelcontextprotocol/server-filesystem","/tmp/mcp-fixtures/express"]},
  {"name":"mem","command":["npx","-y","@modelcontextprotocol/server-memory"]},
  {"name":"ev","url":"http://127.0.0.1:3001/mcp"}
]}
JSON
terse proxy --config multi.json --capture-dir "$CORPUS" --stats-log "$LEDGER"
```

Tools arrive peer-prefixed (`fs__directory_tree`), the terse primer is injected once across
all peers, and the ledger attributes savings per peer.

## Gotchas worth knowing

- **Hold stdin open.** `mcp_probe.py` keeps the pipe open until every response arrives.
  Closing it right after writing the requests tears the child down mid-call: fast servers
  still answer, slow ones (browser launch, HTTP fetch) return *nothing* — which looks like
  "that server is broken" when it is the harness. Raise `PROBE_DEADLINE` (default 300s)
  for slow servers.
- **A server's own requests use their own ids.** `roots/list` / `sampling/createMessage`
  are *requests*, and their ids collide with the probe's. The probe only treats
  `result`/`error` messages as responses and answers inbound requests `-32601`; anything
  looser ends the run early and reports an **empty corpus as a clean measurement**.
- **`isError: true` is a failure, not a payload.** A mistyped path or a wrong argument name
  comes back that way. terse tees it to the corpus *before* the probe sees it, so the probe
  cannot prevent the poisoning — it reports `TOOL ERROR` and **exits non-zero**. On a
  `TOOL ERROR`, discard the corpus dir and the ledger and re-run: those rows otherwise skew
  both the codec % and the `diff_reason` breakdown.
- **The probe checks the artifacts, not just the replies.** §6's numbers come from the
  corpus and the ledger, so a run that answered every request but wrote neither is still a
  failed measurement — the probe verifies both and exits non-zero. (Since #131 terse also
  announces the first failure of each sink kind on stderr without `--debug`; the probe's
  own artifact check predates that and stays, since it catches a sink that silently wrote
  the *wrong* thing as well as one that wrote nothing.)
- **Repeats are serialized**, not pipelined — servers dispatch concurrently and the proxy
  sets its diff base in arrival order, so batched repeats made the diff nondeterministic.
- **Proxy stderr** is teed to `<stats_log>.stderr` and the tail is printed on failure; it
  carries terse's launch errors (the usual cause of an `initialize` failure).
  `PROBE_STDERR=1` inherits it live instead.
- **playwright** needs `--no-sandbox` where user namespaces are restricted (WSL,
  containers), and its pinned playwright version may want a newer Chromium than you have
  cached — install it with
  `npx -y playwright@<version> install chromium` (the version is in
  `npm view @playwright/mcp dependencies`).
- **serena** indexes the project with a language server on first call; allow a generous
  deadline on a large repo.
- Tool argument names are worth reading off `tools/list` rather than guessing — e.g.
  serena's `find_symbol` takes `name_path_pattern`, not `name_path`.
