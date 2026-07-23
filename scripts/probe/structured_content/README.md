# Which MCP result field reaches the model's context? (issue #128)

Issue #128 lists four options for `structuredContent` and says none is obviously right.
All four hinge on one fact nobody had measured: **which field does the MCP client
actually put in the model's context** — the text block terse compresses, the
`structuredContent` it leaves alone, or both?

Three possible answers, three *different* decisions:

| if the client forwards… | then the untouched duplicate… | and #128 becomes |
|---|---|---|
| only the text block | never reaches the model, costs 0 | option 4 — document the shape divergence |
| both | costs ~396 tokens per call | option 2 — drop the redundant mirror |
| only `structuredContent` | is the *only* thing that reaches the model | terse's saving on these tools is **zero** |

So: measure first.

## Result (2026-07-23, `claude` 2.1.218, Linux/WSL2)

**The client forwards `structuredContent` and ignores the text block entirely.**

The fixture emits a spec-compliant pair whose two halves are provably identical at the
source but *textually distinguishable* — the text block uses `json.dumps` default
separators (`, ` / `: `, 2,836 chars), so a compact 2,596-char rendering can only have
come from re-serializing the typed field.

| arm | terse in path? | text block terse emitted | what reached the model |
|---|---|---|---|
| `raw` | no | — (2,836 chars, spaced) | 2,596 chars, compact, plain records |
| `terse` | yes (proven via `--debug-log`) | 1,008 chars, `__terse_dict__`/`__terse_table__` | **the same 2,596 chars** — byte-identical to the raw arm |

The `terse` arm is the control that makes this airtight. If the client were forwarding
the text block — verbatim *or* parsed-and-re-dumped — the terse arm would carry terse's
envelope into the context, because that is what the text block contained. It does not.
The only field that could have produced those bytes is `structuredContent`.

Proof terse really was in the path and really did compress (so the block was discarded by
the *client*, not never produced):

```
tool records  tiers ['minify','tabularize','dictionary']  changed True
  raw chars     2836
  emitted chars 1008
```

### What this means for #128

Worse than the issue's framing. #128 measured the duplicate as *halving* the saving
(70.5% → 56.2%). For this client the saving on a `structuredContent`-emitting tool is
**0%** — terse compresses a field the client throws away.

- **Options 2, 3 and 4 all leave the real saving at zero.** They differ only in how they
  handle a shape divergence the model never sees.
- **Option 1 (compress `structuredContent` itself) is the only one that moves the
  number** — and it is the invasive one, since that field is what clients validate
  against `outputSchema`.
- There was a **fourth consequence the issue does not mention**: `terse stats` derived its
  savings from the text block alone, so on these tools the ledger reported a saving that
  did not exist — the same class of silent measurement error #131 fixed, one layer up.
  **Fixed**: `build_record` now counts the untouched duplicate on both sides, taking the
  reference fixture's reported figure from 58.7% to 33.9%. Note that 33.9% is the *wire*
  truth — what terse can measure without knowing the client — and remains an upper bound
  on what a `structuredContent`-reading client actually saves, which is 0%.

### Scope — read before generalizing

- One client, one version: `claude` **2.1.218**. The MCP spec makes the text block a
  *backwards-compatibility* mirror, so a client preferring `structuredContent` is
  spec-reasonable — but other clients (Codex, Cursor, OpenCode) may differ, and this
  harness has not been pointed at them.
- Cross-checked against the real third-party `@modelcontextprotocol/server-everything`
  (`get-structured-content`), which behaves consistently — though **that server alone
  cannot prove the point**: its text block is already compact and byte-identical to its
  serialized `structuredContent`, so the two hypotheses are indistinguishable there.
  That ambiguity is exactly why the local fixture exists.

## Run it

```bash
./run_capture.sh                  # both arms + verdict
TOOL=weather ./run_capture.sh     # the flat-object case (minify only)
OUTDIR=/path/to/scratch ./run_capture.sh
```

Needs `mitmdump` (a CA at `~/.mitmproxy/`, generated on first run), `claude` and `terse`
on PATH. Each arm spends a small number of real API tokens — it drives a live headless
`claude -p`.

## How it works, and why each piece is the way it is

- **`structured_server.py`** — a dependency-free stdio MCP server emitting a
  spec-compliant `content` + `structuredContent` pair from a single source object, so any
  divergence observed downstream is provably terse's or the client's, never the fixture's.
  Two tools: `weather` (flat object → `minify` only, the benign case) and `records`
  (30 uniform records → `tabularize` + `dictionary` fire, the expensive case).
- **`context_capture.py`** — a **read-only** mitmproxy addon that extracts only
  `tool_result` blocks from outbound `/v1/messages` bodies. It never mutates a flow and
  never touches request headers: the session's OAuth bearer lives there, and this addon
  has no reason to see it. Output is written 0600 to `CAP_OUT` (scratchpad, never the
  repo, never the transcript).
- **`run_capture.sh`** — runs the two arms. It must launch its **own** `claude`:
  `HTTPS_PROXY` is read at process start, so an already-running session can never be
  routed through the proxy mid-flight. `--mcp-config` + `--strict-mcp-config` keep the
  probe off your real `~/.claude.json`, and `terse proxy --no-stats` keeps its synthetic
  calls out of your real savings ledger.
- **`report.py`** — prints per-arm context cost and the verdict. Its sharpest test needs
  no fixture knowledge: if both arms put *byte-identical* text in the model's context,
  interposing terse changed nothing the model sees.

An arm that captures zero `tool_result` blocks **exits non-zero** rather than reporting a
clean 0% — an empty artifact is a failed measurement, not evidence (the #131 lesson).
