# Usage Guide: terse

## What This Does

terse makes an AI agent's tool output smaller without throwing anything away. When
a tool returns a big block of JSON (a list of search results, a directory listing,
an API response), a lot of that text is repetition — the same field names on every
row, the same values over and over, extra spacing. terse rewrites that block into a
denser form the model can still read directly, so it takes fewer tokens while keeping
every piece of information.

It only compresses where it helps. Some tools already return tidy output; terse
leaves those alone rather than doing pointless work. You tell it which tools to
compress (and how hard) with a small policy file.

The one rule terse never breaks: the compressed output can always be turned back
into the exact original. If it ever couldn't, that is treated as a failure, not a
trade-off.

## How to Use It

Everything runs through the `terse` command. Set up once with `uv sync`.

### Check whether a payload is worth compressing

Pipe a tool's JSON output into `terse gate`. It tells you three things: whether the
compression is perfectly reversible, what *shape* the data is, and how many tokens
were saved.

```
cat some-output.json | uv run terse gate -
```

You'll see something like `round-trip lossless: PASS` and `cl100k tokens: 1810 -> 881
(51.3% saved)`. A PASS means nothing was lost. A small or zero saving just means that
payload didn't have much repetition to remove — that's normal and honest.

### Compress a payload through your policy

```
cat some-output.json | uv run terse compress --tool gh.api.repos --policy policy.example.json -
```

The compressed text comes out; a short summary goes to the side (which tiers ran and
the percent saved). `--tool` is the name terse matches against your policy to decide
how to treat that tool. If you leave off `--policy`, terse uses a safe lossless
default for everything.

### Run a tool server "behind" terse (automatic compression)

Instead of compressing outputs by hand, you can put terse in front of a whole tool
server so it compresses that server's results automatically, as they come back:

```
uv run terse proxy --policy policy.example.json -- <the command that starts the server>
```

Everything still works exactly as before from the outside — terse just quietly shrinks
the results the policy says to shrink, and passes everything else through untouched. If
anything ever goes wrong with a compression, terse sends the original result instead, so
a tool call is never lost. Add `--debug` to see what it compressed.

**One proxy per server (stdio only).** Each `terse proxy` wraps exactly one **stdio**
MCP server — the kind launched by a command that talks newline-delimited JSON-RPC over
stdin/stdout. To cover several servers, give each its own wrapper (one `terse proxy --
<server cmd>` entry per server); `install-mcp` below does this for you. terse does **not**
proxy HTTP/SSE servers (the kind configured by a `url`); pointed at one it fails fast with
a clear message instead of hanging, e.g.:

```
$ uv run terse proxy -- https://example.com/mcp
[terse-proxy] 'https://example.com/mcp' looks like a URL — terse proxies a stdio MCP
server (a launchable command), not an HTTP/SSE endpoint. HTTP/SSE transport is not
supported yet (issue #5).
```

Multi-downstream and HTTP/SSE support are tracked in issue #5.

### Wire terse into Claude Code automatically (`install-mcp`)

Rather than editing `~/.claude.json` by hand, let terse wrap your MCP servers for you:

```
# preview the change (writes nothing):
uv run terse install-mcp runecho --policy policy.example.json --print

# apply it (backs up ~/.claude.json first), then restart Claude Code:
uv run terse install-mcp runecho codegraph --policy policy.example.json

# undo — restore the original command(s) exactly:
uv run terse uninstall-mcp runecho        # one server
uv run terse uninstall-mcp --all          # every terse-managed server
```

`install-mcp` rewrites each named `mcpServers` entry so its command becomes
`<python> -m terse proxy --policy <policy> -- <the original command>`, preserving
the entry's `env`/`cwd`/etc. The original is saved verbatim in a sidecar stash
(`.terse-mcp-stash.json` next to the config), so `uninstall-mcp` restores it
byte-for-byte. It's **idempotent** (re-running re-wraps from the stashed original
instead of nesting proxies) and never enables `--diff`. It honors `$CLAUDE_CONFIG`
if your config isn't at `~/.claude.json`. Start with one high-win, read-only
server (e.g. `runecho`) and confirm it works before wrapping more.

### Let terse write the policy for you (`policy generate`)

Authoring `policy.json` by hand is the main chore: add a server, see no compression, then
capture → measure → read the report → edit JSON. `policy generate` does that loop for you
from a corpus of captured outputs (collect one with `proxy --capture-dir` or `install-mcp
--capture-dir`):

```bash
# print a policy to stdout (per-tool decisions go to stderr):
uv run terse policy generate --corpus ~/.config/terse/session-corpus

# or write it straight to a file:
uv run terse policy generate --corpus <dir> --out policy.json
```

It is **conservative and lossless**: for each tool it measures the real per-tier token
savings and enables a tier only where the saving clears `--threshold` (default 5%) *and*
every payload round-trips exactly — otherwise that tool is left as passthrough. The
dictionary tier is added only where its *marginal* saving clears the bar too (so a tool
that tabularizes well but has no repeated values won't carry the dictionary cost). Each
rule is commented with the measured savings, and the summary prints highest-win tools
first:

```
# terse policy generate — 6 tool(s), threshold 5.0%
  gh.api.items        minify,tabularize,dictionary  40.4% saved (dictionary +30.2%)
  kb.read.search      minify,tabularize             31.5% saved (dictionary +0.0% below threshold — dropped)
  status.rate_limit   (passthrough)                 1.2% < 5.0% threshold
```

It also **suggests** (never enables) `drop-to-retrieve` candidates: fields that are large
*and* near-unique, where the lossless tiers are powerless (nothing repeats to fold) but the
field dominates the payload — the classic case being an `embedding`/vector field. These ride
along as an **inactive** `_suggested_fields` block and print under the tool:

```
  kb.read.list_nodes  (passthrough)                 2.6% < 5.0% threshold
      ↳ drop-candidate result[].embedding (~84% of tokens, 100% unique, ~2293 tok/value) — suggested, off by default
```

Because drop is lossy, you opt in by renaming `_suggested_fields` → `fields` in the output —
then confirm the model still answers with `terse fluency`. Until you do, it is a no-op (the
loader ignores `_suggested_fields`). On the measured corpus, enabling the `embedding` drop
turns a +2.6% tool into a +77% one.

terse measures *tokens*, not comprehension — before relying on a generated policy, confirm
the model still reads the compressed form with `terse fluency --corpus <dir>` (see below).

### When a result looks wrong: the replay log

If a compressed result ever looks misshapen, add `--debug-log FILE` to the proxy and it
appends one JSON line per intercepted result — the raw payload, the tier decision, and
exactly what terse emitted:

```bash
uv run terse proxy --debug-log /tmp/terse-audit.jsonl -- uvx some-mcp-server
```

Each line has `{tool, id, diff_mode, tiers, changed, blocks:[{raw, emitted}]}`. Even a
no-op (`changed:false`) is logged, so you can confirm terse left a suspect payload alone.
Replay any line through `terse compress --tool <tool>` on its `raw` to reproduce. Opt-in
and side-effect-only: a log-write failure never affects what the client receives.

### See how well it does across many tools

If you've collected sample outputs (see "Building a sample set" below), these produce
markdown reports:

- `uv run terse measure` — how many tokens are saved, per tool and per data shape.
- `uv run terse probe` — whether there's more to gain from future features.
- `uv run terse validate` — confirms the savings hold across different token counters.
- `uv run terse fluency` — the one that matters for the proxy: does a model *read* the
  compressed form as accurately as raw JSON? (see below.)

Add `--html` to `measure` or `verify` for a charted companion report next to the
markdown (e.g. `reports/verify-report.html`) — inline SVG bar/stacked-bar charts,
zero JS, zero CDN, so it stays offline like everything else in terse. Every chart
has a `<details>` table-view fallback underneath it.

Add `--bars` to `measure` or `verify` for the same savings charts as unicode bars
printed straight to the terminal — no new file, color only when stdout is a tty
(honors `NO_COLOR`). `fluency` also has a `--bars` flag — see below.

### Check that the model still understands the compressed output

Saving tokens is pointless if the model reads the compressed form worse than raw JSON.
`fluency` measures that directly: it asks a model deterministic questions (count a
field, look one up, list them all, take a max) over both forms and scores the answers
against known-correct values — no second model judging.

```
# 1. (optional) build a synthetic corpus that stresses the hardest cases
python scripts/gen_stress_corpus.py corpus-stress

# 2a. keyless: writes an eval pack you can drive by hand, then score
uv run terse fluency --corpus corpus-stress

# 2b. with models (one OpenAI-compatible endpoint, e.g. OpenRouter):
TERSE_FLUENCY_BASE_URL=https://openrouter.ai/api/v1 \
TERSE_FLUENCY_API_KEY=sk-... \
TERSE_FLUENCY_MODELS=google/gemini-2.5-flash,anthropic/claude-haiku-4.5 \
  uv run terse fluency --corpus corpus-stress

# 2c. tighten the verdict: repeat each question N times for a confidence interval
TERSE_FLUENCY_BASE_URL=... TERSE_FLUENCY_API_KEY=... TERSE_FLUENCY_MODELS=... \
  uv run terse fluency --corpus corpus-stress --trials 5
```

The report shows accuracy per model for raw vs compressed vs compressed-with-a-one-time
format note ("primer"), flags which transform (if any) costs comprehension, and gives a
PASS/FAIL gated on the *worst* model. A model that scores 0% on raw JSON is a setup
error (wrong model id, no key) and is excluded from the verdict, not counted as a
comprehension failure. `--trials N` repeats each question N times and reports each
accuracy with a `±` 95% bound, so the verdict is a tight bound rather than directional.

Add `--bars` for the same verdict as a terminal forest plot — a point + 95% CI track
per model (best terse-form vs raw, or diff-form vs full-terse under `--diff`) with a
pass/fail badge, printed straight to the terminal.

### Cross-call diffing and its fluency check

The proxy can emit a lossless **delta** against the prior same-tool result instead of
the full payload — big in agent loops that call the same tool repeatedly (~91% overlap).
It is **opt-in** and stateful:

```
# enable it on the proxy (off by default):
uv run terse proxy --diff -- uvx some-mcp-server

# before trusting it, check a model still reads the diff as well as the full result.
# needs same-tool PAIRS in the corpus (capture a tool 2+ times) + a configured model:
TERSE_FLUENCY_BASE_URL=... TERSE_FLUENCY_API_KEY=... TERSE_FLUENCY_MODELS=... \
  uv run terse fluency --diff --corpus corpus
```

`fluency --diff` reports diff-form accuracy vs full-result accuracy on the same
questions and PASS/FAILs on the worst model — run it before enabling `proxy --diff` for
your consumer. The diff is always lossless and only sent when smaller; it falls back to
the full compressed form whenever no diff applies or the prior result isn't available.

### Building a sample set

To measure your own tools, capture their outputs first:

```
your-tool | uv run terse capture --tool your.tool.name -
```

Each capture is saved locally. **Only capture output you're comfortable storing** —
captured files can contain whatever the tool returned. Do not capture anything with
passwords, personal data, or private documents in it.

### Adjusting the policy

The policy file (`policy.example.json`) is a list of rules. Each rule says: for tools
whose name matches this pattern, run these compression tiers. Patterns use `*` as a
wildcard (`gh.*` matches every GitHub tool). The first matching rule wins, so put more
specific rules first. An empty tier list means "leave this tool's output alone."

## What to Do When Something Breaks

- **"round-trip lossless: FAIL"** — Stop and report it. This should never happen; it
  means the compression and decompression disagree. Note the tool and input that
  triggered it. (The test suite checks this on every change, so a FAIL in normal use
  is a bug worth filing.)

- **"It saved 0%" or a tiny number** — Not a failure. That payload was already compact
  or had little repetition (single objects and already-tidy tools do this). The
  per-tool report will show which tools are worth compressing and which aren't.

- **A `[warn] field ... not implemented yet` message** — The policy asked for `summarize`
  on a field; that mode isn't built yet, so terse safely ignored it and kept everything.
  Nothing was lost.
- **A `[warn] lossy: truncated marked field(s) — output is NOT lossless` message** — Expected
  when a field is marked `{"lossy":"truncate"}`. terse capped that field on purpose and
  annotated the cut (`…⟨+N chars⟩`). Only fields you marked (never `{"critical":true}` ones)
  are affected; if the gate can't prove that, terse falls back to the lossless output and
  says so (`lossy step skipped`).
- **A `[warn] lossy: dropped marked field(s) to retrieve handle(s)` message** — Expected
  when a field is marked `{"lossy":"drop-to-retrieve"}` and terse is running as the proxy.
  terse replaced that field with a `__terse_dropped__` handle and stored the original; the
  model gets it back by calling the injected `terse.retrieve` tool. Outside the proxy (e.g.
  the one-shot CLI) there is no store, so the field is left lossless with a
  `needs the proxy store` warning instead.

- **"no payloads in corpus/"** — You ran a report before capturing any samples. Capture
  some tool outputs first (see "Building a sample set").

- **A token count shows "unavailable"** — The token counter's data file didn't load
  (usually no internet on first run). Reconnect and try again.

For anything else, see the [Technical Reference](TECHNICAL.md) or [README](README.md).

## FAQ

**Does terse delete any of my data?**
No. Today it is fully lossless — the compressed output always reconstructs the exact
original. A future opt-in mode could drop detail you explicitly mark, but it isn't
built, and even the policy slots for it are ignored for now.

**Why didn't my payload get smaller?**
Because there was nothing safe to remove. terse shrinks repetition (repeated field
names, repeated values, extra spacing). A single small object or an already-minimal
response has none of that, so it's left as-is.

**Will the model still understand the compressed output?**
Yes. The compressed form is still readable text — a table with a header, or values
with a small legend at the top — not an encoded blob. The model reads it in place;
it never has to "fetch" anything that was removed.

**Why does it compress some tools and not others?**
Because it was measured to only pay off on some. Compressing already-tidy output wastes
effort, so the policy turns it off there. You control this in the policy file.

**Do I need an Anthropic or OpenAI key?**
No. Everything runs locally. A key is only needed for one optional command that
double-checks token counts against Anthropic directly, and you never need it for normal
use.
