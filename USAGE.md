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

**One proxy, one downstream, either transport.** By default each `terse proxy` wraps one
downstream MCP server — either **stdio** (a command that talks newline-delimited JSON-RPC
over stdin/stdout) or **HTTP/SSE** (MCP Streamable HTTP, configured by a `url`). Point the
same `proxy` command at a URL instead of a command to proxy a remote server:

```
uv run terse proxy --policy policy.example.json -- https://example.com/mcp

# add auth/other headers (repeatable); secret-shaped values are redacted wherever
# terse prints a command line (e.g. install-mcp --print), never in the request itself:
uv run terse proxy --policy policy.example.json \
  --header 'Authorization=Bearer sk-...' -- https://example.com/mcp
```

`install-mcp` (below) wraps a `url`-configured `mcpServers` entry the same way
automatically. For several servers, either give each its own wrapper (one `terse proxy --
<server cmd>` entry per server — `install-mcp` does this by default) or front all of them
from one proxy process with `--config` (see "Fronting multiple servers from one proxy"
below).

**Tell terse which server it's fronting.** If your policy has rules scoped to a server —
`runecho.*`, `codegraph.*` — add `--server-name` so they actually match:

```
uv run terse proxy --policy policy.json --server-name runecho -- runecho-mcp
```

The catch this solves: MCP tool names don't inherently name their server. kb calls its
tools `kb.read.search`, so a `kb.*` rule matches unaided — but runecho calls its tool
plain `structure`, so a `runecho.*` rule matched *nothing* and quietly fell through to
your defaults. The rule looked authored and did nothing. `--server-name` supplies the
missing half so a server-scoped rule means the same thing for every server; it also
labels `terse stats` with the real server rather than the launch command's basename.
`install-mcp` (below) fills it in for you from your MCP config, and `--config` uses each
peer's `name` — so you only pass this by hand when running `proxy` directly.

### Fronting multiple servers from one proxy (`--config`)

`--config` fans one `terse proxy` process out to N downstream peers — any mix of stdio and
HTTP — behind a single policy/primer/process instead of one wrapper per server:

```json
// peers.json
{ "downstreams": [
    { "name": "gh", "policy": "gh-policy.json", "command": ["uvx", "github-mcp"] },
    { "name": "kb", "policy": "kb-policy.json", "url": "https://kb.example/mcp",
      "headers": { "Authorization": "Bearer sk-..." } }
] }
```

```
uv run terse proxy --config peers.json
```

Each peer's tools are advertised prefixed with its `name` (`gh__search_issues`,
`kb__read`, ...) so the client can tell them apart and call the right one; terse strips
the prefix before forwarding. The synthetic `terse.retrieve` tool (drop-to-retrieve) is
advertised once, shared across every peer, regardless of which peer dropped the field.
This is an ergonomics convenience — MCP clients can already talk to several servers
directly — so keep expectations proportionate: broadcast requests (`initialize`,
`tools/list`) wait on every peer up to a bounded timeout before merging what arrived, and
methods outside `initialize`/`tools/list`/`tools/call` fall back to peer 0 for now (a
debug-logged, documented v1 limitation, not a silent gap).

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
instead of nesting proxies). Cross-call diffing is the proxy default, so a plain
wrap inherits it; pass `install-mcp --no-diff` to bake an opt-out into a server's
entry (or `--diff` to override a policy-file `"diff": false`), and
`--diff-keyframe-interval K` to tune re-anchoring. Flags always reflect the latest
install invocation. It honors `$CLAUDE_CONFIG` if your config isn't at
`~/.claude.json`. Start with one high-win, read-only server (e.g. `runecho`) and
confirm it works before wrapping more.

Hand-edits to a wrapped entry (say, an `env.PATH` pin) survive re-installs: the
re-wrap rebuilds `command`/`args` from the stashed original but keeps every other
live key, and reports what it carried (`kept hand-edited key(s) …`). One edge
remains: `uninstall-mcp` restores the **pre-terse original**, which does not carry
such edits — put them on the original entry too if they must survive an uninstall.
This matters most when the edit is what makes the *downstream* server work at all
(a Node-version `env.PATH` pin for a server that crashes on the system's default
runtime, say): kept on the wrapped copy alone, it vanishes the moment you uninstall.

#### After upgrading terse, re-check `mcp-status` (`$TERSE_MCP_CMD`)

A wrapped entry launches terse as `<absolute interpreter> -m terse`, and that
interpreter path is captured at install time. That's the default because it does not
depend on `terse` being on the MCP launcher's `PATH` — an MCP client does not
necessarily start your shell's environment. The cost is that the path is only as
stable as the install behind it. An isolated-tool install (`uv tool`, `pipx`) puts its
interpreter in a **versioned** venv, so an upgrade — or a rename of the distribution —
can move it and leave every wrapped server at once pointing at an interpreter that no
longer exists. They then fail *silently*: the server just shows up with no tools.

If your installer provides a stable console script, point wrapped entries at that
instead. `$TERSE_MCP_CMD` (whitespace-split) overrides what `install-mcp` bakes in:

```
TERSE_MCP_CMD='~/.local/bin/terse' uv run terse install-mcp runecho \
  --policy policy.example.json --print     # confirm the command, then drop --print
```

`~/.local/bin/terse` is the console script installed by both `uv tool install
terse-mcp` and `pipx install terse-mcp`, and it survives upgrades that move the venv.
A leading `~` is expanded for you (a wrapped entry is spawned without a shell, so a
literal tilde would never resolve), and a path that does not exist is **rejected at
install time** rather than written into the config — the same treatment `--policy`
already gets. A bare name like `terse` is passed through untouched, since it resolves
against the launcher's `PATH`, which the installer cannot know.

After any terse upgrade, `terse mcp-status` flags an entry whose launcher stopped
resolving:

```
  runecho              wrapped  policy=/home/you/.config/terse/policy.json
                       wraps=runecho-mcp  diff=default  stats=on
                       launcher=/home/you/.local/share/uv/tools/terse-mcp/bin/python (MISSING) — this entry cannot start; re-run install-mcp
```

Then confirm each server completes an `initialize` + `tools/list` handshake once the
client restarts. A broken wrapper does not announce itself to the client — it only
ever looks like a server that has gone quiet.

Claude Code has three MCP scopes, and `--scope` targets any of them (default
`user`, i.e. today's behavior):

```
# project scope — a .mcp.json checked into the repo and shared with every clone
uv run terse install-mcp runecho --policy policy.example.json --scope project
uv run terse uninstall-mcp runecho --scope project             # --file overrides the path (default ./.mcp.json)

# local scope — personal to one repo on one machine, nested in ~/.claude.json's
# projects."<repo-path>" block; --repo-path defaults to `git rev-parse
# --git-common-dir` (the bare-repo root for a claudew/codexw worktree, so every
# worktree of the same repo shares one entry instead of one per worktree)
uv run terse install-mcp runecho --policy policy.example.json --scope local
uv run terse uninstall-mcp runecho --scope local
```

The sidecar stash is namespaced per scope, so the same server can be
independently managed in more than one scope at once (e.g. wrapped with a
stricter policy at `user` scope and a looser one at `local` scope for one repo)
without the two colliding.

With three scopes, "what's wrapped where" is no longer one file to eyeball —
`mcp-status` checks all three and prints one report, read-only (it never writes
anything):

```
uv run terse mcp-status
# [user] /home/you/.claude.json
#   runecho              wrapped  policy=/home/you/.config/terse/policy.json
#                        wraps=runecho-mcp  diff=default  stats=on
#   codegraph            wrapped  policy=/home/you/.config/terse/gone.json (MISSING)
#                        wraps=codegraph serve --mcp  diff=off  stats=on
#   some-other-server    unwrapped
uv run terse mcp-status --json   # the same rows as JSON, for scripts / CI checks
```

Each server is one of `wrapped` (terse-managed, present), `unwrapped` (present,
not terse's), or `orphaned-stash` (terse has a stash entry but the `mcpServers`
entry it should match is gone — usually a sign the config was hand-edited after
wrapping; `uninstall-mcp --all` won't touch it either since there's nothing to
restore it *into*). For a `wrapped` server a second line shows **what it actually
fronts** (`wraps=…`), whether the cross-call **diff** tier is on/off/default, and
whether the **stats** ledger is on — the things you need when a wrapped server
misbehaves and the bare `wrapped` line can't tell you why. A policy file that has
gone missing since install is flagged `(MISSING)` (the proxy would fail to launch
without it); a *relative* policy path is never flagged, since it resolves against
the launcher's cwd, which a status scan can't know. `--file`/`--repo-path` override
project/local scope the same way `install-mcp` does.

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

### Reaching a long-text tool (`$text.code_blocks`)

Some tools don't return JSON at all — they return markdown whose bulk is verbatim source
(`codegraph_explore` is the canonical case). The lossless tiers have nothing to fold in
prose, so those payloads pass through at **0% saved** no matter how you set `tiers`.

A `$`-sigil field path selects *spans* of the raw text instead of fields of an object:

```json
{
  "match": { "tool": "codegraph.*" },
  "tiers": ["minify", "tabularize", "dictionary"],
  "fields": {
    "$text.code_blocks": { "lossy": "drop-to-retrieve", "min": 400 }
  }
}
```

Every fenced code block of at least `min` characters leaves the wire as a `terse.retrieve`
handle; the exploration summary, blast radius, and file headings stay resident, so the model
keeps the *intelligence* and fetches the *source* only if it actually needs it. On 60 real
captured `codegraph_explore` payloads this took a 0.0%-saved tool to **87.0%**.

This is **lossy and opt-in** — the same bar as any drop:

- Everything is recoverable, and the gate proves it: terse restores the entire payload from
  what it emitted plus the session store and requires byte-for-byte equality with the
  original, or it emits the untouched text and warns.
- It is suppressed — with a warning, never silently — on a `--never-lossy` server, on a
  selector marked `{"critical": true}`, and on a rule with `"tiers": []`.
- A misspelled selector (`$text.codeblocks`) warns rather than quietly compressing nothing.
- Confirm the behavior, not just the token count, with `terse fluency --drop` (below): the
  question that matters for this shape is whether the model *retrieves* the source instead
  of answering from the surrounding prose.

### One-command lossy tuning (`terse tune`)

`terse tune` chains the whole lossy-adoption loop into one command: it runs the generator,
then presents drop-to-retrieve candidates **safe-first**, classified by field role:

```bash
terse tune --corpus corpus/ --out policy.json
# # terse tune — 40 payload(s), 6 tool(s), 3 drop candidate(s)
# SAFE candidates — supporting prose, enable after a dropeval pass:
#   kb.read.nodes    result[].description   ~41% tok, 100% uniq  [prose]
#   → enabling all 1 here: ≈12,400 tok, ~18% of corpus (gross, before the per-record retrieve-handle cost)
# REVIEW candidates — role unknown, may be LOAD-BEARING; verify carefully:
#   kb.read.list_principles  result[].principle  ~36% tok, 100% uniq  [unknown]
#   → enabling all 1 here: ≈9,800 tok, ~14% of corpus (gross, before the per-record retrieve-handle cost)
```

Each bucket ends with a **rollup** — the estimated gross tokens dropping that whole
bucket would evict (`mean field tokens × record count`, summed) and its share of the
corpus's raw tokens. That's the number that answers "is turning on the SAFE set worth
it?" — a per-field `~% tok` alone can't, since it's relative to each tool's own record
list. It's a gross estimate: each dropped field leaves a small `terse.retrieve` handle
behind, so the realized saving is slightly lower.

- **`[prose]`** (evidence, rationale, description, notes, body…) — supporting text, the safe
  drop candidate.
- **`[unknown]`** — the name doesn't reveal the role, so it *may be load-bearing* (dropping it
  forces the model to call `terse.retrieve` for it). A field the model reasons over, like a
  `principle` or a `verdict`, lands here on purpose — a name heuristic can't know it's the
  essence of the record.
- **`[identity]`** fields (id, name, key, path, title…) are never suggested — the record
  needs them in-line.

Verify before enabling — the token win is real, but only a live model can tell you whether a
dropped field was actually needed:

```bash
# runs the real 2-turn retrieve eval on the suggested drops and prints the verdict:
terse tune --corpus corpus/ --out policy.json --drop-eval \
  --base-url $URL --models glm-5.2,deepseek-v4-flash
```

If the worst-case model **PASSES**, enable a field by renaming that tool's
`_suggested_fields` → `fields`. Start with `[prose]`; leave any `[unknown]` that fails.
The dropeval gate — not the role guess — is the real safety net.

### Forbidding lossy on a credential/personal server (`--never-lossy`)

Lossy transforms are **structurally forbidden** on a *never-lossy* server (a credential or
personal store), even if a policy marks one of its fields lossy — a policy typo can't leak a
credential payload through a truncate/drop. Two layers:

- A **built-in name floor** always forbids lossy on servers whose name looks secret-shaped
  (`secret`/`credential`/`vault`/`token`/`password`/`key`/`auth`) — non-overridable.
- For a sensitive store the floor can't catch (a personal KB, a launcher alias), **declare it
  at install** so it's baked into the policy:

```bash
terse install-mcp kb --policy policy.json --never-lossy
# never-lossy: baked kb into the policy's never_lossy_servers — lossy is now forbidden on it
```

`install-mcp` also prints a *hint* when a server looks sensitive but wasn't marked. The
enforcement keys off the server's verified identity (the `--server-name` terse bakes into the
wrap), so it can't be defeated by a mislabeled rule.

### Keeping a tool's output off disk (`"capture": false`)

Some tools return things that should never be written to a file — a credential, a token,
a private key. terse's two disk sinks (`--capture-dir`'s corpus and `--debug-log`'s
replay trace) both store **raw** payloads, so for those tools you want them off:

```json
{ "match": { "tool": "secret-broker.*" },
  "tiers": [],          // don't compress it (nothing to gain, no state kept)
  "capture": false }    // ...and never write it to disk
```

Two things worth knowing:

- **`"tiers": []` alone is not enough.** Passthrough stops compression, but the capture
  tee runs *before* the tiers and would still write the payload out. You need
  `"capture": false` to stop that.
- **You still get the measurement.** The savings ledger (`terse stats`) records sizes and
  decisions only — never content — so a gated tool is still counted, just never quoted.

Why put it in the policy rather than just leaving `--capture-dir` off that server? Because
the policy is durable and reviewable: it survives a re-wrap, it's visible next to
everything else you decided, and it's the only way to express the exclusion when one
proxy fronts several servers (`--config` has a single capture dir). A flag you have to
remember not to pass is one copy-pasted command away from writing secrets to disk.

Terse only ever sees what's already in a tool's result — i.e. what your model was going
to read anyway — so wrapping a server doesn't expose anything new to the *model*. What
`capture: false` controls is what **survives on disk** afterwards.

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

Because those records embed the raw payload, a tool with `"capture": false` in the policy
(above) is skipped here too — one declaration covers both disk sinks.

### How much is terse actually saving me? (`terse stats`)

Every proxy keeps a **live savings ledger** by default — one small JSON line per tool
result recording sizes, token counts, and what terse did (`compressed`, `diff`,
`unchanged`, `passthrough`). It stores **no payload content whatsoever** (that's what
makes always-on safe — unlike the replay log above), it's bounded (rotated at 10 MB,
one prior generation kept), and a write failure can never affect a tool call.

```bash
uv run terse stats               # all recorded history
uv run terse stats --since 7d    # just the last week (30m / 24h / 7d / 1w forms)
uv run terse stats --json        # the raw aggregate, for scripts
```

The report shows total tokens saved, the decision mix (how often the cross-call diff
actually fired), and a per-server/per-tool breakdown — with a `diff%` column (that
tool's cross-call diff hit rate) so you can see *which* tools the diff tier is actually
paying off on. When tiktoken wasn't available at record time the per-tool columns fall
back to characters (`chr raw`/`chr out`), matching the header rather than showing zeros.
Real numbers from your real sessions, complementing the synthetic-corpus `measure` report. The ledger lives at
`$XDG_STATE_HOME/terse/stats.jsonl` (usually `~/.local/state/terse/stats.jsonl`);
redirect it with `proxy --stats-log FILE` or disable it with `proxy --no-stats`
(bake the opt-out into a wrapped entry with `install-mcp --no-stats`).

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

Add `--json` to `verify` for a machine-readable aggregate on stdout instead of the
report — the lossless-gate verdict (`lossless_gate.ok`), cl100k savings totals, and
per-shape/coverage breakdown, from the same numbers as the markdown. Handy in CI:
`terse verify --corpus corpus --json | jq -e .lossless_gate.ok`. It writes no
file/HTML/bars (machine mode), mirroring `stats --json` / `mcp-status --json`.

Add `--history <file.jsonl>` to `measure` to track savings over time, not just one
snapshot — is the win improving, flat, or regressing as the corpus grows? Each run
appends one line (timestamp, payload count, lossless gate, token totals) to the file,
then prints a trend table plus a sparkline across every run recorded there so far:

```
uv run terse measure --corpus corpus --history reports/measure-history.jsonl
# ...
# ## Trend across runs
# | # | timestamp | label | payloads | lossless | raw tok | terse tok | saved % | Δ pts |
# |---|---|---|---|---|---|---|---|---|
# | 1 | 2026-07-01T09:00:00+00:00 | corpus | 12 | 12/12 | 4102 | 2380 | +42.0% | — |
# | 2 | 2026-07-02T09:00:00+00:00 | corpus | 14 | 14/14 | 4890 | 2650 | +45.8% | +3.8 |
#   ▁█   +42.0% -> +45.8%  (range +42.0% .. +45.8%)
```

The file is plain JSONL (one JSON object per line, append-only) — diffable, greppable,
safe to commit if you want savings tracked in git history. It's not currently wired
into `verify` — verify's no-`--corpus` fallback uses a synthetic, deterministic sample,
which would just log the same numbers every time; point `--history` at `measure`
runs over your own captured corpus for a trend that means something.

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
TERSE_FLUENCY_MODELS=google/gemini-2.5-flash,deepseek/deepseek-chat \
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
pass/fail badge, printed straight to the terminal. For a charted HTML version of that
forest plot, add `--html` to any of the paired diff-family evals (`--diff`,
`--diff-soak`, `--text-diff-eval`) — it writes next to `--out` with a `.html` suffix,
same inline-SVG/no-JS/no-CDN form as `measure --html`.

### Does the model actually use `terse.retrieve`? (`fluency --drop-eval`)

A field marked `{"lossy":"drop-to-retrieve"}` is provably recoverable — but only *if* the
model calls the synthetic `terse.retrieve` tool when it needs that field. `--drop-eval`
measures the actual behavior with a live tool-calling model, not just the round-trip gate:

```
# needs a policy with a drop-to-retrieve field, and a tool-capable model:
TERSE_FLUENCY_BASE_URL=... TERSE_FLUENCY_API_KEY=... TERSE_FLUENCY_MODELS=... \
  uv run terse fluency --drop-eval --policy drop-policy.json --corpus corpus
```

The report scores retrieve-recall (did it call retrieve when the answer needed the dropped
field), no-overfetch (did it leave the tool alone when the answer didn't need it), and
final-answer accuracy — gated on the worst model. Run this before enabling
`drop-to-retrieve` in a policy you'll actually deploy.

### Cross-call diffing and its fluency check

The proxy emits a lossless **delta** against the prior same-tool result instead of
the full payload — big in agent loops that call the same tool repeatedly (~91% overlap).
It is stateful and **on by default** (its validation program — pair fluency,
nested-record coverage, and the drift soak — has passed; see TECHNICAL.md):

```
# nothing to enable — a plain proxy diffs. Opt OUT per proxy:
uv run terse proxy --no-diff -- uvx some-mcp-server
# or per policy file: {"diff": false, ...}

# re-run the fluency gate against your own consumer/model anytime.
# needs same-tool PAIRS in the corpus (capture a tool 2+ times) + a configured model:
TERSE_FLUENCY_BASE_URL=... TERSE_FLUENCY_API_KEY=... TERSE_FLUENCY_MODELS=... \
  uv run terse fluency --diff --corpus corpus
```

`fluency --diff` reports diff-form accuracy vs full-result accuracy on the same
questions and PASS/FAILs on the worst model — re-run it for a new/weaker consumer
before trusting the default. The diff is always lossless and only sent when smaller;
it falls back to the full compressed form whenever no diff applies or the prior
result isn't available.

### The depth dimension: `fluency --diff-soak`

`--diff` tests one hop (full result + one diff). In production a model reads up to
`--diff-keyframe-interval` (default 5) **consecutive** diffs off one full anchor
before the proxy re-anchors — so the question that gated the default-flip (and gates
raising the keyframe interval) is whether comprehension *drifts* with chain depth:

```
# real corpus runs, depths 1..5 (6 chain windows per depth, round-robin across tools):
TERSE_FLUENCY_BASE_URL=... TERSE_FLUENCY_API_KEY=... TERSE_FLUENCY_MODELS=... \
  uv run terse fluency --diff-soak --corpus corpus --trials 3
```

The report shows accuracy by depth (chain form vs full-terse control on the same
final-state questions) and PASS/FAILs both overall and at the deepest tested depth,
worst model. The mechanical half of the same soak — hundreds of chained hops
reconstructed exactly, keyframe cadence, reconnect resets — is pinned in
`tests/test_diff_soak.py` and runs in CI.

### Text diffing and its fluency check (`fluency --text-diff-eval`)

The diff above only reasons about JSON. Non-JSON tool output (file reads, source
excerpts, log tails) gets its own diff codec (Tier 0.7, `text_diff.py`) — same
lossless/on-by-default/falls-back-to-full contract, applied to unstructured text
instead of records. `--text-diff-eval` is its behavioral check, the text-payload
analogue of `--diff`:

```
# needs same-tool TEXT payload PAIRS in the corpus (capture a text-producing tool
# 2+ times) + a configured model:
TERSE_FLUENCY_BASE_URL=... TERSE_FLUENCY_API_KEY=... TERSE_FLUENCY_MODELS=... \
  uv run terse fluency --text-diff-eval --corpus corpus
```

It asks whether a model reconstructs the current text as accurately from (previous text
+ text-diff) as from the full current text, and PASS/FAILs on the worst model — re-run
it for a new consumer of text-heavy tools. There's no separate switch: the same
default-on diffing emits a text diff instead of a JSON diff whenever the payload isn't
JSON (and `--no-diff` turns both off together), so this eval is a risk-item check on
existing behavior.

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

A pattern is matched against the tool's own name — so `gh.*` works only because those
tools are *called* `gh.something`. If you're scoping a rule to a **server** whose tools
have plain names (runecho's is just `structure`), the proxy needs `--server-name` to
connect the two, or the rule silently never fires. `install-mcp` handles this
automatically; see "Tell terse which server it's fronting" above.

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

**Do I need an API key?**
No. Everything runs locally. A key is only needed for the optional live fluency eval
(`fluency --diff`/`--drop-eval`), which calls any OpenAI-compatible endpoint you point
it at (broker pool or a loopback gateway). You never need it for normal use.
