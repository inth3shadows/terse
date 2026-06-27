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

### See how well it does across many tools

If you've collected sample outputs (see "Building a sample set" below), these produce
markdown reports:

- `uv run terse measure` — how many tokens are saved, per tool and per data shape.
- `uv run terse probe` — whether there's more to gain from future features.
- `uv run terse validate` — confirms the savings hold across different token counters.

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

- **A `[warn] field ... requests a lossy mode` message** — Expected for now. The policy
  mentions an optional "drop some detail" mode that isn't built yet, so terse safely
  ignored it and kept everything. Nothing was lost.

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
