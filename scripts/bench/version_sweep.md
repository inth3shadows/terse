# Version sweep — Tier-0 savings across releases

What the codec actually delivers on a **fixed** corpus, release by release. Run
`./version_sweep.sh` to regenerate; it checks out each tag in the recorded range in
turn and measures the same nine payloads from [`corpus/`](corpus/). Pass `TAGS=` to
sweep a different range — the default is pinned to `v0.5.1`..`v0.17.0` so re-running
reproduces the table below rather than a moving window of whatever the last twenty
tags happen to be.

The corpus is the tracked one, so every number here is reproducible from git alone.

## Result — 2026-07-31, tags `v0.5.1` … `v0.17.0`

Tier-0 output was **completely flat across twenty releases**: byte-identical token
counts at every tag.

| | raw tok | terse tok | saved | % |
|---|---|---|---|---|
| **ALL** (9 payloads) | 365,144 | 152,411 | +212,733 | **58.3%** |
| array-of-records (6) | 315,103 | 118,122 | +196,981 | +62.5% |
| compact-json (3) | 50,041 | 34,289 | +15,752 | +31.5% |

Tier attribution on the array-of-records bucket, also identical at every tag:
minify +0, tabularize +47,187, dictionary +149,794.

Lossless gate: 9/9 at every tag.

### Why flat is the right answer, not a broken harness

`transforms.py` grew by 133 lines net between `v0.5.1` and `v0.17.0` (643 → 776; 147
insertions against 14 deletions), so twenty identical rows look like a sweep that
measured one installed build twenty times. It isn't — verified 2026-08-04 by
re-running `v0.5.1`'s own source in a fresh worktree against the tracked corpus:

```
v0.5.1 codec: raw=365144  terse=152411  saved=212733  58.3%
```

Exactly the recorded figure. The churn in that range — the opt-in `embedded` tier,
NaN-aware gate equality, the diff tier, marker guards — either defaults off or
does not touch Tier-0 on nine uniform GitHub payloads. The codec math genuinely
did not move.

Review re-measured 11 tags spread wider than the recorded window — `v0.1.0`, `v0.3.1`,
`v0.5.0`, `v0.5.1`, `v0.9.0`, `v0.14.0`, `v0.16.2`, `v0.17.0`, `v0.17.1`, `v0.18.0`,
`v0.18.1` — and every one returns `365144 / 152411 / +58.3%`. So the flat stretch is
wider than the table above claims in both directions; the recorded range is
conservative, not cherry-picked.

### First movement in the whole range: #202

Union-schema tabularize is the first change since `v0.5.1` to shift this corpus, and
so far the only one:

```
v0.5.1 .. v0.17.0 : 152,411 tok  58.3%
v0.19.0 (has #202): 149,486 tok  59.1%   (+0.8pp, +2,925 tok)
```

Modest here **by construction** — these nine payloads are uniform records or compact
JSON, and union-schema only reaches non-uniform ones. The whole +2,925 lands in the
`compact-json` bucket, all of it in `gh_issues`, whose tabularize contribution goes
+0 → +4,718 as union-schema reaches a nested non-uniform array; `array-of-records` is
byte-identical either side. Its real target is `runecho structure` (~10% of live fleet
traffic, previously compressing at 0.6%). That is the point of keeping this sweep: it
says what a change does *not* regress, on a corpus that has not moved in twenty
releases.

**`README.md` and `BENCHMARKS.md` still publish the pre-#202 figures** (58.3% weighted,
`gh_issues` 32.7%). Tracked in #206 — refresh them from one `terse measure` run rather
than by hand-patching the two headline cells.

## Provenance

The 2026-07-31 run wrote twenty per-version reports plus a `results.tsv` into a
`/tmp` scratchpad. Those are gone; this file is the extraction. Two things worth
knowing if you compare against anything written before 2026-08-04:

- The original `results.tsv` was **empty of savings data**. Its harness grepped
  `^(TOTAL|total|weighted)`, which matches `Total payloads captured: **9**` — so it
  recorded the payload count twenty times and the savings series survived only inside
  the per-version reports. `version_sweep.sh` now greps the `| **ALL** |` row.
- The original consumed a hand-built envelope set whose filenames (`{sha8}.json`)
  predate the current `{tool}__{sha8}.json` scheme, so it could not be rebuilt by
  today's `terse capture`. The script now regenerates envelopes from the tracked
  corpus instead, which changes envelope *filenames* but not a single token count —
  `measure` reads each envelope's `raw` field.
