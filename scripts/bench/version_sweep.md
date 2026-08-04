# Version sweep — Tier-0 savings across releases

What the codec actually delivers on a **fixed** corpus, release by release. Run
`./version_sweep.sh` to regenerate; it checks out every release tag in turn and
measures the same nine payloads from [`corpus/`](corpus/).

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

`transforms.py` gained 147 lines between `v0.5.1` and `v0.17.0`, so twenty identical
rows look like a sweep that measured one installed build twenty times. It isn't —
verified 2026-08-04 by re-running `v0.5.1`'s own source in a fresh worktree against
the tracked corpus:

```
v0.5.1 codec: raw=365144  terse=152411  saved=212733  58.3%
```

Exactly the recorded figure. The churn in that range — the opt-in `embedded` tier,
NaN-aware gate equality, the diff tier, marker guards — either defaults off or
does not touch Tier-0 on nine uniform GitHub payloads. The codec math genuinely
did not move.

### First movement in the whole range: #202

Union-schema tabularize is the first change since `v0.5.1` to shift this corpus:

```
v0.5.1 .. v0.17.0 : 152,411 tok  58.3%
main (with #202)  : 149,486 tok  59.1%   (+0.8pp, +2,925 tok)
```

Modest here **by construction** — these nine payloads are uniform records or compact
JSON, and union-schema only reaches non-uniform ones. Its real target is
`runecho structure` (~10% of live fleet traffic, previously compressing at 0.6%).
That is the point of keeping this sweep: it says what a change does *not* regress,
on a corpus that has not moved in twenty releases.

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
