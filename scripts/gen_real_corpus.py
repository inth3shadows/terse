"""Generate a REAL-PAYLOAD corpus for the fluency eval, from the bench GitHub capture.

The stress corpus (`gen_stress_corpus.py`) is adversarial by construction: it maximizes
alias resolution and wide/long table lookup precisely because those are the transforms
most likely to cost comprehension. A verdict drawn from it alone answers "does terse
survive the worst case", not "what does terse cost on the payloads it actually wraps".
#249's operator note makes a real-payload arm a PRECONDITION of shipping any primer
default, not a follow-up.

The obvious construction — cap `scripts/bench/corpus/` by token size — biases the result.
Measured (cl100k, full payloads), an ~8.5k cap keeps four (gh_dir_listing, gh_repo_single,
gh_labels, gh_rate_limit) and cuts five, including terse's four largest SAVINGS by absolute
tokens: gh_pulls, gh_workflow_runs, gh_issues, gh_commits. (By percentage the 4th-largest
is gh_dir_listing, which the cap keeps — the ranking that matters here is tokens saved.)
The per-payload figures are not restated here on purpose: they live in BENCHMARKS.md §1,
which `tests/test_published_benchmarks.py` pins, and a third hand-copied version of that
table would drift silently since `fetch_corpus.sh` can re-fetch the source data.

What the cap leaves is dominated by payloads small enough that repetition never
accumulates, and one survivor (gh_repo_single) compresses by nothing at all. It is not a
clean "keeps only the low-compression shapes" filter — it also cuts gh_commits_flat at
2.4% — but it does remove the top of the range, which is the half the comprehension
question is actually about.

So this takes a RECORD PREFIX of each payload instead. Every payload stays represented,
and each keeps its own nesting and intra-record repetition — only the number of records
shrinks, to bring each into a token band a fluency question can be asked over without the
prompt dominating the run.

Three caveats, stated up front rather than discovered later:

  - THE PREFIX MAKES THE QUESTIONS EASIER, not just the payload smaller. `gen_questions`
    derives the count/enumerate answers from the records present, so an N-record prefix
    asks the model to count to N and to enumerate N values. `questions.py:333` records
    that "under-enumeration of wide tables was terse's measured recall gap" — precisely
    what a short prefix stops testing. This is the flattery channel of this construction
    and it is why the prefixes below are as large as the token budget allows rather than
    as small as the exam permits. It cannot be eliminated: gh_pulls is ~5.0k tok/record,
    so its full 30 records is a 151k-token prompt. Read any verdict from this corpus as
    "at these record counts", never as "at production scale".
  - A short prefix also loses CROSS-record repetition, so the codec here runs below the
    published figure (gh_pulls reaches 68.9% at 6 records against 76.1% at 30). The prefix
    shifts which tier does the work; this is not a reproduction of BENCHMARKS §1.
  - Not every payload contributes to the exam. Some generate no questions at all, and
    `gh_repo_single` compresses to a byte-identical result *when tiktoken is installed*, so
    its questions ask the model the same text in both arms and can only tie. The summary
    table flags both, because a headline codec range computed over payloads that measure
    nothing overstates coverage. Both flags are tokenizer-dependent — without tiktoken,
    `compress` picks candidates by a byte heuristic and the vacuous set changes — so the
    summary says so in that mode.

    python scripts/gen_real_corpus.py [corpus_dir]   # default: corpus-real
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terse.capture import capture_payload, load_corpus  # noqa: E402
from terse.fluency.questions import gen_questions  # noqa: E402
from terse.tokenize import count_cl100k  # noqa: E402
from terse.transforms import compress, minify  # noqa: E402

BENCH_CORPUS = Path(__file__).resolve().parent / "bench" / "corpus"

# Records to keep per payload; None = the whole payload (already small enough, or not a
# record list at all). Sized AS LARGE AS a ~30k cl100k prompt allows, not as small as the
# exam permits — see caveat 1: the record count is the count/enumerate answer, so a short
# prefix quietly converts the hardest payloads into "count to 2". At the previous values
# (pulls 2, runs 2) those two asked exactly that, at 44.3%/55.3% codec against the
# 76.1%/80.3% production emits.
#
# Two hard floors, both load-bearing:
#   - Never lower one of these to 1. `gen_questions` needs >= 2 records to build a
#     record-list question (`transforms.py:182`), so a 1-record prefix silently drops the
#     payload out of the exam entirely — which for gh_pulls and gh_workflow_runs would
#     remove terse's two highest-compression shapes, the exact bias this file exists to
#     avoid.
#   - Raising these costs run time and degraded calls superlinearly (4 arms x trials, and
#     #268's no-content failures scale with prompt length). ~30k is the ceiling that keeps
#     a 4-arm trials=5 panel run tractable.
PREFIXES: dict[str, int | None] = {
    "gh_pulls": 6,
    "gh_workflow_runs": 6,
    "gh_issues": 8,
    "gh_rate_limit": None,
    "gh_dir_listing": None,
    "gh_labels": None,
    "gh_commits": 8,
    "gh_repo_single": None,
    "gh_commits_flat": 25,
}


def prefix_of(obj, n: int | None):
    """Take the first `n` records, or the whole object when it is not a record list."""
    if n is None or not isinstance(obj, list):
        return obj
    return obj[:n]


def _fmt_tok(tok: int | None) -> str:
    """tiktoken absence is a supported configuration (see terse.tokenize), so the
    summary has to render a missing count rather than crash formatting it."""
    return f"{tok:,}" if tok is not None else "n/a"


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus-real")
    rows = []
    for tool, n in PREFIXES.items():
        src = BENCH_CORPUS / f"{tool}.json"
        if not src.exists():
            # `fetch_corpus.sh` re-fetches eight of these from the live GitHub API;
            # `gh_commits_flat.json` has no producer in the repo and is a committed
            # artifact, so restore that one from git rather than by re-fetching.
            print(f"missing {src} — restore it with `git checkout -- {src}`, or "
                  f"re-fetch the API-derived payloads with scripts/bench/fetch_corpus.sh",
                  file=sys.stderr)
            return 1
        obj = prefix_of(json.loads(src.read_text(encoding="utf-8")), n)
        raw = minify(obj)
        # `capture_payload` is the same path the proxy and `terse capture` use, so the
        # envelopes this writes are indistinguishable from a real captured corpus.
        # max_per_tool=1: envelopes are named by content sha, so re-running with a
        # different PREFIXES value would otherwise ADD a second envelope for the same
        # tool rather than replace it, and `load_corpus` would then score that tool
        # twice at two different sizes. Capping at one per tool evicts the stale one.
        capture_payload(tool, raw, out_dir, max_per_tool=1)
        raw_tok = count_cl100k(raw)
        cmp_text = compress(obj)
        cmp_tok = count_cl100k(cmp_text)
        codec = (1 - cmp_tok / raw_tok) if (raw_tok and cmp_tok is not None) else None
        # Report the scope that was actually APPLIED, measured off the result rather than
        # the request: `prefix_of` returns the whole object untouched when it is not a
        # list, and a re-fetch returning fewer records than the prefix would otherwise
        # print "first 25" for 3 records. `len(obj)` can never disagree with the artifact.
        scope = f"first {len(obj)}" if (n is not None and isinstance(obj, list)) else "whole"
        # Two ways a payload can be in the corpus but absent from the exam.
        nq = len(gen_questions(obj))
        vacuous = cmp_text == raw
        rows.append({"tool": tool, "tok": raw_tok, "codec": codec, "scope": scope,
                     "nq": nq, "vacuous": vacuous})

    rows.sort(key=lambda r: -(r["codec"] if r["codec"] is not None else -1))
    print(f"wrote {len(rows)} payloads -> {out_dir}\n")
    print("| payload | tok | codec | scope | questions | note |")
    print("|---|--:|--:|---|--:|---|")
    for r in rows:
        codec = f"{r['codec']:.1%}" if r["codec"] is not None else "n/a"
        if r["nq"] == 0:
            note = "no questions — not in the exam"
        elif r["vacuous"]:
            note = "terse form == raw — arms can only tie"
        else:
            note = ""
        print(f"| {r['tool']} | {_fmt_tok(r['tok'])} | {codec} | {r['scope']} "
              f"| {r['nq']} | {note} |")

    scored = [r for r in rows if r["nq"] and not r["vacuous"]]
    codecs = [r["codec"] for r in scored if r["codec"] is not None]
    nq_total = sum(r["nq"] for r in scored)
    if not scored:
        print("\nno payload contributes a non-vacuous question — the exam would be empty")
    else:
        # The range that matters is the one over payloads the exam can actually fail on.
        # Without tiktoken there are no codecs to range over AND `compress` picks
        # candidates by a byte heuristic instead, which changes which payloads come out
        # vacuous — so the coverage count is tokenizer-dependent too, not just the range.
        # Saying only "range unavailable" would present a shifted count as if it were firm.
        rng = (f"codec range {min(codecs):.1%}–{max(codecs):.1%}" if codecs
               else "codec range and vacuous flags unavailable/approximate (no tiktoken)")
        print(f"\n{len(scored)}/{len(rows)} payloads contribute {nq_total} non-vacuous "
              f"questions, {rng}")

    # The loop reports what THIS run wrote; the eval loads whatever is in the directory.
    # `max_per_tool=1` replaces a tool's stale envelope but cannot remove a tool dropped
    # from PREFIXES, nor anything captured here previously — so a stale tool would be
    # scored while going unmentioned above. Compare against the artifact and say so.
    extra = sorted({e["tool"] for e in load_corpus(out_dir)} - set(PREFIXES))
    if extra:
        print(f"\nWARNING: {out_dir} also holds envelopes the eval WILL score but this "
              f"run did not write: {', '.join(extra)}. Remove them or add them to "
              f"PREFIXES — the table above does not describe them.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
