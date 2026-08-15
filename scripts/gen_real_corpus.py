"""Generate a REAL-PAYLOAD corpus for the fluency eval, from the bench GitHub capture.

The stress corpus (`gen_stress_corpus.py`) is adversarial by construction: it maximizes
alias resolution and wide/long table lookup precisely because those are the transforms
most likely to cost comprehension. A verdict drawn from it alone answers "does terse
survive the worst case", not "what does terse cost on the payloads it actually wraps".
#249's operator note makes a real-payload arm a PRECONDITION of shipping any primer
default, not a follow-up.

The obvious construction — cap `scripts/bench/corpus/` by token size — biases the result.
Measured (cl100k, full payloads), an ~8.5k cap keeps four:

    gh_dir_listing 6,736 tok / 31.4%   gh_repo_single 1,652 / 0.0%
    gh_labels        632 tok / 15.2%   gh_rate_limit    357 / 13.4%

and cuts five, including every one of terse's four largest wins — gh_workflow_runs
(76,032 / 80.3%), gh_pulls (151,165 / 76.1%), gh_issues (48,032 / 38.8%), gh_commits
(69,652 / 26.5%). What survives is dominated by payloads small enough that repetition
never accumulates, and one survivor (gh_repo_single) compresses by nothing at all. The
cap is not a clean "keeps only the low-compression shapes" filter — it also cuts
gh_commits_flat at 2.4% — but it does remove the top of the range, which is the half the
comprehension question is actually about.

So this takes a RECORD PREFIX of each payload instead. Every payload stays represented,
and each keeps its own nesting and intra-record repetition — only the number of records
shrinks, to bring each into a token band a fluency question can be asked over without the
prompt dominating the run.

Two caveats, stated up front rather than discovered later:

  - A 1-record prefix of `gh_pulls` loses the CROSS-record repetition that produced its
    76.1% headline (dozens of repeated `repo` objects folding to one legend entry). It
    still compresses on intra-record repetition, but the prefix shifts which tier does the
    work. This corpus measures comprehension across a realistic compression range; it is
    not a reproduction of BENCHMARKS §1.
  - Not every payload contributes to the exam. Some generate no questions at all, and
    `gh_repo_single` compresses to a byte-identical result, so its questions ask the model
    the same text in both arms and can only tie. The summary table flags both, because a
    headline codec range computed over payloads that measure nothing overstates coverage.

    python scripts/gen_real_corpus.py [corpus_dir]   # default: corpus-real
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terse.capture import capture_payload  # noqa: E402
from terse.fluency.questions import gen_questions  # noqa: E402
from terse.tokenize import count_cl100k  # noqa: E402
from terse.transforms import compress, minify  # noqa: E402

BENCH_CORPUS = Path(__file__).resolve().parent / "bench" / "corpus"

# Records to keep per payload; None = the whole payload (already small enough, or not a
# record list at all). Chosen to land every payload near ~10k cl100k tokens while keeping
# each payload IN THE EXAM. That second constraint is not free: `gen_questions` needs at
# least 2 records to build a record-list question, so a 1-record prefix silently drops a
# payload out of the exam entirely — which for `gh_pulls` and `gh_workflow_runs` would
# have removed terse's two highest-compression shapes, exactly the bias this construction
# exists to avoid. Never lower one of these to 1.
PREFIXES: dict[str, int | None] = {
    "gh_pulls": 2,
    "gh_workflow_runs": 2,
    "gh_issues": 5,
    "gh_rate_limit": None,
    "gh_dir_listing": None,
    "gh_labels": None,
    "gh_commits": 3,
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
        # Report the scope that was actually APPLIED: `prefix_of` returns the whole
        # object untouched when it is not a list, so `n` alone can name a prefix that
        # was never taken (a payload arriving as a dict envelope instead of a array).
        scope = f"first {n}" if (n is not None and isinstance(obj, list)) else "whole"
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
        # Without tiktoken there are no codecs to range over, but the exam is still real —
        # report the coverage and say the range is unavailable rather than implying both.
        rng = (f"codec range {min(codecs):.1%}–{max(codecs):.1%}" if codecs
               else "codec range unavailable (no tiktoken)")
        print(f"\n{len(scored)}/{len(rows)} payloads contribute {nq_total} non-vacuous "
              f"questions, {rng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
