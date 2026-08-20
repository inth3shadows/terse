"""The primer's COST, recorded rather than inferred (#311) — and the phantom bill (#286).

`primer_liability` sizes a primer from the INSTALLED policy and then uses the ledger only
to decide who was called. Its own docstring concedes what that cannot see: a session whose
every compressible result also carried `structuredContent` never reaches the lazy attach,
so the server was called, paid nothing, and was billed anyway. #286 is that bill observed
in production — `searxng-mcp` charged 312 tok/session for a primer it is structurally
incapable of sending.

No read-side cleverness recovers this, because the fact is only known at the attach site.
So the attach site records it. These tests pin the payload-free property, that a primer row
can never reach the published savings figure, and — the point of the exercise — that the
row appears if and ONLY if the primer really went out.
"""

from __future__ import annotations

import json

from terse import transforms
from terse.policy import Policy, Rule
from terse.proxy import PRIMER_HEAD, Interceptor
from terse.stats import (
    PRIMER_CADENCE_ONCE,
    PRIMER_EVENT,
    aggregate,
    build_primer_record,
    primer_liability,
)
from terse.tokenize import count_cl100k

FULL = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))])


def _records_text():
    return json.dumps({"result": [{"id": i, "status": "active", "url": "https://x/api"}
                                  for i in range(20)]}, indent=2)


def _result_msg(mid, text, structured=False):
    result = {"content": [{"type": "text", "text": text}]}
    if structured:
        # The exact shape the lazy attach refuses to prime on: Claude Code discards the
        # text block entirely when a result also carries `structuredContent`, so a primer
        # inserted here would be thrown away with it.
        result["structuredContent"] = {"result": [{"id": 1}]}
    return json.dumps({"jsonrpc": "2.0", "id": mid, "result": result})


def _note_call(inter, mid, name):
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": name}}))


def _primed(structured=False):
    """Drive one compressible result through a lazily-primed Interceptor and return
    (emitted result, recorded primer rows)."""
    rows: list[tuple[str, str]] = []
    inter = Interceptor(FULL, stats_primer=lambda cadence, text: rows.append((cadence, text)))
    _note_call(inter, 2, "gh.api.items")
    out = json.loads(inter.transform_response(
        _result_msg(2, _records_text(), structured=structured)))
    return inter, out, rows


# --- the record itself ---

def test_a_primer_record_never_enters_the_savings_total():
    """THE load-bearing guard, mirroring the retrieve row's. `aggregate` skips any row
    without int raw_chars/out_chars. A primer row must therefore never carry them —
    otherwise every primer would count as a compressed block and fold its bytes into the
    one number terse publishes about itself."""
    rec = build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="P" * 400)
    agg = aggregate([rec])
    assert agg["total"]["blocks"] == 0
    assert agg["total"]["raw_chars"] == 0 and agg["total"]["out_chars"] == 0
    # ...and the cost is not merely discarded — it lands on its own axis.
    assert agg["primers"][0]["emissions"] == 1 and agg["primers"][0]["bytes"] == 400

    # TWO independent guards, both pinned because either alone would pass this test while
    # leaving the other free to rot:
    #   1. the `event` marker, checked first
    assert rec["event"] == PRIMER_EVENT
    assert aggregate([dict(rec, raw_chars=400, out_chars=400)])["total"]["blocks"] == 0
    #   2. the absence of the result-record size keys, which excludes a row whose event
    #      marker was lost (an older writer, a hand-edited ledger)
    assert "raw_chars" not in rec and "out_chars" not in rec
    stripped = {k: v for k, v in rec.items() if k != "event"}
    assert aggregate([stripped])["total"]["blocks"] == 0
    # Control: a row that is BOTH unmarked and result-shaped IS counted — so the two
    # assertions above are about the guards, not about `aggregate` ignoring everything.
    assert aggregate([dict(stripped, raw_chars=400, out_chars=400)])["total"]["blocks"] == 1


def test_the_record_carries_no_payload_only_its_size():
    """Payload-free like every other record: the primer is measured and discarded."""
    rec = build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="SECRET" * 50)
    assert "SECRET" not in json.dumps(rec)
    assert rec["bytes"] == 300 and rec["cadence"] == PRIMER_CADENCE_ONCE


# --- the write site: recorded if and only if it actually went out ---

def test_the_lazy_attach_records_exactly_one_primer():
    inter, out, rows = _primed()
    blocks = out["result"]["content"]
    assert PRIMER_HEAD in blocks[0]["text"]          # it really was attached
    assert transforms.decompress(blocks[1]["text"]) == json.loads(_records_text())
    assert inter._primer_sent is True
    assert len(rows) == 1
    cadence, text = rows[0]
    assert cadence == PRIMER_CADENCE_ONCE
    assert PRIMER_HEAD in text                        # the real text, so the size is real


def test_a_second_result_does_not_bill_the_primer_again():
    """Once per session is the whole point of #211. A per-result record would restore the
    per-turn cost in the ledger even though the wire is correct."""
    inter, _out, rows = _primed()
    _note_call(inter, 3, "gh.api.items")
    inter.transform_response(_result_msg(3, _records_text()))
    assert len(rows) == 1


def test_a_structuredContent_result_records_no_primer():
    """#286, pinned. The lazy attach is gated on the absence of `structuredContent`, so a
    server whose results always carry it pays ZERO forever — and the ledger must say so
    rather than leaving the reader to infer payment from "it was called".

    If the guard at the attach site is ever removed, this test fails: the mutation attaches
    a primer, which records a row, which makes `rows` non-empty.
    """
    inter, out, rows = _primed(structured=True)
    assert inter._primer_sent is False                # nothing was attached...
    assert PRIMER_HEAD not in json.dumps(out)         # ...on the wire either
    assert rows == []                                 # ...so nothing was billed


def test_a_session_that_never_calls_a_wrapped_tool_records_nothing():
    rows: list[tuple[str, str]] = []
    inter = Interceptor(FULL, stats_primer=lambda c, t: rows.append((c, t)))
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                   "params": {}}))
    inter.transform_response(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
        "protocolVersion": "1", "capabilities": {}, "serverInfo": {"name": "s"}}}))
    assert inter._primer_sent is False
    assert rows == []


def test_the_eager_initialize_site_records_nothing_by_design():
    """Scope decision, pinned so it is a choice rather than an omission (#311).

    An eagerly-primed entry injects into `initialize.instructions` unconditionally whenever
    its policy yields a primer — the same predicate `primer_liability` already evaluates
    from the installed policy. Inference there is exact, so a row would cost bytes without
    adding knowledge. Only the LAZY attach is unobservable from outside the process.
    """
    rows: list[tuple[str, str]] = []
    inter = Interceptor(FULL, lazy_primer=False,
                        stats_primer=lambda c, t: rows.append((c, t)))
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                   "params": {}}))
    out = json.loads(inter.transform_response(json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "result": {"protocolVersion": "1", "capabilities": {}, "serverInfo": {"name": "s"}}})))
    assert PRIMER_HEAD in out["result"]["instructions"]   # it DID prime...
    assert rows == []                                     # ...and deliberately did not bill


def test_a_write_failure_never_reaches_the_client():
    """Fail-open, same contract as every other stats path: the ledger is never
    load-bearing, so a broken writer degrades the report and not the proxy."""
    def boom(cadence, text):
        raise RuntimeError("ledger is on fire")

    inter = Interceptor(FULL, stats_primer=boom)
    _note_call(inter, 2, "gh.api.items")
    out = json.loads(inter.transform_response(_result_msg(2, _records_text())))
    blocks = out["result"]["content"]
    assert PRIMER_HEAD in blocks[0]["text"]
    assert transforms.decompress(blocks[1]["text"]) == json.loads(_records_text())


# --- the read side: recorded beats inferred, and the two are never blended ---

def _scan_row(name="gh", state="wrapped", wraps="gh-server"):
    return {"server": name, "state": state, "wraps": wraps, "scope": "user", "policy": None}


def _agg_with(blocks_for="gh", primer_rows=()):
    """An aggregate carrying one compressed block for `blocks_for`, plus any primer rows."""
    recs = [{"server": blocks_for, "tool": "gh.api.items", "raw_chars": 1000,
             "out_chars": 400, "raw_tokens": 250, "out_tokens": 100,
             "decision": "tabularize", "passthrough": False}]
    return aggregate(list(recs) + list(primer_rows))


def test_a_recorded_primer_replaces_the_inferred_size():
    """The correction itself. Inference sizes the primer from policy and assumes a called
    server paid it; a recorded row says what actually went out.

    Asserted against an INDEPENDENTLY computed expectation, not against
    `aggregate([rec])["primers"][0]["tokens"]`. The original version did the latter and was
    structurally incapable of failing — it compared the reader's output to the same sum the
    reader consumed, so it stayed green through the window-sum bug below."""
    rec = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    liab = primer_liability([_scan_row()], _agg_with(blocks_for="gh-server",
                                                     primer_rows=[rec]))
    row = liab["servers"][0]
    assert row["primer_source"] == "recorded"
    assert row["primer_tokens"] == count_cl100k("P" * 40)


def test_a_multi_session_window_reports_ONE_primer_not_the_windows_sum():
    """THE regression this file exists for after review. `primer_tokens` is a PER-SESSION
    charge: `_break_even` divides by it to get "blocks once per session" and the report
    renders it "N tok/session". `aggregate` sums a whole WINDOW, and a standalone proxy is
    one process per session writing one row per session — so assigning the window sum to
    that field over-bills by the session count and flips break-even to NET NEGATIVE on any
    multi-session window, which is the default (`terse stats` with no `--since` reads all
    history).

    Same mis-denomination class #312 was closed for, and it made the "measured" number
    WORSE than the estimate it replaced. Pinned across several session counts because the
    single-row fixture above cannot see it: at one emission the mean and the sum agree."""
    one = count_cl100k("P" * 40)
    for sessions in (1, 2, 5, 20):
        rows = [build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE,
                                    primer="P" * 40) for _ in range(sessions)]
        liab = primer_liability([_scan_row()],
                                _agg_with(blocks_for="gh-server", primer_rows=rows))
        row = liab["servers"][0]
        assert row["primer_source"] == "recorded"
        assert row["primer_tokens"] == one, (
            f"{sessions} sessions reported {row['primer_tokens']} for a {one}-token primer")
        # ...and the consumers that divide by it stay in per-session units too.
        assert liab["session_once_tokens"] == one
    # Break-even must not drift with session count either — it is "blocks ONCE PER SESSION".
    be = [primer_liability([_scan_row()],
                           _agg_with(blocks_for="gh-server",
                                     primer_rows=[build_primer_record(
                                         "gh-server", cadence=PRIMER_CADENCE_ONCE,
                                         primer="P" * 40) for _ in range(n)])
                           )["servers"][0]["blocks_to_break_even"] for n in (1, 20)]
    assert be[0] == be[1]


def test_emissions_with_no_token_count_are_not_published_as_a_measurement():
    """A tiktoken-less terse records that the primer WENT OUT but not what it cost. The
    divisor is tokenized emissions, so a row whose tokens are unknown must not be treated as
    a zero-cost emission — that would drag the mean down and publish a fabricated
    measurement under the `recorded` label."""
    rec = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    rec["tokens"] = None                       # exactly what a tiktoken-less writer emits
    liab = primer_liability([_scan_row()], _agg_with(blocks_for="gh-server",
                                                     primer_rows=[rec]))
    row = liab["servers"][0]
    assert row["primer_source"] == "estimated"     # falls back, does not invent
    assert row["primer_tokens"]                    # still sized from policy

    # A corrupt divisor must also fall back rather than publish an inflated figure. A
    # hand-edited ledger can carry `untokenized > emissions`, making the tokenized-emission
    # count NEGATIVE — and a negative divisor is how an over-bill would re-enter through
    # the very arithmetic the mean exists to prevent.
    #
    # TWO guards stop it and neither is pinned alone here, because either one suffices and
    # a test asserting one would pass with it deleted: the `tokenized_emissions <= 0` skip
    # in the accumulator, and `measured = rec_em > 0` at the point of use. What IS pinned is
    # the property that matters — a nonsense divisor yields the labelled estimate, never a
    # number presented as a measurement.
    good = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    agg = _agg_with(blocks_for="gh-server", primer_rows=[good, good, good])
    for prow in agg["primers"]:
        prow["emissions"] += 1          # a fourth emission...
        prow["untokenized"] += 5        # ...with an untokenized count that cannot be real
    liab2 = primer_liability([_scan_row()], agg)
    corrupt = liab2["servers"][0]
    assert corrupt["primer_source"] == "estimated"
    assert corrupt["primer_tokens"] == row["primer_tokens"]   # the policy size, untouched
    assert corrupt["primer_tokens"] > 0                       # and never negative/zero


def test_a_recorded_emission_beats_the_encoded_zero_inference():
    """#311 review. `_cadence` infers "never paid" from `encoded == 0`, and its own
    docstring names the path where that is wrong: a `passthrough` result whose downstream
    payload quotes a terse marker attaches the primer while `encoded` stays 0. A recorded
    row settles it. Before this, one report called the server MEASURED and listed it under
    "costing nothing at all" simultaneously, and dropped the charge from the total."""
    rec = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    passthrough = {"server": "gh-server", "tool": "gh.api.items", "raw_chars": 900,
                   "out_chars": 900, "raw_tokens": 200, "out_tokens": 200,
                   "decision": "passthrough", "passthrough": True}
    liab = primer_liability([_scan_row()], aggregate([passthrough, rec]))
    row = liab["servers"][0]
    assert row["primer_source"] == "recorded"
    assert row["cadence"] == "once/session"          # NOT "once/session (unpaid)"
    assert row["server"] not in liab["free"]
    assert liab["session_once_tokens"] == count_cl100k("P" * 40)


def test_no_recorded_row_keeps_the_inference_and_says_so():
    """Absence is NOT yet read as evidence of non-payment: a window written before this
    field existed has no primer rows at all and cannot be distinguished here from one whose
    attach never fired. Such an entry keeps the old estimate and is LABELLED, rather than
    being silently mixed in with the measured ones."""
    liab = primer_liability([_scan_row()], _agg_with(blocks_for="gh-server"))
    row = liab["servers"][0]
    assert row["primer_source"] == "estimated"
    assert row["primer_tokens"]           # still sized from policy, as before


def test_measured_and_estimated_are_never_summed_into_one_unlabelled_number():
    """#312 was closed for mis-denominating a ratio. The same error one layer down would be
    adding a measured primer to an inferred one and publishing the total as if both were
    facts. The report has to name the split."""
    rec = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    agg = _agg_with(blocks_for="gh-server", primer_rows=[rec])
    liab = primer_liability([_scan_row()], agg)
    from terse.stats import build_primer_section
    text = "\n".join(build_primer_section(liab))
    assert "MEASURED" in text
    assert "inferred to have paid" in text


def test_the_primer_write_happens_after_the_lock_is_released():
    """#311 review, and the one finding the two reviewers disagreed on.

    Every other sink (capture/audit/stats) is queued into `deferred` and run once
    `_local_lock` is released. The comment above that list says why in terms a try/except
    cannot satisfy: it catches a sink that RAISES, and does nothing for one that BLOCKS —
    a stalled mount or a full disk mid-rotation would hold `_local_lock`, freeze
    `note_request` (same lock) and wedge every later tools/call on the connection.
    `append_stats` stats, may rotate, reads and appends a file, so it is exactly that kind
    of sink. The primer write shipped INSIDE the lock.

    Asserted by having the writer itself try to take the lock: if it still ran under the
    lock this deadlocks (non-reentrant `Lock`), so the test would hang rather than fail —
    hence the explicit non-blocking probe, which turns a hang into a clean assertion.
    """
    seen: list[bool] = []
    inter = Interceptor(FULL)

    def probe(cadence, text):
        # True = the lock was free at write time, i.e. the write is genuinely deferred.
        got = inter._local_lock.acquire(blocking=False)
        seen.append(got)
        if got:
            inter._local_lock.release()

    inter.stats_primer = probe
    _note_call(inter, 2, "gh.api.items")
    out = json.loads(inter.transform_response(_result_msg(2, _records_text())))
    assert PRIMER_HEAD in out["result"]["content"][0]["text"]   # it really attached
    assert seen == [True], "the primer ledger write ran while _local_lock was held"


def test_a_blocking_primer_writer_cannot_wedge_the_next_call():
    """The consequence the deferral buys, as behaviour rather than structure: a slow sink
    delays only its own response's return, and `note_request` stays available. Pinned with a
    writer that BLOCKS (not one that raises) because that is the case try/except never
    covered.

    SYNCHRONISED, not timed (found in re-review). The first version started both threads and
    gave `note_request` a 3s budget, which measured thread-scheduling latency rather than
    lock availability -- on a loaded runner it hit 3.4s and failed while the code was
    correct. Now `slow` signals on ENTRY and the probe does not start until that signal
    arrives, so by construction the sink is executing when `note_request` is attempted. The
    remaining timeouts are deadlock escapes, not measurements: if the write were back inside
    the lock, `entered` would be set while the lock is held and the probe could never finish
    at any budget.
    """
    import threading
    entered = threading.Event()
    release = threading.Event()
    inter = Interceptor(FULL)
    attached: list[str] = []

    def slow(cadence, text):
        attached.append(cadence)
        entered.set()                  # the sink is now running
        release.wait(timeout=10)       # ...and stays running until we say otherwise

    inter.stats_primer = slow
    _note_call(inter, 2, "gh.api.items")
    responder = threading.Thread(
        target=lambda: inter.transform_response(_result_msg(2, _records_text())),
        daemon=True)
    responder.start()
    assert entered.wait(timeout=10), "the primer sink never ran at all"

    # The sink is mid-write RIGHT NOW. If that write held `_local_lock`, this cannot finish.
    done = threading.Event()
    threading.Thread(target=lambda: (_note_call(inter, 3, "gh.api.items"), done.set()),
                     daemon=True).start()
    unblocked = done.wait(timeout=10)
    release.set()
    responder.join(timeout=10)

    # Not a vacuous pass: the earlier version asserted only `unblocked`, which is trivially
    # true if the attach never fires and the sink is never called at all.
    assert attached == ["once/session"], "the primer never attached, so nothing was proven"
    assert unblocked, "note_request was blocked by an in-flight primer-ledger write"


def test_zero_token_emissions_are_never_published_as_a_measured_free_primer():
    """Re-review, flagged for symmetry. A corrupt ledger can carry emissions whose tokens
    total zero. Publishing `primer_tokens: 0` under `primer_source: "recorded"` claims we
    MEASURED a free primer — the same fabrication the accumulator's `tokenized_emissions`
    skip exists to prevent, entered through the numerator instead of the divisor.

    Unreachable from the proxy (an empty primer sets `_primer_sent` at construction, so the
    attach never fires), which is why it is pinned here rather than left to chance."""
    rec = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    rec["tokens"] = 0                      # emissions happened; they cost "nothing"
    liab = primer_liability([_scan_row()], _agg_with(blocks_for="gh-server",
                                                     primer_rows=[rec]))
    row = liab["servers"][0]
    assert row["primer_source"] == "estimated"
    assert row["primer_tokens"] > 0
