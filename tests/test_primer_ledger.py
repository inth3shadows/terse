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
    server paid it; a recorded row says what actually went out."""
    rec = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    liab = primer_liability([_scan_row()], _agg_with(blocks_for="gh-server",
                                                     primer_rows=[rec]))
    row = liab["servers"][0]
    assert row["primer_source"] == "recorded"
    assert row["primer_tokens"] == aggregate([rec])["primers"][0]["tokens"]


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
