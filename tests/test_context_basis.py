"""#420: `terse stats` published a WIRE-basis saving and divided it by a primer charged in
CONTEXT, over-crediting every break-even by the mirror text block the model never read.

`build_record` folds the text block and the typed field into one figure. That fold is a
deliberate wire basis — #141 fixed the opposite bug — and it is not changing. What was
missing is the other basis: `policy.py` already records the measurement that a
`structuredContent`-reading client discards the block (raw 2,596 chars in context ->
"compress" 1,008 -> "replace" 1,008, no change), so `"replace"` saves stdio bytes and
nothing at all off the model's context.

Ledger-wide the two differ by 1.84x on raw and 1.93x on the saving, and `turns_covered` —
the recurring-primer runway a router is judged against — read 5,460 where the context basis
says 2,836.
"""
from __future__ import annotations

from terse.stats import _context_tokens, aggregate, primer_liability

_ABSENT = object()


def _rec(server="kb", raw=100, out=40, structured=None, structured_out=None, **kw):
    """One ledger record. `structured*` are the recoverable typed-field split (#134/#141);
    `raw`/`out` are the FOLDED wire totals, as `build_record` writes them."""
    r = {"server": server, "tool": "t", "decision": "tabularize", "passthrough": False,
         "raw_chars": raw, "out_chars": out, "raw_tokens": raw, "out_tokens": out}
    if structured is not None:
        r["structured_tokens"] = structured
        # Only when asked. Writing it unconditionally is what made the legacy-record test
        # below unable to fail: `structured_out_tokens` was always present, so the
        # `else st` fallback it claimed to cover was never reached, and mutating that
        # fallback to `0` left the whole suite green (review of PR #435).
        if structured_out is not _ABSENT:
            r["structured_out_tokens"] = (structured if structured_out is None
                                          else structured_out)
    r.update(kw)
    return r


def test_a_record_with_no_typed_field_has_the_same_context_and_wire_basis():
    """The 2,266-row half of the live ledger, and every pre-#134 ledger entirely: with no
    typed field the text block IS what the model read, so there is nothing to subtract."""
    assert _context_tokens(_rec(raw=100, out=40), 100, 40) == (100, 40)
    # An explicit zero is "no field", not "a field of zero tokens".
    assert _context_tokens(_rec(raw=100, out=40, structured=0), 100, 40) == (100, 40)


def test_a_typed_field_is_the_whole_context_payload():
    """Where the field exists the model reads it alone. The wire pair carries BOTH halves
    folded together (`build_record`), so the context pair is the typed split by itself."""
    # wire 100/40, of which the typed field is 30 raw -> 12 emitted.
    assert _context_tokens(_rec(raw=100, out=40, structured=30, structured_out=12),
                           100, 40) == (30, 12)


def test_an_untouched_typed_field_lands_the_same_size_on_both_sides():
    """`structured_out` defaults to `structured` in `build_record` for a field terse left
    alone. Reading only one of the two would report a saving on a field nothing touched."""
    assert _context_tokens(_rec(structured=30), 100, 40) == (30, 30)


def test_a_record_predating_the_out_side_split_reads_one_value_on_both_sides():
    """The legacy branch, and it needs its own test because the helper above always writes
    BOTH keys — which is why mutating `else st` to `else 0` once left all 2,370 tests green
    (review of PR #435). 27 rows of the live ledger carry `structured_tokens` with no
    `structured_out_tokens`; reading the missing side as 0 would report a 100% context
    saving on a field nothing touched, which is the inflation #420 exists to remove."""
    legacy = _rec(structured=30, structured_out=_ABSENT)
    assert "structured_out_tokens" not in legacy
    assert _context_tokens(legacy, 100, 40) == (30, 30)
    # A present-but-unreadable out side takes the same fallback, never a fabricated zero.
    assert _context_tokens({"structured_tokens": 30, "structured_out_tokens": None},
                           100, 40) == (30, 30)
    assert _context_tokens({"structured_tokens": 30, "structured_out_tokens": "12"},
                           100, 40) == (30, 30)


def test_an_untokenized_typed_field_falls_back_rather_than_claiming_zero():
    """No tiktoken at write time leaves `structured_tokens` unset or None. That record
    cannot state its context payload, and answering 0 would claim the model received
    nothing — unknown is not zero, the rule this whole module keeps."""
    assert _context_tokens(_rec(structured=None), 100, 40) == (100, 40)
    assert _context_tokens({"structured_tokens": None}, 100, 40) == (100, 40)
    assert _context_tokens({"structured_tokens": "30"}, 100, 40) == (100, 40)


def test_aggregate_carries_both_bases_over_the_same_record_set():
    """Both bases are accumulated in the SAME branch, so they always cover exactly the
    same records. Summing one over a wider set than the other makes them incomparable,
    which is the entire point of publishing them together."""
    agg = aggregate([_rec(raw=100, out=40, structured=30, structured_out=12),
                     _rec(raw=200, out=80)])                      # no typed field
    t = agg["total"]
    assert (t["raw_tokens"], t["out_tokens"]) == (300, 120)
    assert (t["ctx_raw_tokens"], t["ctx_out_tokens"]) == (230, 92)
    row = agg["tools"][0]
    assert (row["ctx_raw_tokens"], row["ctx_out_tokens"]) == (230, 92)


def test_an_untokenized_record_is_in_neither_basis():
    """`untokenized` is excluded from the wire sums already; the context sums must skip the
    same rows or the two stop being comparable."""
    agg = aggregate([_rec(raw=100, out=40, structured=30, structured_out=12),
                     {"server": "kb", "tool": "t", "decision": "tabularize",
                      "raw_chars": 9, "out_chars": 9}])           # no token fields at all
    t = agg["total"]
    assert t["untokenized"] == 1
    assert (t["raw_tokens"], t["ctx_raw_tokens"]) == (100, 30)


def test_break_even_divides_the_CONTEXT_saving_by_the_primer(tmp_path):
    """The defect itself. A primer is charged in context, every turn, so dividing a wire
    saving by it credits the entry with the mirror block it also never paid for. On the
    live ledger that read 5,460 turns of runway where the context basis says 2,836."""
    import json
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "kb.*"},
                                             "tiers": ["minify", "tabularize"]}]}), encoding="utf-8")
    scan = [{"server": "terse", "state": "router", "wraps": "kb", "scope": "user",
             "policy": str(pol)}]
    # Wire saves 60 per record; context saves only 18 of it.
    agg = aggregate([_rec(raw=100, out=40, structured=30, structured_out=12)
                     for _ in range(10)])
    liab = primer_liability(scan, agg)

    assert liab["saved_tokens"] == 180            # CONTEXT
    assert liab["wire_saved_tokens"] == 600       # reported, never divided
    row = liab["servers"][0]
    assert row["saved_per_block"] == 18.0         # NOT 60.0
    assert liab["turns_covered"] == 180 / row["primer_tokens"]


def test_an_agg_without_the_context_keys_falls_back_to_wire(tmp_path):
    """A liability blob can be built from an agg an OLDER terse produced, or a hand-rolled
    one. Those carry no `ctx_*` key, and defaulting a missing key to 0 would publish
    "this server saved nothing" — the unknown-is-not-zero failure, in the direction that
    tells an operator to unwrap a server that is paying for itself."""
    import json
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "policies": []}), encoding="utf-8")
    scan = [{"server": "terse", "state": "router", "wraps": "kb", "scope": "user",
             "policy": str(pol)}]
    legacy = {"total": {"raw_tokens": 1000, "out_tokens": 400, "blocks": 4},
              "tools": [{"server": "kb", "tool": "t", "blocks": 4, "tokenized": 4,
                         "raw_tokens": 1000, "out_tokens": 400}],
              "decisions": {}, "diff_reasons": {}}
    liab = primer_liability(scan, legacy)
    assert liab["saved_tokens"] == 600 == liab["wire_saved_tokens"]
    assert liab["servers"][0]["saved_per_block"] == 150.0


def test_the_report_names_the_basis_and_publishes_the_wire_figure(tmp_path):
    """Both numbers are real and they are different units. A report that prints one without
    saying which is the #141 confusion in the other direction."""
    import json

    from terse.stats import build_primer_section
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "kb.*"},
                                             "tiers": ["minify", "tabularize"]}]}), encoding="utf-8")
    scan = [{"server": "terse", "state": "router", "wraps": "kb", "scope": "user",
             "policy": str(pol)}]
    agg = aggregate([_rec(raw=1000, out=400, structured=300, structured_out=120)
                     for _ in range(20)])
    text = "\n".join(build_primer_section(primer_liability(scan, agg)))

    assert "the figures above and the saved/block column below are CONTEXT tokens" in text
    assert "12,000 tok left the stdio" in text            # the wire figure, named as wire
    assert "3,600 tok of CONTEXT saved" in text
    # Said BELOW the figures it labels, including the break-even table's saved/block column
    # — an earlier draft's comment claimed "every figure above" while printing first.
    assert text.index("3,600 tok of CONTEXT saved") < text.index("are CONTEXT tokens")


def test_an_uncalled_entry_never_prints_the_wire_figure_alone(tmp_path):
    """Review of PR #435. The basis line was gated on the two bases DIFFERING, but the
    context figure only renders inside the turns/session blocks — both None for an entry
    that was never called. So the section printed `… 12,000 tok left the stdio pipe` and
    the context saving appeared nowhere, under a header announcing the context basis. An
    operator reads the wire number as the context one: #141's confusion, re-created."""
    import json

    from terse.stats import build_primer_section
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "kb.*"},
                                             "tiers": ["minify", "tabularize"]}]}),
                   encoding="utf-8")
    # Wrapped, with typed fields in the ledger under a label this entry does NOT own, so
    # both ratios are None.
    scan = [{"server": "kb", "state": "wrapped", "wraps": "other-server", "scope": "user",
             "policy": str(pol)}]
    agg = aggregate([_rec(server="zzz", raw=1000, out=400, structured=300,
                          structured_out=120) for _ in range(20)])
    liab = primer_liability(scan, agg)
    text = "\n".join(build_primer_section(liab))

    assert liab["turns_covered"] is None and liab["session_covered"] is None
    assert "CONTEXT tokens" not in text
    assert "left the stdio" not in text


def test_the_verdict_screen_names_the_basis_too(tmp_path):
    """`--recommend` REPLACES the ledger tables, so the line `build_primer_section` adds
    never reaches it — and that is the screen USAGE documents as the one an operator reads
    for the verdict. Its `coverage` column moved 2,840 -> 1,739 on the live fleet with
    nothing on screen saying why (review of PR #435)."""
    import json

    from terse.stats import build_recommend_report
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "kb.*"},
                                             "tiers": ["minify", "tabularize"]}]}),
                   encoding="utf-8")
    scan = [{"server": "terse", "state": "router", "wraps": "kb", "scope": "user",
             "policy": str(pol)}]
    agg = aggregate([_rec(raw=1000, out=400, structured=300, structured_out=120)
                     for _ in range(20)])
    out = build_recommend_report(agg, log_path="x.jsonl",
                                 liability=primer_liability(scan, agg))
    assert "are CONTEXT tokens" in out
    assert "wire saving (#420)" in out


def test_a_legacy_aggregate_says_its_saving_is_on_the_WIRE_basis(tmp_path):
    """Review of PR #435. The fallback to the wire pair is right — zero would publish "this
    server saved nothing" — but it is not silently equivalent, and an earlier comment said
    it was. The realistic legacy shape is a PRE-#420 aggregate over a POST-#134 ledger: no
    `ctx_*` keys, plenty of typed fields, and the two bases differ by the full inflation
    (3.33x on this fixture). So the row says which basis it is on, the same discipline
    `primer_source` keeps for the primer half."""
    import json
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "kb.*"},
                                             "tiers": ["minify", "tabularize"]}]}),
                   encoding="utf-8")
    scan = [{"server": "terse", "state": "router", "wraps": "kb", "scope": "user",
             "policy": str(pol)}]
    recs = [_rec(raw=1000, out=400, structured=300, structured_out=120) for _ in range(20)]

    modern = primer_liability(scan, aggregate(recs))
    assert modern["saved_basis"] == "context"
    assert modern["saved_tokens"] == 3_600

    legacy = aggregate(recs)
    legacy["total"] = {k: v for k, v in legacy["total"].items() if not k.startswith("ctx_")}
    legacy["tools"] = [{k: v for k, v in r.items() if not k.startswith("ctx_")}
                       for r in legacy["tools"]]
    blob = primer_liability(scan, legacy)
    assert blob["saved_basis"] == "wire"          # says so rather than looking measured
    assert blob["saved_tokens"] == 12_000         # ...and falls back, never to zero


def test_every_saving_figure_in_the_section_names_its_basis(tmp_path):
    """Review of PR #435. Three of the four `saved` render sites were relabelled and the
    fourth — the one-time NET NEGATIVE branch — was missed. That is the single sentence in
    this report designed to stop an operator, and it was the only saving figure left with
    no basis on it.

    Driven through the renderer rather than asserted per line, so a fifth site added later
    fails here without anyone remembering to update a list."""
    import json
    import re

    from terse.stats import build_primer_section
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "kb.*"},
                                             "tiers": ["minify", "tabularize"]}]}),
                   encoding="utf-8")
    # A lazy standalone whose one record saves far less than its one-time primer.
    scan = [{"server": "kb", "state": "wrapped", "wraps": "kb", "scope": "user",
             "policy": str(pol)}]
    agg = aggregate([_rec(server="kb", raw=100, out=60, structured=30, structured_out=20)])
    liab = primer_liability(scan, agg)
    text = "\n".join(build_primer_section(liab))

    assert liab["session_covered"] is not None and liab["session_covered"] < 1
    assert "NET NEGATIVE" in text

    # ...and the ROUTER's net-negative branch, which is a different line in a different
    # block. Both have to be driven or half the vocabulary stays unmeasured.
    router = [{"server": "terse", "state": "router", "wraps": "kb", "scope": "user",
               "policy": str(pol)}]
    r_liab = primer_liability(router, agg)
    r_text = "\n".join(build_primer_section(r_liab))
    assert r_liab["turns_covered"] is not None and r_liab["turns_covered"] < 1
    assert "NET NEGATIVE" in r_text

    # Every "N tok saved"/"N tok covers" figure in either rendering carries the basis word.
    bare = [ln for ln in (text + "\n" + r_text).split("\n")
            if re.search(r"\d[\d,]* tok (saved|covers)", ln) and "CONTEXT" not in ln]
    assert not bare, bare


def test_a_fleet_with_no_typed_fields_prints_no_basis_line(tmp_path):
    """The two bases coincide for a pre-#134 fleet, and a line explaining a distinction
    that does not apply is noise. Gated on the figures actually differing."""
    import json

    from terse.stats import build_primer_section
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1,
                               "policies": [{"match": {"tool": "kb.*"},
                                             "tiers": ["minify", "tabularize"]}]}), encoding="utf-8")
    scan = [{"server": "terse", "state": "router", "wraps": "kb", "scope": "user",
             "policy": str(pol)}]
    agg = aggregate([_rec(raw=1000, out=400) for _ in range(20)])
    text = "\n".join(build_primer_section(primer_liability(scan, agg)))

    assert "are CONTEXT tokens" not in text
    assert "12,000 tok of CONTEXT saved" in text
