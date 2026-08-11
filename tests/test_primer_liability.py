"""#168: the primer is a real cost the ledger never sees, and `terse stats` has to say so.

The ledger charges terse for the payloads it compresses and never for the context it adds,
so `terse stats` could report a win in a session that was a net loss — measured from outside
terse as a 14.0% win at one wrapped server and a 2.1% LOSS at three.

There are TWO cadences and they are never summed (#211 follow-up): a multiproxy router still
primes eagerly at `initialize`, which the client re-reads every turn, while a standalone
`terse proxy` attaches its primer once, to the first compressible result, and not at all if
that result never comes. Charging the second as if it were the first is what these tests
mostly pin now — it inflated the headline by a whole session's worth of turns and told the
operator that the servers #211 made FREE were "pure cost".

What is deliberately NOT built here is a per-turn charge in the ledger: `turns` is not
observable from a stdio proxy, and inventing one would be the #144/#186/#188 defect family
again — a number describing something the code never measured. These tests pin the
break-even framing that replaces it, and the ways sizing it can quietly lie: counting from
the ledger (which cannot see the install that is pure cost), mistaking "unknown" for "never
called", and now mistaking a one-time charge for a recurring one.
"""
from __future__ import annotations

import json

import pytest

from terse.stats import (
    build_primer_section,
    build_recommend_section,
    build_stats_report,
    primer_liability,
)


def _policy(tmp_path, name="p.json", tool="kb.*", tiers=("minify", "tabularize")):
    path = tmp_path / name
    path.write_text(json.dumps({"version": 1,
                                "policies": [{"match": {"tool": tool},
                                              "tiers": list(tiers)}]}), encoding="utf-8")
    return str(path)


def _scan(server, state, wraps, policy, scope="user"):
    return {"scope": scope, "server": server, "state": state, "wraps": wraps,
            "policy": policy}


def _agg(*rows):
    """Minimal `aggregate()`-shaped result: (ledger_label, blocks, raw, out) tuples."""
    tools = [{"server": s, "tool": "t", "blocks": b, "raw_tokens": r, "out_tokens": o,
              "raw_chars": r, "out_chars": o, "diffs": 0} for s, b, r, o in rows]
    return {"total": {"blocks": sum(t["blocks"] for t in tools),
                      "raw_tokens": sum(t["raw_tokens"] for t in tools),
                      "out_tokens": sum(t["out_tokens"] for t in tools),
                      "raw_chars": 0, "out_chars": 0, "untokenized": 0},
            "decisions": {}, "diff_reasons": {}, "tools": tools}


def test_liability_is_sized_from_the_INSTALL_not_from_the_ledger(tmp_path):
    """The whole point of scanning the config: a ROUTER nobody called still ships its union
    primer every turn and contributes zero ledger rows. Sizing this from the ledger would
    hide exactly the worst case — the install that is pure cost.

    Uses a router, not a `wrapped` entry, because after #211 only a router still pays for
    being installed; the standalone case is the test below."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol)],
                            _agg())          # empty ledger
    assert liab["per_turn_tokens"] > 0
    assert liab["idle"] == ["terse"]         # pays every turn, banked nothing
    assert liab["turns_covered"] == 0.0


def test_an_uncalled_standalone_entry_costs_nothing_rather_than_a_turn_of_primer(tmp_path):
    """The #211 follow-up. A lazily-primed `terse proxy` attaches its primer to the first
    compressible result, so an entry that never handles one pays NOTHING — not a primer
    every turn.

    The old model billed it into a `tok/turn` headline and then listed it under "pure cost
    until they handle a compressible result", which is precisely inverted: #211 is what made
    these free. An operator acting on that line would unwrap the servers costing them least."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb-server --stdio", pol)], _agg())
    assert liab["servers"][0]["primer_tokens"] > 0     # it still HAS a primer...
    assert liab["per_turn_tokens"] == 0                # ...and is charged for none of it
    assert liab["session_once_tokens"] == 0
    assert liab["idle"] == []                          # NOT "pure cost"
    assert liab["free"] == ["kb"]
    assert "costing nothing at all" in "\n".join(build_primer_section(liab))


def test_a_called_standalone_entry_is_billed_once_per_session_not_per_turn(tmp_path):
    """The other half: once it handles a compressible result the primer IS paid — once. It
    belongs to the session bucket, never to the per-turn one, and the two are not summed
    because `tok/turn` and `tok/session` are different units."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb-server", pol)],
                            _agg(("kb-server", 4, 4_000, 1_000)))
    srv = liab["servers"][0]
    assert srv["cadence"] == "once/session"
    assert liab["session_once_tokens"] == srv["primer_tokens"] > 0
    assert liab["per_turn_tokens"] == 0
    assert liab["free"] == [] and liab["idle"] == []
    text = "\n".join(build_primer_section(liab))
    assert "tok/session" in text and f"{'1x':>9}" in text


def test_the_two_cadences_are_reported_separately_and_never_added(tmp_path):
    """A mixed install is where summing them did the visible damage: the router's genuinely
    recurring cost and the standalone's one-time cost landed in one `tok/turn` total that was
    true of neither."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol),
                             _scan("rc", "wrapped", "runecho-mcp", pol)],
                            _agg(("kb", 2, 2_000, 500), ("runecho-mcp", 2, 2_000, 500)))
    by_name = {s["server"]: s for s in liab["servers"]}
    assert by_name["terse"]["cadence"] == "per-turn"
    assert by_name["rc"]["cadence"] == "once/session"
    assert liab["per_turn_tokens"] == by_name["terse"]["primer_tokens"]
    assert liab["session_once_tokens"] == by_name["rc"]["primer_tokens"]
    # The old bug in one line: the headline is no longer the sum of the two.
    assert liab["per_turn_tokens"] != (by_name["terse"]["primer_tokens"]
                                       + by_name["rc"]["primer_tokens"])


def test_an_unlabelled_standalone_entry_is_uncertain_not_free(tmp_path):
    """`blocks is None` means no ledger label was recoverable, so whether the lazy attach ever
    fired is unknown. Collapsing that to "never called" would move it into `free` and
    under-report the install — the same None-vs-0 distinction `_break_even` already keeps."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))], _agg())
    assert liab["servers"][0]["cadence"] == "once/session (?)"
    assert liab["free"] == [] and liab["uncertain"] == ["x"]
    assert liab["session_once_tokens"] == 0       # not billed either — it is unknown
    assert "unknown whether the lazy primer ever attached" in "\n".join(
        build_primer_section(liab))


def test_folded_peers_do_not_double_charge_behind_their_router(tmp_path):
    """A folded peer is stashed BEHIND a router: the router pays one union primer for the
    fleet and the peers pay nothing. Counting them would charge the same primer four times
    for a three-peer install."""
    pol = _policy(tmp_path)
    rows = [_scan("terse", "router", "codegraph, kb, runecho", pol),
            *[_scan(p, "folded", "", pol) for p in ("codegraph", "kb", "runecho")]]
    liab = primer_liability(rows, _agg())
    assert [s["server"] for s in liab["servers"]] == ["terse"]


def test_a_router_is_sized_by_the_union_over_PEER_names_not_its_own(tmp_path):
    """Each peer's policy is gated against its OWN name. Gating the union on the router's
    name tests a rule like `kb.*` against "terse", which matches nothing — so the default
    policy applies to every unmatched tool and the primer is sized LARGER than the one the
    router actually ships."""
    pol = _policy(tmp_path)
    router = primer_liability([_scan("terse", "router", "kb", pol)], _agg())
    direct = primer_liability([_scan("kb", "wrapped", "kb-server", pol)], _agg())
    # Compared per-server, not via the totals: those now live in different cadence buckets
    # (#211 follow-up), and this test is about SIZING, not about who is billed.
    assert router["servers"][0]["primer_tokens"] == direct["servers"][0]["primer_tokens"]
    # Pin the failure mode itself: sizing it as `build_primer(pol, "terse")` is bigger.
    from terse.policy import load_policy
    from terse.proxy import build_primer
    from terse.tokenize import count_cl100k
    assert count_cl100k(build_primer(load_policy(pol), "terse")) > router["per_turn_tokens"]


def test_a_busy_router_is_not_reported_as_never_called(tmp_path):
    """`wraps` means two different things by state — the downstream COMMAND for a wrapped
    entry, the comma-joined peer NAMES for a router. Reading a router's peer list as a
    command is how a router fronting three busy peers reported "pure cost, never called"
    against a ledger full of its own traffic."""
    pol = _policy(tmp_path)
    liab = primer_liability(
        [_scan("terse", "router", "codegraph, kb, runecho", pol)],
        _agg(("kb", 5, 1000, 400), ("codegraph", 2, 500, 500)))
    row = liab["servers"][0]
    assert row["ledger_labels"] == ["codegraph", "kb", "runecho"]
    assert row["blocks"] == 7          # summed across the peers that DID work
    assert liab["idle"] == []


def test_a_wrapped_entrys_ledger_label_comes_from_its_downstream_command(tmp_path):
    """The ledger keys on `server_label` of the wrapped command, not the MCP entry name."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("my-entry", "wrapped", "/opt/bin/kb-server --stdio", pol)],
                            _agg(("kb-server", 3, 900, 300)))
    assert liab["servers"][0]["ledger_labels"] == ["kb-server"]
    assert liab["servers"][0]["blocks"] == 3


def test_an_unknown_label_is_not_an_accusation_of_being_idle(tmp_path):
    """"Unknown" and "never called" are different claims, and only the second one accuses
    an install of being pure cost. A row with nothing to derive a label from reports
    `blocks: None` and stays out of `idle`."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))], _agg())
    assert liab["servers"][0]["blocks"] is None
    assert liab["idle"] == []


def test_an_unreadable_policy_is_excluded_and_the_total_labelled_a_lower_bound(tmp_path):
    """Substituting the built-in default would OVERSTATE (the default emits every form).
    Leave it out, count it, and say the number is a floor."""
    good = _policy(tmp_path)
    # Both entries are CALLED, so the exclusion shows up in a bucket that is actually billed
    # — against an empty ledger every standalone entry is free and the total would be 0
    # whether or not the unreadable policy was excluded, proving nothing.
    liab = primer_liability([_scan("a", "wrapped", "a", good),
                             _scan("b", "wrapped", "b", str(tmp_path / "nope.json"))],
                            _agg(("a", 2, 900, 300), ("b", 2, 900, 300)))
    assert liab["unresolved"] == 1
    assert liab["servers"][1]["primer_tokens"] is None
    assert liab["session_once_tokens"] == liab["servers"][0]["primer_tokens"]
    assert "lower bound" in "\n".join(build_primer_section(liab))


def test_a_server_that_can_emit_no_compressed_form_pays_nothing(tmp_path):
    """`build_primer` returns "" for a policy that can emit no compressed form — a
    default-deny server explains nothing because it produces nothing. Sizing every server
    at one shared constant would invent a cost it does not pay."""
    deny = _policy(tmp_path, name="deny.json", tool="*", tiers=())
    liab = primer_liability([_scan("secret-broker", "wrapped", "sb", deny)], _agg())
    assert liab["servers"][0]["primer_tokens"] == 0
    assert liab["per_turn_tokens"] == 0
    # Nothing to break even on — not "infinite turns", which would read as a measurement.
    assert liab["turns_covered"] is None
    assert liab["idle"] == []


def test_duplicate_names_across_scopes_are_one_server(tmp_path):
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol, scope="project"),
                             _scan("kb", "wrapped", "kb", pol, scope="user")], _agg())
    assert len(liab["servers"]) == 1 and liab["servers"][0]["scope"] == "project"


def test_the_report_calls_out_a_net_negative_window(tmp_path):
    """The line the issue exists for: savings that do not cover even one turn of primer."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol)],
                            _agg(("kb", 1, 110, 100)))    # 10 tok saved, ~200 tok/turn
    assert liab["turns_covered"] < 1
    text = "\n".join(build_primer_section(liab))
    assert "NET NEGATIVE" in text


def test_a_standalone_only_install_still_gets_a_net_negative_verdict(tmp_path):
    """With no router there is no recurring charge, so `turns_covered` is None and the old
    renderer printed no bottom line at all — silence on exactly the install shape #211 made
    the common one. The verdict is per SESSION there, against the one-time charge."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 1, 110, 100)))    # 10 tok saved vs a ~200 tok primer
    assert liab["turns_covered"] is None
    assert liab["session_covered"] < 1
    text = "\n".join(build_primer_section(liab))
    # The shortfall in tokens, not a ratio: `~0x over` would round to a zero that reads as a
    # measurement (the print-as-zero hole review of #197 closed for `saved/block`).
    assert "NET NEGATIVE" in text
    assert f"does not cover the {liab['session_once_tokens']:,} tok one-time charge" in text
    assert "0x over" not in text


def test_the_one_time_charge_is_NOT_netted_out_of_the_recurring_ratio(tmp_path):
    """Review of this change caught the first attempt doing exactly that. It looks prudent —
    both charges come out of the same savings — but `session_once_tokens` is charged once per
    SESSION while `saved_tokens` spans the whole window, and a window covers an unknown number
    of sessions (a `terse proxy` is one process per session and `_primer_sent` re-arms at every
    `initialize`). Subtracting one primer where K were paid under-bills by the session count,
    and `sessions` is no more observable from this ledger than `turns` is — which is the very
    reason there is no per-turn charge in it. Divide like against like, and state the
    cross-cadence comparison as a bound instead."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol),
                             _scan("rc", "wrapped", "runecho-mcp", pol)],
                            _agg(("kb", 2, 10_000, 0), ("runecho-mcp", 2, 10_000, 0)))
    assert liab["session_once_tokens"] > 0            # there IS a one-time charge...
    assert liab["turns_covered"] == liab["saved_tokens"] / liab["per_turn_tokens"]


def test_the_one_time_coverage_is_labelled_a_ceiling_not_a_measurement(tmp_path):
    """`saved / once` treats the whole window as ONE session, which is the most favourable
    reading available. Said plainly it would over-credit terse by the session count — the
    unsafe direction — so it is rendered as a ceiling, and alongside the recurring line rather
    than instead of it."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol),
                             _scan("rc", "wrapped", "runecho-mcp", pol)],
                            _agg(("kb", 2, 10_000, 0), ("runecho-mcp", 2, 10_000, 0)))
    text = "\n".join(build_primer_section(liab))
    assert "at most" in text and "more than one session" in text
    assert "turn(s) of the recurring primer" in text   # both cadences reported, not one


def test_a_router_only_install_is_not_told_about_a_charge_it_does_not_pay(tmp_path):
    """`once == 0` for an install with no lazy entry, so every sentence about a one-time
    charge is about a charge that does not exist. The stanza and the "settles the one-time
    charge" clause are both gated on there BEING one — the same reason the report says
    `no primer` rather than `0` elsewhere: absent and zero are different claims."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol)],
                            _agg(("kb", 20, 40_000, 10_000)))
    text = "\n".join(build_primer_section(liab))
    assert "tok/turn" in text
    assert "one-time" not in text and "tok/session" not in text
    assert "settles the one-time charge" not in text
    assert "not summed" not in text


def test_a_liability_blob_from_a_pre_cadence_terse_still_renders(tmp_path):
    """`build_primer_section` is public and `--json` emits this dict, so a blob round-tripped
    through a terse that predates the cadence split carries no `cadence`/`session_once_tokens`
    at all. It measured the eager model, so it renders as the recurring half — degrading, not
    raising, and never guessing a cadence it did not record."""
    liab = primer_liability([_scan("terse", "router", "kb", _policy(tmp_path))],
                            _agg(("kb", 5, 5_000, 1_000)))
    legacy = {k: v for k, v in liab.items()
              if k not in ("session_once_tokens", "session_covered", "free", "uncertain")}
    legacy["servers"] = [{k: v for k, v in s.items() if k != "cadence"}
                         for s in liab["servers"]]
    text = "\n".join(build_primer_section(legacy))
    assert "tok/turn" in text and "tok/session" not in text
    assert f"{'–':>9}" in text           # cadence cell blank rather than guessed


def test_the_liability_still_prints_when_the_ledger_is_empty(tmp_path):
    """An install with wrapped servers and NO recorded results is the purest net-negative
    case there is; suppressing the line here would hide it behind "nothing to report"."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], _agg())
    out = build_stats_report(_agg(), log_path="/tmp/x.jsonl", liability=liab)
    assert "no results recorded" in out
    assert "primer liability" in out


def test_the_liability_is_absent_rather_than_zero_when_nothing_is_wrapped():
    assert build_primer_section(primer_liability([], _agg())) == []


# --- per-server break-even (#175) -----------------------------------------------------
#
# #175 stated the rule ("wrap a server when its typical payload saves more than
# `primer x turns-per-call`") and then computed the table proving it BY HAND. These pin the
# same arithmetic inside `terse stats`, and — more importantly — the four cases where the
# honest answer is not a number.


def test_break_even_divides_this_servers_savings_by_its_own_primer(tmp_path):
    """The headline number: 600 tok banked over 10 blocks is 60/call, so a primer of P
    tokens is repaid by P/60 calls. This fixture is a `wrapped` (standalone) entry, so the
    cadence is once/session — the break-even is reached once, not re-earned per turn."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 10, 1_000, 400)))
    srv = liab["servers"][0]
    assert srv["saved_per_block"] == 60.0
    assert srv["blocks_to_break_even"] == srv["primer_tokens"] / 60.0
    assert "to break even" in "\n".join(build_primer_section(liab))


def test_a_server_that_never_breaks_even_says_so_instead_of_printing_a_huge_number(tmp_path):
    """A non-positive rate does not break even at ANY call volume. Rendering that as
    `999,999.00 calls/turn` invites an operator to read it as merely expensive; it is the
    one verdict in this table that should stop them, so it is a word."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 4, 100, 100)))    # codec banked nothing
    srv = liab["servers"][0]
    assert srv["saved_per_block"] == 0.0
    assert srv["blocks_to_break_even"] is None
    assert srv["break_even_verdict"] == "never"
    # not merely `"never" in text` — the `never called` verdict also matches that, and so
    # does the surrounding prose. Match the right-aligned CELL (width 21).
    assert f"{'never':>17}" in "\n".join(build_primer_section(liab))


def test_an_untokenized_ledger_reports_no_token_data_not_a_zero_rate(tmp_path):
    """Rows recorded without tiktoken carry char totals only. Savings in TOKENS are then
    unknown, not zero — and dividing a cl100k primer by a char-derived rate would silently
    mix units. The distinction matters: `0` accuses the server of being incompressible."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 7, 0, 0)))        # blocks, but no token counts
    srv = liab["servers"][0]
    assert srv["blocks"] == 7                              # it was called...
    assert srv["saved_per_block"] is None                   # ...but we cannot rate it
    assert srv["blocks_to_break_even"] is None
    assert "no token data" in "\n".join(build_primer_section(liab))


def test_a_zero_primer_server_breaks_even_at_no_calls_at_all(tmp_path):
    """A default-deny server ships no primer, so there is nothing to earn back — distinct
    from "never", which is the opposite verdict."""
    deny = _policy(tmp_path, name="deny2.json", tool="*", tiers=())
    liab = primer_liability([_scan("sb", "wrapped", "sb", deny)], _agg(("sb", 2, 100, 90)))
    srv = liab["servers"][0]
    assert srv["primer_tokens"] == 0
    assert srv["blocks_to_break_even"] == 0.0
    assert srv["break_even_verdict"] == "no primer"
    assert "no primer" in "\n".join(build_primer_section(liab))


def test_a_routers_rate_pools_every_peer_it_fronts(tmp_path):
    """A router pays ONE union primer for the fleet, so its break-even is against the
    pooled savings of all its peers — charging each peer separately would be the
    double-charge `_PAYS_PRIMER` already refuses upstream."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb, runecho", pol)],
                            _agg(("kb", 2, 500, 300), ("runecho", 2, 500, 100)))
    srv = liab["servers"][0]
    assert srv["blocks"] == 4
    assert srv["saved_per_block"] == 150.0          # (200 + 400) / 4


def test_the_table_is_suppressed_when_no_server_has_a_rate(tmp_path):
    """An install with nothing in the ledger renders four dashes and no information; the
    `idle` line above already made that point."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], _agg())
    text = "\n".join(build_primer_section(liab))
    assert "primer liability" in text
    assert "to break even" not in text


def test_an_unreadable_policy_is_primer_unknown_not_never(tmp_path):
    """Review of #197: `never` and "we could not read the policy" both produced
    `calls_to_break_even is None`, and the renderer read that single sentinel as the first.
    A server saving 600 tok/call was condemned as `never` — the one verdict here meant to
    stop an operator — on evidence about a missing FILE."""
    liab = primer_liability([_scan("b", "wrapped", "b", str(tmp_path / "gone.json"))],
                            _agg(("b", 5, 5_000, 2_000)))
    srv = liab["servers"][0]
    assert srv["primer_tokens"] is None                 # policy unreadable
    assert srv["saved_per_block"] == 600.0               # ...but the rate is real
    assert srv["break_even_verdict"] == "primer unknown"
    text = "\n".join(build_primer_section(liab))
    # The `never` that must be absent is the VERDICT CELL, not the word anywhere in the
    # section — the surrounding prose legitimately uses it.
    assert "primer unknown" in text and f"{'never':>17}" not in text


def test_the_rate_divides_by_TOKENIZED_blocks_not_by_every_block(tmp_path):
    """Review of #197: `aggregate` counts every record in `blocks` but only tokenized ones
    in the token sums, so a ledger spanning one offline session (`count_cl100k` -> None) and
    later online ones divided a partial numerator by a full denominator. The error is always
    pessimistic, i.e. it argues for unwrapping a server that is paying for itself."""
    from terse.stats import aggregate

    recs = [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4}
            for _ in range(90)]                                   # recorded without tiktoken
    recs += [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4,
              "raw_tokens": 1_000, "out_tokens": 400} for _ in range(10)]
    agg = aggregate(recs)
    assert agg["tools"][0]["blocks"] == 100 and agg["tools"][0]["tokenized"] == 10

    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], agg)
    srv = liab["servers"][0]
    assert srv["blocks"] == 100 and srv["tokenized_blocks"] == 10
    assert srv["saved_per_block"] == 600.0            # 6,000 tok over the 10 MEASURED calls
    assert srv["blocks_to_break_even"] == srv["primer_tokens"] / 600.0
    # And the contaminated denominator is visible where it changed the number.
    assert "10/100" in "\n".join(build_primer_section(liab))


def test_an_unrecoverable_ledger_label_is_not_the_tiktoken_accusation(tmp_path):
    """Review of #197: `blocks is None` (no label could be derived) and `blocks == 0` (never
    called) both rendered `no token data`, which blames the tokenizer for a missing label."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))],
                            _agg(("kb", 4, 400, 100)))
    srv = liab["servers"][0]
    assert srv["blocks"] is None
    assert srv["break_even_verdict"] == "no ledger label"


def test_a_losing_server_sorts_below_one_that_merely_breaks_even(tmp_path):
    """Review of #197: the `or -1` sort key coerced a 0.0 rate (falsy) to -1 while leaving
    -0.5 (truthy) alone, so the server actively EXPANDING payloads printed above the one
    that merely broke even. The gap only opens for losses inside (-1, 0), which is why this
    fixture uses -0.5 rather than a large loss."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("losing", "wrapped", "losing", pol),
                             _scan("flat", "wrapped", "flat", pol)],
                            _agg(("losing", 4, 100, 102), ("flat", 4, 100, 100)))
    rows = [ln for ln in build_primer_section(liab) if "losing" in ln or "flat" in ln]
    assert len(rows) == 2 and "flat" in rows[0] and "losing" in rows[1]


def test_a_sub_unit_rate_does_not_print_as_zero(tmp_path):
    """Review of #197: `:,.0f` rendered every rate in (-1, 1) as `0`/`-0`, which sat beside
    a finite break-even and read as a contradiction."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 10, 1_004, 1_000)))     # 0.4 tok/call
    assert liab["servers"][0]["saved_per_block"] == 0.4
    assert "0.4" in "\n".join(build_primer_section(liab))


def test_the_denominator_column_is_labelled_blocks_because_that_is_what_it_counts(tmp_path):
    """Round-2 review of #197: the header was renamed `blocks` -> `calls` over a counter
    `aggregate` increments once per emitted tool-result BLOCK (>=1 per call, #141). A server
    emitting three blocks per call would have had its break-even overstated 3x with nothing
    in the report disclosing the unit."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))],
                            _agg(("kb", 30, 10_000, 4_000)))
    text = "\n".join(build_primer_section(liab))
    assert "to break even" in text and "calls" not in text
    assert "saved/block" in text and "saved/call" not in text
    assert "a BLOCK is one emitted tool-result text block" in text


def test_a_rate_too_small_to_render_is_a_bound_not_a_zero(tmp_path):
    """Round-2 review of #197: the first fix moved the print-as-zero hole from (-1, 1) to
    (-0.05, 0.05) rather than closing it, so `0.0` still sat beside a finite break-even."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 1_000, 100_004, 100_000)))    # 0.004 tok/block
    assert liab["servers"][0]["saved_per_block"] == 0.004
    text = "\n".join(build_primer_section(liab))
    assert "<0.01" in text


def test_an_unrecoverable_label_leaves_the_tokenized_count_unknown_too(tmp_path):
    """Round-2 review of #197: `blocks` got the `if labels else None` guard and `tokenized`
    did not, so `--json` carried `{"blocks": null, "tokenized_blocks": 0}` — a 0 claiming
    "nothing was tokenized" where the truth is that no rows were ever found to ask."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))],
                            _agg(("kb", 4, 400, 100)))
    srv = liab["servers"][0]
    assert srv["blocks"] is None and srv["tokenized_blocks"] is None
    # And it still renders, rather than being suppressed alongside a genuinely idle install.
    rendered = build_primer_section(primer_liability(
        [_scan("x", "wrapped", "", _policy(tmp_path)),
         _scan("kb", "wrapped", "kb", _policy(tmp_path))], _agg(("kb", 4, 400, 100))))
    assert "no ledger label" in "\n".join(rendered)


def test_a_liability_blob_without_a_verdict_degrades_instead_of_raising(tmp_path):
    """Round-2 review of #197: `build_primer_section` is public and `--json` emits this exact
    dict shape, so a blob round-tripped through an older terse carries no `break_even_verdict`
    and took the numeric branch with a None. A report is never load-bearing — it degrades."""
    from terse.stats import _fmt_break_even

    assert _fmt_break_even({"saved_per_block": 0.0, "blocks_to_break_even": None}) == ("0", "–")


# --- review round 3 ------------------------------------------------------------------


def test_a_pre_cadence_blob_gets_the_recurring_legend_not_the_standalone_one(tmp_path):
    """The prose gated on `s.get("cadence") or _PER_TURN` and the table's legend on a bare
    `s.get("cadence")`. On the backward-compat path both were written for — a blob from a
    terse with no `cadence` at all — that set was `{None}`, so the table suppressed the
    `/turn` legend and printed the standalone one instead, directly under prose declaring
    the whole figure recurring. One helper now serves both."""
    liab = primer_liability([_scan("terse", "router", "kb", _policy(tmp_path))],
                            _agg(("kb", 10, 1_000, 400)))
    legacy = dict(liab)
    legacy["servers"] = [{k: v for k, v in s.items() if k != "cadence"}
                         for s in liab["servers"]]
    text = "\n".join(build_primer_section(legacy))
    assert "/turn = an eagerly-primed router" in text
    assert "1x = a lazily-primed standalone entry" not in text


def test_the_break_even_row_stays_inside_eighty_columns(tmp_path):
    """The `blocks` cell was narrowed to 11 on the reasoning that it only holds `N` or
    `tokenized/N` — but the LIVE ledger already rendered `1,790/1,799`, exactly 11, and one
    more order of magnitude would have overflowed and broken the 80-column guarantee the
    table's own comment makes. Sized against a million-block ledger, with the separators
    dropped from the pair form where the cell is widest."""
    from terse.stats import aggregate

    recs = [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4,
             "decision": "compressed"} for _ in range(7)]
    recs += [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4,
              "decision": "compressed", "raw_tokens": 10, "out_tokens": 4}
             for _ in range(3)]
    agg = aggregate(recs)
    # Blow the counts up to a million-block ledger without writing a million records.
    agg["tools"][0].update(blocks=1_234_567, tokenized=1_000_000)
    liab = primer_liability([_scan("a-very-long-server-name", "wrapped", "kb",
                                   _policy(tmp_path))], agg)
    rows = [ln for ln in build_primer_section(liab) if "1000000/1234567" in ln]
    assert len(rows) == 1
    assert max(len(ln) for ln in build_primer_section(liab)
               if ln.startswith("  ") and "1000000/1234567" in ln) <= 80


def test_a_called_server_that_never_shipped_a_wire_form_is_not_billed_a_primer(tmp_path):
    """Review finding, and the same mis-bucketing this split exists to fix, in the other
    direction. The lazy primer attaches to a result carrying a terse wire form, so an entry
    called a thousand times that never produced one — all-passthrough policy, non-JSON
    payloads, a shape the codec never wins on — paid NOTHING. `blocks` counts every emitted
    block regardless of decision and cannot see that; `encoded` counts every block except
    the `passthrough`/`unchanged` ones, so an unreadable `decision` still counts (see
    `aggregate`). Evidence, not proof — `_cadence`'s docstring names the path where a
    `passthrough` block attaches a primer anyway."""
    from terse.stats import aggregate

    agg = aggregate([{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 10,
                      "raw_tokens": 10, "out_tokens": 10, "decision": "passthrough"}
                     for _ in range(20)])
    assert agg["tools"][0]["blocks"] == 20 and agg["tools"][0]["encoded"] == 0
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], agg)
    assert liab["servers"][0]["cadence"] == "once/session (unpaid)"
    assert liab["session_once_tokens"] == 0
    assert liab["free"] == ["kb"]


def test_one_encoded_block_is_enough_to_bill_it(tmp_path):
    """The inference is deliberately one-directional. `encoded == 0` PROVES the primer could
    not have attached; `encoded > 0` does not prove it did (a minify-only `compressed` block
    carries no marker). So a non-zero count bills — the over-billing direction this module
    already argues is the safe one."""
    from terse.stats import aggregate

    recs = [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 10,
             "raw_tokens": 10, "out_tokens": 10, "decision": "passthrough"}
            for _ in range(20)]
    recs.append({"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4,
                 "raw_tokens": 10, "out_tokens": 4, "decision": "compressed"})
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))],
                            aggregate(recs))
    assert liab["servers"][0]["cadence"] == "once/session"
    assert liab["session_once_tokens"] > 0


def test_a_row_that_cannot_report_encoded_falls_back_to_the_old_coarser_behaviour(tmp_path):
    """A hand-rolled agg, or one written before the counter existed, must not be read as
    "this server never shipped a wire form" — that would report an install free on the
    strength of a row that simply could not say. `_cadence` falls back to `blocks`."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))],
                            _agg(("kb", 4, 400, 100)))   # `_agg` emits no `encoded`
    assert "encoded" not in _agg(("kb", 4, 400, 100))["tools"][0]
    assert liab["servers"][0]["cadence"] == "once/session"
    assert liab["session_once_tokens"] > 0


def test_a_mixed_install_is_told_the_two_lines_are_not_jointly_true(tmp_path):
    """Review finding. Both lines credit the SAME savings in full against their own charge.
    Not netting them is right (the units differ), but silence let a mixed install read two
    individually-true lines as jointly true, while it pays `per_turn*turns + once`."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol),
                             _scan("rc", "wrapped", "runecho-mcp", pol)],
                            _agg(("kb", 2, 10_000, 0), ("runecho-mcp", 2, 10_000, 0)))
    text = "\n".join(build_primer_section(liab))
    assert "credits the SAME savings against its own charge alone" in text
    assert "clearing one of them is not clearing the pair" in text
    # ...and a single-cadence install is not told about a pair it does not have.
    solo = primer_liability([_scan("terse", "router", "kb", pol)],
                            _agg(("kb", 2, 10_000, 0)))
    assert "clearing the pair" not in "\n".join(build_primer_section(solo))


def test_a_passthrough_block_can_still_attach_a_primer__known_gap(tmp_path):
    """The limit of the `encoded` heuristic, executable rather than asserted in a comment.

    An earlier revision of `_cadence`'s docstring claimed `encoded == 0` PROVES the primer
    never attached. It does not, and this is the counterexample (found in review, reproduced
    here rather than taken on faith): the attach guard fires on `'"__terse_'` appearing
    anywhere in the FINAL content, and that text can come from the DOWNSTREAM payload — a
    code-search tool returning terse's own source, a doubly wrapped peer. The result
    classifies as `passthrough`, so `encoded` stays 0 and the server is filed under `free`
    with "costs nothing at all", when it did pay.

    Pinned as a GAP, not as desired behaviour: the assertions below are what the code does
    today, and the last one is the wrong answer. Closing it needs the ledger to record
    whether the attach fired, which is a shape change and a separate decision."""
    import json as _json

    from terse.policy import Policy, Rule
    from terse.proxy import Interceptor
    from terse.stats import aggregate, classify_decision

    pol = Policy(rules=[Rule("echo.*", ())])            # explicit passthrough, no tiers
    inter = Interceptor(pol, server_name="probe")
    payload = _json.dumps({"source": '{"__terse_table__":1,"n":1,"cols":["a"],"rows":[[1]]}'})
    inter.note_request(_json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": "echo.thing"}}),
                       tool_name="echo.thing")
    out = inter.transform_response(_json.dumps(
        {"jsonrpc": "2.0", "id": 1,
         "result": {"content": [{"type": "text", "text": payload}]}}))

    # The primer DID attach — an extra leading block, and the latch is set.
    assert len(_json.loads(out)["result"]["content"]) == 2
    assert inter._primer_sent is True
    # ...and the ledger records the block as `passthrough`, so `encoded` cannot see it.
    assert classify_decision(payload, payload, passthrough=True) == "passthrough"
    agg = aggregate([{"server": "probe", "tool": "echo.thing", "raw_chars": len(payload),
                      "out_chars": len(payload), "raw_tokens": 50, "out_tokens": 50,
                      "decision": "passthrough"}])
    assert agg["tools"][0]["encoded"] == 0
    liab = primer_liability([_scan("probe", "wrapped", "probe", _policy(tmp_path))], agg)
    # THE WRONG ANSWER, pinned so the day someone fixes it this test fails and says why.
    assert liab["servers"][0]["cadence"] == "once/session (unpaid)"
    assert liab["free"] == ["probe"]


# --- wrap/don't-wrap verdict (#238) ---------------------------------------------------
#
# #175 stated the rule and #197 put both halves of the arithmetic in `terse stats` as two
# numeric columns, leaving the operator to do the comparison. These pin the rollup: one word
# per INSTALLED ENTRY, computed from the fields already published, thresholded exactly where
# `build_primer_section` already prints NET NEGATIVE.


def _one(tmp_path, *rows, state="wrapped", policy=None, wraps=None, server="kb"):
    """One installed entry against a hand-built agg, returned as its liability ROW.

    A helper rather than four lines repeated seventeen times, because every test below is
    about the VERDICT and none of them is about the scan/agg plumbing that produces it."""
    pol = policy if policy is not None else _policy(tmp_path)
    liab = primer_liability([_scan(server, state, server if wraps is None else wraps, pol)],
                            _agg(*rows))
    return liab["servers"][0]


def test_a_server_that_cleared_its_primer_this_window_reads_KEEP(tmp_path):
    """The happy path, and the shape the whole verdict exists to name in one word: a
    positive rate that carried `tokenized_blocks` past `blocks_to_break_even`. Before this
    the operator read two numeric columns and did the comparison themselves."""
    srv = _one(tmp_path, ("kb", 10, 10_000, 1_000))
    assert srv["break_even_verdict"] is None          # a real, finite break-even...
    assert srv["tokenized_blocks"] > srv["blocks_to_break_even"]
    assert srv["verdict"] == "KEEP"
    assert srv["verdict_reason"] == "cleared"
    assert srv["break_even_coverage"] > 1


def test_a_server_short_of_its_break_even_reads_TUNE_rather_than_UNWRAP(tmp_path):
    """The shortfall here is a VOLUME gap, not a structural one: the rate is positive, so
    more blocks at today's rate do clear the primer. `UNWRAP` would tell an operator to
    throw away a server that is one busy session from paying for itself — the same
    over-pessimism that argued for unwrapping a paying server in review of #197."""
    srv = _one(tmp_path, ("kb", 1, 110, 100))         # 10 tok saved against a ~248 primer
    assert srv["saved_per_block"] > 0                  # ...reachable, just not reached
    assert srv["tokenized_blocks"] < srv["blocks_to_break_even"]
    assert srv["verdict"] == "TUNE"
    assert srv["verdict_reason"] == "short of break-even"
    assert 0 < srv["break_even_coverage"] < 1


def test_a_non_positive_rate_reads_UNWRAP_at_any_block_volume(tmp_path):
    """`break_even_verdict == "never"` is the ONLY structural impossibility in this table —
    no block volume earns this primer back, which is exactly what separates `UNWRAP` from
    `TUNE`. The block count is deliberately large: volume is not the missing ingredient."""
    srv = _one(tmp_path, ("kb", 5_000, 100_000, 100_000))
    assert srv["blocks"] == 5_000 and srv["saved_per_block"] == 0.0
    assert srv["break_even_verdict"] == "never"
    assert srv["verdict"] == "UNWRAP"
    assert srv["verdict_reason"] == "never"


def test_exactly_clearing_the_break_even_reads_KEEP_not_TUNE(tmp_path):
    """The boundary at `tokenized_blocks == blocks_to_break_even`. A `>` where the code has
    `>=` would file a server that EXACTLY paid for itself as needing a tune, and the issue's
    own framing is "break-even already cleared".

    The fixture is constructed so the equality is exact rather than approximately true: the
    primer is read back from a probe run and then banked verbatim over a single block, so
    `rate == primer` and `blocks_to_break_even == primer / primer == 1.0` with no rounding
    anywhere in the chain."""
    pol = _policy(tmp_path)
    primer = _one(tmp_path, policy=pol)["primer_tokens"]
    assert primer > 0
    srv = _one(tmp_path, ("kb", 1, primer + 100, 100), policy=pol)
    assert srv["tokenized_blocks"] == srv["blocks_to_break_even"] == 1
    assert srv["verdict"] == "KEEP"
    assert srv["verdict_reason"] == "cleared"
    assert srv["break_even_coverage"] == 1.0


def test_every_break_even_reason_that_names_missing_data_reads_INSUFFICIENT(tmp_path):
    """A `KEEP` or an `UNWRAP` on any of these would be a verdict about a missing FILE or a
    missing tokenizer rather than about the server — the #197 review finding, one level up.
    `never called` is deliberately NOT in this family: it is a real measurement, and the test
    below splits it on cadence."""
    cases = {
        # no label to look the ledger rows up by — we never found the rows to ask.
        "no ledger label": _one(tmp_path, ("kb", 4, 400, 100), wraps=""),
        # called, but every matching row was recorded without tiktoken.
        "no token data": _one(tmp_path, ("kb", 7, 0, 0)),
        # a real rate, but the bar it has to clear is unreadable.
        "primer unknown": _one(tmp_path, ("b", 5, 5_000, 2_000), server="b",
                               policy=str(tmp_path / "gone.json")),
    }
    for reason, srv in cases.items():
        assert srv["break_even_verdict"] == reason, reason
        assert srv["verdict"] == "INSUFFICIENT", reason
        assert srv["verdict_reason"] == reason, reason
        # Never a fabricated 0: a consumer would read that as a measured total loss.
        assert srv["break_even_coverage"] is None, reason


def test_an_idle_ROUTER_reads_UNWRAP_while_an_idle_STANDALONE_entry_reads_INSUFFICIENT(
        tmp_path):
    """THE CADENCE SPLIT. Both rows carry `break_even_verdict == "never called"` and they get
    different verdicts, because who pays for idling differs.

    An eagerly-primed router ships its union primer every turn whether or not anybody calls
    it, so zero calls is not missing data — it is a COMPLETE measurement of a total loss, and
    `primer_liability` already reports exactly that as `idle`, "pure cost".

    A lazily-primed standalone entry paid NOTHING (#211, `free`, "costing nothing at all"),
    and "unwrap the servers costing you least" is the precise inversion the #211 follow-up
    exists to prevent. Flattening these two to one verdict re-introduces it."""
    pol = _policy(tmp_path)
    router = _one(tmp_path, state="router", server="terse", wraps="kb", policy=pol)
    standalone = _one(tmp_path, policy=pol)
    assert router["break_even_verdict"] == standalone["break_even_verdict"] == "never called"
    assert router["primer_tokens"] > 0 and standalone["primer_tokens"] > 0
    assert router["cadence"] == "per-turn"
    assert router["verdict"] == "UNWRAP" and router["verdict_reason"] == "never called"
    assert standalone["cadence"] == "once/session (unpaid)"
    assert standalone["verdict"] == "INSUFFICIENT"
    assert standalone["verdict_reason"] == "never called"


def test_an_idle_router_with_no_primer_to_pay_is_not_condemned(tmp_path):
    """Mirrors `idle`'s own truthy-`primer_tokens` guard: a default-deny router emits no
    compressed form, so it ships no primer and costs nothing to leave idle. Condemning it
    would be condemning a server for a charge it does not pay."""
    deny = _policy(tmp_path, name="deny238.json", tool="*", tiers=())
    srv = _one(tmp_path, state="router", server="terse", wraps="kb", policy=deny)
    assert srv["primer_tokens"] == 0 and srv["cadence"] == "per-turn"
    assert srv["break_even_verdict"] == "never called"
    assert srv["verdict"] == "INSUFFICIENT"


def test_a_zero_primer_server_that_still_saves_tokens_reads_KEEP(tmp_path):
    """`no primer` means there is nothing to earn back "at any rate, positive or negative"
    (`_break_even`), so there is no case for unwrapping it. The coverage must stay None:
    `blocks_to_break_even` is 0.0 and the ratio is undefined, so an infinity or a 0 there
    would both be fabrications."""
    deny = _policy(tmp_path, name="deny238b.json", tool="*", tiers=())
    srv = _one(tmp_path, ("kb", 2, 100, 90), policy=deny)
    assert srv["primer_tokens"] == 0 and srv["blocks_to_break_even"] == 0.0
    assert srv["break_even_verdict"] == "no primer"
    assert srv["verdict"] == "KEEP"
    assert srv["verdict_reason"] == "no primer"      # "no reason to unwrap", not "profitable"
    assert srv["break_even_coverage"] is None


def test_a_zero_primer_server_that_is_EXPANDING_payloads_still_reads_UNWRAP(tmp_path):
    """The one deliberate divergence from `_break_even`'s ordering, and it is a rendering
    decision rather than an arithmetic change.

    `_break_even` short-circuits on `primer == 0` BEFORE it tests the rate, so a server with
    a free primer that is actively making payloads LARGER still reports `no primer`. The
    primer is free; the expansion is not, and `KEEP` there is the one flatly wrong word this
    table can print. Pins that the verdict layer reads BOTH published fields rather than
    trusting the break-even string alone."""
    deny = _policy(tmp_path, name="deny238c.json", tool="*", tiers=())
    srv = _one(tmp_path, ("kb", 2, 100, 110), policy=deny)
    assert srv["break_even_verdict"] == "no primer"   # ...what the arithmetic layer says
    assert srv["saved_per_block"] < 0                 # ...and the fact it stepped over
    assert srv["verdict"] == "UNWRAP"
    assert srv["verdict_reason"] == "expanding"


def test_a_router_gets_ONE_verdict_for_the_whole_fleet_and_its_peers_get_none(tmp_path):
    """THE PER-ENTRY REQUIREMENT, BUILT AS A TRAP.

    A router pays ONE union primer for the whole fleet. A per-peer verdict would charge each
    peer that full shared primer and then tell the operator to unwrap peers that are
    collectively paying for themselves — the inversion `primer_liability`'s docstring warns
    about, one level up.

    The fixture is sized so it actually discriminates rather than passing by luck: each peer
    banks ~0.45 of the shared primer, so every peer taken alone is short of it while the pool
    clears it comfortably. A per-peer implementation therefore produces three TUNEs and the
    assertions below fail loudly. Both directions are asserted, so the fixture cannot rot
    into one that would pass either way."""
    pol = _policy(tmp_path, name="all238.json", tool="*")
    peers = ("codegraph", "kb", "runecho")
    rows = [_scan("terse", "router", ", ".join(peers), pol)]
    primer = primer_liability(rows, _agg())["servers"][0]["primer_tokens"]
    share = int(primer * 0.45)
    assert 0 < share < primer
    liab = primer_liability(rows, _agg(*((p, 2, share + 100, 100) for p in peers)))

    # (a) exactly one verdict row, on the installed entry, and it clears.
    assert [s["server"] for s in liab["servers"]] == ["terse"]
    row = liab["servers"][0]
    assert row["verdict"] == "KEEP" and row["break_even_coverage"] >= 1

    # (b) the peers carry evidence and nothing that could become a verdict.
    assert [c["label"] for c in row["contributors"]] == sorted(peers)
    for c in row["contributors"]:
        for forbidden in ("verdict", "verdict_reason", "break_even_coverage",
                          "primer_tokens", "blocks_to_break_even", "break_even_verdict"):
            assert forbidden not in c, forbidden

    # (c) the fixture discriminates: each peer ALONE is short of the shared primer, so a
    #     per-peer verdict would print three TUNEs where the truth is one KEEP.
    for c in row["contributors"]:
        assert c["saved_tokens"] / primer < 1, "fixture no longer traps a per-peer verdict"
    assert sum(c["saved_tokens"] for c in row["contributors"]) / primer >= 1


def test_a_folded_peer_never_receives_a_verdict_because_it_pays_no_primer(tmp_path):
    """The other half of the double-charge guard, and the strongest of the three: `_PAYS_PRIMER`
    filters `folded` peers out BEFORE a row is ever built, so there is structurally nothing to
    attach a verdict to. Extends the `test_folded_peers_do_not_double_charge_behind_their_router`
    fixture — same install, now asserting the verdict surface as well as the cost one."""
    pol = _policy(tmp_path, name="folded238.json", tool="*")
    peers = ("codegraph", "kb", "runecho")
    rows = [_scan("terse", "router", ", ".join(peers), pol),
            *[_scan(p, "folded", "", pol) for p in peers]]
    liab = primer_liability(rows, _agg(*((p, 4, 4_000, 1_000) for p in peers)))
    assert [s["server"] for s in liab["servers"]] == ["terse"]
    assert liab["servers"][0]["verdict"] == "KEEP"
    # And the peers appear only as evidence under it — never as rows with words of their own.
    assert [c["label"] for c in liab["servers"][0]["contributors"]] == sorted(peers)
    text = "\n".join(build_recommend_section(liab))
    for peer in peers:
        assert f"  {peer:<14} " not in text


def test_the_coverage_counts_only_the_blocks_the_rate_was_measured_over(tmp_path):
    """`saved = rate x tokenized` is an exact identity, so `tokenized` is the only honest
    numerator. Dividing by `blocks` would credit savings to blocks that were never measured —
    manufacturing coverage out of an offline session — and this fixture is sized so that
    error would flip the verdict rather than merely nudge a ratio: 10 measured blocks out of
    100 means a `blocks` numerator reads 10x high, turning a genuine TUNE into a KEEP."""
    from terse.stats import aggregate

    pol = _policy(tmp_path)
    primer = _one(tmp_path, policy=pol)["primer_tokens"]
    saved = primer // 2                                    # coverage 0.5 -> TUNE
    recs = [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4}
            for _ in range(90)]                            # recorded without tiktoken
    recs += [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4,
              "raw_tokens": saved + 100, "out_tokens": 100} for _ in range(1)]
    recs += [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4,
              "raw_tokens": 100, "out_tokens": 100} for _ in range(9)]
    agg = aggregate(recs)
    assert agg["tools"][0]["blocks"] == 100 and agg["tools"][0]["tokenized"] == 10

    srv = primer_liability([_scan("kb", "wrapped", "kb", pol)], agg)["servers"][0]
    assert srv["blocks"] == 100 and srv["tokenized_blocks"] == 10
    assert srv["break_even_coverage"] == pytest.approx(saved / primer)
    assert srv["verdict"] == "TUNE"
    # The error this pins, stated as the number it would have produced: 10x, and over the bar.
    assert srv["blocks"] / srv["blocks_to_break_even"] > 1


def test_a_liability_blob_from_a_pre_verdict_terse_degrades_to_a_dash_instead_of_raising(
        tmp_path):
    """`build_recommend_section` is public and `--json` emits this exact dict, so a blob
    round-tripped through a terse that predates #238 carries none of the four new keys. It
    must render, not raise — a report is never load-bearing (#197). Mirrors
    `test_a_liability_blob_without_a_verdict_degrades_instead_of_raising` one layer up."""
    from terse.stats import _fmt_verdict

    liab = primer_liability([_scan("terse", "router", "kb", _policy(tmp_path))],
                            _agg(("kb", 10, 10_000, 1_000)))
    legacy = dict(liab)
    legacy["servers"] = [{k: v for k, v in s.items()
                          if k not in ("verdict", "verdict_reason", "break_even_coverage",
                                       "contributors")}
                         for s in liab["servers"]]
    text = "\n".join(build_recommend_section(legacy))
    assert f"  {'terse':<14} {'–':<12}" in text        # the verdict cell, dashed out
    assert _fmt_verdict({}) == ("–", "–", "–")


def test_the_TUNE_legend_points_at_autotune_rather_than_claiming_a_change_was_modelled(
        tmp_path):
    """`TUNE` is a REACHABILITY statement and nothing more. terse modelled no policy change
    and structurally cannot — the ledger is payload-free by design, so re-encoding a real
    payload under a hypothetical gate is not something any code in `stats.py` can do.

    Wording that implies terse tested a change would be the #144/#186/#188 defect family: a
    claim about something the code never did. The failure mode is a future edit quietly
    upgrading the sentence, which is why the wording itself is pinned."""
    srv_row = _one(tmp_path, ("kb", 1, 110, 100))
    assert srv_row["verdict"] == "TUNE"
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))],
                            _agg(("kb", 1, 110, 100)))
    text = "\n".join(build_recommend_section(liab))
    assert "terse policy autotune" in text
    assert "terse has tested no policy change here" in text
    assert "would" not in text and "will improve" not in text


def test_the_recommend_row_stays_inside_eighty_columns(tmp_path):
    """The same discipline as `test_the_break_even_row_stays_inside_eighty_columns`, and the
    reason this table exists at all: `INSUFFICIENT` is 12 characters and the break-even row
    was already 79 of its 80, so the verdict could not be a column there.

    Sized against every extreme at once — an over-long server name, the longest verdict word,
    the longest reason string in the closed vocabulary, and a million-fold coverage."""
    pol = _policy(tmp_path, name="wide238.json", tool="*")
    liab = primer_liability(
        [_scan("a-very-long-server-name", "wrapped", "unlabelled", pol),
         _scan("short", "wrapped", "short", pol),
         _scan("massive", "wrapped", "massive", pol)],
        _agg(("short", 1, 110, 100), ("massive", 1_000_000, 10_000_000, 0)))
    lines = build_recommend_section(liab)
    rows = lines[1:1 + len(liab["servers"])]
    assert len(rows) == 3
    assert "INSUFFICIENT" in "\n".join(rows)
    assert "short of break-even" in "\n".join(rows)
    assert max(len(ln) for ln in rows) <= 80, rows


def test_the_recommend_section_is_empty_when_nothing_pays_a_primer():
    """Absent and zero are different claims — the same reason `build_primer_section` returns
    `[]` rather than a table of dashes. There is no verdict to give about an install with no
    wrapped entry in it."""
    assert build_recommend_section(primer_liability([], _agg())) == []


def test_the_verdicts_sort_by_what_needs_action_not_by_rate(tmp_path):
    """Deliberately a DIFFERENT order from the break-even table directly above it, which
    sorts by rate. That table answers "which server is the best codec fit"; this one answers
    "what should I change today", so the row needing nothing (`KEEP`) sorts last and the one
    that can never pay for itself (`UNWRAP`) sorts first. Sorting this by rate would bury the
    only row an operator has to act on underneath the ones that are already fine."""
    pol = _policy(tmp_path, name="sort238.json", tool="*")
    liab = primer_liability(
        [_scan("keeper", "wrapped", "keeper", pol),
         _scan("nolabel", "wrapped", "", pol),
         _scan("shortfall", "wrapped", "shortfall", pol),
         _scan("loser", "wrapped", "loser", pol)],
        _agg(("keeper", 10, 10_000, 1_000), ("shortfall", 1, 110, 100),
             ("loser", 20, 1_000, 1_000)))
    rows = build_recommend_section(liab)[1:1 + len(liab["servers"])]
    assert [ln.split()[0] for ln in rows] == ["loser", "shortfall", "nolabel", "keeper"]
    assert [ln.split()[1] for ln in rows] == ["UNWRAP", "TUNE", "INSUFFICIENT", "KEEP"]
