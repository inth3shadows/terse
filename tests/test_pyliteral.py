"""The opt-in `pyliteral` tier (#460): a text block that is the exact Python `repr` of
JSON-native data is read as that data and compressed like any payload.

`mcp-server-sqlite` and `postgres-mcp` return query rows as `str(results)`, which is not
JSON, so terse passed them through at 0%. The fold is taken only when `repr` of the parse
rebuilds the text byte for byte, and the envelope decodes back to that text exactly.
"""

from __future__ import annotations

import ast
import json
import sqlite3

import pytest

from terse import measure, policy, transforms
from terse.policy import Policy, Rule
from terse.policy_gen import generate_policy
from terse.proxy import (
    PRIMER_PYLITERAL,
    Interceptor,
    build_primer,
    union_primer,
    wire_needs_primer,
)

TIERS = ("minify", "tabularize", "dictionary", "pyliteral")
POL = Policy(rules=[Rule("sqlite.*", TIERS)])


def _sqlite_rows(n: int = 40) -> str:
    """What `mcp-server-sqlite`'s `read_query` returns: `str([dict(row) ...])`."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("create table users (id integer, name text, email text, plan text, "
               "active integer, score real, note text)")
    db.executemany("insert into users values (?,?,?,?,?,?,?)",
                   [(i, f"user {i}", f"user{i}@example.com",
                     "pro" if i % 3 else "free", i % 2, i * 1.5,
                     None if i % 4 else "it's \"quoted\"") for i in range(n)])
    rows = [dict(r) for r in db.execute("select * from users").fetchall()]
    return str(rows)


def test_the_sqlite_fixture_is_lossless_and_compresses():
    raw = _sqlite_rows()
    applied = policy.apply(raw, "sqlite.read_query", POL)
    assert not applied.skipped and "pyliteral" in applied.tiers
    assert transforms.decompress(applied.text) == raw
    assert '{"__terse_py__":1,' in applied.text
    assert len(applied.text) < len(raw) * 0.7


def test_the_tier_is_opt_in():
    raw = _sqlite_rows()
    off = Policy(rules=[Rule("sqlite.*", ("minify", "tabularize", "dictionary"))])
    applied = policy.apply(raw, "sqlite.read_query", off)
    assert applied.skipped and applied.text == raw


def test_a_json_payload_never_reports_the_tier():
    applied = policy.apply(json.dumps([{"a": i} for i in range(20)]), "sqlite.x", POL)
    assert not applied.skipped and "pyliteral" not in applied.tiers


@pytest.mark.parametrize("raw", [
    str([(1, 2), (3, 4)]),                    # tuple: JSON would bring back a list
    str([{1, 2}]),                            # set
    str([{"a": b"x"}]),                       # bytes
    str([{1: "a"}]),                          # non-str key
    "[{'a':1}]",                              # not repr's own spacing
    '[{"a": 1, "b": \'x\'}]',                 # mixed quoting
    str([{"a": 1}]) + "\n",                   # trailing newline
    "('a', 'b')",                             # tuple at the top
    "'just a string'",
    "plain prose [with brackets]",
    str([{"__terse_py__": 1, "v": 2}] * 20),  # carries a reserved marker
])
def test_what_repr_cannot_rebuild_passes_through_unchanged(raw):
    assert transforms.parse_pyliteral(raw) is None
    assert transforms.fold_pyliteral(raw) is None
    applied = policy.apply(raw, "sqlite.read_query", POL)
    assert applied.text == raw


def test_a_literal_too_small_to_win_passes_through():
    raw = str({"a": 1.0, "b": 2})
    assert transforms.parse_pyliteral(raw) == {"a": 1.0, "b": 2}
    assert transforms.fold_pyliteral(raw) is None
    assert policy.apply(raw, "sqlite.read_query", POL).text == raw


def test_a_json_payload_carrying_the_marker_is_left_alone():
    raw = json.dumps({"__terse_py__": 1, "v": [{"a": 1}] * 30})
    assert policy.apply(raw, "sqlite.x", POL).text == raw


@pytest.mark.parametrize("raw", [
    "[" * (transforms.MAX_DEPTH + 1) + "]" * (transforms.MAX_DEPTH + 1),
    "[" + "1, " * (transforms.MAX_PYLITERAL_CHARS // 3) + "1]",
])
def test_size_and_depth_bombs_are_refused_before_literal_eval(raw, monkeypatch):
    def boom(_):
        raise AssertionError("literal_eval reached")
    monkeypatch.setattr(ast, "literal_eval", boom)
    assert transforms.parse_pyliteral(raw) is None


def test_a_payload_just_under_both_caps_is_parsed():
    depth = transforms.MAX_DEPTH
    assert transforms.parse_pyliteral("[" * depth + "]" * depth) is not None


def test_an_envelope_that_does_not_decode_to_the_text_is_never_emitted(monkeypatch):
    """The fold's own gate, behind the `repr` check: a codec bug that changed the data
    must fall back to the raw text, not ship."""
    monkeypatch.setattr(transforms, "compress_with",
                        lambda *a, **k: transforms.minify({transforms.PY_MARKER: 1, "v": []}))
    assert transforms.fold_pyliteral(_sqlite_rows()) is None


def test_types_round_trip_exactly():
    raw = str([{"i": -3, "f": -0.0, "big": 10**30, "e": 1e-07, "t": True, "n": None,
                "s": "tab\there é \\ 'q' \"d\"", "l": [1, [2, {}]]}] * 12)
    out = transforms.fold_pyliteral(raw)
    assert out is not None and transforms.decompress(out) == raw


# --- through the proxy ---

def _transform(inter, mid, text):
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": "sqlite.read_query"}}))
    out = inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}]}}))
    return [b["text"] for b in json.loads(out)["result"]["content"]]


def test_the_proxy_emits_the_envelope_and_records_why():
    """The reason is the diff path's label, so it shows with diffing on (off, every
    single-block row reads `diff_off`)."""
    rows: list = []
    pol = Policy(rules=[Rule("sqlite.*", TIERS)], diff=True)
    inter = Interceptor(pol, lazy_primer=False,
                        stats=lambda *a, **k: rows.append(a))
    inter.diff = True
    raw = _sqlite_rows()
    (text,) = _transform(inter, 1, raw)
    assert transforms.decompress(text) == raw
    assert rows[0][4] == "pyliteral"            # diff_reason


def test_a_python_literal_result_clears_the_diff_bases():
    pol = Policy(rules=[Rule("sqlite.*", TIERS)], diff=True)
    inter = Interceptor(pol, lazy_primer=False)
    inter.diff = True
    _transform(inter, 1, json.dumps([{"id": i, "s": "x"} for i in range(30)]))
    _transform(inter, 2, "some plain text " * 20)
    assert "sqlite.read_query" in inter.last_text
    inter.last["sqlite.read_query"] = [1]
    _transform(inter, 3, _sqlite_rows())
    assert "sqlite.read_query" not in inter.last
    assert "sqlite.read_query" not in inter.last_text


def test_the_primer_explains_the_envelope_only_when_the_tier_is_reachable():
    assert PRIMER_PYLITERAL in build_primer(POL)
    off = Policy(rules=[Rule("sqlite.*", ("minify", "tabularize", "dictionary"))])
    assert PRIMER_PYLITERAL not in build_primer(off)
    assert PRIMER_PYLITERAL in union_primer([(off, "a"), (POL, "b")])


def test_auto_primer_mode_always_explains_the_envelope():
    assert wire_needs_primer(transforms.fold_pyliteral(_sqlite_rows()) or "")


# --- measure and policy generate ---

def test_measure_credits_the_fold_only_on_a_python_literal():
    raw = _sqlite_rows()
    row = measure.measure_payload(raw)
    saved = row["saved_cl100k"]
    assert not row["applicable"] and saved["pyliteral"] > 0
    assert saved["tier_total"] == saved["pyliteral"]
    assert measure.measure_payload(json.dumps({"a": 1}))["saved_cl100k"]["pyliteral"] == 0
    assert measure.measure_payload("plain prose")["saved_cl100k"]["pyliteral"] == 0


def test_policy_generate_offers_the_tier_only_where_it_wins():
    envs = ([{"tool": "sqlite.read_query", "raw": _sqlite_rows(n)} for n in (20, 30, 40)]
            + [{"tool": "gh.list", "raw": json.dumps([{"id": i, "state": "open"}
                                                      for i in range(30)])}])
    _doc, rows = generate_policy(envs)
    by_tool = {r["tool"]: r for r in rows}
    assert "pyliteral" in by_tool["sqlite.read_query"]["tiers"]
    assert "non-JSON" not in by_tool["sqlite.read_query"]["reason"]
    assert "pyliteral" not in by_tool["gh.list"]["tiers"]


# --- review of #559 ---

def test_an_int_too_long_to_print_is_refused_not_raised():
    """`literal_eval` reads `0xfff...` as an int `repr` cannot print (Python's int-to-str
    digit limit). That ValueError used to escape into `terse measure` and `policy
    generate`, which run the check on every non-JSON payload whatever the policy."""
    raw = "[0x" + "f" * 5000 + "]"
    assert transforms.parse_pyliteral(raw) is None
    assert transforms.fold_pyliteral(raw) is None
    assert measure.measure_payload(raw)["saved_cl100k"]["pyliteral"] == 0


def test_the_size_cap_keeps_the_parse_small():
    """The AST is what costs memory: 8 MiB of `[0, 0, ...]` peaked at ~2.6 GB."""
    assert transforms.MAX_PYLITERAL_CHARS <= 256 << 10


def test_brackets_inside_a_string_cannot_smuggle_depth_past_the_cap(monkeypatch):
    """The pre-parse bracket count ignores quotes, so `)` inside a string lowers it. Today
    CPython's own parser refuses nesting past ~200 ("too many nested parentheses"), so the
    parse cannot return deeper than `MAX_DEPTH`; the parsed value is checked anyway, for a
    parser that allows more. Simulated: the parse returns a value one level past the cap
    whose `repr` is exactly the text."""
    deep: list = [1]
    for _ in range(transforms.MAX_DEPTH):
        deep = [deep]
    obj = [")" * 400, deep]
    raw = repr(obj)
    assert not transforms._bracket_depth_exceeds(raw, transforms.MAX_DEPTH)
    assert transforms.exceeds_depth(obj)
    monkeypatch.setattr(ast, "literal_eval", lambda _: obj)
    assert transforms.parse_pyliteral(raw) is None


def test_a_tier_below_its_bar_does_not_lift_the_tool_over_the_threshold():
    """Like `emb_fail`: a saving the generated rule will not deliver must not decide
    whether the tool is compressed at all."""
    envs = ([{"tool": "mixed", "raw": _sqlite_rows(30)}]
            + [{"tool": "mixed", "raw": json.dumps({"items": [{"id": i, "state": "open"}
                                                             for i in range(8)]}, indent=1)}]
            + [{"tool": "mixed", "raw": "plain prose line\n" * 400}])
    (probe,) = generate_policy(envs, threshold=0.0)[1]
    py, rest = probe["py_pct"], probe["saved_pct"] - probe["py_pct"]
    assert py > 0 and rest > 0
    bar = (max(py, rest) + py + rest) / 2      # each alone is under it, together over
    (row,) = generate_policy(envs, threshold=bar)[1]
    assert row["tiers"] == [], row["reason"]
