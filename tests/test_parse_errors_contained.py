"""#582 item 8: one malformed payload or file never aborts a whole command. A >4300-digit
integer raises a plain ValueError, deep nesting a RecursionError, a torn multibyte write a
UnicodeDecodeError; readers used to catch only JSONDecodeError."""
from __future__ import annotations

import json

import pytest

from terse import transforms

HUGE_INT = '{"n": ' + "9" * 5000 + "}"
DEEP = "[" * 800 + "]" * 800
VERY_DEEP = "[" * 100_000 + "]" * 100_000


@pytest.mark.parametrize("raw", [HUGE_INT, DEEP, VERY_DEEP, "nope", None])
def test_loads_within_depth_raises_only_parse_errors(raw):
    with pytest.raises(transforms.PARSE_ERRORS):
        transforms.loads_within_depth(raw)


def test_loads_within_depth_passes_ordinary_json():
    assert transforms.loads_within_depth('{"a": [1, 2]}') == {"a": [1, 2]}


def _env(raw, sha="s1", tool="t.get"):
    return {"raw": raw, "sha": sha, "tool": tool, "shape": "records"}


def test_load_corpus_skips_bad_utf8_and_deep_files(tmp_path):
    from terse.capture import load_corpus
    (tmp_path / "bad.json").write_bytes(b'{"raw": "\xff"}')
    (tmp_path / "deep.json").write_text(VERY_DEEP, encoding="utf-8")
    (tmp_path / "ok.json").write_text(json.dumps(_env('{"a":1}')), encoding="utf-8")
    assert [e["sha"] for e in load_corpus(tmp_path)] == ["s1"]


def test_load_history_survives_bad_bytes_and_non_dict_lines(tmp_path):
    from terse.history import load_history
    p = tmp_path / "h.jsonl"
    p.write_bytes(b'{"ts": 1}\n\xff\xfe\n[1]\n5\n{"ts": 2}\n')
    assert [r["ts"] for r in load_history(p)] == [1, 2]


def test_load_stats_survives_bad_bytes_and_huge_ints(tmp_path):
    from terse.stats import load_stats
    p = tmp_path / "stats.jsonl"
    p.write_bytes(b'{"ts": 1, "tool": "a"}\n\xff\n' + HUGE_INT.encode() + b'\n{"ts": 2}\n')
    assert [r["ts"] for r in load_stats(p)] == [1, 2]


def test_cross_tokenizer_savings_passes_a_deep_payload_through():
    from terse.measure import cross_tokenizer_savings
    rows = cross_tokenizer_savings([_env(DEEP)])
    assert len(rows) == 1


@pytest.mark.parametrize("raw", [HUGE_INT, DEEP])
def test_fluency_pack_and_harness_skip_unparseable_payloads(raw):
    from terse.fluency.harnesses import build_chain_windows, diff_pairs
    from terse.fluency.pack import build_pack
    envs = [_env(raw, "a"), _env(raw, "b")]
    build_pack(envs)
    diff_pairs(envs)
    build_chain_windows(envs)


def test_run_fluency_skips_an_unparseable_payload():
    from terse.fluency.harnesses import run_fluency
    run_fluency([_env(HUGE_INT)], {"m": lambda system, user: "1"})
