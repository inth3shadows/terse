"""A JSON integer past Python's 4300-digit limit is non-JSON to terse, never a crash (#563).

`json.loads` raises a plain `ValueError` for it, not the `JSONDecodeError` subclass the
parse sites caught, so it escaped `policy.apply` and crashed `terse compress`, the evals
and `policy generate`. The proxy failed open around it; the rest did not.
"""

from __future__ import annotations

import json

import pytest

from terse import dropeval, measure, policy, policy_gen
from terse.cli import main
from terse.policy import Policy, Rule
from terse.proxy import Interceptor

BIG = "[" + "1" * 5000 + "]"
POL = Policy(rules=[Rule("x", ("minify", "tabularize", "dictionary"))])


def test_the_input_really_is_past_the_limit():
    with pytest.raises(ValueError) as exc:
        json.loads(BIG)
    assert not isinstance(exc.value, json.JSONDecodeError)


def test_policy_apply_passes_it_through():
    applied = policy.apply(BIG, "x", POL)
    assert applied.skipped and applied.text == BIG


def test_apply_joined_refuses_it_as_non_json():
    assert policy.apply_joined([BIG, "[1]"], "x", POL)[2] == policy.JOIN_REFUSED_NON_JSON


def test_terse_compress_passes_it_through(tmp_path, capsys):
    f = tmp_path / "big.json"
    f.write_text(BIG)
    assert main(["compress", "--tool", "x", str(f)]) == 0
    assert capsys.readouterr().out.strip() == BIG


def test_measure_and_policy_generate_do_not_crash():
    assert measure.measure_joined([BIG, "[1]"]) is None
    measure.cross_tokenizer_savings([{"tool": "x", "raw": BIG}])
    envs = [{"tool": "x", "raw": BIG}, {"tool": "x", "raw": json.dumps([{"a": 1}] * 9)}]
    policy_gen.generate_policy(envs)
    assert policy_gen._drop_candidates([BIG, BIG]) is not None


def test_dropeval_reads_it_as_text():
    rule = Policy(rules=[Rule("x", ("minify",))]).select
    _probe, _rule, payload, is_json = dropeval._probe_envelope(
        {"tool": "x", "raw": BIG}, lambda tool, server=None: rule(tool, server))
    assert not is_json and payload == BIG


def test_a_multi_block_result_holding_one_still_compresses_the_rest():
    """`_partial_join`'s record check raised too, failing the whole result open."""
    inter = Interceptor(POL, lazy_primer=False)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "x"}}))
    rec = [json.dumps({"id": i, "state": "open", "url": "https://example.com/a/b/c"})
           for i in range(12)]
    content = [{"type": "text", "text": t} for t in [*rec, BIG]]
    out = json.loads(inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"content": content}})))
    texts = [b["text"] for b in out["result"]["content"]]
    assert BIG in texts and any("__terse_" in t for t in texts)


def test_the_codec_eval_skips_it_instead_of_crashing():
    from terse.codeceval import run_codec_fluency
    run = run_codec_fluency([{"tool": "x", "raw": BIG}], {}, preflight=False, policy=POL)
    assert run is not None
