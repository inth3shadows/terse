"""Guards for the `synthetic.*` payloads in `scripts/gen_stress_corpus.py` (#403 Blocker 2).

`secret.list_credentials` is the largest codec saver in the live ledger and has zero
captured payloads by design: the shipped policy says `capture: false` for every
secret-broker tool. The generator reproduces the STRUCTURE (`secret_broker/ops/meta.py`)
and invents the values, so the codec can be measured on the shape without a credential
inventory reaching a corpus that feeds a published benchmark.

Three things have to stay true or the payload is worse than nothing: it must remain
labelled as synthetic, it must keep the shape it claims to reproduce, and it must actually
reach the codec verdict (a question, an encoded payload, and enough of them for a cell).

`scripts/` is not a package, so the generator is loaded by path, once — the same idiom
`test_gen_real_corpus.py` uses, and for the same reason (the script prepends `src/` to
`sys.path` on every load).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from terse import capture, codeceval, fluency
from terse.report import _CODEC_MIN_TRIALS
from terse.transforms import minify

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_stress_corpus.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_stress_corpus", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gsc = _load()

# The keys `list_credentials` returns per row (`secret_broker/ops/meta.py:46-110`): an
# undeclared vault item has no `kind`/`env_var` and carries a `note` instead.
DECLARED = {"name", "kind", "source", "env_var", "has_value",
            "call_count", "last_used", "last_ok"}
UNDECLARED = {"name", "source", "has_value", "call_count", "last_used", "last_ok", "note"}


def test_the_synthetic_payloads_are_labelled_as_synthetic():
    # The verdict table prints the tool name, so this prefix is what stops a cell measured
    # on invented values from reading as production coverage.
    assert gsc.SYNTHETIC, "no synthetic payloads registered"
    assert all(tool.startswith("synthetic.") for tool, _ in gsc.SYNTHETIC)
    assert not any(t.startswith("synthetic.") for t in gsc.PAYLOADS)


@pytest.mark.parametrize("obj", [obj for _, obj in gsc.SYNTHETIC])
def test_each_variant_keeps_the_shape_the_broker_returns(obj):
    assert set(obj) == {"credentials", "services"}
    keysets = {frozenset(r) for r in obj["credentials"]}
    assert keysets <= {frozenset(DECLARED), frozenset(UNDECLARED)}, keysets
    assert frozenset(DECLARED) in keysets, "no declared rows"
    # Required per VARIANT, not just somewhere in the set: a variant with no undeclared rows
    # has no union schema and no repeated note, yet still contributes its trials to the
    # cell's verdict (review of #403 Blocker 2 caught exactly that mutation surviving).
    assert frozenset(UNDECLARED) in keysets, "no undeclared rows"
    # A never-used credential carries explicit nulls, not absent keys — the distinction a
    # deref/enumerate failure destroys, and one the real tool emits (`meta.py:46`). Asserted
    # on DECLARED rows: undeclared ones are null by construction, so they would carry this
    # for free.
    declared = [r for r in obj["credentials"] if "env_var" in r]
    assert any(r["last_used"] is None for r in declared)
    assert any(r["last_used"] is not None for r in declared)
    # `extract_records` walks dict keys in order, so `credentials` is the cell under test —
    # not `services`, whose container columns would make it a different question.
    assert capture.extract_records(obj) == obj["credentials"]


@pytest.mark.parametrize("obj", [obj for _, obj in gsc.SYNTHETIC])
def test_each_variant_reaches_the_codec_verdict(obj):
    # A question the eval admits, on a payload the codec actually encodes. Without both,
    # `run_codec_fluency` skips it and the fixture measures nothing.
    assert [q.qtype for q in codeceval.gen_codec_questions(obj)] == ["enumerate"]
    assert codeceval.codec_changes(obj)


def test_the_largest_variant_exercises_absent_columns_and_the_legend():
    # Union-schema absent cells (undeclared rows drop two keys) and the repeated `note`
    # string are the two encodings this shape is here to test.
    obj = max((o for _, o in gsc.SYNTHETIC), key=lambda o: len(o["credentials"]))
    text = fluency.compress(obj)
    assert "absent_cols" in text and "__terse_dict__" in text
    # The `note` specifically must be what the legend folds: a legend fires on any repeated
    # value, so `__terse_dict__` alone would still be there if every note were unique.
    note = next(r["note"] for r in obj["credentials"] if "note" in r)
    assert minify(obj).count(note) > 1, "the note is not repeated in the raw payload"
    assert text.count(note) == 1, "the repeated note was not aliased into the legend"


@pytest.mark.parametrize("obj", [obj for _, obj in gsc.SYNTHETIC])
def test_env_var_is_both_explicitly_null_and_absent(obj):
    # The one column where the real payload forces the encoder to tell an ABSENT cell from
    # an explicit null: declared rows may have no env_var (`config.py:182` types it
    # `str | None`), undeclared rows omit the key entirely.
    declared = [r for r in obj["credentials"] if "env_var" in r]
    assert any(r["env_var"] is None for r in declared), "no declared row with a null env_var"
    assert any(r["env_var"] for r in declared), "every declared env_var is null"
    assert any("env_var" not in r for r in obj["credentials"]), "no row omits env_var"
    assert "__terse_absent__" in fluency.compress(obj), "the absent-vs-null branch is unused"


def test_the_largest_variant_is_the_size_the_real_tool_returns():
    # The shipped policy's own note calls `list_credentials` "~70 uniform records". A
    # handful of rows would not exercise the table the real payload produces.
    assert max(len(o["credentials"]) for _, o in gsc.SYNTHETIC) >= 50


def test_the_variants_are_distinct_payloads_pooled_into_one_cell():
    # Same tool + shape = one verdict cell; distinct content = separate envelopes, since
    # `capture_payload` names files by content sha.
    tools = {tool for tool, _ in gsc.SYNTHETIC}
    assert len(tools) == 1, tools
    bodies = {minify(obj) for _, obj in gsc.SYNTHETIC}
    assert len(bodies) == len(gsc.SYNTHETIC)


def test_there_are_enough_variants_for_a_cell_to_resolve():
    # One question per payload, so the count of payloads is what decides whether a cell can
    # clear the floor at all. Three at `--trials 7` reach 21; one never could.
    assert len(gsc.SYNTHETIC) * 7 >= _CODEC_MIN_TRIALS
    assert _CODEC_MIN_TRIALS > 7, "one payload at --trials 7 must not be able to resolve"


def test_the_fleet_shapes_are_opt_in(tmp_path, capsys):
    # `terse verify`'s zero-setup sample shells out to this script with no flag, and that
    # sample is an adopter-facing claim about ordinary traffic. These three payloads are 78%
    # of its tokens and moved its headline from +37.2% to +38.7%, so they stay out by
    # default (review of #403 Blocker 2).
    assert gsc.main(str(tmp_path)) == 0
    capsys.readouterr()
    envelopes = capture.load_corpus(tmp_path)
    assert len(envelopes) == len(gsc.PAYLOADS)
    assert not [e for e in envelopes if e["tool"].startswith("synthetic.")]


def _run_script(tmp_path, *args):
    """The script as `terse verify` runs it: a subprocess with positional args, NOT `main`.

    The opt-in lives in the `__main__` argv block, and `cli.py`'s `_cmd_verify` shells out
    (`subprocess.run([sys.executable, script, sample_dir])`). Pinning only `main`'s keyword
    left that layer unwatched — a mutation defaulting it to on passed the whole suite while
    `terse verify`'s headline moved from +37.2% to +38.7% (review of #403 Blocker 2).
    """
    return subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), *args],
                          capture_output=True, text=True)


def test_the_command_line_keeps_the_fleet_shapes_off_by_default(tmp_path):
    r = _run_script(tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(capture.load_corpus(tmp_path)) == len(gsc.PAYLOADS)
    assert "synthetic." not in r.stdout


def test_the_command_line_writes_the_fleet_shapes_when_asked(tmp_path):
    r = _run_script(tmp_path, gsc.FLEET_FLAG)
    assert r.returncode == 0, r.stderr
    assert len(capture.load_corpus(tmp_path)) == len(gsc.PAYLOADS) + len(gsc.SYNTHETIC)
    assert "synthetic.secret.list_credentials" in r.stdout


def test_a_mistyped_flag_is_refused_rather_than_named_as_the_corpus_dir(tmp_path):
    # It used to positional-ize: exit 0, a corpus in a directory called `--fleetshapes`,
    # no fleet shapes in it, and nothing saying so — then a codec-verdict run measuring
    # nothing about the shape the flag exists to measure.
    r = _run_script(tmp_path, "--fleetshapes")
    assert r.returncode == 2, r.stdout
    assert "unknown option" in r.stderr and gsc.FLEET_FLAG in r.stderr
    assert not list(tmp_path.iterdir()), "a refused run must write nothing"


def test_the_written_line_survives_a_payload_with_no_record_list(tmp_path, capsys, monkeypatch):
    # `verify` turns a traceback here into "the bundled sample generator failed", so the
    # size line must not assume a `credentials` key.
    monkeypatch.setattr(gsc, "PAYLOADS", {})
    assert gsc.main(str(tmp_path), with_fleet_shapes=True) == 0
    assert "(70 records)" in capsys.readouterr().out
    monkeypatch.setattr(gsc, "SYNTHETIC", [("synthetic.other", {"a": 1, "b": 2})])
    assert gsc.main(str(tmp_path), with_fleet_shapes=True) == 0
    assert "(2 keys)" in capsys.readouterr().out


def test_the_generator_writes_the_fleet_shapes_when_asked(tmp_path, capsys):
    assert gsc.main(str(tmp_path), with_fleet_shapes=True) == 0
    capsys.readouterr()
    envelopes = capture.load_corpus(tmp_path)
    assert len(envelopes) == len(gsc.PAYLOADS) + len(gsc.SYNTHETIC)
    synthetic = [e for e in envelopes if e["tool"].startswith("synthetic.")]
    assert len(synthetic) == len(gsc.SYNTHETIC)
    assert len({e["sha"] for e in synthetic}) == len(gsc.SYNTHETIC)
