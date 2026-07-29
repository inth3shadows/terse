"""Statistical correctness of the A/B session harness.

`scripts/bench/ab_session.py` prints the numbers a ship/no-ship call is made on, so
the failure mode that matters is not a crash — it is a row that reads confidently and
says the wrong thing. Every test here pins one such row.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "ab_session.py"


def _load():
    spec = importlib.util.spec_from_file_location("ab_session", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load()


def _transcript(path: Path, *, turns: int, cache_write: int = 0, cache_read: int = 0,
                output: int = 100, inp: int = 10) -> Path:
    """A minimal Claude Code transcript: `turns` assistant records, each billed once."""
    with path.open("w") as fh:
        for i in range(turns):
            fh.write(json.dumps({
                "type": "assistant",
                "requestId": f"req{i}",
                "message": {
                    "id": f"msg{i}",
                    "model": "claude-opus-5",
                    "content": [],
                    "usage": {
                        "input_tokens": inp,
                        "cache_creation_input_tokens": cache_write,
                        "cache_read_input_tokens": cache_read,
                        "output_tokens": output,
                    },
                },
            }) + "\n")
    return path


def _arm(tmp_path: Path, label: str, specs: list[dict]):
    paths = [_transcript(tmp_path / f"{label}{i}.jsonl", **s) for i, s in enumerate(specs)]
    return ab.Arm(paths, label)


# ------------------------------------------------------------------ zero control mean

def test_zero_control_mean_reports_na_not_zero_percent(tmp_path):
    """A control arm that never wrote cache has no percentage. Printing +0.0% beside a
    five-figure absolute delta reads as "no change" — the opposite of the row's own
    delta column."""
    a = _arm(tmp_path, "a", [{"turns": 3, "cache_write": 0}] * 2)
    b = _arm(tmp_path, "b", [{"turns": 3, "cache_write": 5000}] * 2)
    line, clears = ab._row("cache write", a, b, "cache_write")
    assert "n/a" in line
    assert "+0.0%" not in line
    assert "+15,000" in line  # 3 turns x 5000, the delta the row actually found
    assert clears is True


def test_nonzero_control_mean_still_reports_a_percentage(tmp_path):
    a = _arm(tmp_path, "a", [{"turns": 2, "output": 100}] * 2)
    b = _arm(tmp_path, "b", [{"turns": 2, "output": 200}] * 2)
    line, _ = ab._row("output", a, b, "output")
    assert "+100.0%" in line
    assert "n/a" not in line


# ------------------------------------------------------------------- zero pooled spread

def test_zero_variance_delta_is_signal_not_withheld(tmp_path):
    """Both arms perfectly reproducible is the STRONGEST evidence a delta is real. The
    earlier `pooled > 0` guard forced it to the same verdict as "too noisy to tell"."""
    a = _arm(tmp_path, "a", [{"turns": 3, "output": 200}] * 3)
    b = _arm(tmp_path, "b", [{"turns": 3, "output": 400}] * 3)
    line, clears = ab._row("output", a, b, "output")
    assert clears is True
    assert "SIGNAL" in line
    assert "noise" not in line


def test_zero_variance_and_zero_delta_is_not_signal(tmp_path):
    """The flip side: identical arms must not become SIGNAL just because sd is 0."""
    a = _arm(tmp_path, "a", [{"turns": 3, "output": 200}] * 2)
    b = _arm(tmp_path, "b", [{"turns": 3, "output": 200}] * 2)
    line, clears = ab._row("output", a, b, "output")
    assert clears is False
    assert "SIGNAL" not in line


def test_zero_variance_arm_verdict_is_not_inconclusive(tmp_path, capsys):
    a = _arm(tmp_path, "a", [{"turns": 3, "output": 200}] * 2)
    b = _arm(tmp_path, "b", [{"turns": 3, "output": 400}] * 2)
    ab.report(a, b)
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" not in out
    assert "TERSE LOSES" in out  # B spends more output; sign convention is d = B - A


def test_single_run_arm_still_withholds_judgement(tmp_path):
    """n=1 reports sd=0 too, but for the opposite reason — no spread was observed at
    all. That must stay `n<2`, or the zero-variance fix would hand it false confidence."""
    a = _arm(tmp_path, "a", [{"turns": 3, "output": 200}])
    b = _arm(tmp_path, "b", [{"turns": 3, "output": 400}])
    line, clears = ab._row("output", a, b, "output")
    assert clears is False
    assert "n<2" in line


# ------------------------------------------------------------------------ duplicate runs

def test_duplicate_paths_are_not_counted_as_replicates(tmp_path):
    """The same transcript twice is n=1 wearing an n=2 costume: it reports sd=0 from
    non-independent data and would clear the noise floor on its own repetition."""
    p = _transcript(tmp_path / "dup.jsonl", turns=3, output=200)
    arm = ab.Arm([p, p], "a")
    assert len(arm.runs) == 1
    assert len(arm.kept) == 1


def test_distinct_paths_with_identical_content_are_kept(tmp_path):
    """Dedup is on path identity, not content — two real runs may legitimately tie."""
    p1 = _transcript(tmp_path / "r1.jsonl", turns=3, output=200)
    p2 = _transcript(tmp_path / "r2.jsonl", turns=3, output=200)
    assert len(ab.Arm([p1, p2], "a").runs) == 2


# ------------------------------------------------------------------------- outlier drop

def test_modal_turn_filter_is_symmetric_across_arms(tmp_path):
    """The mode is computed on the union so it cannot be tuned to favor one side."""
    a = _arm(tmp_path, "a", [{"turns": 3}, {"turns": 3}, {"turns": 18}])
    b = _arm(tmp_path, "b", [{"turns": 3}, {"turns": 3}])
    assert ab.modal_turns(a, b) == 3
    a.restrict_to_turns(3)
    b.restrict_to_turns(3)
    assert len(a.kept) == 2 and len(a.dropped) == 1
    assert len(b.kept) == 2 and not b.dropped


def test_modal_turn_tie_resolves_deterministically_to_the_lower_count(tmp_path):
    a = _arm(tmp_path, "a", [{"turns": 5}, {"turns": 9}])
    b = _arm(tmp_path, "b", [{"turns": 9}, {"turns": 5}])
    assert ab.modal_turns(a, b) == 5


def test_arm_emptied_by_the_filter_reports_no_comparable_runs(tmp_path, capsys):
    a = _arm(tmp_path, "a", [{"turns": 3}, {"turns": 3}])
    b = _arm(tmp_path, "b", [{"turns": 7}, {"turns": 9}])
    rc = ab.report(a, b)
    assert rc == 2
    assert "NO COMPARABLE RUNS" in capsys.readouterr().out
