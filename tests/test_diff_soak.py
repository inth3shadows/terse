"""Long-chain drift soak for the cross-call diff tier (#8/#20 follow-up).

The per-hop guarantee is proven at encode time (a diff is accepted only if it
reconstructs curr exactly) and pinned by the test_proxy/test_diff unit tests. What
none of those cover is DRIFT: a client that reconstructs hop N against its own hop
N-1 state, hundreds of hops deep, across interleaved tools, keyframes, shape flips,
error interludes, and a mid-session reconnect. This soak drives the real Interceptor
over a seeded evolving workload while an independent `ModelView` mirrors exactly
what the model must do with each emitted text — and asserts view == the raw
downstream payload at EVERY hop, so any base-bookkeeping bug (wrong base, missed
eviction, stale keyframe counter) surfaces as a first-divergence step index."""

from __future__ import annotations

import json
import random

from terse import text_diff, transforms
from terse.policy import Policy, Rule
from terse.proxy import Interceptor

TIERS = ("minify", "tabularize", "dictionary")


def _req(mid, name):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                       "params": {"name": name}})


def _init_req(mid):
    return json.dumps({"jsonrpc": "2.0", "id": mid, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05"}})


def _result_msg(mid, text):
    return json.dumps({"jsonrpc": "2.0", "id": mid,
                       "result": {"content": [{"type": "text", "text": text}]}})


class ModelView:
    """The client side of the diff protocol: per tool, keep the last reconstructed
    value and apply each emitted text per the documented terse semantics (diff →
    apply to the previous same-tool result; anything else → a self-contained full).
    Deliberately implemented over the public decode functions only — it must never
    peek at the Interceptor's internal state, or the soak proves nothing."""

    def __init__(self):
        self.state: dict[str, object] = {}   # tool -> last full JSON value
        self.text_state: dict[str, str] = {}  # tool -> last full raw text

    def reset(self):
        """A client reconnect: the context (and every prior result) is gone."""
        self.state.clear()
        self.text_state.clear()

    def see(self, tool: str, emitted: str):
        """Reconstruct the full raw value this emission denotes, updating state.
        Returns the value (JSON) or text (str) the model would now believe in."""
        try:
            obj = json.loads(emitted)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict) and obj.get(transforms.DIFF_MARKER) == 1:
            curr = transforms.diff_decode(self.state[tool], obj)
            self.state[tool] = curr
            return curr
        if isinstance(obj, dict) and obj.get(text_diff.DIFF_MARKER) == 1:
            curr = text_diff.text_diff_decode(self.text_state[tool], obj)
            self.text_state[tool] = curr
            return curr
        if obj is not None:
            curr = transforms.decompress(emitted)
            self.state[tool] = curr
            return curr
        self.text_state[tool] = emitted     # non-JSON: the raw text IS the full form
        return emitted


class Soak:
    """Drives one Interceptor + one ModelView and asserts equality at every hop."""

    def __init__(self, interval: int):
        pol = Policy(rules=[Rule("*", TIERS)], diff=True,
                     diff_keyframe_interval=interval)
        self.inter = Interceptor(pol)
        self.view = ModelView()
        self.mid = 0
        self.consec: dict[str, int] = {}     # tool -> consecutive emitted diffs
        self.max_consec: dict[str, int] = {}
        self.diffs = 0                       # total diff emissions observed

    def call(self, tool: str, payload) -> str:
        """One tools/call round trip. `payload` is a JSON value or a raw str.
        Asserts the ModelView reconstructs it exactly. Returns the emitted text."""
        self.mid += 1
        raw = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
        self.inter.note_request(_req(self.mid, tool))
        out = self.inter.transform_response(_result_msg(self.mid, raw))
        emitted = json.loads(out)["result"]["content"][0]["text"]
        got = self.view.see(tool, emitted)
        want = payload
        assert got == want, (f"drift at step {self.mid} tool={tool}: "
                             f"reconstruction diverged from the raw payload")
        is_diff = (transforms.DIFF_MARKER in emitted
                   or text_diff.DIFF_MARKER in emitted)
        self.consec[tool] = self.consec.get(tool, 0) + 1 if is_diff else 0
        self.max_consec[tool] = max(self.max_consec.get(tool, 0), self.consec[tool])
        self.diffs += is_diff
        return emitted

    def reconnect(self):
        """Client re-handshake (#20): both sides drop everything."""
        self.mid += 1
        self.inter.note_request(_init_req(self.mid))
        self.view.reset()
        self.consec.clear()


# --------------------------------------------------------------------- workloads
class RecordWorld:
    """An evolving uniform record list under {"result": [...]} — the agent-loop
    shape the row diff targets. Mutations keep the id column unique and append new
    rows at the end (the representable pattern), with occasional reorders that
    force the diff to bow out to a full form (which must also reconstruct)."""

    def __init__(self, rng: random.Random, n: int = 30, reorders: bool = True):
        self.rng = rng
        self.next_id = n
        self.reorders = reorders
        self.rows = [self._row(i) for i in range(n)]

    def _row(self, i):
        return {"id": i, "status": "active", "score": i % 11,
                "url": "https://x.example/api/items"}

    def step(self):
        r = self.rng.random()
        if r < 0.55 and self.rows:                       # mutate one field
            self.rng.choice(self.rows)["status"] = self.rng.choice(
                ["active", "closed", "stale", "merged"])
        elif r < 0.75:                                   # append a new row
            self.rows.append(self._row(self.next_id))
            self.next_id += 1
        elif r < 0.90 and len(self.rows) > 3:            # delete one row
            del self.rows[self.rng.randrange(len(self.rows))]
        elif r < 0.95 or not self.reorders:              # no-op: identical payload
            pass
        else:                                            # reorder: diff bows out
            self.rng.shuffle(self.rows)
        return {"result": [dict(row) for row in self.rows]}


class MapWorld:
    """A structure-like dict-map (path -> {symbols: [...]}) with non-uniform inner
    records — the nested shape #71/#72 was about. Exercises the coarse keys diff."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.n = 0
        self.files = {f"src/m{i}.py": self._syms(i) for i in range(6)}

    def _syms(self, i):
        self.n += 1
        return {"symbols": [{"name": f"fn_{i}_{j}", "kind": "function",
                             "line": 10 * j + 1, "hash": f"h{self.n}{j}"}
                            for j in range(4)]
                + [{"name": "os", "kind": "import"}]}

    def step(self):
        r = self.rng.random()
        keys = list(self.files)
        if r < 0.6:                                      # edit one file's symbols
            k = self.rng.choice(keys)
            self.files[k] = self._syms(len(keys))
        elif r < 0.8:                                    # new file
            self.files[f"src/m{self.n}.py"] = self._syms(self.n)
        elif len(keys) > 2:                              # file removed
            del self.files[self.rng.choice(keys)]
        return {"files": {k: json.loads(json.dumps(v))
                          for k, v in self.files.items()}}


class LogWorld:
    """A growing, occasionally edited plain-text log — the CDC text-diff domain."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.lines = [f"[{i:04d}] worker heartbeat ok, queue_depth={i % 7}"
                      for i in range(120)]

    def step(self):
        r = self.rng.random()
        if r < 0.6:                                      # append (tail -f shape)
            n = len(self.lines)
            self.lines.append(f"[{n:04d}] worker heartbeat ok, queue_depth={n % 7}")
        elif r < 0.85 and self.lines:                    # edit one line in place
            i = self.rng.randrange(len(self.lines))
            self.lines[i] = f"[{i:04d}] RETRY backoff=2 attempt={self.rng.randrange(9)}"
        else:                                            # truncate (log rotation)
            self.lines = self.lines[-60:]
        return "\n".join(self.lines)


def test_soak_400_steps_interleaved_tools_no_drift():
    # The headline soak: 400 hops, three tools interleaved, production keyframe
    # interval (5), plus error interludes and a mid-soak client reconnect. Every
    # hop asserts the client reconstruction equals the raw payload exactly.
    rng = random.Random(42)
    soak = Soak(interval=5)
    worlds = {"gh.items": RecordWorld(rng), "repo.map": MapWorld(rng),
              "fs.log": LogWorld(rng)}
    tools = list(worlds)

    for step in range(400):
        tool = tools[step % 3] if rng.random() < 0.7 else rng.choice(tools)
        soak.call(tool, worlds[tool].step())

        if step in (150, 290) and tool == "gh.items":
            # a non-JSON interlude (upstream error) on a JSON tool: becomes the
            # visible "previous result", so the JSON base must evict (#8) …
            soak.call(tool, "upstream error: rate limited")
            # … and the next JSON result must re-anchor as a full, never a diff
            # against the now-invisible pre-error base.
            emitted = soak.call(tool, worlds[tool].step())
            assert transforms.DIFF_MARKER not in emitted

        if step == 200:
            soak.reconnect()                            # client context wiped (#20)
            for t in tools:
                # first post-reconnect emission per tool must be self-contained
                emitted = soak.call(t, worlds[t].step())
                assert (transforms.DIFF_MARKER not in emitted
                        and text_diff.DIFF_MARKER not in emitted)

    # the soak must have actually exercised long diff chains, not degenerated to
    # fulls — and the keyframe bound must have held for every tool throughout.
    assert soak.diffs > 150, f"only {soak.diffs} diffs emitted — workload too weak"
    assert soak.max_consec and max(soak.max_consec.values()) == 5
    assert all(v <= 5 for v in soak.max_consec.values())


def test_soak_unbounded_chain_interval_zero_exact_at_depth_300():
    # keyframe interval 0 = never re-anchor: a single tool's chain runs hundreds of
    # diffs deep off ONE full result. Reconstruction must stay exact at every depth
    # — this is the pure accumulation-drift case the keyframe normally bounds.
    rng = random.Random(7)
    soak = Soak(interval=0)
    world = RecordWorld(rng, n=40, reorders=False)   # every step representable
    soak.call("gh.items", {"result": [dict(r) for r in world.rows]})   # the anchor
    for _ in range(300):
        soak.call("gh.items", world.step())
    # small seeded mutations diff nearly every hop; prove real chain depth was hit
    assert soak.diffs >= 250
    assert soak.max_consec["gh.items"] >= 200


def test_soak_text_only_long_chain():
    # The CDC text path keeps its own base/keyframe state (#25); soak it separately
    # at the production interval so a text-side bookkeeping bug can't hide behind
    # the JSON assertions of the interleaved soak.
    rng = random.Random(99)
    soak = Soak(interval=5)
    world = LogWorld(rng)
    for _ in range(200):
        soak.call("fs.log", world.step())
    assert soak.diffs > 100
    assert max(soak.max_consec.values()) <= 5
