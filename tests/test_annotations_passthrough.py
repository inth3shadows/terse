"""#467 gate: a downstream tool's `annotations` (readOnlyHint / idempotentHint /
openWorldHint) must reach the client unchanged through `tools/list`, for both the
single proxy and the multiproxy merge. Glama's TDQS scores tool `annotations`
directly, so a proxy that silently dropped them would cap every wrapped server's
score, not just the demo's.

`fake_annotated_server.py` is a dedicated fixture: neither `fake_mcp_server.py` nor
(pre-#467) the demo server ever emitted `annotations`, so nothing else in the suite
could have caught a regression here.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

from terse.multiproxy import run_multi_proxy
from terse.policy import default_policy
from terse.proxy import run_proxy

ROOT = pathlib.Path(__file__).resolve().parent
FAKE = ROOT / "fake_annotated_server.py"

EXPECTED_ANNOTATIONS = {
    "readOnlyHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}


def _drive_single(*requests: dict) -> list[dict]:
    cin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    cout = io.StringIO()
    rc = run_proxy([sys.executable, str(FAKE)], default_policy(), stdin=cin, stdout=cout)
    assert rc == 0
    return [json.loads(ln) for ln in cout.getvalue().splitlines() if ln.strip()]


def test_single_proxy_tools_list_passes_annotations_through_unchanged():
    (listed,) = _drive_single({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {t["name"]: t for t in listed["result"]["tools"]}
    assert tools["annotated.tool"]["annotations"] == EXPECTED_ANNOTATIONS


def test_multiproxy_tools_list_passes_annotations_through_unchanged_when_renamed(tmp_path):
    # Two peers exporting the SAME bare tool name forces #168's qualification path —
    # `_expose_names` rewrites `name` to `{peer}__annotated.tool` via `{**it, "name":
    # exposed}` — which is exactly the spread that must not drop `annotations`.
    cfg = tmp_path / "multi.json"
    cfg.write_text(json.dumps({"downstreams": [
        {"name": "ann1", "command": [sys.executable, str(FAKE)]},
        {"name": "ann2", "command": [sys.executable, str(FAKE)]},
    ]}), encoding="utf-8")
    cin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
    cout = io.StringIO()
    rc = run_multi_proxy(str(cfg), default_policy(), stdin=cin, stdout=cout)
    assert rc == 0
    (listed,) = [json.loads(ln) for ln in cout.getvalue().splitlines() if ln.strip()]
    tools = {t["name"]: t for t in listed["result"]["tools"]}
    assert set(tools) == {"ann1__annotated.tool", "ann2__annotated.tool"}
    for name in tools:
        assert tools[name]["annotations"] == EXPECTED_ANNOTATIONS
