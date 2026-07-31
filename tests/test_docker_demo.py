"""The container's default path: `terse proxy -- python examples/demo_mcp_server.py`.

`Dockerfile`'s ENTRYPOINT+CMD is exactly that command, and it is the ONLY thing a
registry inspector ever runs against this image. Nothing else in the suite covers it —
`tests/fake_mcp_server.py` is a different file with a different tool set — so a rename,
a broken JSON-RPC reply, or a demo payload that stops being record-shaped would show up
as a dead listing rather than a failing test.

The proxy is driven with `default_policy()` and no field drop, because that is what a
flagless `terse proxy` uses (cli.py: `load_policy(args.policy) if args.policy else
default_policy()`).
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import sys

from terse.policy import default_policy
from terse.proxy import run_proxy
from terse.transforms import decompress

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO = ROOT / "examples" / "demo_mcp_server.py"
DOCKERFILE = ROOT / "Dockerfile"


def _drive(*requests: dict) -> list[dict]:
    cin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    cout = io.StringIO()
    rc = run_proxy([sys.executable, str(DEMO)], default_policy(), stdin=cin, stdout=cout)
    assert rc == 0
    return [json.loads(ln) for ln in cout.getvalue().splitlines() if ln.strip()]


def _exec_form(directive: str) -> list[str]:
    m = re.search(rf'^{directive} \[(.+)\]', DOCKERFILE.read_text(), re.M)
    assert m, f"Dockerfile lost its {directive}"
    return [p.strip().strip('"') for p in m.group(1).split(",")]


def test_dockerfile_entrypoint_and_cmd_form_the_command_this_module_drives():
    # Neither directive can fail the BUILD — both resolve at run time — so dropping
    # `proxy` or the `--` yields an image that starts and quietly does nothing useful.
    # The `--` is load-bearing: it is what makes CMD the downstream server command.
    assert _exec_form("ENTRYPOINT") == ["terse", "proxy", "--"]
    assert DEMO.exists()
    assert _exec_form("CMD")[-1].endswith("examples/demo_mcp_server.py")


def test_demo_server_answers_the_handshake_a_registry_inspector_performs():
    # initialize -> tools/list -> tools/call is the whole introspection exchange. Tools
    # without an inputSchema cannot be rendered or invoked by an inspector at all, which
    # is the specific reason this demo is not tests/fake_mcp_server.py.
    init, listed = _drive(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert init["result"]["serverInfo"]["name"] == "terse-demo"
    # the client's protocol version is echoed, not a hardcoded one that would mismatch
    assert init["result"]["protocolVersion"] == "2025-06-18"
    tools = {t["name"]: t for t in listed["result"]["tools"]}
    assert set(tools) == {"demo_orders", "demo_logs"}
    assert all(t.get("description") and t.get("inputSchema") for t in tools.values())


def test_demo_orders_reaches_the_model_compressed_not_as_raw_json():
    # The demo's entire point is that the reply is terse's form. If the payload ever
    # stops folding, the image still "works" while demonstrating nothing.
    (call,) = _drive({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "demo_orders", "arguments": {}}})
    text = call["result"]["content"][0]["text"]
    assert "__terse_table__" in text and "__terse_dict__" in text
    raw = json.dumps({"orders": _orders()}, indent=2)
    assert len(text) < len(raw) / 2  # measured -65%; half is a loose floor, not the claim


def test_demo_tools_survive_junk_arguments_without_killing_the_stream():
    # A traceback on stdout would corrupt the JSON-RPC wire, not merely fail one call —
    # so out-of-range and wrong-typed arguments must clamp, and an unknown tool must
    # come back as a normal isError result.
    limit, lines, unknown = _drive(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "demo_orders", "arguments": {"limit": -5}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "demo_logs", "arguments": {"lines": "not-a-number"}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "nope", "arguments": {}}},
    )
    assert not limit["result"]["isError"] and not lines["result"]["isError"]
    assert unknown["result"]["isError"] is True
    # and the clamp actually clamps: limit=-5 lands on the floor of 1, it does not fall
    # through to `ORDERS[:-5]`, which would quietly serve 35 records for a request of -5.
    orders = decompress(limit["result"]["content"][0]["text"])["orders"]
    assert len(orders) == 1


def test_malformed_requests_are_answered_not_fatal():
    # Each of these used to raise an uncaught AttributeError inside the demo server. That
    # is far worse than a failed call: the process dies and every LATER request on the
    # stream goes unanswered, which an inspector reads as a dead server. So the real
    # assertion is the trailing well-formed tools/list — it proves the wire survived.
    replies = _drive(
        # a JSON-RPC BATCH: legal 2.0, part of MCP through 2025-03-26, unsupported here
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}],           # type: ignore[arg-type]
        # `arguments` present but not an object — truthy, so `or {}` does not catch it
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "demo_orders", "arguments": [1, 2]}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": "not-an-object"},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/list"},
    )
    assert replies[-1]["result"]["tools"], "the server died and stopped answering"
    # a batch cannot be correlated to an id, so it draws no reply at all; the other two
    # are answerable and must come back as errors or as clamped-to-default successes
    assert {r["id"] for r in replies} == {2, 3, 4}


def test_a_notification_with_a_known_method_draws_no_reply():
    # `{"method": "tools/list"}` with no id is a legal NOTIFICATION. Dispatching it and
    # replying `"id": null` is a protocol violation a strict client may reject, so the
    # id check has to sit ahead of the method dispatch, not behind it.
    replies = _drive(
        {"jsonrpc": "2.0", "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 9, "method": "tools/list"},
    )
    assert [r["id"] for r in replies] == [9]


def _orders() -> list[dict]:
    sys.path.insert(0, str(DEMO.parent))
    try:
        import demo_mcp_server
    finally:
        sys.path.pop(0)
    return demo_mcp_server.ORDERS
