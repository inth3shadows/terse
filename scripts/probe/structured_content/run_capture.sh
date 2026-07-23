#!/usr/bin/env bash
# Measures which MCP result field reaches the model's context (issue #128).
#
# Launches a SEPARATE headless Claude Code through a read-only mitmproxy, wired via
# --mcp-config/--strict-mcp-config to the structured-content fixture server. It has to be
# a separate process: HTTPS_PROXY is read at startup, so an already-running session can
# never be routed through the proxy mid-flight.
#
# Two arms, same prompt, same fixture:
#   raw   — the fixture server directly            (what the client does with the original)
#   terse — the fixture server behind terse proxy  (what the client does with a compressed
#                                                   text block + an untouched duplicate)
#
# The delta between the arms' tool_result token counts IS the answer: if terse's saving
# shows up in full, the client forwards only the text block; if it shows up halved, the
# client forwards both and #128's duplicate is a real cost.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8089}"
OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/terse-128-capture}"
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude)}"
TERSE_BIN="${TERSE_BIN:-$(command -v terse)}"
CA="${CA:-$HOME/.mitmproxy/mitmproxy-ca-cert.pem}"
# One call per arm is enough — the fixture is deterministic — but the prompt must make the
# model call the tool and nothing else, or the run costs tokens without adding evidence.
PROMPT="${PROMPT:-Call the ${TOOL:-records} tool once. Then reply with exactly: DONE. Do not explain, do not call any other tool.}"
TOOL="${TOOL:-records}"

[ -n "$CLAUDE_BIN" ] || { echo "no claude on PATH; set CLAUDE_BIN" >&2; exit 2; }
[ -n "$TERSE_BIN" ]  || { echo "no terse on PATH; set TERSE_BIN"  >&2; exit 2; }
[ -f "$CA" ] || { echo "mitmproxy CA missing at $CA — run mitmdump once to generate it" >&2; exit 2; }

mkdir -p "$OUTDIR"; chmod 700 "$OUTDIR"

# --no-stats: this probe must not write into the user's real savings ledger. Its calls are
# synthetic and would skew `terse stats` for every genuinely-wrapped server.
cat > "$OUTDIR/mcp-raw.json" <<JSON
{"mcpServers":{"sfix":{"command":"python3","args":["$HERE/structured_server.py"]}}}
JSON
# POLICY=<file> threads a policy through to the terse arm — the way to measure what
# `"structured": "compress"` (#128) actually does to the model's context, which is the
# only number that counts here: the ledger and the benchmark can both disagree with it.
POLICY_ARGS=""
if [ -n "${POLICY:-}" ]; then
  POLICY_ARGS="\"--policy\",\"$POLICY\","
fi
cat > "$OUTDIR/mcp-terse.json" <<JSON
{"mcpServers":{"sfix":{"command":"$TERSE_BIN","args":["proxy",$POLICY_ARGS"--no-stats","--server-name","sfix","--","python3","$HERE/structured_server.py"]}}}
JSON

MITM_PID=""
cleanup() { [ -n "$MITM_PID" ] && kill "$MITM_PID" 2>/dev/null || true; }
trap cleanup EXIT

port_is_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

run_arm() {
  # Separate statements on purpose: bash expands every word of a single `local` before
  # executing it, so `local arm="$1" out="...$arm..."` reads $arm while it is still unset
  # and dies under `set -u`.
  local arm="$1"
  local out="$OUTDIR/$arm.jsonl"
  local log="$OUTDIR/$arm.mitm.log"
  rm -f "$out"

  # Refuse a port that is ALREADY serving. Otherwise mitmdump fails to bind and dies,
  # while the readiness probe below connects happily to whatever else is there — the arm
  # then runs against a foreign proxy, writes no CAP_OUT, and fails downstream with a
  # message about auth or the tool call. Wrong diagnosis for a port collision.
  if port_is_open "$PORT"; then
    echo "  FAIL: 127.0.0.1:$PORT is already in use — set PORT=<free port>." >&2
    return 1
  fi

  CAP_OUT="$out" mitmdump -q --listen-host 127.0.0.1 -p "$PORT" \
      -s "$HERE/context_capture.py" >"$log" 2>&1 &
  MITM_PID=$!
  # Wait for OUR listener rather than sleeping a guessed interval. The liveness check is
  # the other half of the guard above: mitmdump can also die AFTER binding (bad addon,
  # missing CA), and without it the loop would just spin down to a timeout.
  local ready=""
  for _ in $(seq 1 50); do
    if ! kill -0 "$MITM_PID" 2>/dev/null; then
      echo "  FAIL: mitmdump exited during startup; see $log" >&2
      MITM_PID=""; return 1
    fi
    if port_is_open "$PORT"; then ready=1; break; fi
    sleep 0.2
  done
  if [ -z "$ready" ]; then
    echo "  FAIL: mitmdump never accepted a connection on $PORT; see $log" >&2
    return 1
  fi

  echo "== arm: $arm =="
  HTTPS_PROXY="http://127.0.0.1:$PORT" HTTP_PROXY="http://127.0.0.1:$PORT" \
  NODE_EXTRA_CA_CERTS="$CA" \
    "$CLAUDE_BIN" -p "$PROMPT" \
      --mcp-config "$OUTDIR/mcp-$arm.json" --strict-mcp-config \
      --allowedTools "mcp__sfix__$TOOL" \
      >"$OUTDIR/$arm.stdout" 2>"$OUTDIR/$arm.stderr" || {
        echo "  claude exited non-zero; see $OUTDIR/$arm.stderr" >&2; }

  kill "$MITM_PID" 2>/dev/null || true; wait "$MITM_PID" 2>/dev/null || true; MITM_PID=""

  # An empty artifact is a FAILED measurement, not a clean one (the #131 lesson).
  if [ ! -s "$out" ]; then
    echo "  FAIL: no tool_result blocks captured for arm '$arm'." >&2
    echo "        Check $OUTDIR/$arm.stderr (auth? tool not called?) and $log (proxy?)." >&2
    return 1
  fi
  echo "  captured $(wc -l < "$out") tool_result block(s) -> $out"
}

run_arm raw
run_arm terse

echo
"$HERE/report.py" "$OUTDIR/raw.jsonl" "$OUTDIR/terse.jsonl"
