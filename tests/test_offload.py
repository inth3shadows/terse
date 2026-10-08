"""Results the client offloads to a file instead of putting in context (item 3).

Claude Code saves an MCP result over its limit (25,000 tokens by default,
`MAX_MCP_OUTPUT_TOKENS` to change it) to a file and hands the model only the path. Two
consequences for terse: a lazy primer attached to such a result may never be read, and the
ledger's saving for it never reached context. Measured on the live ledger before this: ~20%
of the reported saving sat in offloaded results.
"""
from __future__ import annotations

import json

import pytest

from terse.policy import Policy, Rule
from terse.proxy import (
    OFFLOAD_DEFAULT_CHARS,
    OFFLOAD_DEFAULT_TOKENS,
    PRIMER_HEAD,
    Interceptor,
    PrimerLatch,
    offload_limit,
    over_limit,
    union_primer,
)
from terse.stats import build_stats_writer, context_tokens, load_stats

POL = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))])
LIMIT = 1500   # tokens; the fixtures below are sized against it


def _rows(n: int) -> str:
    # cl100k, raw -> compressed: 12 rows 376 -> 196, 60 rows 1,864 -> 779,
    # 200 rows 6,204 -> 2,459. The primer is ~555.
    return json.dumps({"result": [{"id": i, "owner": {"name": f"user-{i:02d}",
                                                      "team": "platform-infrastructure"},
                                   "count": i * 3} for i in range(n)]})


def _drive(inter: Interceptor, mid: int, text: str) -> dict:
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    line = inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": mid,
         "result": {"content": [{"type": "text", "text": text}]}}))
    return json.loads(line)["result"]


def _has_primer(result: dict) -> bool:
    return any(PRIMER_HEAD in b.get("text", "") for b in result["content"])


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setenv("MAX_MCP_OUTPUT_TOKENS", str(LIMIT))


# --- the limit ---

def test_only_an_offloading_client_has_a_limit():
    assert offload_limit(None, {}) is None
    assert offload_limit("some-other-client", {}) is None
    assert offload_limit("claude-code", {}) == OFFLOAD_DEFAULT_TOKENS == 25_000


@pytest.mark.parametrize("env, want", [("50000", 50_000), ("", 25_000), ("lots", 25_000),
                                       ("0", 25_000), ("-5", 25_000)])
def test_the_limit_follows_max_mcp_output_tokens(env, want):
    assert offload_limit("claude-code", {"MAX_MCP_OUTPUT_TOKENS": env}) == want


def test_over_limit_counts_tokens_not_bytes():
    assert not over_limit("x" * 10_000, None)
    assert not over_limit("short", 3)                  # 5 bytes, but 1 token
    assert over_limit(_rows(200), 1000)
    assert not over_limit(_rows(200), 100_000)


# --- the primer ---

def test_the_primer_skips_an_offloaded_result_and_attaches_to_the_next(small_limit):
    inter = Interceptor(POL)
    inter.client_name = "claude-code"
    first = _drive(inter, 1, _rows(200))               # 2,459 compressed > 1,500
    assert '"__terse_' in first["content"][0]["text"]  # still compressed, just unprimed
    assert not _has_primer(first)
    assert _has_primer(_drive(inter, 2, _rows(12)))    # the latch stayed armed


def test_the_primer_skips_a_result_that_only_the_primer_pushes_over(monkeypatch):
    # 779 compressed fits under 1,000; with the ~555-token primer it would not.
    monkeypatch.setenv("MAX_MCP_OUTPUT_TOKENS", "1000")
    inter = Interceptor(POL)
    inter.client_name = "claude-code"
    assert not _has_primer(_drive(inter, 1, _rows(60)))
    assert _has_primer(_drive(inter, 2, _rows(12)))


def test_a_client_that_never_offloads_gets_the_primer_on_any_result(small_limit):
    inter = Interceptor(POL)
    inter.client_name = "some-other-client"
    assert _has_primer(_drive(inter, 1, _rows(200)))


def test_a_router_does_not_spend_its_shared_latch_on_an_offloaded_result(small_limit):
    latch = PrimerLatch()
    a, b = (Interceptor(POL, server_name=n, lazy_primer=False, shared_primer=latch)
            for n in ("a", "b"))
    latch.set_text(union_primer([(POL, "a"), (POL, "b")], structured_wrap=True))
    a.client_name = b.client_name = "claude-code"
    assert not _has_primer(_drive(a, 1, _rows(200)))
    assert latch.pending()
    assert _has_primer(_drive(b, 2, _rows(12)))
    assert not latch.pending()


# --- the ledger ---

def _ledger(tmp_path, client: str | None, text: str) -> list[dict]:
    log = tmp_path / "stats.jsonl"
    inter = Interceptor(POL, stats=build_stats_writer(log, "gh"))
    inter.client_name = client
    _drive(inter, 1, text)
    return [r for r in load_stats(log) if not r.get("event")]


def test_a_result_offloaded_after_terse_is_tagged_out(tmp_path, small_limit):
    (row,) = _ledger(tmp_path, "claude-code", _rows(200))
    assert row["offload"] == "out"


def test_a_result_terse_kept_under_the_limit_is_tagged_raw(tmp_path, small_limit):
    (row,) = _ledger(tmp_path, "claude-code", _rows(60))      # 1,864 raw -> 779
    assert row["offload"] == "raw"


def test_an_inline_result_and_a_non_offloading_client_carry_no_tag(tmp_path, small_limit):
    (inline,) = _ledger(tmp_path, "claude-code", _rows(12))
    (other,) = _ledger(tmp_path / "x", "some-other-client", _rows(200))
    assert "offload" not in inline and "offload" not in other


def test_a_writer_from_before_the_field_still_works(small_limit):
    # A stats callback with the old seven-argument signature must not be handed `offload=`
    # for an ordinary result.
    seen: list = []
    inter = Interceptor(POL, stats=lambda *a: seen.append(a))
    inter.client_name = "claude-code"
    _drive(inter, 1, _rows(12))
    assert len(seen) == 1 and len(seen[0]) == 7


# --- the context basis ---

def test_an_offloaded_row_saves_nothing_in_context():
    assert context_tokens({"offload": "out"}, 30_000, 27_000) == (0, 0)
    assert context_tokens({"offload": "raw"}, 30_000, 9_000) == (9_000, 9_000)
    assert context_tokens({}, 3_000, 900) == (3_000, 900)


def test_a_kept_inline_row_with_a_rewritten_typed_field_pays_its_typed_size():
    rec = {"offload": "raw", "structured_tokens": 28_000, "structured_out_tokens": 8_000}
    assert context_tokens(rec, 40_000, 12_000) == (8_000, 8_000)


def _drive_typed(inter: Interceptor, mid: int, rows: int) -> dict:
    typed = json.loads(_rows(rows))
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    line = inter.transform_response(json.dumps(
        {"jsonrpc": "2.0", "id": mid,
         "result": {"content": [{"type": "text", "text": json.dumps(typed)}],
                    "structuredContent": typed}}))
    return json.loads(line)["result"]


def test_the_typed_wrapper_skips_an_offloaded_result_and_wraps_the_next(small_limit):
    # #463's second attach site: a structured-reading client gets the primer inside the
    # typed field. An over-limit one reverts to the raw hold and keeps the latch armed.
    latch = PrimerLatch()
    peer = Interceptor(POL, server_name="p0", lazy_primer=False, shared_primer=latch)
    latch.set_text(union_primer([(POL, "p0")], structured_wrap=True))
    peer.client_name = "claude-code"
    big = _drive_typed(peer, 1, 200)
    assert "__terse_primer__" not in json.dumps(big["structuredContent"])
    assert latch.pending()
    small = _drive_typed(peer, 2, 12)
    assert "__terse_primer__" in small["structuredContent"]
    assert not latch.pending()


def test_a_tokenizer_that_raises_never_breaks_forwarding(small_limit, monkeypatch):
    # Under the primer guard an exception from the tokenizer would kill the proxy's
    # reader thread. It must read as inline instead.
    import terse.proxy as proxy_mod

    def refuse(text):
        raise ValueError("tokenizer refused the text")
    monkeypatch.setattr(proxy_mod, "count_cl100k", refuse)
    assert not over_limit("log tail: <|endoftext|> " + "x" * 30_000, LIMIT)
    inter = Interceptor(POL)
    inter.client_name = "claude-code"
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "gh.api.items"}}))
    line = inter.transform_response(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
        "content": [{"type": "text", "text": _rows(12)},
                    {"type": "text", "text": "log tail: <|endoftext|> " + "x" * 30_000}]}}))
    assert json.loads(line)["result"]["content"]


def test_an_over_limit_text_spelling_a_special_token_is_seen_as_over_limit():
    # The tokenizer used to raise on this text, which the guard reads as inline: the
    # primer was then attached to a result the client was about to save to a file.
    text = "<|endoftext|> " + "word " * 4000
    assert len(text) <= OFFLOAD_DEFAULT_CHARS      # decided by the token count, not the size
    assert over_limit(text, LIMIT)


def test_a_result_spelling_a_special_token_is_still_compressed():
    # The codec's size comparisons used to raise on this text; the proxy then failed open
    # and forwarded the result whole.
    rows = json.dumps({"result": [{"id": i, "owner": {"name": f"user-{i:02d}",
                                                      "team": "platform-infrastructure"},
                                   "note": "tail <|endoftext|> marker"} for i in range(60)]})
    inter = Interceptor(POL, lazy_primer=False)
    out = _drive(inter, 1, rows)["content"][-1]["text"]
    assert "__terse_table__" in out and len(out) < len(rows) // 2


def test_the_tokenizer_pass_is_bounded(monkeypatch):
    seen: list[int] = []
    import terse.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "count_cl100k", lambda t: seen.append(len(t)) or 0)
    assert over_limit("y" * 2_000_000, 1000)   # past the char cutoff: no tokenizer pass at all
    assert seen == []
    over_limit("y" * 40_000, 1000)             # under it: only a limit*8 prefix is tokenized
    assert seen == [8000]


# --- #496: the cutoff is ~50,000 characters, not only 25k tokens ---

_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"]


def _prose(n: int) -> str:
    # cl100k, raw -> compressed: 200 rows 46,770 chars / 13,679 tok -> 43,628 / 12,299;
    # 260 rows 60,852 / 17,784 -> 56,750 / 15,984. Prose tokenizes long (~3.5 chars per
    # token), so these cross 50,000 chars while staying far under 25,000 tokens -- the shape
    # of the Opus run's 57 KB `list_principles` result that Claude Code offloaded (#496).
    return json.dumps({"result": [{"id": i, "note": " ".join(
        _WORDS[(i * 7 + k) % 26] for k in range(i % 5 + 30)) + f" item {i}"}
        for i in range(n)]})


def test_a_result_over_fifty_thousand_chars_is_offloaded_under_the_token_limit():
    # Measured on the operator's transcripts: largest MCP result shown inline 49,034 chars,
    # smallest offloaded 51,246 ("Output too large (50KB)"), Claude Code 2.1.284-2.1.287.
    assert over_limit(_prose(260), OFFLOAD_DEFAULT_TOKENS)        # 60,852 chars, ~17.8k tok
    assert not over_limit(_prose(200), OFFLOAD_DEFAULT_TOKENS)    # 46,770 chars, ~13.7k tok


@pytest.fixture
def default_limits(monkeypatch):
    # These tests run inside Claude Code sessions too, which may export its own limit.
    monkeypatch.delenv("MAX_MCP_OUTPUT_TOKENS", raising=False)


def test_the_primer_skips_a_result_over_fifty_thousand_chars_at_the_default_limits(
        default_limits):
    inter = Interceptor(POL)
    inter.client_name = "claude-code"
    big = _drive(inter, 1, _prose(260))                # 56,750 chars compressed
    assert not _has_primer(big)
    assert _has_primer(_drive(inter, 2, _prose(200)))  # 43,628 + the primer still fits


def test_a_result_over_fifty_thousand_chars_is_tagged_out_at_the_default_limits(
        tmp_path, default_limits):
    (row,) = _ledger(tmp_path, "claude-code", _prose(260))
    assert row["offload"] == "out"
