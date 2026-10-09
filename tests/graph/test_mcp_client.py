import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from src.graph.mcp_client import (
    ORDER_TOOLS,
    READ_ONLY_TOOLS,
    ToolSelectionError,
    open_research_tools,
    open_tools,
    parse_tool_result,
    select_tools,
)

STUB = str(Path(__file__).parent / "stub_mcp_server.py")
STUB_CONNECTIONS = {"market": {"transport": "stdio", "command": sys.executable, "args": [STUB]}}


# ------------------------------------------------------------------ constants


def test_research_tools_never_include_order_tools():
    assert READ_ONLY_TOOLS.isdisjoint(ORDER_TOOLS)
    assert "place_paper_order" not in READ_ONLY_TOOLS


# ------------------------------------------------------------ parse_tool_result


def test_parse_content_blocks_with_json():
    raw = [{"type": "text", "text": '{"ok": true, "price": 10.5}', "id": "lc_1"}]
    assert parse_tool_result(raw) == {"ok": True, "price": 10.5}


def test_parse_plain_json_string():
    assert parse_tool_result('{"ok": false, "code": "X"}') == {"ok": False, "code": "X"}


def test_parse_dict_with_and_without_ok():
    assert parse_tool_result({"ok": True, "a": 1}) == {"ok": True, "a": 1}
    assert parse_tool_result({"a": 1})["ok"] is False


def test_parse_tool_message():
    msg = ToolMessage(content='{"ok": true, "price": 1}', tool_call_id="c1")
    assert parse_tool_result(msg) == {"ok": True, "price": 1}


def test_parse_server_side_exception_text_is_not_ok():
    raw = [{"type": "text", "text": "Error executing tool explode: kaput"}]
    out = parse_tool_result(raw)
    assert out["ok"] is False
    assert out["code"] == "TOOL_ERROR"
    assert "kaput" in out["message"]


@pytest.mark.parametrize("raw", [None, "", "   ", [], [{"type": "image", "data": "x"}], 42])
def test_parse_empty_or_unsupported_is_not_ok(raw):
    assert parse_tool_result(raw)["ok"] is False


def test_parse_json_that_is_not_an_object_or_lacks_ok():
    assert parse_tool_result("[1, 2, 3]")["ok"] is False
    assert parse_tool_result('{"price": 10}')["ok"] is False


def test_parse_keeps_server_error_fields():
    raw = '{"ok": false, "code": "UPSTREAM", "message": "m", "retryable": true}'
    out = parse_tool_result(raw)
    assert out["code"] == "UPSTREAM" and out["retryable"] is True


def test_parse_long_error_text_is_truncated():
    assert len(parse_tool_result("x" * 5000)["message"]) == 500


# ---------------------------------------------------------------- select_tools


def _fake(*names):
    return [SimpleNamespace(name=n) for n in names]


def test_select_tools_filters_and_sorts():
    out = select_tools(_fake("b", "a", "c"), {"a", "b"})
    assert [t.name for t in out] == ["a", "b"]


def test_select_tools_missing_tool_raises():
    with pytest.raises(ToolSelectionError, match="get_quote"):
        select_tools(_fake("get_history"), {"get_quote", "get_history"})


# --------------------------------------------- real stdio session against stub


@pytest.mark.asyncio
async def test_research_tools_are_read_only_and_complete():
    async with open_research_tools(connections=STUB_CONNECTIONS) as tools:
        names = {t.name for t in tools}
    assert names == set(READ_ONLY_TOOLS)
    assert names.isdisjoint(ORDER_TOOLS)


@pytest.mark.asyncio
async def test_tool_call_roundtrip_and_single_server_process():
    async with open_research_tools(connections=STUB_CONNECTIONS) as tools:
        quote = next(t for t in tools if t.name == "get_quote")
        first = parse_tool_result(await quote.ainvoke({"symbol": "TCS"}))
        second = parse_tool_result(await quote.ainvoke({"symbol": "INFY"}))
    assert first["ok"] is True and first["price"] == 123.0 and first["symbol"] == "TCS"
    assert first["pid"] == second["pid"]  # one process, so the server cache would survive


@pytest.mark.asyncio
async def test_server_side_exception_surfaces_as_not_ok():
    async with open_tools({"explode"}, connections=STUB_CONNECTIONS) as tools:
        raw = await tools[0].ainvoke({})
    assert parse_tool_result(raw)["ok"] is False


@pytest.mark.asyncio
async def test_asking_for_unknown_tool_fails_loudly():
    with pytest.raises(ToolSelectionError):
        async with open_tools({"get_quote", "nope"}, connections=STUB_CONNECTIONS):
            pytest.fail("should not get here")
