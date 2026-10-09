import pytest
from fakes import ExplodingChatModel, ScriptedChatModel, ai_text, ai_tool_calls
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from src.graph.research import (
    HISTORY_ARGS,
    baseline_history_id,
    baseline_portfolio_id,
    baseline_quote_id,
    evidence_from_messages,
    fetch_baseline,
    make_research_node,
    tokens_used,
)
from src.graph.state import initial_state

# ----------------------------------------------------------------- fake tools


def make_tools(calls: list, *, fail: set[str] = frozenset(), raise_in: set[str] = frozenset()):
    """In-process stand-ins for the MCP tools. `calls` records (name, args)."""

    def build(name, fn, description="stub"):
        async def run(**kwargs):
            calls.append((name, kwargs))
            if name in raise_in:
                raise RuntimeError(f"{name} blew up")
            if name in fail:
                return {"ok": False, "code": "UPSTREAM_ERROR", "message": "down", "retryable": True}
            return fn(**kwargs)

        return StructuredTool.from_function(
            coroutine=run, name=name, description=description, args_schema=fn.schema
        )

    from pydantic import BaseModel

    class Sym(BaseModel):
        symbol: str

    class Hist(BaseModel):
        symbol: str
        period: str = "1y"
        interval: str = "1d"
        last_n: int = 60

    class News(BaseModel):
        query: str
        limit: int = 10

    class NoArgs(BaseModel):
        pass

    def spec(schema, fn):
        fn.schema = schema
        return fn

    defs = [
        ("get_quote", spec(Sym, lambda symbol: {"ok": True, "price": 100.0, "source": "stub"})),
        ("get_history", spec(Hist, lambda **kw: {"ok": True, "candles": [], "source": "stub"})),
        ("get_fundamentals", spec(Sym, lambda symbol: {"ok": True, "pe": 20.0, "source": "stub"})),
        (
            "search_news",
            spec(News, lambda **kw: {"ok": True, "untrusted": True, "content": "headline"}),
        ),
        ("get_portfolio", spec(NoArgs, lambda: {"ok": True, "cash": 100000.0})),
    ]
    return [build(n, f) for n, f in defs]


def state_for(*symbols):
    return initial_state("Is RELIANCE a fit for me?", candidates=symbols)


# ------------------------------------------------------------------- baseline


@pytest.mark.asyncio
async def test_baseline_fetches_quote_history_per_symbol_and_portfolio_once():
    calls: list = []
    items = await fetch_baseline(make_tools(calls), ["RELIANCE", "TCS"])
    assert [n for n, _ in calls] == [
        "get_quote",
        "get_history",
        "get_quote",
        "get_history",
        "get_portfolio",
    ]
    ids = {i.id for i in items}
    assert baseline_quote_id("RELIANCE") in ids
    assert baseline_history_id("TCS") in ids
    assert baseline_portfolio_id() in ids
    history_call = next(a for n, a in calls if n == "get_history")
    assert history_call["last_n"] == HISTORY_ARGS["last_n"] == 250
    assert all(i.ok for i in items)


@pytest.mark.asyncio
async def test_baseline_tool_exception_becomes_failed_evidence_and_continues():
    calls: list = []
    items = await fetch_baseline(make_tools(calls, raise_in={"get_quote"}), ["TCS"])
    by_id = {i.id: i for i in items}
    q = by_id[baseline_quote_id("TCS")]
    assert q.ok is False and q.error_code == "TOOL_CALL_FAILED"
    assert by_id[baseline_history_id("TCS")].ok is True  # later calls still ran


@pytest.mark.asyncio
async def test_baseline_missing_tool_is_an_error():
    tools = [t for t in make_tools([]) if t.name != "get_history"]
    with pytest.raises(ValueError, match="get_history"):
        await fetch_baseline(tools, ["TCS"])


# ---------------------------------------------------------------- messages


def test_evidence_from_messages_pairs_calls_with_results():
    msgs = [
        HumanMessage("hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_fundamentals", "args": {"symbol": "TCS"}, "id": "c1"}],
        ),
        ToolMessage(content='{"ok": true, "pe": 25.0}', tool_call_id="c1", name="get_fundamentals"),
        ToolMessage(content="Error executing tool x: bad", tool_call_id="c9", name="x"),
    ]
    items = evidence_from_messages(msgs)
    assert items[0].id == "get_fundamentals:symbol=TCS" and items[0].ok is True
    assert items[0].data["pe"] == 25.0
    assert items[1].ok is False  # unparsable text is never ok
    assert items[1].tool == "x"


def test_tool_message_with_error_status_is_not_ok():
    msgs = [
        AIMessage(content="", tool_calls=[{"name": "search_news", "args": {}, "id": "c1"}]),
        ToolMessage(content='{"ok": true}', tool_call_id="c1", name="search_news", status="error"),
    ]
    assert evidence_from_messages(msgs)[0].ok is False


def test_tokens_used_sums_and_flags_missing_usage():
    a = ai_text(tokens=7)
    b = ai_tool_calls(("get_quote", {"symbol": "X"}), tokens=3)
    assert tokens_used([a, b]) == (10, False)
    assert tokens_used([a, ai_text(tokens=None)]) == (7, True)


# ---------------------------------------------------------------- the node


@pytest.mark.asyncio
async def test_node_happy_path():
    calls: list = []
    model = ScriptedChatModel(
        script=[
            ai_tool_calls(
                ("get_fundamentals", {"symbol": "RELIANCE"}),
                ("search_news", {"query": "Reliance Industries", "limit": 5}),
                tokens=100,
            ),
            ai_text("DONE", tokens=20),
        ]
    )
    node = make_research_node(model, make_tools(calls))
    out = await node(state_for("RELIANCE"))

    ev = out["evidence"]
    assert baseline_quote_id("RELIANCE") in ev and baseline_history_id("RELIANCE") in ev
    assert baseline_portfolio_id() in ev
    assert "get_fundamentals:symbol=RELIANCE" in ev
    news = ev["search_news:limit=5,query=Reliance Industries"]
    assert news["untrusted"] is True
    assert out["step_count"] == 1
    assert out["token_spend"] == 120
    assert out["flags"] == []
    agent_calls = [n for n, _ in calls if n in {"get_fundamentals", "search_news"}]
    assert sorted(agent_calls) == ["get_fundamentals", "search_news"]


@pytest.mark.asyncio
async def test_agent_never_sees_quote_history_or_portfolio_tools():
    model = ScriptedChatModel(
        script=[ai_tool_calls(("get_history", {"symbol": "TCS"})), ai_text()],
    )
    calls: list = []
    node = make_research_node(model, make_tools(calls))
    await node(state_for("TCS"))
    # get_history ran once (baseline). The agent's attempt to call it was not honoured.
    assert [n for n, _ in calls].count("get_history") == 1


@pytest.mark.asyncio
async def test_model_call_budget_stops_a_looping_agent():
    calls: list = []
    looping = ScriptedChatModel(
        script=[ai_tool_calls(("get_fundamentals", {"symbol": "TCS"}), tokens=5)], cycle=True
    )
    node = make_research_node(looping, make_tools(calls), max_model_calls=3)
    out = await node(state_for("TCS"))
    assert "research_budget_hit" in out["flags"]
    assert looping.calls <= 3
    assert [n for n, _ in calls].count("get_fundamentals") <= 3
    assert out["token_spend"] <= 15  # only real model calls are counted


@pytest.mark.asyncio
async def test_agent_failure_keeps_baseline_and_flags():
    node = make_research_node(ExplodingChatModel(script=[ai_text()]), make_tools([]))
    out = await node(state_for("TCS"))
    assert baseline_quote_id("TCS") in out["evidence"]
    assert "research_agent_failed:RuntimeError" in out["flags"]
    assert out["token_spend"] == 0
    assert "research_no_evidence" not in out["flags"]  # baseline succeeded


@pytest.mark.asyncio
async def test_everything_failing_raises_no_evidence_flag():
    node = make_research_node(
        ScriptedChatModel(script=[ai_text()]),
        make_tools([], fail={"get_quote", "get_history", "get_portfolio"}),
    )
    out = await node(state_for("TCS"))
    assert "research_no_evidence" in out["flags"]
    assert all(v["ok"] is False for v in out["evidence"].values())


@pytest.mark.asyncio
async def test_missing_token_usage_is_flagged_not_zeroed_silently():
    node = make_research_node(ScriptedChatModel(script=[ai_text(tokens=None)]), make_tools([]))
    out = await node(state_for("TCS"))
    assert "token_usage_unknown" in out["flags"]


@pytest.mark.asyncio
async def test_no_candidates_makes_no_calls():
    calls: list = []
    node = make_research_node(ScriptedChatModel(script=[ai_text()]), make_tools(calls))
    out = await node(state_for())
    assert calls == [] and out["flags"] == ["research_no_candidates"]
    assert out["step_count"] == 1


# ------------------------------------------------------- read-only enforcement


def test_order_tools_are_rejected():
    order = StructuredTool.from_function(
        func=lambda: {"ok": True}, name="place_paper_order", description="x"
    )
    with pytest.raises(ValueError, match="read-only"):
        make_research_node(ScriptedChatModel(script=[ai_text()]), [*make_tools([]), order])


def test_unknown_tools_are_rejected():
    odd = StructuredTool.from_function(
        func=lambda: {"ok": True}, name="delete_all", description="x"
    )
    with pytest.raises(ValueError, match="allowlist"):
        make_research_node(ScriptedChatModel(script=[ai_text()]), [*make_tools([]), odd])


def test_bad_budgets_rejected():
    with pytest.raises(ValueError):
        make_research_node(ScriptedChatModel(script=[ai_text()]), make_tools([]), max_model_calls=0)
