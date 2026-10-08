import json

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.graph.state import (
    AgentState,
    EvidenceItem,
    UserProfile,
    evidence_from_tool_result,
    evidence_id,
    evidence_items,
    initial_state,
    merge_dicts,
    merge_nested,
    profile_from_state,
)
from src.risk.rules import ProfileName

# ------------------------------------------------------------------ reducers


def test_merge_dicts_right_wins_and_handles_none():
    assert merge_dicts({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}
    assert merge_dicts(None, {"a": 1}) == {"a": 1}
    assert merge_dicts({"a": 1}, None) == {"a": 1}


def test_merge_nested_keeps_both_kinds_for_same_symbol():
    left = {"RELIANCE.NS": {"technical": {"x": 1}}}
    right = {"RELIANCE.NS": {"risk": {"y": 2}}, "TCS.NS": {"risk": {"z": 3}}}
    out = merge_nested(left, right)
    assert out["RELIANCE.NS"] == {"technical": {"x": 1}, "risk": {"y": 2}}
    assert out["TCS.NS"] == {"risk": {"z": 3}}
    assert left == {"RELIANCE.NS": {"technical": {"x": 1}}}  # inputs not mutated


# --------------------------------------------------------------- UserProfile


def test_profile_all_missing_by_default():
    p = UserProfile()
    assert p.missing() == ["risk", "horizon_days", "budget_inr"]


def test_profile_partial_and_complete():
    p = UserProfile(risk="moderate", budget_inr=5000)
    assert p.risk is ProfileName.MODERATE
    assert p.missing() == ["horizon_days"]
    assert UserProfile(risk="aggressive", horizon_days=90, budget_inr=1000).missing() == []


@pytest.mark.parametrize(
    "bad",
    [
        {"risk": "yolo"},
        {"horizon_days": 0},
        {"budget_inr": -5},
        {"budget_inr": 0},
        {"budget_inr": float("nan")},
        {"budget_inr": float("inf")},
    ],
)
def test_profile_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        UserProfile(**bad)


# ------------------------------------------------------------------ evidence


def test_evidence_id_ignores_argument_order():
    a = evidence_id("get_history", {"symbol": "TCS", "period": "1y"})
    b = evidence_id("get_history", {"period": "1y", "symbol": "TCS"})
    assert a == b == "get_history:period=1y,symbol=TCS"
    assert evidence_id("get_portfolio", {}) == "get_portfolio"


def test_evidence_from_ok_result():
    result = {
        "ok": True,
        "symbol": "RELIANCE.NS",
        "price": 2500.5,
        "as_of": "2026-10-08T09:30:00+00:00",
        "source": "yfinance",
    }
    item = evidence_from_tool_result("get_quote", {"symbol": "RELIANCE"}, result)
    assert item.ok is True
    assert item.source == "yfinance"
    assert item.as_of is not None and item.as_of.tzinfo is not None
    assert item.error_code is None
    assert item.data["price"] == 2500.5
    assert item.untrusted is False


def test_evidence_from_error_result_is_kept_but_not_ok():
    result = {"ok": False, "code": "UPSTREAM_ERROR", "message": "boom", "retryable": True}
    item = evidence_from_tool_result("get_quote", {"symbol": "X"}, result)
    assert item.ok is False
    assert item.error_code == "UPSTREAM_ERROR"
    assert item.source is None and item.as_of is None


def test_missing_ok_key_is_not_ok():
    item = evidence_from_tool_result("get_quote", {"symbol": "X"}, {"price": 10})
    assert item.ok is False


def test_news_is_marked_untrusted():
    item = evidence_from_tool_result(
        "search_news", {"query": "Reliance"}, {"ok": True, "untrusted": True, "content": "..."}
    )
    assert item.untrusted is True


def test_bad_as_of_becomes_none_not_now():
    item = evidence_from_tool_result("get_quote", {}, {"ok": True, "as_of": "garbage"})
    assert item.as_of is None


# ------------------------------------------------------------- initial_state


def test_initial_state_defaults_and_json_safe():
    s = initial_state("Should I buy RELIANCE?", candidates=["RELIANCE", " RELIANCE ", "", "TCS"])
    assert s["candidates"] == ["RELIANCE", "TCS"]
    assert s["step_count"] == 0 and s["token_spend"] == 0
    assert s["evidence"] == {} and s["analyses"] == {}
    assert s["draft_recommendation"] is None and s["approval"] is None
    assert s["confidence"] is None
    json.dumps(s)  # must not raise


def test_initial_state_profile_roundtrip():
    s = initial_state("goal", user_profile=UserProfile(risk="conservative", budget_inr=2000))
    assert s["user_profile"]["risk"] == "conservative"
    assert profile_from_state(s).budget_inr == 2000


def test_initial_state_rejects_empty_goal_and_string_candidates():
    with pytest.raises(ValueError):
        initial_state("   ")
    with pytest.raises(TypeError):
        initial_state("goal", candidates="RELIANCE")


def test_evidence_items_parses_state():
    item = evidence_from_tool_result("get_quote", {"symbol": "X"}, {"ok": True, "source": "s"})
    s = initial_state("goal")
    s["evidence"] = {item.id: item.model_dump(mode="json")}
    parsed = evidence_items(s)
    assert len(parsed) == 1 and isinstance(parsed[0], EvidenceItem)


# ----------------------------------------- reducers inside a real graph run


@pytest.mark.asyncio
async def test_parallel_nodes_merge_through_reducers():
    def node_a(state):
        return {
            "evidence": {"get_quote:symbol=X": {"v": 1}},
            "analyses": {"X": {"technical": {"trend": "up"}}},
            "step_count": 1,
            "token_spend": 100,
            "flags": ["a_ran"],
        }

    def node_b(state):
        return {
            "evidence": {"get_history:symbol=X": {"v": 2}},
            "analyses": {"X": {"risk": {"cleared": True}}},
            "step_count": 1,
            "token_spend": 50,
            "flags": ["b_ran"],
        }

    g = StateGraph(AgentState)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)

    out = await g.compile().ainvoke(initial_state("goal", candidates=["X"]))

    assert set(out["evidence"]) == {"get_quote:symbol=X", "get_history:symbol=X"}
    assert set(out["analyses"]["X"]) == {"technical", "risk"}  # both kinds survived
    assert out["step_count"] == 2
    assert out["token_spend"] == 150
    assert sorted(out["flags"]) == ["a_ran", "b_ran"]
