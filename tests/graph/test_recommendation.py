import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.graph.recommendation import (
    CONFIDENCE_PENALTIES,
    DISCLAIMER,
    Action,
    Draft,
    Recommendation,
    RecommendationError,
    allowed_actions,
    build_recommendation,
    compute_confidence,
    confidence_label,
    numbers_in,
    resolve_action,
)
from src.graph.state import evidence_from_tool_result, evidence_id
from src.risk.assess import assess

NOW = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)  # the day after the last candle below


# ------------------------------------------------------------------ fixtures


def _prices(sigma: float, n: int = 300, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, sigma, n)
    index = pd.bdate_range(end="2026-10-07", periods=n)
    return pd.Series(100 * np.cumprod(1 + rets), index=index)


def _assess(prices: pd.Series, symbol: str = "TEST"):
    return assess(
        symbol,
        prices,
        profile="conservative",
        portfolio_value=100_000,
        budget=5_000,
        max_trade_inr=5_000,
    )


@pytest.fixture(scope="module")
def calm():
    report = _assess(_prices(0.006))
    assert report.cleared, f"fixture assumption broken: {report.blocking_reasons}"
    assert report.sizing.max_amount_inr > 0
    return report


@pytest.fixture(scope="module")
def risky():
    report = _assess(_prices(0.03), "RISKY")
    assert any(r.status.value == "fail" for r in report.rules), "fixture assumption broken"
    return report


@pytest.fixture(scope="module")
def sparse():
    report = _assess(_prices(0.006).tail(30), "SPARSE")
    assert any(r.status.value == "unavailable" for r in report.rules), "fixture assumption broken"
    return report


def _ev(tool, args, result):
    item = evidence_from_tool_result(tool, args, result)
    return item.id, item.model_dump(mode="json")


def make_evidence(symbol="TEST", *, fundamentals=True, news=True, quote=True):
    ev = {}
    if quote:
        k, v = _ev(
            "get_quote",
            {"symbol": symbol},
            {"ok": True, "price": 123.456, "source": "stub", "as_of": "2026-10-07T10:00:00+00:00"},
        )
        ev[k] = v
    if fundamentals:
        k, v = _ev("get_fundamentals", {"symbol": symbol}, {"ok": True, "pe": 20.0, "source": "s"})
        ev[k] = v
    if news:
        k, v = _ev("search_news", {"query": "Test Co", "limit": 5}, {"ok": True, "untrusted": True})
        ev[k] = v
    k, v = _ev("get_quote", {"symbol": "BROKEN"}, {"ok": False, "code": "UPSTREAM_ERROR"})
    ev[k] = v
    return ev


def draft(**kw):
    base = {
        "action": Action.CONSIDER_BUY,
        "rationale": "Steady price behaviour and decent fundamentals fit the profile.",
        "key_risks": ["Broad market weakness could hit all equities."],
        "source_ids": [evidence_id("get_fundamentals", {"symbol": "TEST"})],
    }
    return Draft(**{**base, **kw})


def build(d, report, evidence=None, **kw):
    return build_recommendation(
        d,
        symbol=kw.pop("symbol", "TEST"),
        report=report,
        evidence=make_evidence() if evidence is None else evidence,
        now=NOW,
        **kw,
    )


# ------------------------------------------------------------------ risk gate


def test_gate_cleared_allows_buy(calm):
    assert allowed_actions(calm) == {Action.CONSIDER_BUY, Action.WATCH, Action.AVOID}
    assert resolve_action(Action.CONSIDER_BUY, calm) == (Action.CONSIDER_BUY, False)


def test_gate_failed_rules_block_buy(risky):
    assert Action.CONSIDER_BUY not in allowed_actions(risky)
    assert resolve_action(Action.CONSIDER_BUY, risky) == (Action.AVOID, True)
    assert resolve_action(Action.WATCH, risky) == (Action.WATCH, False)


def test_gate_unavailable_metrics_mean_insufficient_data(sparse):
    assert Action.CONSIDER_BUY not in allowed_actions(sparse)
    assert resolve_action(Action.CONSIDER_BUY, sparse) == (Action.INSUFFICIENT_DATA, True)


def test_gate_without_a_report_is_insufficient_data():
    assert allowed_actions(None) == {Action.INSUFFICIENT_DATA}
    assert resolve_action(Action.AVOID, None) == (Action.INSUFFICIENT_DATA, True)


# ----------------------------------------------------------------- confidence


def test_confidence_no_report_is_zero():
    assert compute_confidence(None, make_evidence(), "TEST", [], NOW) == 0.0


def test_confidence_penalties_are_relative_to_a_full_baseline(calm):
    full = compute_confidence(calm, make_evidence(), "TEST", [], NOW)
    no_fund = compute_confidence(calm, make_evidence(fundamentals=False), "TEST", [], NOW)
    no_news = compute_confidence(calm, make_evidence(news=False), "TEST", [], NOW)
    p = CONFIDENCE_PENALTIES
    assert round(full - no_fund, 2) == p["no_fundamentals"]
    assert round(full - no_news, 2) == p["no_news"]


def test_confidence_stale_data_and_research_flags_lower_it(calm):
    base = compute_confidence(calm, make_evidence(), "TEST", [], NOW)
    later = datetime(2026, 11, 30, tzinfo=UTC)
    stale = compute_confidence(calm, make_evidence(), "TEST", [], later)
    flagged = compute_confidence(
        calm, make_evidence(), "TEST", ["research_agent_failed:RuntimeError"], NOW
    )
    assert round(base - stale, 2) == CONFIDENCE_PENALTIES["stale_data"]
    assert round(base - flagged, 2) == CONFIDENCE_PENALTIES["per_research_issue"]


def test_confidence_unavailable_items_are_capped_and_floored(sparse):
    score = compute_confidence(
        sparse,
        make_evidence(fundamentals=False, news=False),
        "TEST",
        ["research_agent_failed:X", "research_budget_hit", "research_no_evidence"],
        datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert 0.0 <= score < 0.5


def test_confidence_matches_symbol_with_exchange_suffix(calm):
    ev = make_evidence(symbol="TEST.NS")
    with_suffix = compute_confidence(calm, ev, "TEST", [], NOW)
    assert with_suffix == compute_confidence(calm, make_evidence(), "TEST", [], NOW)


def test_confidence_labels():
    assert confidence_label(0.9) == "high"
    assert confidence_label(0.75) == "high"
    assert confidence_label(0.6) == "medium"
    assert confidence_label(0.5) == "medium"
    assert confidence_label(0.49) == "low"


# ----------------------------------------------------------------- numbers_in


def test_numbers_in():
    assert numbers_in("no digits here") == []
    assert numbers_in("up 12.5% to INR 1,234.50 and 52 weeks") == ["12.5%", "1,234.50", "52"]


# -------------------------------------------------------------- happy path


def test_build_happy_path_fills_numbers_from_code(calm):
    rec = build(draft(), calm)
    assert rec.action is Action.CONSIDER_BUY
    assert rec.max_amount_inr_ceiling == calm.sizing.max_amount_inr
    assert rec.profile == calm.profile
    assert rec.disclaimer == DISCLAIMER
    assert rec.flags == []
    assert rec.confidence_label in {"medium", "high"}
    figures = {f.label: f for f in rec.key_figures}
    assert figures["Annualised volatility"].source == "risk_engine"
    assert figures["Latest price"].display == "INR 123.46"
    assert figures["Latest price"].source == evidence_id("get_quote", {"symbol": "TEST"})
    assert "Suggested maximum amount (a ceiling, not a target)" in figures
    assert rec.scenarios is not None


def test_build_is_json_safe(calm):
    rec = build(draft(), calm)
    dumped = rec.model_dump(mode="json")
    json.dumps(dumped)
    assert Recommendation.model_validate(dumped).action is Action.CONSIDER_BUY


def test_latest_price_figure_is_omitted_when_quote_failed(calm):
    rec = build(draft(), calm, evidence=make_evidence(quote=False))
    assert "Latest price" not in {f.label for f in rec.key_figures}


# ----------------------------------------------------------- risk gate in build


def test_risky_buy_is_overridden_and_has_no_ceiling(risky):
    rec = build(draft(), risky, symbol="RISKY")
    assert rec.action is Action.AVOID
    assert rec.max_amount_inr_ceiling is None
    assert "action_overridden_by_risk_gate:consider_buy->avoid" in rec.flags
    assert rec.blocking_reasons


def test_sparse_data_buy_becomes_insufficient_data(sparse):
    rec = build(draft(), sparse, symbol="SPARSE")
    assert rec.action is Action.INSUFFICIENT_DATA
    assert rec.max_amount_inr_ceiling is None
    assert rec.confidence < 1.0


def test_watch_from_llm_is_kept_without_ceiling(calm):
    rec = build(draft(action=Action.WATCH), calm)
    assert rec.action is Action.WATCH and rec.max_amount_inr_ceiling is None
    assert rec.flags == []


# ---------------------------------------------------------- rejecting bad drafts


def test_missing_required_fields_are_all_reported(calm):
    bad = Draft(action=Action.WATCH, rationale="  ", key_risks=[" ", ""], source_ids=[])
    with pytest.raises(RecommendationError) as exc:
        build(bad, calm)
    text = " | ".join(exc.value.problems)
    assert "rationale is empty" in text
    assert "key_risks" in text
    assert "source_ids" in text


def test_unknown_source_id_is_rejected(calm):
    with pytest.raises(RecommendationError, match="does not exist"):
        build(draft(source_ids=["get_fundamentals:symbol=MADE_UP"]), calm)


def test_failed_tool_call_cannot_be_cited(calm):
    failed_id = evidence_id("get_quote", {"symbol": "BROKEN"})
    with pytest.raises(RecommendationError, match="failed tool call"):
        build(draft(source_ids=[failed_id]), calm)


def test_missing_report_is_rejected():
    with pytest.raises(RecommendationError, match="no risk report"):
        build(draft(), None)


def test_overlong_text_is_rejected(calm):
    with pytest.raises(RecommendationError, match="longer than"):
        build(draft(rationale="x" * 5000), calm)
    with pytest.raises(RecommendationError, match="more than"):
        build(draft(key_risks=["r"] * 20), calm)


def test_duplicate_sources_are_collapsed(calm):
    sid = evidence_id("get_fundamentals", {"symbol": "TEST"})
    assert build(draft(source_ids=[sid, sid, f" {sid} "]), calm).source_ids == [sid]


def test_numbers_in_prose_are_flagged_not_rejected(calm):
    rec = build(draft(rationale="Volatility is about 12% which looks fine."), calm)
    assert "prose_contains_numbers" in rec.flags
    assert rec.action is Action.CONSIDER_BUY


# --------------------------------------------------- model-level safety invariants


def test_model_refuses_buy_when_blocked(risky):
    good = build(draft(action=Action.AVOID), risky, symbol="RISKY").model_dump()
    with pytest.raises(ValidationError, match="consider_buy"):
        Recommendation(**{**good, "action": Action.CONSIDER_BUY, "max_amount_inr_ceiling": 100.0})


def test_model_refuses_buy_without_ceiling(calm):
    good = build(draft(), calm).model_dump()
    with pytest.raises(ValidationError, match="ceiling"):
        Recommendation(**{**good, "max_amount_inr_ceiling": None})


def test_model_refuses_ceiling_on_non_buy(calm):
    good = build(draft(action=Action.WATCH), calm).model_dump()
    with pytest.raises(ValidationError, match="only allowed"):
        Recommendation(**{**good, "max_amount_inr_ceiling": 100.0})


def test_model_refuses_empty_required_lists(calm):
    good = build(draft(), calm).model_dump()
    for field in ("key_risks", "source_ids"):
        with pytest.raises(ValidationError):
            Recommendation(**{**good, field: []})
    with pytest.raises(ValidationError):
        Recommendation(**{**good, "rationale": ""})
