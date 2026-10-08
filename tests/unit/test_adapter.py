from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.mcp_server.models import Candle, History, Holding, Portfolio, ToolError
from src.risk.adapter import assess_from_tools, history_to_series, position_value
from src.risk.metrics import InsufficientDataError

NOW = datetime(2026, 10, 7, tzinfo=UTC)


def make_history(symbol="AAA.NS", n=300, mu=0.0005, sigma=0.004, seed=1, interval="1d"):
    rets = np.random.default_rng(seed).normal(mu, sigma, n)
    vals = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    dates = pd.bdate_range("2023-01-02", periods=len(vals))
    candles = [
        Candle(date=d.to_pydatetime(), open=v, high=v, low=v, close=float(v), volume=1000)
        for d, v in zip(dates, vals)
    ]
    return History(symbol=symbol, interval=interval, candles=candles, as_of=NOW, source="test")


def holding(symbol, value, qty=10):
    return Holding(
        symbol=symbol,
        quantity=qty,
        # Safely handle None for avg_cost
        avg_cost=None if value is None else value / qty,
        last_price=None if value is None else value / qty,
        market_value=value,
    )


def make_portfolio(cash=100_000.0, holdings=None, equity=None, complete=True):
    holdings = holdings or []
    mv = sum(h.market_value or 0.0 for h in holdings)
    return Portfolio(
        cash=cash,
        holdings=holdings,
        cost_basis=mv,
        market_value=mv,
        total_equity=cash + mv if equity is None else equity,
        valuation_complete=complete,
        as_of=NOW,
    )


def run(history=None, portfolio=None, **overrides):
    kwargs = {
        "profile": "conservative",
        "portfolio": make_portfolio() if portfolio is None else portfolio,
        "max_trade_inr": 20_000.0,
        "benchmark_history": make_history("^NSEI", seed=2, sigma=0.005),
    }
    kwargs.update(overrides)
    return assess_from_tools("AAA.NS", make_history() if history is None else history, **kwargs)


# ---------- history_to_series ----------


def test_series_is_sorted_deduped_float_and_naive():
    h = make_history(n=10)
    shuffled = list(reversed(h.candles))
    dup = shuffled[0].model_copy(update={"close": 999.0})
    h2 = h.model_copy(update={"candles": [shuffled[0], *shuffled, dup]})
    s = history_to_series(h2)
    assert s.index.is_monotonic_increasing
    assert s.index.is_unique
    assert s.index.tz is None
    assert s.dtype == float
    assert len(s) == 11
    assert s.iloc[-1] == 999.0  # last duplicate wins


def test_tz_aware_candles_become_ist_dates():
    c = Candle(date=datetime(2024, 1, 1, 18, 30, tzinfo=UTC), open=1, high=1, low=1, close=1)
    h = History(symbol="X.NS", interval="1d", candles=[c], as_of=NOW, source="t")
    assert history_to_series(h).index[0] == pd.Timestamp("2024-01-02")


def test_non_daily_interval_raises():
    with pytest.raises(InsufficientDataError):
        history_to_series(make_history(interval="1h"))


def test_empty_history_gives_empty_series():
    h = make_history().model_copy(update={"candles": []})
    assert history_to_series(h).empty


# ---------- position_value ----------


def test_position_value_cases():
    p = make_portfolio(holdings=[holding("AAA.NS", 5_000.0), holding("BBB.NS", None)])
    assert position_value(p, "AAA") == 5_000.0  # .NS stripped on both sides
    assert position_value(p, "AAA.NS") == 5_000.0
    assert position_value(p, "BBB.NS") is None  # held but unpriced
    assert position_value(p, "ZZZ.NS") == 0.0  # not held
    assert position_value(ToolError(code="TIMEOUT", message="x"), "AAA") is None
    assert position_value(None, "AAA") is None


# ---------- assess_from_tools ----------


def test_happy_path_is_cleared():
    r = run()
    assert r.cleared is True
    assert r.sizing.max_amount_inr == pytest.approx(10_000.0)
    assert r.metrics.beta is not None


def test_budget_defaults_to_cash_and_is_capped_by_it():
    r = run(portfolio=make_portfolio(cash=3_000.0, equity=100_000.0))
    assert r.sizing.max_amount_inr == pytest.approx(3_000.0)
    assert r.sizing.binding_constraint == "budget"
    r2 = run(portfolio=make_portfolio(cash=3_000.0, equity=100_000.0), budget=50_000.0)
    assert r2.sizing.max_amount_inr == pytest.approx(3_000.0)


def test_incomplete_valuation_blocks():
    r = run(portfolio=make_portfolio(complete=False))
    assert r.cleared is False
    assert r.sizing.binding_constraint == "portfolio_value_unavailable"
    assert "incomplete" in r.unavailable["portfolio"]


def test_portfolio_tool_error_blocks_with_reason():
    r = run(portfolio=ToolError(code="UPSTREAM_ERROR", message="db down"))
    assert r.cleared is False
    assert r.unavailable["portfolio"] == "UPSTREAM_ERROR: db down"


def test_history_tool_error_blocks_with_reason():
    r = run(history=ToolError(code="NO_DATA", message="no candles for symbol"))
    assert r.cleared is False
    assert r.unavailable["prices"] == "NO_DATA: no candles for symbol"


def test_benchmark_tool_error_only_makes_beta_unavailable():
    r = run(benchmark_history=ToolError(code="TIMEOUT", message="slow", retryable=True))
    assert r.metrics.beta is None
    assert r.unavailable["benchmark"] == "TIMEOUT: slow"
    assert r.cleared is True


def test_existing_position_in_candidate_reduces_room_and_is_not_self_correlated():
    p = make_portfolio(cash=95_000.0, holdings=[holding("AAA.NS", 5_000.0)], equity=100_000.0)
    r = run(portfolio=p)
    assert r.sizing.max_amount_inr == pytest.approx(5_000.0)
    assert r.metrics.correlations == {}


def test_held_symbols_are_correlation_checked_and_missing_history_is_unknown():
    p = make_portfolio(
        cash=80_000.0,
        holdings=[holding("BBB.NS", 10_000.0), holding("CCC.NS", 10_000.0)],
        equity=100_000.0,
    )
    r = run(portfolio=p, holdings_history={"BBB.NS": make_history("BBB.NS", seed=3)})
    assert r.metrics.correlations["BBB.NS"] is not None
    assert r.metrics.correlations["CCC.NS"] is None
    assert r.unavailable["holding:CCC.NS"] == "no history supplied"


def test_daily_remaining_applies_only_when_both_values_given():
    r = run(daily_cap_inr=15_000.0, daily_spent_inr=14_000.0)
    assert r.sizing.max_amount_inr == pytest.approx(1_000.0)
    assert r.sizing.binding_constraint == "daily_remaining"
    r2 = run(daily_cap_inr=15_000.0, daily_spent_inr=None)
    assert r2.sizing.caps["daily_remaining"] is None


def test_proposed_amount_flows_through():
    r = run(proposed_amount_inr=15_000.0)
    assert r.proposed_within_ceiling is False
    assert r.cleared is False
