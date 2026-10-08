"""Adapter: MCP tool outputs -> inputs for assess().

Pure conversion, no network. A failed tool call (ToolError) becomes an
'unavailable' note and never raises, so the proposal is blocked, not crashed.
"""

from __future__ import annotations

import pandas as pd

from src.mcp_server.models import History, Portfolio, ToolError
from src.risk.assess import RiskReport, assess
from src.risk.metrics import InsufficientDataError
from src.risk.rules import ProfileLimits, ProfileName

IST = "Asia/Kolkata"
DAILY_INTERVALS = {"1d", "d", "daily"}


def _norm(symbol: str) -> str:
    return symbol.strip().upper().removesuffix(".NS")


def _empty() -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def history_to_series(history: History) -> pd.Series:
    """Daily closes as a float Series on a tz-naive, sorted, de-duplicated date index."""
    if history.interval.strip().lower() not in DAILY_INTERVALS:
        raise InsufficientDataError(f"daily candles required, got interval {history.interval!r}")
    if not history.candles:
        return _empty()

    idx = pd.DatetimeIndex([c.date for c in history.candles])
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    idx = idx.normalize()

    s = pd.Series([c.close for c in history.candles], index=idx, dtype=float, name=history.symbol)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _series_or_empty(obj: object, key: str, notes: dict[str, str]) -> pd.Series:
    if isinstance(obj, History):
        try:
            s = history_to_series(obj)
        except InsufficientDataError as exc:
            notes[key] = str(exc)
            return _empty()
        if s.empty:
            notes[key] = "history has no candles"
        return s
    if isinstance(obj, ToolError):
        notes[key] = f"{obj.code}: {obj.message}"
    elif obj is None:
        notes[key] = "no history supplied"
    else:
        notes[key] = f"unsupported history type {type(obj).__name__}"
    return _empty()


def position_value(portfolio: Portfolio | ToolError | None, symbol: str) -> float | None:
    """Market value held in `symbol`: 0.0 if not held, None if unknown."""
    if not isinstance(portfolio, Portfolio):
        return None
    for h in portfolio.holdings:
        if _norm(h.symbol) == _norm(symbol):
            return h.market_value  # None when the holding is unpriced
    return 0.0


def assess_from_tools(
    symbol: str,
    history: History | ToolError | None,
    *,
    profile: ProfileName | str,
    portfolio: Portfolio | ToolError | None,
    max_trade_inr: float,
    budget: float | None = None,
    daily_cap_inr: float | None = None,
    daily_spent_inr: float | None = None,
    benchmark_history: History | ToolError | None = None,
    holdings_history: dict[str, History | ToolError] | None = None,
    proposed_amount_inr: float | None = None,
    as_of: str | pd.Timestamp | None = None,
    horizon_days: int = 63,
    seed: int = 42,
    limits: ProfileLimits | None = None,
) -> RiskReport:
    notes: dict[str, str] = {}

    prices = _series_or_empty(history, "prices", notes)
    benchmark = (
        None
        if benchmark_history is None
        else _series_or_empty(benchmark_history, "benchmark", notes)
    )

    portfolio_value = None
    cash = None
    current_value = None
    held: list[str] = []
    if isinstance(portfolio, Portfolio):
        cash = portfolio.cash
        portfolio_value = portfolio.total_equity if portfolio.valuation_complete else None
        if portfolio_value is None:
            notes["portfolio"] = "portfolio valuation is incomplete (a price is missing)"
        current_value = position_value(portfolio, symbol)
        held = [h.symbol for h in portfolio.holdings if _norm(h.symbol) != _norm(symbol)]
    elif isinstance(portfolio, ToolError):
        notes["portfolio"] = f"{portfolio.code}: {portfolio.message}"
    else:
        notes["portfolio"] = "no portfolio supplied"

    # Budget can never exceed available cash.
    if cash is None:
        effective_budget = None
    elif budget is None:
        effective_budget = cash
    else:
        effective_budget = min(budget, cash)

    daily_remaining = None
    if daily_cap_inr is not None and daily_spent_inr is not None:
        daily_remaining = max(0.0, daily_cap_inr - daily_spent_inr)

    # Every held symbol gets a correlation check; no history means 'could not be checked'.
    supplied = {_norm(k): v for k, v in (holdings_history or {}).items()}
    holdings_prices = {
        sym: _series_or_empty(supplied.get(_norm(sym)), f"holding:{sym}", notes) for sym in held
    }

    report = assess(
        symbol,
        prices,
        profile=profile,
        portfolio_value=portfolio_value,
        budget=effective_budget,
        max_trade_inr=max_trade_inr,
        current_position_value=current_value,
        benchmark_prices=benchmark,
        holdings_prices=holdings_prices,
        proposed_amount_inr=proposed_amount_inr,
        daily_remaining_inr=daily_remaining,
        as_of=as_of,
        horizon_days=horizon_days,
        seed=seed,
        limits=limits,
    )
    return report.model_copy(update={"unavailable": {**report.unavailable, **notes}})
