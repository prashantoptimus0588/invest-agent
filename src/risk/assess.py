"""assess(): one typed risk report for a candidate asset.

Runs metrics, indicators, sizing, profile rules and scenarios. Bad or missing
data never raises and never passes: it becomes None plus a reason, and the
rules show 'unavailable', which blocks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.risk.indicators import IndicatorReport, compute_indicators
from src.risk.metrics import (
    InsufficientDataError,
    _check_prices,
    annualised_volatility,
    beta,
    correlation_with_holdings,
    daily_returns,
    historical_var,
    max_drawdown,
    truncate_to,
)
from src.risk.rules import (
    ProfileLimits,
    ProfileName,
    RuleResult,
    blocking,
    evaluate_rules,
    position_weight,
)
from src.risk.scenarios import ScenarioRange, simulate_scenarios
from src.risk.sizing import SizingResult, suggest_max_amount

MIN_RULE_OBS = 60  # daily returns needed before volatility / drawdown are trusted


class MetricsBlock(BaseModel):
    volatility: float | None
    max_drawdown: float | None
    var_95: float | None
    beta: float | None
    correlations: dict[str, float | None] = Field(default_factory=dict)


class RiskReport(BaseModel):
    symbol: str
    profile: ProfileName
    as_of: datetime | None
    metrics: MetricsBlock
    indicators: IndicatorReport | None
    rules: list[RuleResult]
    sizing: SizingResult
    proposed_amount_inr: float | None
    proposed_within_ceiling: bool | None
    scenarios: ScenarioRange | None
    cleared: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    unavailable: dict[str, str] = Field(default_factory=dict)


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _need(returns: pd.Series, n: int) -> None:
    if len(returns) < n:
        raise InsufficientDataError(
            f"need at least {n} daily returns for a reliable estimate, got {len(returns)}"
        )


def assess(
    symbol: str,
    prices: pd.Series,
    *,
    profile: ProfileName | str,
    portfolio_value: float | None,
    budget: float | None,
    max_trade_inr: float | None,
    current_position_value: float | None = 0.0,
    benchmark_prices: pd.Series | None = None,
    holdings_prices: Mapping[str, pd.Series] | None = None,
    proposed_amount_inr: float | None = None,
    daily_remaining_inr: float | None = None,
    as_of: str | pd.Timestamp | None = None,
    horizon_days: int = 63,
    seed: int = 42,
    limits: ProfileLimits | None = None,
) -> RiskReport:
    name = ProfileName(profile)  # unknown profile raises ValueError
    if proposed_amount_inr is not None and (
        not math.isfinite(proposed_amount_inr) or proposed_amount_inr <= 0
    ):
        raise ValueError("proposed_amount_inr must be a finite number above 0")

    unavailable: dict[str, str] = {}

    def attempt(key: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except InsufficientDataError as exc:
            unavailable[key] = str(exc)
            return None

    def cut(s: pd.Series) -> pd.Series:
        return s if as_of is None else truncate_to(s, as_of)

    # ----- candidate history -----
    px = attempt("prices", lambda: _check_prices(cut(prices)))
    rets = daily_returns(px) if px is not None else None

    def on_rets(key: str, fn: Callable[[pd.Series], Any]) -> Any:
        if rets is None:
            unavailable[key] = "price history unusable"
            return None
        return attempt(key, lambda: fn(rets))

    def vol_fn(r: pd.Series) -> float:
        _need(r, MIN_RULE_OBS)
        return annualised_volatility(r)

    def mdd_fn(r: pd.Series) -> float:
        _need(r, MIN_RULE_OBS)
        return max_drawdown(px)

    volatility = on_rets("volatility", vol_fn)
    drawdown = on_rets("max_drawdown", mdd_fn)
    var_95 = on_rets("var_95", lambda r: historical_var(r, 0.95))

    # ----- beta vs benchmark -----
    beta_value = None
    if benchmark_prices is None:
        unavailable["beta"] = "no benchmark supplied"
    else:
        bpx = attempt("benchmark", lambda: _check_prices(cut(benchmark_prices)))
        if bpx is None:
            unavailable["beta"] = "benchmark history unusable"
        else:
            bench_rets = daily_returns(bpx)
            beta_value = on_rets("beta", lambda r: beta(r, bench_rets))

    # ----- correlation with existing holdings (candidate excluded) -----
    correlations: dict[str, float | None] = {}
    holdings_rets: dict[str, pd.Series] = {}
    for sym, hp in (holdings_prices or {}).items():
        if sym == symbol:
            continue
        hpx = attempt(f"holding:{sym}", lambda hp=hp: _check_prices(cut(hp)))
        if hpx is None:
            correlations[sym] = None
        else:
            holdings_rets[sym] = daily_returns(hpx)
    if holdings_rets:
        if rets is None:
            correlations.update({s: None for s in holdings_rets})
        else:
            correlations.update(correlation_with_holdings(rets, holdings_rets))

    # ----- indicators -----
    indicators = attempt("indicators", lambda: compute_indicators(px)) if px is not None else None

    # ----- sizing -----
    last_price = float(px.iloc[-1]) if px is not None else None
    sizing = suggest_max_amount(
        name,
        portfolio_value=portfolio_value,
        budget=budget,
        max_trade_inr=max_trade_inr,
        current_position_value=current_position_value,
        price=last_price,
        daily_remaining_inr=daily_remaining_inr,
        limits=limits,
    )
    effective = proposed_amount_inr if proposed_amount_inr is not None else sizing.max_amount_inr
    within = (
        None if proposed_amount_inr is None else proposed_amount_inr <= sizing.max_amount_inr + 1e-9
    )

    # ----- rules -----
    position_pct = None
    if _finite(current_position_value) and _finite(portfolio_value):
        position_pct = position_weight(current_position_value + effective, portfolio_value)
    rules = evaluate_rules(
        name,
        volatility=volatility,
        max_drawdown=drawdown,
        position_pct=position_pct,
        correlations=correlations,
        limits=limits,
    )

    # ----- scenarios -----
    amount_for_scenarios = effective if effective > 0 else None
    scenarios = on_rets(
        "scenarios",
        lambda r: simulate_scenarios(
            r, horizon_days=horizon_days, seed=seed, amount_inr=amount_for_scenarios
        ),
    )

    # ----- verdict -----
    reasons = [f"{r.rule}: {r.reason}" for r in blocking(rules)]
    if sizing.max_amount_inr <= 0:
        reasons.append(f"sizing: {sizing.reason}")
    if within is False:
        reasons.append(
            f"sizing: proposed INR {proposed_amount_inr:,.2f} exceeds the ceiling of "
            f"INR {sizing.max_amount_inr:,.2f} (limited by {sizing.binding_constraint})"
        )

    return RiskReport(
        symbol=symbol,
        profile=name,
        as_of=px.index[-1].to_pydatetime() if px is not None else None,
        metrics=MetricsBlock(
            volatility=volatility,
            max_drawdown=drawdown,
            var_95=var_95,
            beta=beta_value,
            correlations=correlations,
        ),
        indicators=indicators,
        rules=rules,
        sizing=sizing,
        proposed_amount_inr=proposed_amount_inr,
        proposed_within_ceiling=within,
        scenarios=scenarios,
        cleared=not reasons,
        blocking_reasons=reasons,
        unavailable=unavailable,
    )
