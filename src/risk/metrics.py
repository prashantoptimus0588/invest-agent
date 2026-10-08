"""Core risk metrics.

Pure functions: no I/O, no LLM, no globals. Every number the agent quotes about
risk must come from here (or from tool evidence), never from a prompt.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_OBS = 30  # minimum overlapping observations for VaR, beta, correlation


class InsufficientDataError(ValueError):
    """A metric cannot be computed honestly from the data supplied."""


# ---------- helpers ----------


def truncate_to(series: pd.Series, as_of: str | pd.Timestamp) -> pd.Series:
    """Keep only data up to and including `as_of` (prevents look-ahead leakage)."""
    ts = pd.Timestamp(as_of)
    idx = series.index
    if getattr(idx, "tz", None) is not None and ts.tzinfo is None:
        ts = ts.tz_localize(idx.tz)
    elif getattr(idx, "tz", None) is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return series.loc[idx <= ts]


def _check_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if len(prices) < 2:
        raise InsufficientDataError("need at least 2 prices")
    if prices.isna().any():
        raise InsufficientDataError("prices contain NaN; clean the data first")
    if (prices <= 0).any():
        raise InsufficientDataError("prices must be strictly positive")
    if not prices.index.is_monotonic_increasing:
        raise InsufficientDataError("price index must be sorted oldest to newest")
    return prices.astype(float)


def _check_returns(returns: pd.Series, min_obs: int) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    r = returns.dropna()
    if len(r) < min_obs:
        raise InsufficientDataError(f"need at least {min_obs} returns, got {len(r)}")
    return r.astype(float)


# ---------- metrics ----------


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns: p_t / p_{t-1} - 1."""
    p = _check_prices(prices)
    return p.pct_change().dropna()


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Sample std of daily returns x sqrt(periods_per_year)."""
    r = _check_returns(returns, min_obs=2)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough fall as a negative fraction (-0.35 = -35%)."""
    p = _check_prices(prices)
    drawdown = p / p.cummax() - 1.0
    return float(drawdown.min())


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical one-day VaR as a positive loss fraction.

    0.03 at 95% means: on the worst 5% of days, the loss was about 3% or more.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    r = _check_returns(returns, min_obs=MIN_OBS)
    q = float(np.quantile(r, 1.0 - confidence))
    return max(0.0, -q)


def beta(asset_returns: pd.Series, market_returns: pd.Series, min_obs: int = MIN_OBS) -> float:
    """Beta = cov(asset, market) / var(market), on dates both series share."""
    aligned = pd.concat([asset_returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < min_obs:
        raise InsufficientDataError(
            f"need at least {min_obs} overlapping returns, got {len(aligned)}"
        )
    var_m = aligned.iloc[:, 1].var(ddof=1)
    if var_m == 0 or np.isnan(var_m):
        raise InsufficientDataError("market returns have zero variance")
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / var_m)


def correlation_with_holdings(
    candidate_returns: pd.Series,
    holdings_returns: Mapping[str, pd.Series],
    min_obs: int = MIN_OBS,
) -> dict[str, float | None]:
    """Correlation of the candidate with each current holding.

    A holding with too little overlap or zero variance maps to None
    (unknown), never to 0.
    """
    out: dict[str, float | None] = {}
    for symbol, other in holdings_returns.items():
        aligned = pd.concat([candidate_returns, other], axis=1, join="inner").dropna()
        if len(aligned) < min_obs:
            out[symbol] = None
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        out[symbol] = None if np.isnan(corr) else float(corr)
    return out
