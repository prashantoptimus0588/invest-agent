"""Technical indicators written with pandas.

Everything here is causal: the value at date t depends only on prices up to t,
so these functions are safe to reuse in backtests. Flags describe a state in
plain language; they never recommend an action.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.risk.metrics import _check_prices

SMA_SHORT = 50
SMA_LONG = 200
RSI_PERIOD = 14
MOMENTUM_LOOKBACK = 63  # ~3 months of trading days
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0


class IndicatorFlag(BaseModel):
    code: str
    message: str


class IndicatorReport(BaseModel):
    as_of: datetime
    last_price: float
    sma_50: float | None
    sma_200: float | None
    rsi_14: float | None
    momentum_63d: float | None  # fractional return over the lookback, 0.2 = +20%
    flags: list[IndicatorFlag] = Field(default_factory=list)


# ---------- series functions ----------


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average; NaN until `window` prices exist."""
    p = _check_prices(prices)
    return p.rolling(window=window, min_periods=window).mean()


def rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI. 100 if there were only gains; NaN if the price never moved."""
    p = _check_prices(prices)
    delta = p.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    no_losses = np.where(avg_gain > 0, 100.0, np.nan)
    return out.where(avg_loss != 0.0, other=no_losses)


def momentum(prices: pd.Series, lookback: int = MOMENTUM_LOOKBACK) -> float | None:
    """Simple return over the last `lookback` trading days; None if too short."""
    p = _check_prices(prices)
    if len(p) <= lookback:
        return None
    return float(p.iloc[-1] / p.iloc[-1 - lookback] - 1.0)


def _last(series: pd.Series) -> float | None:
    v = series.iloc[-1]
    return None if pd.isna(v) else float(v)


# ---------- report ----------


def compute_indicators(prices: pd.Series) -> IndicatorReport:
    """Latest indicator values plus plain-language flags."""
    p = _check_prices(prices)
    last_price = float(p.iloc[-1])
    sma_50 = _last(sma(p, SMA_SHORT))
    sma_200 = _last(sma(p, SMA_LONG))
    rsi_14 = _last(rsi(p))
    mom = momentum(p)

    flags: list[IndicatorFlag] = []

    def add(code: str, message: str) -> None:
        flags.append(IndicatorFlag(code=code, message=message))

    if sma_50 is None:
        add("insufficient_history_50", "Fewer than 50 days of history: 50-day average unavailable")
    if sma_200 is None:
        add(
            "insufficient_history_200",
            "Fewer than 200 days of history: 200-day average unavailable",
        )
    if mom is None:
        add("insufficient_history_momentum", "Not enough history to measure 3-month momentum")

    if sma_200 is not None:
        if last_price < sma_200:
            add(
                "price_below_sma200",
                "Price is below its 200-day average (long-term downtrend signal)",
            )
        elif last_price > sma_200:
            add(
                "price_above_sma200",
                "Price is above its 200-day average (long-term uptrend signal)",
            )

    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            add(
                "sma50_above_sma200",
                "50-day average is above the 200-day average (medium-term strength)",
            )
        elif sma_50 < sma_200:
            add(
                "sma50_below_sma200",
                "50-day average is below the 200-day average (medium-term weakness)",
            )

    if rsi_14 is not None:
        if rsi_14 >= RSI_OVERBOUGHT:
            add(
                "rsi_overbought_zone",
                "RSI is above 70: recent gains have been strong (often called overbought)",
            )
        elif rsi_14 <= RSI_OVERSOLD:
            add(
                "rsi_oversold_zone",
                "RSI is below 30: recent losses have been heavy (often called oversold)",
            )

    if mom is not None:
        if mom > 0:
            add("momentum_positive", "Price is higher than it was 3 months ago")
        elif mom < 0:
            add("momentum_negative", "Price is lower than it was 3 months ago")

    return IndicatorReport(
        as_of=p.index[-1].to_pydatetime(),
        last_price=last_price,
        sma_50=sma_50,
        sma_200=sma_200,
        rsi_14=rsi_14,
        momentum_63d=mom,
        flags=flags,
    )
