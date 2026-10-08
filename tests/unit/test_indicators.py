import numpy as np
import pandas as pd
import pytest

from src.risk.indicators import compute_indicators, momentum, rsi, sma


def series(values):
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)))


def codes(report):
    return {f.code for f in report.flags}


def test_sma_known_values():
    rep = compute_indicators(series(np.arange(1, 201)))
    assert rep.sma_50 == pytest.approx(175.5)  # mean of 151..200
    assert rep.sma_200 == pytest.approx(100.5)  # mean of 1..200


def test_rising_series_flags():
    rep = compute_indicators(series(np.arange(1, 251)))
    assert rep.rsi_14 == pytest.approx(100.0)
    assert {
        "price_above_sma200",
        "sma50_above_sma200",
        "rsi_overbought_zone",
        "momentum_positive",
    } <= codes(rep)


def test_falling_series_flags():
    rep = compute_indicators(series(np.arange(250, 0, -1)))
    assert rep.rsi_14 == pytest.approx(0.0)
    assert {
        "price_below_sma200",
        "sma50_below_sma200",
        "rsi_oversold_zone",
        "momentum_negative",
    } <= codes(rep)


def test_flat_series_rsi_is_none_not_fifty():
    rep = compute_indicators(series(np.full(250, 100.0)))
    assert rep.rsi_14 is None
    assert rep.sma_200 == pytest.approx(100.0)
    assert "rsi_overbought_zone" not in codes(rep)
    assert "rsi_oversold_zone" not in codes(rep)


def test_alternating_series_rsi_near_fifty():
    vals = [100 + (i % 2) for i in range(100)]
    assert rsi(series(vals)).iloc[-1] == pytest.approx(50.0, abs=5)


def test_momentum_known_value():
    vals = np.full(100, 100.0)
    vals[-1] = 120.0
    assert momentum(series(vals), lookback=63) == pytest.approx(0.2)


def test_short_history_gives_none_and_flags_not_zero():
    rep = compute_indicators(series(np.linspace(100, 110, 30)))
    assert rep.sma_50 is None
    assert rep.sma_200 is None
    assert rep.momentum_63d is None
    assert {
        "insufficient_history_50",
        "insufficient_history_200",
        "insufficient_history_momentum",
    } <= codes(rep)


def test_indicators_are_causal_no_lookahead():
    rng = np.random.default_rng(3)
    full = series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    prefix = full.iloc[:220]
    assert sma(full, 50).iloc[219] == pytest.approx(sma(prefix, 50).iloc[-1])
    assert rsi(full).iloc[219] == pytest.approx(rsi(prefix).iloc[-1])


def test_report_carries_as_of_and_last_price():
    s = series(np.arange(1, 251))
    rep = compute_indicators(s)
    assert rep.as_of == s.index[-1].to_pydatetime()
    assert rep.last_price == 250.0
