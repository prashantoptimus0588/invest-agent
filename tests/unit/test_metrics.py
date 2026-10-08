import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import (
    InsufficientDataError,
    annualised_volatility,
    beta,
    correlation_with_holdings,
    daily_returns,
    historical_var,
    max_drawdown,
    truncate_to,
)


def prices_from_returns(returns, start=100.0, start_date="2024-01-01"):
    vals = start * np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns))])
    idx = pd.bdate_range(start_date, periods=len(vals))
    return pd.Series(vals, index=idx)


def test_constant_price_has_zero_volatility():
    prices = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=60))
    assert annualised_volatility(daily_returns(prices)) == 0.0


def test_volatility_scaling():
    r = pd.Series([0.01, -0.01])
    expected = 0.01 * np.sqrt(2) * np.sqrt(252)
    assert annualised_volatility(r) == pytest.approx(expected)


def test_monotonic_decline_drawdown():
    prices = pd.Series(np.linspace(100, 50, 50), index=pd.bdate_range("2024-01-01", periods=50))
    assert max_drawdown(prices) == pytest.approx(-0.5)


def test_rising_prices_have_zero_drawdown():
    prices = pd.Series(np.linspace(50, 100, 50), index=pd.bdate_range("2024-01-01", periods=50))
    assert max_drawdown(prices) == 0.0


def test_drawdown_uses_running_peak():
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.Series([100, 120, 90, 110, 100], index=idx, dtype=float)
    assert max_drawdown(prices) == pytest.approx(90 / 120 - 1)


def test_var_known_value():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))
    assert historical_var(r, 0.95) == pytest.approx(0.045)


def test_var_never_negative_when_all_returns_positive():
    assert historical_var(pd.Series(np.full(50, 0.01))) == 0.0


def test_var_too_few_observations():
    with pytest.raises(InsufficientDataError):
        historical_var(pd.Series([0.01, -0.02, 0.0]))


def test_beta_known_value_from_returns():
    rng = np.random.default_rng(42)
    market = pd.Series(rng.normal(0, 0.01, 250), index=pd.bdate_range("2024-01-01", periods=250))
    assert beta(2.0 * market, market) == pytest.approx(2.0)


def test_beta_known_value_from_prices():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 0.01, 250)
    mkt = daily_returns(prices_from_returns(m))
    asset = daily_returns(prices_from_returns(0.5 * m))
    assert beta(asset, mkt) == pytest.approx(0.5, abs=1e-9)


def test_beta_needs_overlap():
    a = pd.Series([0.01] * 40, index=pd.bdate_range("2024-01-01", periods=40))
    b = pd.Series([0.01] * 40, index=pd.bdate_range("2025-01-01", periods=40))
    with pytest.raises(InsufficientDataError):
        beta(a, b)


def test_correlation_with_holdings():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=100)
    base = pd.Series(rng.normal(0, 0.01, 100), index=idx)
    holdings = {
        "SAME": base,
        "OPPOSITE": -base,
        "FLAT": pd.Series(0.0, index=idx),
        "SHORT": base.iloc[:5],
    }

    # Catch the expected runtime warnings gracefully
    with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
        out = correlation_with_holdings(base, holdings)

    assert out["SAME"] == pytest.approx(1.0)
    assert out["OPPOSITE"] == pytest.approx(-1.0)
    assert out["FLAT"] is None
    assert out["SHORT"] is None


def test_bad_prices_rejected():
    idx = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(InsufficientDataError):
        daily_returns(pd.Series([100.0, np.nan, 101.0], index=idx))
    with pytest.raises(InsufficientDataError):
        daily_returns(pd.Series([100.0, 0.0, 101.0], index=idx))


def test_truncate_to_prevents_lookahead():
    idx = pd.bdate_range("2024-01-01", periods=10)
    s = pd.Series(range(10), index=idx, dtype=float)
    cut = truncate_to(s, "2024-01-05")
    assert cut.index.max() == pd.Timestamp("2024-01-05")
    assert len(cut) == 5


def test_truncate_to_handles_tz_aware_index():
    idx = pd.bdate_range("2024-01-01", periods=10, tz="Asia/Kolkata")
    s = pd.Series(range(10), index=idx, dtype=float)
    assert len(truncate_to(s, "2024-01-05")) == 5
