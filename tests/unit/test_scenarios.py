import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import InsufficientDataError
from src.risk.scenarios import DISCLAIMER, simulate_scenarios


def rets(values):
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)))


def noisy(mean=0.0, std=0.01, n=400, seed=5):
    return rets(np.random.default_rng(seed).normal(mean, std, n))


def test_zero_returns_give_zero_scenarios():
    s = simulate_scenarios(rets(np.zeros(100)))
    assert s.p5 == s.p50 == s.p95 == 0.0
    assert s.prob_loss == 0.0


def test_constant_return_compounds_exactly():
    s = simulate_scenarios(rets(np.full(100, 0.001)), horizon_days=63)
    expected = 1.001**63 - 1
    assert s.p5 == pytest.approx(expected)
    assert s.p50 == pytest.approx(expected)
    assert s.p95 == pytest.approx(expected)


def test_constant_loss_gives_prob_loss_one():
    s = simulate_scenarios(rets(np.full(100, -0.001)))
    assert s.prob_loss == 1.0


def test_percentiles_are_ordered():
    s = simulate_scenarios(noisy())
    assert s.p5 <= s.p50 <= s.p95


def test_same_seed_is_deterministic_and_different_seed_differs():
    r = noisy()
    a = simulate_scenarios(r, seed=1)
    b = simulate_scenarios(r, seed=1)
    c = simulate_scenarios(r, seed=2)
    assert a == b
    assert (a.p5, a.p95) != (c.p5, c.p95)


def test_more_volatile_asset_has_wider_range():
    calm = simulate_scenarios(noisy(std=0.01))
    wild = simulate_scenarios(noisy(std=0.03))
    assert (wild.p95 - wild.p5) > (calm.p95 - calm.p5)


def test_demean_removes_historical_drift():
    drifting = rets(0.002 + 0.01 * np.where(np.arange(400) % 2, 1.0, -1.0))
    with_drift = simulate_scenarios(drifting, horizon_days=63)
    no_drift = simulate_scenarios(drifting, horizon_days=63, demean=True)
    assert with_drift.p50 > 0.10
    assert abs(no_drift.p50) < 0.02
    assert no_drift.demeaned and not with_drift.demeaned


def test_amount_converts_to_rupee_values():
    s = simulate_scenarios(noisy(), amount_inr=10_000.0)
    assert s.values_inr["p50"] == pytest.approx(10_000.0 * (1 + s.p50))
    assert s.values_inr["p5"] <= s.values_inr["p50"] <= s.values_inr["p95"]
    assert simulate_scenarios(noisy()).values_inr is None


def test_too_little_history_raises_not_guesses():
    with pytest.raises(InsufficientDataError):
        simulate_scenarios(rets(np.zeros(30)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_days": 0},
        {"horizon_days": 5000},
        {"n_sims": 10},
        {"amount_inr": -1.0},
        {"amount_inr": float("nan")},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_scenarios(noisy(), **kwargs)


def test_labelled_as_scenarios_not_predictions():
    s = simulate_scenarios(noisy())
    assert s.disclaimer == DISCLAIMER
    assert "not a prediction" in s.disclaimer
    assert len(s.assumptions) == 3
    assert any("independent" in a for a in s.assumptions)


def test_nan_returns_are_dropped_not_propagated():
    r = noisy()
    r.iloc[10] = np.nan
    s = simulate_scenarios(r)
    assert s.lookback_obs == 399
    assert np.isfinite([s.p5, s.p50, s.p95]).all()
