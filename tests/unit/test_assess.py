import numpy as np
import pandas as pd
import pytest

from src.risk.assess import RiskReport, assess
from src.risk.rules import RuleStatus


def make_prices(n=300, mu=0.0005, sigma=0.004, seed=1, start="2023-01-02"):
    rets = np.random.default_rng(seed).normal(mu, sigma, n)
    vals = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def run(prices=None, **overrides):
    kwargs = {
        "profile": "conservative",
        "portfolio_value": 100_000.0,
        "budget": 50_000.0,
        "max_trade_inr": 20_000.0,
        "benchmark_prices": make_prices(seed=2, sigma=0.005),
        "holdings_prices": {"BBB": make_prices(seed=3), "CCC": make_prices(seed=4)},
    }
    kwargs.update(overrides)
    p = make_prices(seed=1) if prices is None else prices
    return assess("AAA", p, **kwargs)


def rule(report, name):
    return next(r for r in report.rules if r.rule == name)


def test_calm_asset_is_cleared_with_full_report():
    r = run()
    assert r.cleared is True
    assert r.blocking_reasons == []
    assert r.sizing.max_amount_inr == pytest.approx(10_000.0)
    assert r.indicators is not None and r.scenarios is not None
    assert r.metrics.beta is not None and r.metrics.var_95 is not None
    assert set(r.metrics.correlations) == {"BBB", "CCC"}
    assert r.unavailable == {}
    assert RiskReport.model_validate_json(r.model_dump_json()) == r


def test_volatile_asset_fails_volatility_rule():
    r = run(make_prices(sigma=0.04, seed=1))
    assert rule(r, "max_volatility").status is RuleStatus.FAIL
    assert r.cleared is False
    assert any("max_volatility" in b for b in r.blocking_reasons)


def test_very_short_history_never_raises_and_never_clears():
    r = run(make_prices(n=19))
    assert rule(r, "max_volatility").status is RuleStatus.UNAVAILABLE
    assert rule(r, "max_drawdown").status is RuleStatus.UNAVAILABLE
    assert r.scenarios is None and "scenarios" in r.unavailable
    assert r.cleared is False


def test_medium_history_has_var_but_not_volatility():
    r = run(make_prices(n=40))
    assert r.metrics.var_95 is not None
    assert r.metrics.volatility is None
    assert "volatility" in r.unavailable
    assert r.cleared is False


def test_no_benchmark_means_beta_unavailable_but_does_not_block():
    r = run(benchmark_prices=None)
    assert r.metrics.beta is None
    assert r.unavailable["beta"] == "no benchmark supplied"
    assert r.cleared is True


def test_candidate_is_excluded_from_its_own_correlations():
    p = make_prices(seed=1)
    r = run(p, holdings_prices={"AAA": p})
    assert r.metrics.correlations == {}
    assert r.cleared is True


def test_perfectly_correlated_holding_blocks():
    p = make_prices(seed=1)
    r = run(p, holdings_prices={"BBB": p})
    assert rule(r, "max_correlation").status is RuleStatus.FAIL
    assert r.metrics.correlations["BBB"] == pytest.approx(1.0)
    assert r.cleared is False


def test_unusable_holding_history_is_unknown_not_zero():
    bad = make_prices(n=3)
    r = run(holdings_prices={"BBB": bad, "CCC": make_prices(seed=4)})
    assert r.metrics.correlations["BBB"] is None
    assert "1 holding(s) could not be checked" in rule(r, "max_correlation").reason


def test_proposed_amount_above_ceiling_blocks():
    r = run(proposed_amount_inr=15_000.0)
    assert r.proposed_within_ceiling is False
    assert rule(r, "max_position").status is RuleStatus.FAIL
    assert r.cleared is False
    assert any("exceeds the ceiling" in b for b in r.blocking_reasons)


def test_proposed_amount_within_ceiling_clears_and_drives_scenarios():
    r = run(proposed_amount_inr=5_000.0)
    assert r.proposed_within_ceiling is True
    assert r.cleared is True
    assert r.scenarios.values_inr["p50"] == pytest.approx(5_000.0 * (1 + r.scenarios.p50))


def test_position_already_over_cap_blocks_with_no_room():
    r = run(current_position_value=15_000.0)
    assert r.sizing.max_amount_inr == 0.0
    assert r.cleared is False
    assert any("No room" in b for b in r.blocking_reasons)


def test_missing_portfolio_value_blocks_without_raising():
    r = run(portfolio_value=None)
    assert r.sizing.max_amount_inr == 0.0
    assert rule(r, "max_position").status is RuleStatus.UNAVAILABLE
    assert r.cleared is False


def test_as_of_gives_same_report_as_truncated_input():
    full = make_prices(n=400)
    bench = make_prices(n=400, seed=2, sigma=0.005)
    cut_date = full.index[299]
    a = assess(
        "AAA",
        full,
        profile="moderate",
        portfolio_value=100_000.0,
        budget=50_000.0,
        max_trade_inr=20_000.0,
        benchmark_prices=bench,
        as_of=cut_date,
    )
    b = assess(
        "AAA",
        full.iloc[:300],
        profile="moderate",
        portfolio_value=100_000.0,
        budget=50_000.0,
        max_trade_inr=20_000.0,
        benchmark_prices=bench.iloc[:300],
    )
    assert a == b
    assert a.as_of == cut_date.to_pydatetime()


def test_deterministic():
    assert run() == run()


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_invalid_proposed_amount_raises(bad):
    with pytest.raises(ValueError):
        run(proposed_amount_inr=bad)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        run(profile="yolo")
