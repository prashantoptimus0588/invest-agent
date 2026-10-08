import numpy as np
import pytest

from src.risk.rules import ProfileLimits
from src.risk.sizing import suggest_max_amount

BASE = {"portfolio_value": 100_000.0, "budget": 50_000.0, "max_trade_inr": 20_000.0}


def size(profile="conservative", **overrides):
    return suggest_max_amount(profile, **{**BASE, **overrides})


def test_profile_cap_binds():
    r = size("conservative")  # 10% of 100k = 10k, below budget and per-trade cap
    assert r.max_amount_inr == pytest.approx(10_000.0)
    assert r.binding_constraint == "profile_room"


def test_budget_cap_binds():
    r = size("aggressive", budget=4_000.0)
    assert r.max_amount_inr == pytest.approx(4_000.0)
    assert r.binding_constraint == "budget"


def test_max_trade_cap_binds():
    r = size("aggressive", max_trade_inr=5_000.0)  # aggressive room = 30k
    assert r.max_amount_inr == pytest.approx(5_000.0)
    assert r.binding_constraint == "max_trade_inr"


def test_daily_remaining_binds_when_given():
    r = size("aggressive", daily_remaining_inr=1_500.0)
    assert r.max_amount_inr == pytest.approx(1_500.0)
    assert r.binding_constraint == "daily_remaining"
    assert size("aggressive").caps["daily_remaining"] is None


def test_existing_position_reduces_room():
    r = size("conservative", current_position_value=7_000.0)
    assert r.max_amount_inr == pytest.approx(3_000.0)


def test_position_already_over_cap_gives_zero_not_negative():
    r = size("conservative", current_position_value=15_000.0)
    assert r.max_amount_inr == 0.0
    assert r.binding_constraint == "profile_room"
    assert "No room" in r.reason


def test_quantity_rounds_down_to_whole_shares():
    r = size("conservative", price=3_000.0)  # ceiling 10,000 -> 3 shares
    assert r.max_quantity == 3
    assert r.estimated_cost_inr == pytest.approx(9_000.0)


def test_share_price_above_ceiling_gives_zero_quantity():
    r = size("conservative", price=25_000.0)
    assert r.max_quantity == 0
    assert r.estimated_cost_inr == 0.0
    assert "one share costs" in r.reason


def test_no_price_means_no_quantity():
    r = size("conservative")
    assert r.max_quantity is None and r.estimated_cost_inr is None


@pytest.mark.parametrize(
    "field", ["portfolio_value", "budget", "max_trade_inr", "current_position_value"]
)
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_inputs_give_zero_never_a_spendable_amount(field, bad):
    r = size("aggressive", **{field: bad})
    assert r.max_amount_inr == 0.0
    assert r.binding_constraint == f"{field}_unavailable"


def test_non_positive_portfolio_gives_zero():
    assert size(portfolio_value=0.0).max_amount_inr == 0.0
    assert size(portfolio_value=-5.0).max_amount_inr == 0.0


def test_negative_budget_treated_as_zero():
    assert size("aggressive", budget=-100.0).max_amount_inr == 0.0


def test_amount_is_rounded_down_not_up():
    r = size("aggressive", budget=1_234.5678)
    assert r.max_amount_inr == pytest.approx(1_234.56)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        suggest_max_amount("yolo", **BASE)


def test_limits_override():
    tight = ProfileLimits(
        max_volatility=0.1, max_drawdown=0.1, max_position_pct=0.01, max_correlation=0.5
    )
    assert size("aggressive", limits=tight).max_amount_inr == pytest.approx(1_000.0)


def test_never_exceeds_any_cap_random_inputs():
    rng = np.random.default_rng(11)
    for _ in range(300):
        kwargs = {
            "portfolio_value": float(rng.uniform(1_000, 1_000_000)),
            "budget": float(rng.uniform(0, 500_000)),
            "max_trade_inr": float(rng.uniform(0, 100_000)),
            "current_position_value": float(rng.uniform(0, 200_000)),
            "daily_remaining_inr": float(rng.uniform(0, 100_000)),
            "price": float(rng.uniform(10, 5_000)),
        }
        for profile in ("conservative", "moderate", "aggressive"):
            r = suggest_max_amount(profile, **kwargs)
            assert r.max_amount_inr >= 0
            for cap in r.caps.values():
                if cap is not None:
                    assert r.max_amount_inr <= cap + 1e-9
            assert r.estimated_cost_inr <= r.max_amount_inr + 1e-9
