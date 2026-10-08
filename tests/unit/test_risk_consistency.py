import numpy as np

from src.risk.rules import RuleStatus, evaluate_rules, position_weight
from src.risk.sizing import suggest_max_amount


def position_rule(profile, position_pct):
    results = evaluate_rules(
        profile, volatility=0.0, max_drawdown=0.0, position_pct=position_pct, correlations={}
    )
    return next(r for r in results if r.rule == "max_position")


def test_sizing_ceiling_never_fails_the_position_rule():
    rng = np.random.default_rng(21)
    for _ in range(300):
        pv = float(rng.uniform(10_000, 1_000_000))
        current = float(rng.uniform(0, 0.05 * pv))
        for profile in ("conservative", "moderate", "aggressive"):
            s = suggest_max_amount(
                profile,
                portfolio_value=pv,
                budget=float(rng.uniform(0, 500_000)),
                max_trade_inr=float(rng.uniform(0, 200_000)),
                current_position_value=current,
            )
            pct = position_weight(current + s.max_amount_inr, pv)
            assert position_rule(profile, pct).status is RuleStatus.PASS


def test_one_rupee_above_a_profile_bound_ceiling_fails_the_position_rule():
    rng = np.random.default_rng(22)
    checked = 0
    for _ in range(300):
        pv = float(rng.uniform(10_000, 1_000_000))
        for profile in ("conservative", "moderate", "aggressive"):
            s = suggest_max_amount(
                profile,
                portfolio_value=pv,
                budget=10_000_000.0,
                max_trade_inr=10_000_000.0,
            )
            assert s.binding_constraint == "profile_room"
            pct = position_weight(s.max_amount_inr + 1.0, pv)
            assert position_rule(profile, pct).status is RuleStatus.FAIL
            checked += 1
    assert checked == 900
