import pytest

from src.risk.rules import (
    PROFILES,
    ProfileLimits,
    ProfileName,
    RuleStatus,
    blocking,
    cleared,
    evaluate_rules,
    get_limits,
    position_weight,
)

GOOD = {
    "volatility": 0.15,
    "max_drawdown": -0.10,
    "position_pct": 0.05,
    "correlations": {"TCS": 0.3},
}


def run(profile="conservative", **overrides):
    args = {**GOOD, **overrides}
    return {r.rule: r for r in evaluate_rules(profile, **args)}


def test_all_pass_for_safe_inputs():
    results = evaluate_rules("conservative", **GOOD)
    assert cleared(results)
    assert blocking(results) == []
    assert len(results) == 4


# --- one passing and one failing case per rule ---


def test_volatility_pass_and_fail():
    assert run(volatility=0.15)["max_volatility"].status is RuleStatus.PASS
    r = run(volatility=0.32)["max_volatility"]
    assert r.status is RuleStatus.FAIL
    assert "32.0%" in r.reason and "20.0%" in r.reason


def test_drawdown_pass_and_fail():
    assert run(max_drawdown=-0.20)["max_drawdown"].status is RuleStatus.PASS
    r = run(max_drawdown=-0.30)["max_drawdown"]
    assert r.status is RuleStatus.FAIL
    assert r.value == pytest.approx(0.30)  # stored as positive tolerance comparison


def test_position_pass_and_fail():
    assert run(position_pct=0.08)["max_position"].status is RuleStatus.PASS
    assert run(position_pct=0.15)["max_position"].status is RuleStatus.FAIL


def test_correlation_pass_and_fail():
    assert run(correlations={"A": 0.5})["max_correlation"].status is RuleStatus.PASS
    r = run(correlations={"A": 0.5, "B": 0.85})["max_correlation"]
    assert r.status is RuleStatus.FAIL
    assert "B" in r.reason  # names the worst offender


# --- boundaries and missing data ---


def test_value_exactly_at_limit_passes():
    res = run(volatility=0.20, max_drawdown=-0.25, position_pct=0.10, correlations={"A": 0.70})
    assert all(r.status is RuleStatus.PASS for r in res.values())


@pytest.mark.parametrize(
    "field,rule",
    [
        ("volatility", "max_volatility"),
        ("max_drawdown", "max_drawdown"),
        ("position_pct", "max_position"),
    ],
)
def test_missing_metric_is_unavailable_not_pass(field, rule):
    for missing in (None, float("nan")):
        res = run(**{field: missing})
        assert res[rule].status is RuleStatus.UNAVAILABLE
        assert res[rule].value is None
        assert not cleared(list(res.values()))


def test_correlation_no_holdings_passes():
    r = run(correlations={})["max_correlation"]
    assert r.status is RuleStatus.PASS
    assert r.value is None


def test_correlation_all_unknown_is_unavailable():
    r = run(correlations={"A": None, "B": None})["max_correlation"]
    assert r.status is RuleStatus.UNAVAILABLE


def test_correlation_partial_unknown_checks_known_and_notes_gap():
    r = run(correlations={"A": 0.4, "B": None})["max_correlation"]
    assert r.status is RuleStatus.PASS
    assert "1 holding(s) could not be checked" in r.reason


def test_blocking_lists_fail_and_unavailable():
    results = evaluate_rules("conservative", **{**GOOD, "volatility": 0.5, "position_pct": None})
    names = {r.rule for r in blocking(results)}
    assert names == {"max_volatility", "max_position"}


# --- profiles ---


def test_same_inputs_different_verdict_by_profile():
    kwargs = {
        "volatility": 0.28,
        "max_drawdown": -0.30,
        "position_pct": 0.15,
        "correlations": {"A": 0.75},
    }

    assert not cleared(evaluate_rules("conservative", **kwargs))
    assert cleared(evaluate_rules("moderate", **kwargs))


def test_profiles_get_looser_in_order():
    c, m, a = (PROFILES[p] for p in ProfileName)
    for field in ("max_volatility", "max_drawdown", "max_position_pct", "max_correlation"):
        assert getattr(c, field) < getattr(m, field) < getattr(a, field)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        get_limits("yolo")
    with pytest.raises(ValueError):
        evaluate_rules("yolo", **GOOD)


def test_limits_override_and_validation():
    tight = ProfileLimits(
        max_volatility=0.05, max_drawdown=0.05, max_position_pct=0.01, max_correlation=0.1
    )
    assert not cleared(evaluate_rules("aggressive", limits=tight, **GOOD))
    with pytest.raises(ValueError):
        ProfileLimits(
            max_volatility=-1, max_drawdown=0.1, max_position_pct=0.1, max_correlation=0.5
        )


def test_position_weight():
    assert position_weight(500, 10_000) == pytest.approx(0.05)
    assert position_weight(500, 0) is None
