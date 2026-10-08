"""Profile-based risk rules.

Every rule is evaluated separately and returns pass / fail / unavailable with a
reason. A metric that could not be computed is 'unavailable', which is never
treated as a pass. Limits live in the PROFILES table, not in the logic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

_EPS = 1e-12  # float tolerance so a value exactly at the limit passes


class ProfileName(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ProfileLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_volatility: float = Field(gt=0, le=2.0)  # annualised, 0.20 = 20%
    max_drawdown: float = Field(gt=0, le=1.0)  # tolerance as a positive fraction
    max_position_pct: float = Field(gt=0, le=1.0)  # share of portfolio in one asset
    max_correlation: float = Field(gt=0, le=1.0)  # vs any single existing holding


PROFILES: dict[ProfileName, ProfileLimits] = {
    ProfileName.CONSERVATIVE: ProfileLimits(
        max_volatility=0.20, max_drawdown=0.25, max_position_pct=0.10, max_correlation=0.70
    ),
    ProfileName.MODERATE: ProfileLimits(
        max_volatility=0.30, max_drawdown=0.40, max_position_pct=0.20, max_correlation=0.80
    ),
    ProfileName.AGGRESSIVE: ProfileLimits(
        max_volatility=0.45, max_drawdown=0.60, max_position_pct=0.30, max_correlation=0.90
    ),
}


class RuleStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


class RuleResult(BaseModel):
    rule: str
    status: RuleStatus
    value: float | None
    limit: float
    reason: str


def get_limits(profile: ProfileName | str) -> ProfileLimits:
    """Look up a profile's limits. Unknown names raise ValueError."""
    return PROFILES[ProfileName(profile)]


def position_weight(position_value: float, portfolio_value: float) -> float | None:
    """Fraction of the portfolio in one asset; None if the portfolio value is not positive."""
    if portfolio_value is None or portfolio_value <= 0:
        return None
    return position_value / portfolio_value


def _missing(x: float | None) -> bool:
    return x is None or math.isnan(x)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _check_max(
    rule: str, label: str, value: float | None, limit: float, profile: str
) -> RuleResult:
    """Shared logic for 'value must not exceed limit' rules."""
    if _missing(value):
        return RuleResult(
            rule=rule,
            status=RuleStatus.UNAVAILABLE,
            value=None,
            limit=limit,
            reason=f"{label} could not be computed, so this rule is not cleared",
        )
    if value <= limit + _EPS:
        return RuleResult(
            rule=rule,
            status=RuleStatus.PASS,
            value=value,
            limit=limit,
            reason=f"{label} {_pct(value)} is within the {_pct(limit)} limit for the {profile} profile",
        )
    return RuleResult(
        rule=rule,
        status=RuleStatus.FAIL,
        value=value,
        limit=limit,
        reason=f"{label} {_pct(value)} exceeds the {_pct(limit)} limit for the {profile} profile",
    )


def _check_correlation(
    correlations: Mapping[str, float | None], limit: float, profile: str
) -> RuleResult:
    rule = "max_correlation"
    if not correlations:
        return RuleResult(
            rule=rule,
            status=RuleStatus.PASS,
            value=None,
            limit=limit,
            reason="No existing holdings, so there is nothing to be correlated with",
        )

    known = {s: c for s, c in correlations.items() if not _missing(c)}
    unknown = len(correlations) - len(known)
    if not known:
        return RuleResult(
            rule=rule,
            status=RuleStatus.UNAVAILABLE,
            value=None,
            limit=limit,
            reason="Correlation with existing holdings could not be computed, so this rule is not cleared",
        )

    worst_symbol = max(known, key=lambda s: known[s])
    worst = known[worst_symbol]
    note = f" ({unknown} holding(s) could not be checked)" if unknown else ""
    if worst <= limit + _EPS:
        return RuleResult(
            rule=rule,
            status=RuleStatus.PASS,
            value=worst,
            limit=limit,
            reason=f"Highest correlation {worst:.2f} (with {worst_symbol}) is within the {limit:.2f} limit for the {profile} profile{note}",
        )
    return RuleResult(
        rule=rule,
        status=RuleStatus.FAIL,
        value=worst,
        limit=limit,
        reason=f"Correlation {worst:.2f} with {worst_symbol} exceeds the {limit:.2f} limit for the {profile} profile{note}",
    )


def evaluate_rules(
    profile: ProfileName | str,
    *,
    volatility: float | None,
    max_drawdown: float | None,
    position_pct: float | None,
    correlations: Mapping[str, float | None],
    limits: ProfileLimits | None = None,
) -> list[RuleResult]:
    """Evaluate every rule separately.

    `max_drawdown` is the negative number from metrics.max_drawdown; its absolute
    value is compared with the profile's tolerance. `limits` overrides the table
    (useful in tests).
    """
    name = ProfileName(profile)
    lim = limits or PROFILES[name]
    dd = None if _missing(max_drawdown) else abs(max_drawdown)
    return [
        _check_max("max_volatility", "Annualised volatility", volatility, lim.max_volatility, name),
        _check_max("max_drawdown", "Maximum drawdown", dd, lim.max_drawdown, name),
        _check_max("max_position", "Position size", position_pct, lim.max_position_pct, name),
        _check_correlation(correlations, lim.max_correlation, name),
    ]


def cleared(results: list[RuleResult]) -> bool:
    """True only if every rule passed. 'unavailable' blocks, same as 'fail'."""
    return all(r.status is RuleStatus.PASS for r in results)


def blocking(results: list[RuleResult]) -> list[RuleResult]:
    """The rules that stopped the proposal (fail or unavailable)."""
    return [r for r in results if r.status is not RuleStatus.PASS]
