"""Position sizing.

Returns a CEILING on how much could be committed, never a target to chase.
The smallest of the profile, budget, per-trade and daily caps wins.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from src.risk.rules import ProfileLimits, ProfileName, get_limits


class SizingResult(BaseModel):
    max_amount_inr: float  # a ceiling, not a recommendation to spend it
    max_quantity: int | None  # whole shares at `price`, None if no price given
    estimated_cost_inr: float | None
    binding_constraint: str
    caps: dict[str, float | None]
    reason: str


def _bad(x: float | None) -> bool:
    return x is None or not math.isfinite(x)


def _floor2(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def _zero(constraint: str, caps: dict[str, float | None], reason: str) -> SizingResult:
    return SizingResult(
        max_amount_inr=0.0,
        max_quantity=None,
        estimated_cost_inr=None,
        binding_constraint=constraint,
        caps=caps,
        reason=reason,
    )


def suggest_max_amount(
    profile: ProfileName | str,
    *,
    portfolio_value: float | None,
    budget: float | None,
    max_trade_inr: float | None,
    current_position_value: float | None = 0.0,
    price: float | None = None,
    daily_remaining_inr: float | None = None,
    limits: ProfileLimits | None = None,
) -> SizingResult:
    """Maximum amount (INR) that could be put into one asset.

    portfolio_value = holdings market value + cash, before the trade.
    """
    lim = limits or get_limits(profile)

    required = {
        "portfolio_value": portfolio_value,
        "budget": budget,
        "max_trade_inr": max_trade_inr,
        "current_position_value": current_position_value,
    }
    for name, val in required.items():
        if _bad(val):
            return _zero(
                f"{name}_unavailable",
                {},
                f"{name} is missing or invalid, so no amount can be suggested",
            )
    if portfolio_value <= 0:
        return _zero(
            "portfolio_value_unavailable",
            {},
            "Portfolio value is not positive, so no amount can be suggested",
        )
    if daily_remaining_inr is not None and not math.isfinite(daily_remaining_inr):
        return _zero(
            "daily_remaining_unavailable",
            {},
            "Daily remaining cap is invalid, so no amount can be suggested",
        )

    caps: dict[str, float | None] = {
        "profile_room": max(0.0, lim.max_position_pct * portfolio_value - current_position_value),
        "budget": max(0.0, budget),
        "max_trade_inr": max(0.0, max_trade_inr),
        "daily_remaining": None if daily_remaining_inr is None else max(0.0, daily_remaining_inr),
    }
    active = {k: v for k, v in caps.items() if v is not None}
    binding = min(active, key=lambda k: active[k])  # first wins on ties
    amount = _floor2(active[binding])

    quantity: int | None = None
    cost: float | None = None
    if price is not None and math.isfinite(price) and price > 0:
        quantity = math.floor(amount / price + 1e-9)
        cost = round(quantity * price, 2)

    if amount <= 0:
        reason = f"No room to add: the {binding} cap leaves nothing to commit"
    else:
        reason = (
            f"Ceiling of INR {amount:,.2f}, limited by {binding}. This is a maximum, not a target"
        )
        if quantity == 0:
            reason += f"; one share costs INR {price:,.2f}, which is above the ceiling"

    return SizingResult(
        max_amount_inr=amount,
        max_quantity=quantity,
        estimated_cost_inr=cost,
        binding_constraint=binding,
        caps=caps,
        reason=reason,
    )
