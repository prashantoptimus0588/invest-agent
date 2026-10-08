"""Scenario ranges via bootstrap of historical daily returns.

These are illustrative scenarios built from past returns. They are not
predictions and must never be presented as such.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.risk.metrics import _check_returns

MIN_BOOTSTRAP_OBS = 60

DISCLAIMER = (
    "Illustrative scenarios built from past returns. This is not a prediction "
    "or a guarantee of future results."
)


class ScenarioRange(BaseModel):
    horizon_days: int
    n_sims: int
    seed: int
    lookback_obs: int
    demeaned: bool
    p5: float  # terminal return, 0.10 = +10%
    p50: float
    p95: float
    prob_loss: float  # share of simulated paths ending below zero
    values_inr: dict[str, float] | None = None  # p5 / p50 / p95 for amount_inr
    disclaimer: str = DISCLAIMER
    assumptions: list[str] = Field(default_factory=list)


def simulate_scenarios(
    returns: pd.Series,
    *,
    horizon_days: int = 63,
    n_sims: int = 5000,
    seed: int = 42,
    amount_inr: float | None = None,
    demean: bool = False,
) -> ScenarioRange:
    """Bootstrap terminal-return scenarios over `horizon_days` trading days."""
    if not 1 <= horizon_days <= 756:
        raise ValueError("horizon_days must be between 1 and 756")
    if not 1000 <= n_sims <= 50_000:
        raise ValueError("n_sims must be between 1000 and 50000")
    if amount_inr is not None and (not math.isfinite(amount_inr) or amount_inr < 0):
        raise ValueError("amount_inr must be a finite, non-negative number")

    r = _check_returns(returns, min_obs=MIN_BOOTSTRAP_OBS)
    sample = r.to_numpy(dtype=float)
    if demean:
        sample = sample - sample.mean()

    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(n_sims, horizon_days), replace=True)
    terminal = np.prod(1.0 + draws, axis=1) - 1.0

    p5, p50, p95 = (float(x) for x in np.percentile(terminal, [5, 50, 95]))
    prob_loss = float(np.mean(terminal < 0))

    values = None
    if amount_inr is not None:
        values = {
            "p5": amount_inr * (1 + p5),
            "p50": amount_inr * (1 + p50),
            "p95": amount_inr * (1 + p95),
        }

    assumptions = [
        (
            f"Bootstrap of {len(sample)} past daily returns, {n_sims} simulated paths, "
            f"{horizon_days} trading days each"
        ),
        (
            "Days are treated as independent, so volatility clustering is ignored and "
            "real tail risk can be worse"
        ),
        "Zero drift (average return removed)"
        if demean
        else "Keeps the historical average return, which may flatter assets that rose a lot",
    ]

    return ScenarioRange(
        horizon_days=horizon_days,
        n_sims=n_sims,
        seed=seed,
        lookback_obs=len(sample),
        demeaned=demean,
        p5=p5,
        p50=p50,
        p95=p95,
        prob_loss=prob_loss,
        values_inr=values,
        assumptions=assumptions,
    )
