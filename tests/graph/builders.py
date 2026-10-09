"""Shared builders for graph tests: real RiskReports (via assess), evidence and state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.graph.state import AgentState, evidence_from_tool_result, initial_state
from src.graph.synthesize import RISK_KEY
from src.risk.assess import RiskReport, assess


def prices(sigma: float, n: int = 300, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, sigma, n)
    index = pd.bdate_range(end="2026-10-07", periods=n)
    return pd.Series(100 * np.cumprod(1 + rets), index=index)


def make_report(sigma: float, symbol: str = "TEST", n: int = 300) -> RiskReport:
    return assess(
        symbol,
        prices(sigma, n),
        profile="conservative",
        portfolio_value=100_000,
        budget=5_000,
        max_trade_inr=5_000,
    )


def make_evidence(symbol: str = "TEST") -> dict[str, dict]:
    """A realistic evidence dict: ok quote, fundamentals, history, news, plus one failure."""
    items = [
        ("get_quote", {"symbol": symbol}, {"ok": True, "price": 123.456, "source": "stub"}),
        ("get_fundamentals", {"symbol": symbol}, {"ok": True, "pe": 20.0, "source": "stub"}),
        (
            "get_history",
            {"symbol": symbol, "period": "1y", "interval": "1d", "last_n": 250},
            {"ok": True, "source": "stub", "candles": [{"date": "2026-10-07", "close": 1.0}] * 3},
        ),
        (
            "search_news",
            {"query": "Test Co", "limit": 5},
            {
                "ok": True,
                "untrusted": True,
                "source": "rss",
                "content": "Test Co posts steady results.",
            },
        ),
        ("get_quote", {"symbol": "BROKEN"}, {"ok": False, "code": "UPSTREAM_ERROR"}),
    ]
    out = {}
    for tool, args, result in items:
        item = evidence_from_tool_result(tool, args, result)
        out[item.id] = item.model_dump(mode="json")
    return out


def make_state(
    report: RiskReport | None,
    *,
    symbol: str = "TEST",
    evidence: dict | None = None,
    flags: list[str] | None = None,
    candidates: list[str] | None = None,
) -> AgentState:
    state = initial_state(
        "Is TEST a sensible small buy for me?",
        candidates=candidates if candidates is not None else [symbol],
    )
    state["evidence"] = make_evidence(symbol) if evidence is None else evidence
    if report is not None:
        state["analyses"] = {symbol: {RISK_KEY: report.model_dump(mode="json")}}
    state["flags"] = list(flags or [])
    return state
