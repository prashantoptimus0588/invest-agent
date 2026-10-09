"""Phase 3.5a: the typed Recommendation and the code that guards it. No LLM in here.

Principle: the LLM writes WORDS and makes a CHOICE; code supplies every NUMBER and decides
what the choice is allowed to be.

  Draft           what the synthesizer LLM may fill in: action, rationale, risks, sources.
  Recommendation  the final typed result. Figures, ceiling, confidence, rules, scenarios and
                  the disclaimer are all filled by code from the RiskReport and the evidence.

Guards (all deterministic, all tested):
  * risk gate    the LLM cannot choose CONSIDER_BUY unless the risk engine cleared it.
  * ceiling      the suggested amount is the sizing ceiling from the risk engine, never a
                 number the LLM wrote.
  * sources      every cited id must exist in the evidence and have succeeded.
  * confidence   a data-quality score from a penalty table, not the model's self-report.
  * invariants   the Recommendation model itself refuses an unsafe CONSIDER_BUY.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.graph.state import evidence_id
from src.risk.assess import RiskReport
from src.risk.rules import ProfileName, RuleResult, RuleStatus

DISCLAIMER = (
    "Educational analysis using paper-trading data. Not financial advice. "
    "Scenarios are illustrations of possible ranges, not predictions."
)

MAX_RATIONALE_CHARS = 2000
MAX_RISK_ITEMS = 8
MAX_RISK_CHARS = 400


class Action(StrEnum):
    CONSIDER_BUY = "consider_buy"
    WATCH = "watch"
    AVOID = "avoid"
    INSUFFICIENT_DATA = "insufficient_data"


class Draft(BaseModel):
    """What the synthesizer LLM fills in. Words and a choice, never numbers."""

    action: Action
    rationale: str
    key_risks: list[str]
    source_ids: list[str]


class KeyFigure(BaseModel):
    label: str
    value: float | None
    display: str  # formatted by code, e.g. "24.3%"
    source: str  # "risk_engine" or an evidence id


class Recommendation(BaseModel):
    symbol: str
    profile: ProfileName
    action: Action
    rationale: str = Field(min_length=1)
    key_risks: list[str] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)  # data-quality score, NOT a probability
    confidence_label: Literal["low", "medium", "high"]
    max_amount_inr_ceiling: float | None  # a ceiling, never a target
    key_figures: list[KeyFigure]
    rules: list[RuleResult]
    blocking_reasons: list[str]
    scenarios: dict[str, Any] | None  # from the risk engine; not predictions
    as_of: datetime | None
    generated_at: datetime
    flags: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _buy_needs_clearance(self) -> Recommendation:
        if self.action is Action.CONSIDER_BUY:
            if self.blocking_reasons or any(r.status is not RuleStatus.PASS for r in self.rules):
                raise ValueError("consider_buy requires every risk rule to pass")
            if self.max_amount_inr_ceiling is None or self.max_amount_inr_ceiling <= 0:
                raise ValueError("consider_buy requires a positive sizing ceiling")
        elif self.max_amount_inr_ceiling is not None:
            raise ValueError("a ceiling is only allowed with consider_buy")
        return self


class RecommendationError(ValueError):
    """The draft is unusable. `problems` lists every reason (fed back to the LLM on retry)."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


# ---------------------------------------------------------------- risk gate


def allowed_actions(report: RiskReport | None) -> frozenset[Action]:
    """What the risk engine permits. Unknown never counts as safe."""
    if report is None:
        return frozenset({Action.INSUFFICIENT_DATA})
    if report.cleared:
        return frozenset({Action.CONSIDER_BUY, Action.WATCH, Action.AVOID})
    if any(r.status is RuleStatus.FAIL for r in report.rules):
        return frozenset({Action.AVOID, Action.WATCH})
    if any(r.status is RuleStatus.UNAVAILABLE for r in report.rules):
        return frozenset({Action.INSUFFICIENT_DATA, Action.WATCH})
    return frozenset({Action.AVOID, Action.WATCH})  # blocked by sizing only


def _fallback_action(allowed: frozenset[Action]) -> Action:
    for candidate in (Action.INSUFFICIENT_DATA, Action.AVOID, Action.WATCH):
        if candidate in allowed:
            return candidate
    raise AssertionError("allowed_actions never returns an empty set")


def resolve_action(chosen: Action, report: RiskReport | None) -> tuple[Action, bool]:
    """(final action, whether the risk gate overrode the LLM's choice)."""
    allowed = allowed_actions(report)
    if chosen in allowed:
        return chosen, False
    return _fallback_action(allowed), True


# --------------------------------------------------------------- confidence

# Limits are data. Each penalty is subtracted from 1.0; the score is floored at 0.
CONFIDENCE_PENALTIES: dict[str, float] = {
    "per_unavailable_item": 0.10,
    "unavailable_cap": 0.40,
    "no_fundamentals": 0.15,
    "no_news": 0.10,
    "stale_data": 0.20,
    "per_research_issue": 0.10,
}
STALE_AFTER_DAYS = 5  # covers a weekend plus one holiday
HIGH_AT = 0.75
MEDIUM_AT = 0.50
_RESEARCH_ISSUE_PREFIXES = ("research_agent_failed", "research_budget_hit", "research_no_evidence")


def _norm(symbol: str) -> str:
    s = symbol.strip().upper()
    for suffix in (".NS", ".BO"):
        s = s.removesuffix(suffix)
    return s


def _ok_items(evidence: Mapping[str, Mapping[str, Any]], tool: str) -> list[Mapping[str, Any]]:
    return [v for v in evidence.values() if v.get("tool") == tool and v.get("ok") is True]


def _has_ok(evidence: Mapping[str, Mapping[str, Any]], tool: str, symbol: str | None) -> bool:
    for item in _ok_items(evidence, tool):
        if symbol is None or _norm(str(item.get("args", {}).get("symbol", ""))) == _norm(symbol):
            return True
    return False


def _age_days(as_of: datetime, now: datetime) -> float:
    if as_of.tzinfo is None:
        now = now.replace(tzinfo=None)
    return (now - as_of).total_seconds() / 86400


def compute_confidence(
    report: RiskReport | None,
    evidence: Mapping[str, Mapping[str, Any]],
    symbol: str,
    flags: Sequence[str],
    now: datetime,
) -> float:
    """Heuristic data-quality score in [0, 1]. Says how complete the inputs were."""
    if report is None:
        return 0.0
    p = CONFIDENCE_PENALTIES
    score = 1.0
    score -= min(len(report.unavailable) * p["per_unavailable_item"], p["unavailable_cap"])
    if not _has_ok(evidence, "get_fundamentals", symbol):
        score -= p["no_fundamentals"]
    if not _ok_items(evidence, "search_news"):
        score -= p["no_news"]
    if report.as_of is not None and _age_days(report.as_of, now) > STALE_AFTER_DAYS:
        score -= p["stale_data"]
    issues = sum(1 for f in flags if f.startswith(_RESEARCH_ISSUE_PREFIXES))
    score -= issues * p["per_research_issue"]
    return round(max(score, 0.0), 2)


def confidence_label(score: float) -> Literal["low", "medium", "high"]:
    if score >= HIGH_AT:
        return "high"
    if score >= MEDIUM_AT:
        return "medium"
    return "low"


# ------------------------------------------------------------ key figures


def _pct(x: float | None, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "unavailable"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _fig(label: str, value: float | None, display: str, source: str) -> KeyFigure:
    clean = value if value is not None and math.isfinite(value) else None
    return KeyFigure(label=label, value=clean, display=display, source=source)


def key_figures(
    report: RiskReport, evidence: Mapping[str, Mapping[str, Any]], symbol: str
) -> list[KeyFigure]:
    """Every number the user sees comes from here: the risk engine or a tool result."""
    m = report.metrics
    figs = [
        _fig("Annualised volatility", m.volatility, _pct(m.volatility), "risk_engine"),
        _fig("Maximum drawdown", m.max_drawdown, _pct(m.max_drawdown, signed=True), "risk_engine"),
        _fig("1-day value at risk (95%)", m.var_95, _pct(m.var_95), "risk_engine"),
        _fig(
            "Beta vs benchmark",
            m.beta,
            "unavailable" if m.beta is None else f"{m.beta:.2f}",
            "risk_engine",
        ),
    ]
    quote_id = evidence_id("get_quote", {"symbol": symbol})
    quote = evidence.get(quote_id)
    if quote and quote.get("ok") is True:
        price = quote.get("data", {}).get("price")
        if isinstance(price, int | float) and math.isfinite(price):
            figs.append(_fig("Latest price", float(price), f"INR {price:,.2f}", quote_id))
    ceiling = report.sizing.max_amount_inr
    if ceiling > 0:
        figs.append(
            _fig(
                "Suggested maximum amount (a ceiling, not a target)",
                ceiling,
                f"INR {ceiling:,.2f}",
                "risk_engine",
            )
        )
    return figs


# ------------------------------------------------------------ number scanning

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def numbers_in(text: str) -> list[str]:
    """Numeric tokens in free text. Reused by the Phase 4 verifier."""
    return _NUMBER_RE.findall(text)


# -------------------------------------------------------------------- assembly


def _clean_ids(ids: Sequence[str]) -> list[str]:
    """Strip whitespace, drop blanks and duplicates, keep order."""
    return list(dict.fromkeys(i.strip() for i in ids if i and i.strip()))


def _problems_in(draft: Draft, evidence: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    rationale = draft.rationale.strip()
    if not rationale:
        problems.append("rationale is empty")
    elif len(rationale) > MAX_RATIONALE_CHARS:
        problems.append(f"rationale is longer than {MAX_RATIONALE_CHARS} characters")

    risks = [r.strip() for r in draft.key_risks if r and r.strip()]
    if not risks:
        problems.append("key_risks must list at least one risk")
    if len(risks) > MAX_RISK_ITEMS:
        problems.append(f"key_risks has more than {MAX_RISK_ITEMS} items")
    if any(len(r) > MAX_RISK_CHARS for r in risks):
        problems.append(f"a risk item is longer than {MAX_RISK_CHARS} characters")

    source_ids = _clean_ids(draft.source_ids)
    if not source_ids:
        problems.append("source_ids must cite at least one evidence id")
    for sid in source_ids:
        item = evidence.get(sid)
        if item is None:
            problems.append(f"source id '{sid}' does not exist in the evidence")
        elif item.get("ok") is not True:
            problems.append(f"source id '{sid}' is a failed tool call and cannot be cited")
    return problems


def build_recommendation(
    draft: Draft,
    *,
    symbol: str,
    report: RiskReport | None,
    evidence: Mapping[str, Mapping[str, Any]],
    flags: Sequence[str] = (),
    now: datetime | None = None,
) -> Recommendation:
    """Validate the draft and assemble the final Recommendation. Raises RecommendationError."""
    now = now or datetime.now(UTC)
    problems = _problems_in(draft, evidence)
    if report is None:
        problems.append("no risk report is available for this symbol")
    if problems:
        raise RecommendationError(problems)
    assert report is not None  # narrowed by the problems check above

    action, overridden = resolve_action(draft.action, report)
    out_flags = list(flags)
    if overridden:
        out_flags.append(f"action_overridden_by_risk_gate:{draft.action.value}->{action.value}")

    rationale = draft.rationale.strip()
    risks = [r.strip() for r in draft.key_risks if r and r.strip()]
    if numbers_in(rationale) or any(numbers_in(r) for r in risks):
        out_flags.append("prose_contains_numbers")  # the Phase 4 verifier checks these

    score = compute_confidence(report, evidence, symbol, out_flags, now)
    try:
        return Recommendation(
            symbol=symbol,
            profile=report.profile,
            action=action,
            rationale=rationale,
            key_risks=risks,
            source_ids=_clean_ids(draft.source_ids),
            confidence=score,
            confidence_label=confidence_label(score),
            max_amount_inr_ceiling=report.sizing.max_amount_inr
            if action is Action.CONSIDER_BUY
            else None,
            key_figures=key_figures(report, evidence, symbol),
            rules=report.rules,
            blocking_reasons=report.blocking_reasons,
            scenarios=report.scenarios.model_dump(mode="json") if report.scenarios else None,
            as_of=report.as_of,
            generated_at=now,
            flags=out_flags,
        )
    except ValueError as exc:  # the model's own safety invariants
        raise RecommendationError([str(exc)]) from exc
