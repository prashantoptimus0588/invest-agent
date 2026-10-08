"""Phase 3.2: the shared graph state.

State values are JSON-safe (dicts, lists, numbers, strings) so checkpointers and
SSE streaming work without custom serialisation. Pydantic models validate at the
edges and are dumped with model_dump(mode="json") before going into state.

Reducers make parallel writes (Send fan-out, Phase 4) merge instead of clash:
  evidence       dict merge, later write of the same id wins (fresher fetch)
  analyses       two-level merge: analyses[symbol][kind]
  critic_notes   append
  flags          append
  step_count     sum of deltas   (nodes return {"step_count": 1})
  token_spend    sum of deltas
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from src.risk.rules import ProfileName

# ------------------------------------------------------------------ reducers


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Shallow merge; keys in `right` win."""
    return {**(left or {}), **(right or {})}


def merge_nested(left: dict | None, right: dict | None) -> dict:
    """Two-level merge for analyses[symbol][kind]; inner keys in `right` win."""
    out: dict[str, dict] = {k: dict(v) for k, v in (left or {}).items()}
    for key, inner in (right or {}).items():
        out.setdefault(key, {}).update(inner)
    return out


# -------------------------------------------------------------------- models


class UserProfile(BaseModel):
    """What the user told us. None means 'not provided', never a default."""

    risk: ProfileName | None = None
    horizon_days: int | None = Field(default=None, gt=0)
    budget_inr: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    def missing(self) -> list[str]:
        """Field names still unknown (the clarifier asks about these)."""
        return [name for name in type(self).model_fields if getattr(self, name) is None]


class EvidenceItem(BaseModel):
    """One tool result, with provenance. The verifier matches numbers against these."""

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error_code: str | None = None
    source: str | None = None
    as_of: datetime | None = None  # when the data is from
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    untrusted: bool = False  # third-party text (news): evidence only, never instructions
    data: dict[str, Any] = Field(default_factory=dict)


def evidence_id(tool: str, args: Mapping[str, Any]) -> str:
    """Stable id from tool + args, independent of argument order."""
    parts = ",".join(f"{k}={args[k]}" for k in sorted(args))
    return f"{tool}:{parts}" if parts else tool


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None  # unknown stays unknown


def evidence_from_tool_result(
    tool: str, args: Mapping[str, Any], result: Mapping[str, Any]
) -> EvidenceItem:
    """Wrap a raw MCP tool result dict. Failures are kept as evidence too (audit)."""
    return EvidenceItem(
        id=evidence_id(tool, args),
        tool=tool,
        args=dict(args),
        ok=result.get("ok") is True,  # missing or odd value is not ok
        error_code=result.get("code") if result.get("ok") is not True else None,
        source=result.get("source"),
        as_of=_parse_dt(result.get("as_of")),
        untrusted=bool(result.get("untrusted")),
        data=dict(result),
    )


# --------------------------------------------------------------------- state


class AgentState(TypedDict):
    user_profile: dict[str, Any]  # UserProfile dumped
    goal: str
    candidates: list[str]  # symbols under consideration
    evidence: Annotated[dict[str, dict[str, Any]], merge_dicts]  # id -> EvidenceItem dump
    analyses: Annotated[dict[str, dict[str, Any]], merge_nested]  # symbol -> kind -> result
    draft_recommendation: dict[str, Any] | None
    critic_notes: Annotated[list[str], add]
    confidence: float | None
    approval: dict[str, Any] | None
    step_count: Annotated[int, add]
    token_spend: Annotated[int, add]
    flags: Annotated[list[str], add]


def initial_state(
    goal: str,
    *,
    user_profile: UserProfile | None = None,
    candidates: Sequence[str] = (),
) -> AgentState:
    """Build a valid starting state for one run."""
    if isinstance(candidates, str):
        raise TypeError("candidates must be a sequence of symbols, not a single string")
    goal = goal.strip()
    if not goal:
        raise ValueError("goal must not be empty")
    cleaned = dict.fromkeys(c.strip() for c in candidates if c and c.strip())  # dedupe, keep order
    profile = user_profile or UserProfile()
    return AgentState(
        user_profile=profile.model_dump(mode="json"),
        goal=goal,
        candidates=list(cleaned),
        evidence={},
        analyses={},
        draft_recommendation=None,
        critic_notes=[],
        confidence=None,
        approval=None,
        step_count=0,
        token_spend=0,
        flags=[],
    )


def profile_from_state(state: Mapping[str, Any]) -> UserProfile:
    return UserProfile.model_validate(state["user_profile"])


def evidence_items(state: Mapping[str, Any]) -> list[EvidenceItem]:
    return [EvidenceItem.model_validate(v) for v in state["evidence"].values()]
