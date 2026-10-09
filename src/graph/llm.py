"""Phase 3.4b: one place that builds chat models.

Every node asks for a model by ROLE: get_model(LLMRole.SYNTH). Phase 3 uses one model for
every role (the roadmap's advice: stabilise the graph first). Phase 4 gives each role its
own model by changing model_name_for() and adding settings, with no call-site changes.

Rules:
  * Model name, temperature, timeout and retries come from Settings, never from code.
  * A missing API key fails here, loudly, not on the first request.
  * Retries are explicit and small: the client's default (6) would multiply requests on
    a free tier with 15 requests per minute.
"""

from __future__ import annotations

from enum import StrEnum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import Settings, get_settings


class LLMRole(StrEnum):
    ROUTER = "router"
    RESEARCH = "research"
    SYNTH = "synth"
    CRITIC = "critic"
    VERIFIER = "verifier"


class LLMConfigError(RuntimeError):
    """The LLM is not configured well enough to make a model."""


def model_name_for(role: LLMRole, settings: Settings) -> str:
    """Phase 3: one model for all roles. Phase 4 adds per-role overrides here."""
    return settings.llm_model


def get_model(
    role: LLMRole | str = LLMRole.RESEARCH,
    *,
    settings: Settings | None = None,
) -> BaseChatModel:
    role = LLMRole(role)  # unknown role raises ValueError
    cfg = settings or get_settings()
    if not cfg.llm_api_key.strip():
        raise LLMConfigError("LLM_API_KEY is empty. Set it in .env (Google AI Studio key).")
    name = model_name_for(role, cfg)
    if not name.strip():
        raise LLMConfigError(f"no model configured for role '{role}'")
    return ChatGoogleGenerativeAI(
        model=name,
        google_api_key=cfg.llm_api_key,
        temperature=cfg.llm_temperature,
        timeout=cfg.llm_timeout_s,
        max_retries=cfg.llm_max_retries,
    )


# ---------------------------------------------------------------- live check


async def _live_check() -> None:
    """Two real requests: plain reply, then a tool call. Not part of pytest."""
    from langchain_core.tools import tool

    @tool
    def get_quote(symbol: str) -> str:
        """Latest price for an NSE symbol."""
        return "{}"

    model = get_model(LLMRole.RESEARCH)
    print("model:", getattr(model, "model", "?"))

    reply = await model.ainvoke("Reply with exactly: OK")
    print("plain reply:", reply.content, "| usage:", reply.usage_metadata)

    with_tools = model.bind_tools([get_quote])
    call = await with_tools.ainvoke("What is the latest price of TCS? Use the tool.")
    print(
        "tool calls:",
        call.tool_calls,
        "| usage:",
        call.usage_metadata,
        "plain reply:",
        call.content,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(_live_check())
