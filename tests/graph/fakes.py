"""Test doubles for LLM-driven nodes. No network, no API key, fully deterministic."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per model call.

    cycle=False: running past the end of the script is a test bug and raises.
    cycle=True: loops forever (used to test step budgets).
    `seen` records the messages shown on each call, so tests can assert on prompts.
    """

    script: list[AIMessage]
    cycle: bool = False
    calls: int = 0
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self  # tool schemas are irrelevant: the script decides

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        index = self.calls
        self.calls += 1
        if index >= len(self.script):
            if not self.cycle:
                raise AssertionError("scripted model ran out of responses")
            index %= len(self.script)
        msg = self.script[index].model_copy(update={"id": None})  # fresh id per call
        return ChatResult(generations=[ChatGeneration(message=msg)])


class ExplodingChatModel(ScriptedChatModel):
    """Raises on every call, like a provider outage or a 429."""

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


def ai_tool_calls(*calls: tuple[str, dict], tokens: int | None = 10) -> AIMessage:
    """An AIMessage that asks for tool calls: ai_tool_calls(("get_fundamentals", {...}))."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{i}"} for i, (n, a) in enumerate(calls)],
        usage_metadata=None
        if tokens is None
        else {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


def ai_text(text: str = "DONE", tokens: int | None = 5) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata=None
        if tokens is None
        else {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )
