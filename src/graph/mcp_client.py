"""Phase 3.3: connect the graph to the custom MCP server.

* One stdio session per run, so the server process (and its TTL cache) stays alive
  across tool calls. The default adapter behaviour spawns a new process per call.
* An explicit allowlist decides which tools the agent may see. The research agent
  gets read-only tools only; order tools are never loaded for it.
* parse_tool_result() turns whatever the adapter returns into a plain dict that
  always has an `ok` key. Anything we cannot parse becomes ok=False, never data.

Async client side only. The MCP server itself stays sync (see ASYNC RULES).
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Collection, Iterable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

SERVER_NAME = "market"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Tools the research agent may call: they read data and never move (paper) money.
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {"get_quote", "get_history", "get_fundamentals", "search_news", "get_portfolio"}
)
# Tools reserved for the executor node (Phase 5+), behind human approval.
ORDER_TOOLS: frozenset[str] = frozenset({"place_paper_order", "get_order_status"})


class ToolSelectionError(RuntimeError):
    """The server does not expose the tools we expected."""


def server_connection() -> dict[str, Any]:
    """Stdio connection that starts our own server as a module from the project root."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "src.mcp_server.server"],
        "cwd": str(PROJECT_ROOT),
    }


def select_tools(tools: Iterable[BaseTool], allowed: Collection[str]) -> list[BaseTool]:
    """Keep exactly the allowed tools. A missing one is an error, never silently skipped."""
    by_name = {t.name: t for t in tools}
    missing = sorted(set(allowed) - by_name.keys())
    if missing:
        raise ToolSelectionError(f"MCP server is missing expected tools: {missing}")
    return [by_name[name] for name in sorted(allowed)]


@asynccontextmanager
async def open_tools(
    allowed: Collection[str],
    *,
    connections: Mapping[str, Any] | None = None,
) -> AsyncIterator[list[BaseTool]]:
    """Open one MCP session and yield the allowed tools, bound to that session.

    The tools only work inside the `async with` block.
    """
    client = MultiServerMCPClient(dict(connections or {SERVER_NAME: server_connection()}))
    selection_error: ToolSelectionError | None = None
    async with client.session(SERVER_NAME) as session:
        tools = await load_mcp_tools(session)
        try:
            selected = select_tools(tools, allowed)
        except ToolSelectionError as exc:
            # Raising inside the session would be wrapped in an ExceptionGroup by the
            # adapter's task group, so raise it once the session has closed.
            selection_error = exc
        else:
            yield selected
    if selection_error is not None:
        raise selection_error


def open_research_tools(*, connections: Mapping[str, Any] | None = None):
    """Read-only tools for the research agent."""
    return open_tools(READ_ONLY_TOOLS, connections=connections)


# ------------------------------------------------------------- result parsing


def _error(message: str, code: str = "TOOL_ERROR") -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message[:500], "retryable": False}


def _text_of(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return None


def parse_tool_result(raw: Any) -> dict[str, Any]:
    """Normalise a tool result (content blocks, string, dict or ToolMessage) to a dict."""
    content = getattr(raw, "content", raw)  # ToolMessage -> its content
    if isinstance(content, Mapping):
        return dict(content) if "ok" in content else _error("tool result has no 'ok' field")
    text = _text_of(content)
    if text is None or not text.strip():
        return _error("empty or unsupported tool result")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # e.g. "Error executing tool x: ..." when the server code raised
        return _error(text)
    if not isinstance(parsed, dict):
        return _error("tool result is not a JSON object")
    if "ok" not in parsed:
        return _error("tool result has no 'ok' field")
    return parsed


# ---------------------------------------------------------------- live check


async def _live_check() -> None:
    expected = READ_ONLY_TOOLS | ORDER_TOOLS
    async with open_tools(expected) as tools:
        print(f"Loaded {len(tools)} tools:")
        for t in tools:
            first_line = (t.description or "").strip().splitlines()[0]
            print(f"  {t.name:18} {first_line[:70]}")
        by_name = {t.name: t for t in tools}
        raw = await by_name["get_quote"].ainvoke({"symbol": "RELIANCE"})
        print("get_quote RELIANCE ->", parse_tool_result(raw))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_live_check())
