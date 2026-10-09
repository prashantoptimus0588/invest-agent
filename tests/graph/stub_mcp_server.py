"""Offline stand-in for the real MCP server: same tool names, canned data."""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("market")


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """Stub quote. Includes the server pid to prove one process serves many calls."""
    return {"ok": True, "symbol": symbol, "price": 123.0, "source": "stub", "pid": os.getpid()}


@mcp.tool()
def get_history(symbol: str) -> dict:
    """Stub history."""
    return {"ok": True, "symbol": symbol, "candles": []}


@mcp.tool()
def get_fundamentals(symbol: str) -> dict:
    """Stub fundamentals."""
    return {"ok": True, "symbol": symbol}


@mcp.tool()
def search_news(query: str) -> dict:
    """Stub news."""
    return {"ok": True, "untrusted": True, "content": "headline"}


@mcp.tool()
def get_portfolio() -> dict:
    """Stub portfolio."""
    return {"ok": True, "cash": 100000.0}


@mcp.tool()
def place_paper_order(symbol: str) -> dict:
    """Stub order tool (must never reach the research agent)."""
    return {"ok": True}


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Stub order status."""
    return {"ok": False, "code": "NOT_FOUND", "message": "no such order", "retryable": False}


@mcp.tool()
def explode() -> dict:
    """Raises, to see how the adapter reports server-side failures."""
    raise RuntimeError("kaput")


if __name__ == "__main__":
    mcp.run()
