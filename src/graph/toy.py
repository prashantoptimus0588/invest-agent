"""Phase 3.1: throwaway graphs that demonstrate LangGraph primitives.

Nothing here is used by the real agent. It exists to show, in tests:
  * state flowing through nodes (TypedDict, overwrite semantics)
  * a reducer (Annotated[list, add]) merging parallel writes
  * a conditional edge choosing between two destinations
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class ToyState(TypedDict):
    symbol: str
    notes: Annotated[list[str], add]  # reducer: appends instead of overwriting
    score: int  # no reducer: last write wins


def fetch_node(state: ToyState) -> dict:
    # .strip() handles whitespace strings like "  "
    symbol = state.get("symbol", "").strip()
    return {"notes": [f"fetched:{symbol or '<none>'}"], "score": 1}


def analyse_node(state: ToyState) -> dict:
    return {
        "notes": [f"analysed:{state['symbol']}"],
        "score": state["score"] + 1,
    }


def route_after_fetch(state: ToyState) -> Literal["analyse", "stop"]:
    """Conditional edge: no symbol means there is nothing to analyse."""
    return "analyse" if state["symbol"].strip() else "stop"


def build_toy_graph():
    g = StateGraph(ToyState)
    g.add_node("fetch", fetch_node)
    g.add_node("analyse", analyse_node)
    g.add_edge(START, "fetch")
    g.add_conditional_edges("fetch", route_after_fetch, {"analyse": "analyse", "stop": END})
    g.add_edge("analyse", END)
    return g.compile()


# ---------------------------------------------------------------- reducers


class ParallelState(TypedDict):
    notes: Annotated[list[str], add]


class UnsafeParallelState(TypedDict):
    notes: list[str]  # NO reducer: two writers in one step is an error


def _node_a(state) -> dict:
    return {"notes": ["from_a"]}


def _node_b(state) -> dict:
    return {"notes": ["from_b"]}


def _build_parallel(state_type):
    g = StateGraph(state_type)
    g.add_node("a", _node_a)
    g.add_node("b", _node_b)
    g.add_edge(START, "a")  # both start in the same step
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    return g.compile()


def build_parallel_graph():
    """Two parallel nodes, merged safely by the `add` reducer."""
    return _build_parallel(ParallelState)


def build_unsafe_parallel_graph():
    """Same shape, no reducer. Invoking it raises InvalidUpdateError."""
    return _build_parallel(UnsafeParallelState)
