import pytest
from langgraph.errors import InvalidUpdateError

from src.graph.toy import (
    build_parallel_graph,
    build_toy_graph,
    build_unsafe_parallel_graph,
)


def test_toy_graph_happy_path():
    out = build_toy_graph().invoke({"symbol": "TCS", "notes": [], "score": 0})
    assert out["notes"] == ["fetched:TCS", "analysed:TCS"]  # reducer appended
    assert out["score"] == 2  # overwritten twice, last write wins


def test_conditional_edge_stops_on_empty_symbol():
    out = build_toy_graph().invoke({"symbol": "  ", "notes": [], "score": 0})
    assert out["notes"] == ["fetched:<none>"]  # analyse never ran
    assert out["score"] == 1


def test_reducer_merges_parallel_writes():
    out = build_parallel_graph().invoke({"notes": []})
    assert sorted(out["notes"]) == ["from_a", "from_b"]


def test_parallel_writes_without_reducer_fail():
    with pytest.raises(InvalidUpdateError):
        build_unsafe_parallel_graph().invoke({"notes": []})
