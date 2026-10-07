import json
import sqlite3
from contextlib import closing

import pytest

from src.mcp_server.store import PortfolioStore


@pytest.fixture
def store(tmp_path):
    return PortfolioStore(tmp_path / "t.db")


def raw(store, sql, params=()):
    """Run SQL on a separate connection that is always closed (no leaked locks)."""
    with closing(sqlite3.connect(store.db_path, timeout=2)) as c:
        try:
            c.execute(sql, params)
            c.commit()
        except Exception:
            c.rollback()
            raise


def test_starts_with_paper_cash_and_no_holdings(store):
    assert store.get_cash() == 100_000.0 and store.get_holdings() == []


def test_reopening_does_not_reset_cash_or_holdings(tmp_path):
    path = tmp_path / "t.db"
    PortfolioStore(path).upsert_holding("TCS.NS", 5, 3000.0)
    again = PortfolioStore(path, starting_cash=1.0)  # ignored: account already exists
    assert again.get_cash() == 100_000.0 and again.get_holdings()[0].quantity == 5


def test_holdings_are_sorted_and_upsert_overwrites(store):
    store.upsert_holding("TCS.NS", 5, 3000.0)
    store.upsert_holding("INFY.NS", 2, 1500.0)
    store.upsert_holding("TCS.NS", 8, 2900.0)
    assert [(h.symbol, h.quantity) for h in store.get_holdings()] == [("INFY.NS", 2), ("TCS.NS", 8)]


@pytest.mark.parametrize("qty, cost", [(-5, 100.0), (0, 100.0), (5, 0.0), (5, -1.0)])
def test_constraints_reject_impossible_holdings(store, qty, cost):
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_holding("BAD.NS", qty, cost)


def test_constraint_rejects_negative_cash(store):
    with pytest.raises(sqlite3.IntegrityError):
        raw(store, "UPDATE account SET cash = -1 WHERE id = 1")


def test_account_table_holds_only_one_row(store):
    with pytest.raises(sqlite3.IntegrityError):
        raw(store, "INSERT INTO account (id, cash, updated_at) VALUES (2, 5, 'x')")


ORDER_SQL = (
    "INSERT INTO orders (order_id, idempotency_key, symbol, side, quantity, status, "
    "created_at) VALUES (?, ?, 'TCS.NS', ?, 1, 'FILLED', 'x')"
)


def test_orders_reject_bad_side(store):
    with pytest.raises(sqlite3.IntegrityError):
        raw(store, ORDER_SQL, ("o1", "key-0001", "HOLD"))


def test_orders_reject_duplicate_idempotency_key(store):
    raw(store, ORDER_SQL, ("o1", "key-0001", "BUY"))
    with pytest.raises(sqlite3.IntegrityError):
        raw(store, ORDER_SQL, ("o2", "key-0001", "BUY"))


def test_audit_log_is_append_only(store):
    store.log_event("TEST", allowed=True, reason="r")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw(store, "UPDATE audit_log SET reason = 'tampered'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw(store, "DELETE FROM audit_log")
    assert len(store.recent_audit()) == 1


def test_audit_log_records_details_and_returns_newest_first(store):
    store.log_event("FIRST", allowed=True)
    store.log_event("SECOND", allowed=False, reason="why", symbol="TCS.NS", details={"x": 1})
    store.log_event("THIRD", allowed=None)
    rows = store.recent_audit(limit=2)
    assert [r["event"] for r in rows] == ["THIRD", "SECOND"]
    assert rows[0]["allowed"] is None and rows[1]["allowed"] == 0
    assert json.loads(rows[1]["details"]) == {"x": 1} and rows[1]["reason"] == "why"
