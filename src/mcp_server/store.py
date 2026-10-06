import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .models import Holding, utcnow

# <repo root>/data/invest.db, independent of the directory you launch from
DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "invest.db"
STARTING_CASH = 100_000.0  # paper rupees

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    cash       REAL NOT NULL CHECK (cash >= 0),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS holdings (
    symbol     TEXT PRIMARY KEY,
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    avg_cost   REAL NOT NULL CHECK (avg_cost > 0),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT NOT NULL UNIQUE,
    idempotency_key TEXT NOT NULL UNIQUE,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    limit_price     REAL,
    fill_price      REAL,
    status          TEXT NOT NULL CHECK (status IN ('FILLED', 'REJECTED')),
    reason          TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    event   TEXT NOT NULL,
    allowed INTEGER CHECK (allowed IN (0, 1)),
    reason  TEXT,
    symbol  TEXT,
    details TEXT
);

CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


class PortfolioStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB, starting_cash: float = STARTING_CASH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            c.execute(
                "INSERT OR IGNORE INTO account (id, cash, updated_at) VALUES (1, ?, ?)",
                (starting_cash, utcnow().isoformat()),
            )

    @contextmanager
    def _conn(self):
        """One short-lived connection per operation: thread-safe, commit or rollback."""
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA journal_mode = WAL")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # ---- reads -------------------------------------------------------------
    def get_cash(self) -> float:
        with self._conn() as c:
            return float(c.execute("SELECT cash FROM account WHERE id = 1").fetchone()["cash"])

    def get_holdings(self) -> list[Holding]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT symbol, quantity, avg_cost FROM holdings ORDER BY symbol"
            ).fetchall()
        return [
            Holding(symbol=r["symbol"], quantity=r["quantity"], avg_cost=r["avg_cost"])
            for r in rows
        ]

    def recent_audit(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- writes ------------------------------------------------------------
    def upsert_holding(self, symbol: str, quantity: int, avg_cost: float) -> None:
        """Low-level setter. The broker in step 1.7 decides the new values."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO holdings (symbol, quantity, avg_cost, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(symbol) DO UPDATE SET "
                "quantity = excluded.quantity, avg_cost = excluded.avg_cost, "
                "updated_at = excluded.updated_at",
                (symbol, quantity, avg_cost, utcnow().isoformat()),
            )

    def log_event(
        self,
        event: str,
        allowed: bool | None = None,
        reason: str | None = None,
        symbol: str | None = None,
        details: dict | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit_log (ts, event, allowed, reason, symbol, details) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    utcnow().isoformat(),
                    event,
                    None if allowed is None else int(allowed),
                    reason,
                    symbol,
                    json.dumps(details, default=str) if details else None,
                ),
            )
