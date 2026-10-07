import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.mcp_server.broker import BrokerSettings, PaperBroker
from src.mcp_server.models import Candle, History, Quote, ToolError, utcnow
from src.mcp_server.store import PortfolioStore

OPEN = datetime(2026, 10, 6, 6, 0, tzinfo=UTC)  # Tue 11:30 IST, market open
CLOSED = datetime(2026, 10, 6, 11, 0, tzinfo=UTC)  # Tue 16:30 IST, after close
SATURDAY = datetime(2026, 10, 10, 6, 0, tzinfo=UTC)  # Sat 11:30 IST
NEXT_DAY = datetime(2026, 10, 7, 6, 0, tzinfo=UTC)  # Wed 11:30 IST, new trading day


class FakeMarket:
    """Stands in for ResilientProvider: fixed prices, no network."""

    def __init__(self, price=1000.0, prices=None, fail=()):
        self.price, self.prices, self.fail = price, prices or {}, set(fail)

    def get_quote(self, symbol):
        if symbol in self.fail:
            return ToolError(code="NO_DATA", message="simulated outage", retryable=True)
        return Quote(
            symbol=symbol, price=self.prices.get(symbol, self.price), as_of=utcnow(), source="fake"
        )

    def get_history(self, symbol, period="1y", interval="1d"):
        candles = [
            Candle(date=utcnow(), open=1.0, high=2.0, low=0.5, close=1.5, volume=10)
            for _ in range(100)
        ]
        return History(
            symbol=symbol, interval=interval, candles=candles, as_of=utcnow(), source="fake"
        )


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip retry backoff delays so retry tests run instantly."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)


@pytest.fixture
def clocks():
    return SimpleNamespace(open=OPEN, closed=CLOSED, saturday=SATURDAY, next_day=NEXT_DAY)


@pytest.fixture
def fake_market():
    return lambda **kw: FakeMarket(**kw)


@pytest.fixture
def make_broker(tmp_path):
    """Factory: each call builds a fresh broker + temp database + fake market."""
    counter = {"n": 0}

    def _make(clock=OPEN, cash=100_000.0, price=1000.0, **cfg):
        counter["n"] += 1
        store = PortfolioStore(tmp_path / f"broker{counter['n']}.db", starting_cash=cash)
        market = FakeMarket(price)
        broker = PaperBroker(
            market,
            store,
            settings_factory=lambda: BrokerSettings(_env_file=None, **cfg),
            clock=lambda: clock,
        )
        return broker, store, market

    return _make
