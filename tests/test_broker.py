import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from src.mcp_server.broker import BrokerSettings, PaperBroker, is_market_open, ist_day_window
from src.mcp_server.models import ToolError

SETTING_ENV = [
    "TRADING_ENABLED",
    "PAPER_MODE",
    "MAX_TRADE_INR",
    "DAILY_CAP_INR",
    "PRICE_BAND_PCT",
    "ALLOWED_SYMBOLS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Shell environment variables must never change a test result."""
    for name in SETTING_ENV:
        monkeypatch.delenv(name, raising=False)


# ---- helpers ---------------------------------------------------------------
def order(b, key, qty=10, price=1000.0, sym="RELIANCE", side="BUY"):
    return b.place_order(sym, side, qty, price, key)


def snapshot(store):
    with store._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    holdings = [(h.symbol, h.quantity, h.avg_cost) for h in store.get_holdings()]
    return store.get_cash(), holdings, n


def expect_blocked(store, code, call):
    """Run `call`, then prove it was blocked, changed nothing and was audited."""
    before = snapshot(store)
    res = call()
    assert res.ok is False and res.status == "BLOCKED"
    assert res.block_code == code, res.message
    assert res.order_id is None and res.message
    assert snapshot(store) == before, "a blocked order must not change any state"
    last = store.recent_audit(1)[0]
    assert (last["event"], last["reason"], last["allowed"]) == ("ORDER_BLOCKED", code, 0)
    return res


def broker_for(store, market, clock, **cfg):
    """A second broker sharing the same store, with its own clock and settings."""
    return PaperBroker(
        market,
        store,
        clock=lambda: clock,
        settings_factory=lambda: BrokerSettings(_env_file=None, **cfg),
    )


# ---- fills -----------------------------------------------------------------
def test_buy_fills_at_live_price_and_updates_state(make_broker):
    b, s, _m = make_broker()
    r = order(b, "key-0001")
    assert r.ok and r.status == "FILLED" and r.mode == "paper" and not r.idempotent_replay
    assert r.order_id.startswith("ORD-") and r.fill_price == 1000.0
    assert r.notional == 10000.0 and r.cash_after == 90000.0
    assert s.get_cash() == 90000.0
    h = s.get_holdings()[0]
    assert (h.symbol, h.quantity, h.avg_cost) == ("RELIANCE.NS", 10, 1000.0)


def test_symbol_and_side_are_normalised(make_broker):
    b, _s, _m = make_broker()
    r = b.place_order(" reliance ", "buy", 1, 1000.0, "key-0001")
    assert r.status == "FILLED" and r.symbol == "RELIANCE.NS" and r.side == "BUY"


def test_second_buy_uses_weighted_average_cost(make_broker):
    b, s, m = make_broker()
    order(b, "key-0001")
    m.price = 1100.0
    order(b, "key-0002", price=1100.0)
    h = s.get_holdings()[0]
    assert h.quantity == 20 and h.avg_cost == pytest.approx(1050.0)
    assert s.get_cash() == 100_000 - 10_000 - 11_000


def test_sell_reduces_holding_and_credits_cash(make_broker):
    b, s, _m = make_broker()
    order(b, "key-0001")
    r = order(b, "key-0002", qty=4, side="SELL")
    assert r.status == "FILLED" and r.side == "SELL" and r.cash_after == 94000.0
    h = s.get_holdings()[0]
    assert h.quantity == 6 and h.avg_cost == 1000.0  # sells never change average cost


def test_selling_everything_removes_the_holding(make_broker):
    b, s, _m = make_broker()
    order(b, "key-0001")
    order(b, "key-0002", qty=10, side="SELL")
    assert s.get_holdings() == [] and s.get_cash() == 100_000.0


def test_sell_at_a_higher_price_books_the_gain(make_broker):
    b, s, m = make_broker()
    order(b, "key-0001")
    m.price = 1100.0
    order(b, "key-0002", price=1100.0, side="SELL")
    assert s.get_cash() == 101_000.0


# ---- one test per limit ----------------------------------------------------
def test_kill_switch_blocks(make_broker):
    b, s, _m = make_broker(trading_enabled=False)
    expect_blocked(s, "TRADING_DISABLED", lambda: order(b, "key-0001"))


def test_kill_switch_is_re_read_for_every_order(tmp_path, make_broker):
    env = tmp_path / ".env"
    env.write_text("TRADING_ENABLED=true\n")
    b, s, _m = make_broker()
    b.settings_factory = lambda: BrokerSettings(_env_file=env)
    assert order(b, "key-0001", qty=1).status == "FILLED"
    env.write_text("TRADING_ENABLED=false\n")  # flipped without a restart
    expect_blocked(s, "TRADING_DISABLED", lambda: order(b, "key-0002", qty=1))


def test_live_mode_is_refused(make_broker):
    b, s, _m = make_broker(paper_mode=False)
    expect_blocked(s, "LIVE_MODE_NOT_SUPPORTED", lambda: order(b, "key-0001"))


def test_symbol_not_on_allowlist_blocks(make_broker):
    b, s, _m = make_broker()
    expect_blocked(s, "SYMBOL_NOT_ALLOWED", lambda: order(b, "key-0001", sym="ZOMATO"))


def test_custom_allowlist_is_respected(make_broker):
    b, s, _m = make_broker(allowed_symbols="TCS")
    expect_blocked(s, "SYMBOL_NOT_ALLOWED", lambda: order(b, "key-0001", sym="RELIANCE"))
    assert order(b, "key-0002", sym="TCS").status == "FILLED"


def test_market_closed_in_the_evening_blocks_and_is_retryable(make_broker, clocks):
    b, s, _m = make_broker(clock=clocks.closed)
    res = expect_blocked(s, "MARKET_CLOSED", lambda: order(b, "key-0001"))
    assert res.retryable


def test_market_closed_on_saturday_blocks(make_broker, clocks):
    b, s, _m = make_broker(clock=clocks.saturday)
    expect_blocked(s, "MARKET_CLOSED", lambda: order(b, "key-0001"))


def test_per_trade_cap_blocks_but_exactly_at_cap_is_allowed(make_broker):
    b, s, _m = make_broker()
    expect_blocked(s, "PER_TRADE_CAP", lambda: order(b, "key-0001", qty=21))  # 21,000
    assert order(b, "key-0002", qty=20).status == "FILLED"  # 20,000


def test_daily_cap_blocks_but_exactly_at_cap_is_allowed(make_broker):
    b, s, _m = make_broker()
    for i, qty in enumerate((20, 20, 10)):  # 20k + 20k + 10k = 50k
        assert order(b, f"cap-{i:04d}", qty=qty).status == "FILLED"
    expect_blocked(s, "DAILY_CAP", lambda: order(b, "cap-9999", qty=1))


def test_daily_cap_resets_on_the_next_ist_day(make_broker, clocks):
    b, s, m = make_broker()
    for i, qty in enumerate((20, 20, 10)):
        order(b, f"cap-{i:04d}", qty=qty)
    expect_blocked(s, "DAILY_CAP", lambda: order(b, "cap-9999", qty=1))
    tomorrow = broker_for(s, m, clocks.next_day)
    assert order(tomorrow, "cap-0100", qty=1).status == "FILLED"


def test_price_band_blocks_prices_far_from_live(make_broker):
    b, s, _m = make_broker()
    expect_blocked(s, "PRICE_BAND", lambda: order(b, "key-0001", price=1100.0))  # +10%
    expect_blocked(s, "PRICE_BAND", lambda: order(b, "key-0002", price=900.0))  # -10%


def test_price_within_band_fills_at_the_live_price_not_the_agents_price(make_broker):
    b, _s, _m = make_broker()
    r = order(b, "key-0001", price=1015.0)  # 1.5% away
    assert r.status == "FILLED" and r.fill_price == 1000.0


def test_price_band_is_configurable(make_broker):
    b, _s, _m = make_broker(price_band_pct=10.0)
    assert order(b, "key-0001", price=1050.0).status == "FILLED"


def test_insufficient_cash_blocks(make_broker):
    b, s, _m = make_broker(cash=5000.0)
    expect_blocked(s, "INSUFFICIENT_CASH", lambda: order(b, "key-0001"))


def test_cannot_sell_what_is_not_held(make_broker):
    b, s, _m = make_broker()
    expect_blocked(s, "INSUFFICIENT_HOLDINGS", lambda: order(b, "key-0001", qty=5, side="SELL"))


def test_cannot_sell_more_than_held(make_broker):
    b, s, _m = make_broker()
    order(b, "key-0001")
    expect_blocked(s, "INSUFFICIENT_HOLDINGS", lambda: order(b, "key-0002", qty=11, side="SELL"))


def test_missing_live_price_blocks_and_is_retryable(make_broker):
    b, s, m = make_broker()
    m.fail.add("RELIANCE.NS")
    res = expect_blocked(s, "PRICE_UNAVAILABLE", lambda: order(b, "key-0001"))
    assert res.retryable


@pytest.mark.parametrize("case", ["kill", "allowlist", "closed"])
def test_cheap_checks_run_before_any_price_lookup(make_broker, clocks, case):
    cfg = {"kill": {"trading_enabled": False}, "allowlist": {}, "closed": {}}[case]
    clock = clocks.closed if case == "closed" else clocks.open
    b, _s, m = make_broker(clock=clock, **cfg)
    calls = []
    m.get_quote = lambda symbol: calls.append(symbol)
    res = order(b, "key-0001", sym="ZOMATO" if case == "allowlist" else "RELIANCE")
    assert res.status == "BLOCKED" and calls == []


# ---- bad arguments ---------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        ("RELIANCE", "BUY", 0, 1000.0, "bad-00001"),  # zero quantity
        ("RELIANCE", "BUY", -3, 1000.0, "bad-00002"),  # negative quantity
        ("RELIANCE", "BUY", True, 1000.0, "bad-00003"),  # bool is not a quantity
        ("RELIANCE", "BUY", "5", 1000.0, "bad-00004"),  # string quantity
        ("RELIANCE", "HOLD", 1, 1000.0, "bad-00005"),  # unknown side
        ("RELIANCE", "BUY", 1, -5.0, "bad-00006"),  # negative price
        ("RELIANCE", "BUY", 1, 0, "bad-00007"),  # zero price
        ("RELIANCE", "BUY", 1, float("nan"), "bad-00008"),  # NaN price
        ("RELIANCE", "BUY", 1, "abc", "bad-00009"),  # non-numeric price
        ("   ", "BUY", 1, 1000.0, "bad-00010"),  # empty symbol
        ("RELIANCE", "BUY", 1, 1000.0, "short"),  # key too short
        ("RELIANCE", "BUY", 1, 1000.0, "k" * 65),  # key too long
        ("RELIANCE", "BUY", 1, 1000.0, None),  # missing key
    ],
)
def test_bad_arguments_are_blocked(make_broker, args):
    b, s, _m = make_broker()
    expect_blocked(s, "BAD_ARGUMENT", lambda: b.place_order(*args))


# ---- idempotency -----------------------------------------------------------
def test_replay_returns_the_original_order_without_a_second_fill(make_broker):
    b, s, _m = make_broker()
    first = order(b, "key-0001")
    again = order(b, "key-0001")
    assert again.idempotent_replay and again.status == "FILLED"
    assert again.order_id == first.order_id
    assert snapshot(s) == (90_000.0, [("RELIANCE.NS", 10, 1000.0)], 1)


def test_key_is_trimmed_before_use(make_broker):
    b, _s, _m = make_broker()
    first = order(b, "key-0001")
    assert order(b, "  key-0001  ").order_id == first.order_id


def test_replay_still_works_after_the_market_closes_or_the_kill_switch_trips(make_broker, clocks):
    b, s, m = make_broker()
    first = order(b, "key-0001")
    late = broker_for(s, m, clocks.closed)
    killed = broker_for(s, m, clocks.open, trading_enabled=False)
    for other in (late, killed):
        again = order(other, "key-0001")
        assert again.idempotent_replay and again.order_id == first.order_id
    assert snapshot(s)[0] == 90_000.0


@pytest.mark.parametrize("change", [{"qty": 5}, {"sym": "TCS"}, {"side": "SELL"}])
def test_same_key_with_a_different_order_is_blocked(make_broker, change):
    b, s, _m = make_broker()
    order(b, "key-0001")
    expect_blocked(s, "IDEMPOTENCY_KEY_REUSED", lambda: order(b, "key-0001", **change))


def test_blocked_attempt_does_not_reserve_its_key(make_broker, clocks):
    b, s, m = make_broker()
    closed = broker_for(s, m, clocks.closed)
    expect_blocked(s, "MARKET_CLOSED", lambda: order(closed, "key-0001"))
    assert order(b, "key-0001").status == "FILLED"  # same key works once open


def test_eight_simultaneous_submits_create_exactly_one_order(make_broker):
    b, s, _m = make_broker()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: order(b, "race-0001"), range(8)))
    assert all(r.status == "FILLED" for r in results)
    assert len({r.order_id for r in results}) == 1
    assert sum(r.idempotent_replay for r in results) == 7
    assert snapshot(s) == (90_000.0, [("RELIANCE.NS", 10, 1000.0)], 1)


def test_concurrent_orders_cannot_jointly_exceed_the_daily_cap(make_broker):
    b, s, _m = make_broker()
    with ThreadPoolExecutor(10) as pool:  # 10 x 10,000 against a 50,000 cap
        results = list(pool.map(lambda i: order(b, f"cap-{i:04d}"), range(10)))
    filled = [r for r in results if r.status == "FILLED"]
    blocked = [r for r in results if r.status == "BLOCKED"]
    assert len(filled) == 5 and len(blocked) == 5
    assert {r.block_code for r in blocked} == {"DAILY_CAP"}
    cash, holdings, n = snapshot(s)
    assert cash == 50_000.0 and n == 5 and holdings[0][1] == 50


# ---- audit trail and robustness --------------------------------------------
def test_every_attempt_is_audited(make_broker):
    b, s, _m = make_broker()
    order(b, "key-0001")  # FILLED
    order(b, "key-0001")  # REPLAY
    order(b, "key-0002", qty=25)  # BLOCKED: per-trade cap
    order(b, "key-0001", qty=5)  # BLOCKED: key reuse
    events = [e["event"] for e in s.recent_audit(50)]
    assert sorted(events) == ["ORDER_BLOCKED", "ORDER_BLOCKED", "ORDER_FILLED", "ORDER_REPLAY"]


def test_blocked_audit_row_keeps_the_request_and_message(make_broker):
    b, s, _m = make_broker()
    order(b, "key-0001", qty=25)
    row = s.recent_audit(1)[0]
    details = json.loads(row["details"])
    assert row["symbol"] == "RELIANCE.NS" and row["reason"] == "PER_TRADE_CAP"
    assert details["quantity"] == 25 and "per-trade cap" in details["message"]


def test_unexpected_store_error_becomes_a_blocked_result(make_broker, monkeypatch):
    b, s, _m = make_broker()

    def boom(**kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(s, "fill_order", boom)
    res = expect_blocked(s, "INTERNAL_ERROR", lambda: order(b, "key-0001"))
    assert "disk on fire" in res.message


def test_get_order_status(make_broker):
    b, _s, _m = make_broker()
    placed = order(b, "key-0001")
    found = b.get_order_status(placed.order_id)
    assert found.status == "FILLED" and found.quantity == 10 and found.fill_price == 1000.0
    missing = b.get_order_status("ORD-NOPE")
    assert isinstance(missing, ToolError) and missing.code == "ORDER_NOT_FOUND"


def test_blocked_orders_have_no_order_status(make_broker):
    b, _s, _m = make_broker()
    res = order(b, "key-0001", qty=25)
    assert res.order_id is None
    assert isinstance(b.get_order_status(""), ToolError)


# ---- settings and time helpers ---------------------------------------------
def test_settings_defaults_are_conservative():
    cfg = BrokerSettings(_env_file=None)
    assert cfg.trading_enabled and cfg.paper_mode
    assert (cfg.max_trade_inr, cfg.daily_cap_inr, cfg.price_band_pct) == (20_000, 50_000, 2.0)
    assert "RELIANCE.NS" in cfg.allowlist


def test_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv("MAX_TRADE_INR", "123.5")
    monkeypatch.setenv("TRADING_ENABLED", "false")
    cfg = BrokerSettings(_env_file=None)
    assert cfg.max_trade_inr == 123.5 and cfg.trading_enabled is False


def test_allowlist_is_normalised():
    cfg = BrokerSettings(_env_file=None, allowed_symbols="tcs, infy.ns ,")
    assert cfg.allowlist == {"TCS.NS", "INFY.NS"}


@pytest.mark.parametrize(
    "when, expected",
    [
        (datetime(2026, 10, 6, 3, 44, tzinfo=UTC), False),  # 09:14 IST
        (datetime(2026, 10, 6, 3, 45, tzinfo=UTC), True),  # 09:15 IST, open
        (datetime(2026, 10, 6, 10, 0, tzinfo=UTC), True),  # 15:30 IST, last minute
        (datetime(2026, 10, 6, 10, 1, tzinfo=UTC), False),  # 15:31 IST
        (datetime(2026, 10, 9, 6, 0, tzinfo=UTC), True),  # Friday
        (datetime(2026, 10, 10, 6, 0, tzinfo=UTC), False),  # Saturday
        (datetime(2026, 10, 11, 6, 0, tzinfo=UTC), False),  # Sunday
    ],
)
def test_market_hours_boundaries(when, expected):
    assert is_market_open(when) is expected


def test_ist_day_window_runs_from_ist_midnight_to_midnight(clocks):
    start, end = ist_day_window(clocks.open)
    assert (start, end) == ("2026-10-05T18:30:00+00:00", "2026-10-06T18:30:00+00:00")
