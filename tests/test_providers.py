import pandas as pd
import pytest

from src.mcp_server.providers import yfinance_provider as yfp
from src.mcp_server.providers.base import ProviderError, normalise_symbol
from src.mcp_server.providers.yfinance_provider import YFinanceProvider, _clean

COLS = ["Open", "High", "Low", "Close", "Volume"]
INFO = {
    "trailingPE": 21.14,
    "marketCap": 1.58e13,
    "debtToEquity": 46.278,
    "returnOnEquity": 0.12,
    "fiftyTwoWeekHigh": 1611.8,
    "fiftyTwoWeekLow": 1160.8,
}


def make_df(closes):
    closes = [float(c) for c in closes]
    idx = pd.date_range("2026-09-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=idx,
    )


@pytest.fixture
def fake_yf(monkeypatch):
    """Replace yfinance.Ticker. Edit the returned dict to script each test."""
    cfg = {
        "fast_info": {
            "last_price": 1167.7,
            "previous_close": 1187.0,
            "year_high": 1500.0,
            "year_low": 1000.0,
        },
        "history": make_df([100, 110]),
        "info": dict(INFO),
    }

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def fast_info(self):
            if cfg["fast_info"] is None:
                raise RuntimeError("fast_info unavailable")
            return cfg["fast_info"]

        def history(self, period="1y", interval="1d"):
            return cfg["history"]

        @property
        def info(self):
            return cfg["info"]

    monkeypatch.setattr(yfp.yf, "Ticker", FakeTicker)
    return cfg


# ---- helpers ---------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("RELIANCE", "RELIANCE.NS"),
        ("reliance", "RELIANCE.NS"),
        ("  tcs ", "TCS.NS"),
        ("TCS.NS", "TCS.NS"),
        ("TCS.BO", "TCS.BO"),
        ("^NSEI", "^NSEI"),
    ],
)
def test_normalise_symbol(raw, expected):
    assert normalise_symbol(raw) == expected


def test_normalise_symbol_rejects_empty():
    with pytest.raises(ProviderError) as e:
        normalise_symbol("   ")
    assert e.value.code == "BAD_SYMBOL" and not e.value.retryable


def test_clean_keeps_zero_but_drops_missing():
    assert _clean(0) == 0.0 and _clean(0) is not None  # zero is a real value
    assert _clean(None) is None
    assert _clean(float("nan")) is None
    assert _clean("abc") is None
    assert _clean("5") == 5.0


# ---- quote -----------------------------------------------------------------
def test_quote_uses_fast_info_and_rounds(fake_yf):
    fake_yf["fast_info"]["last_price"] = 1215.0999755859375
    q = YFinanceProvider().get_quote("RELIANCE")
    assert q.symbol == "RELIANCE.NS" and q.price == 1215.1
    assert q.prev_close == 1187.0 and q.source == "yfinance"


def test_quote_falls_back_to_history_when_fast_info_fails(fake_yf):
    fake_yf["fast_info"] = None
    q = YFinanceProvider().get_quote("TCS")
    assert q.price == 110.0 and q.prev_close == 100.0


def test_quote_falls_back_when_price_is_nan(fake_yf):
    fake_yf["fast_info"]["last_price"] = float("nan")
    assert YFinanceProvider().get_quote("TCS").price == 110.0


def test_quote_no_data_raises_retryable(fake_yf):
    fake_yf["fast_info"] = None
    fake_yf["history"] = pd.DataFrame(columns=COLS)
    with pytest.raises(ProviderError) as e:
        YFinanceProvider().get_quote("ZZZZ")
    assert e.value.code == "NO_DATA" and e.value.retryable


# ---- history ---------------------------------------------------------------
def test_history_drops_incomplete_candles_and_rounds(fake_yf):
    df = make_df([100.123456, 200.0, 300.0])
    df.loc[df.index[1], "High"] = float("nan")  # one corrupted candle
    fake_yf["history"] = df
    h = YFinanceProvider().get_history("TCS")
    assert len(h.candles) == 2 and h.candles[0].close == 100.12
    assert h.symbol == "TCS.NS" and h.source == "yfinance"


def test_history_empty_raises_no_data(fake_yf):
    fake_yf["history"] = pd.DataFrame(columns=COLS)
    with pytest.raises(ProviderError) as e:
        YFinanceProvider().get_history("TCS")
    assert e.value.code == "NO_DATA"


# ---- fundamentals ----------------------------------------------------------
def test_fundamentals_converts_debt_to_equity_to_ratio(fake_yf):
    f = YFinanceProvider().get_fundamentals("RELIANCE")
    assert f.debt_to_equity == pytest.approx(0.46278)  # yfinance gives a percentage
    assert f.pe == 21.14 and f.missing_fields == []


def test_fundamentals_missing_field_is_none_and_listed(fake_yf):
    del fake_yf["info"]["returnOnEquity"]
    f = YFinanceProvider().get_fundamentals("RELIANCE")
    assert f.roe is None and f.missing_fields == ["roe"]


def test_fundamentals_nan_counts_as_missing(fake_yf):
    fake_yf["info"]["trailingPE"] = float("nan")
    f = YFinanceProvider().get_fundamentals("RELIANCE")
    assert f.pe is None and "pe" in f.missing_fields


def test_fundamentals_52_week_range_falls_back_to_fast_info(fake_yf):
    del fake_yf["info"]["fiftyTwoWeekHigh"], fake_yf["info"]["fiftyTwoWeekLow"]
    f = YFinanceProvider().get_fundamentals("RELIANCE")
    assert f.week52_high == 1500.0 and f.week52_low == 1000.0


def test_fundamentals_empty_info_is_retryable_no_data(fake_yf):
    # Documents current behaviour: yfinance can't tell "no such symbol" from an outage,
    # so empty data stays retryable instead of being cached as a success.
    fake_yf["info"], fake_yf["fast_info"] = {}, None
    with pytest.raises(ProviderError) as e:
        YFinanceProvider().get_fundamentals("ZZZZ")
    assert e.value.code == "NO_DATA" and e.value.retryable
