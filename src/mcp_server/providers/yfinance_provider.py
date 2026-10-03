import logging
import math

import yfinance as yf

from ..models import Candle, Fundamentals, History, Quote, utcnow
from .base import MarketDataProvider, ProviderError, normalise_symbol

log = logging.getLogger(__name__)

SOURCE = "yfinance"


def _clean(x):
    """NaN/None -> None. Never turn 'missing' into 0."""
    if x is None:
        return None
    try:
        return None if math.isnan(float(x)) else float(x)
    except (TypeError, ValueError):
        return None


class YFinanceProvider(MarketDataProvider):
    def get_quote(self, symbol: str) -> Quote:
        sym = normalise_symbol(symbol)
        t = yf.Ticker(sym)

        price, prev = None, None
        try:
            fi = t.fast_info
            price = _clean(fi["last_price"])
            prev = _clean(fi["previous_close"])
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            log.debug("fast_info unavailable for %s, falling back to history: %s", symbol, e)

        if price is None:
            df = t.history(period="5d", interval="1d").dropna(subset=["Close"])
            if not df.empty:
                price = float(df["Close"].iloc[-1])
                if len(df) > 1:
                    prev = float(df["Close"].iloc[-2])

        if price is None:
            raise ProviderError("NO_DATA", f"No price for {sym}", retryable=True)

        return Quote(symbol=sym, price=price, prev_close=prev, as_of=utcnow(), source=SOURCE)

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> History:
        sym = normalise_symbol(symbol)
        df = yf.Ticker(sym).history(period=period, interval=interval)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if df.empty:
            raise ProviderError("NO_DATA", f"No history for {sym}", retryable=True)
        candles = [
            Candle(
                date=idx.to_pydatetime(),
                open=float(r.Open),
                high=float(r.High),
                low=float(r.Low),
                close=float(r.Close),
                volume=int(r.Volume) if _clean(r.Volume) is not None else None,
            )
            for idx, r in df.iterrows()
        ]
        return History(
            symbol=sym, interval=interval, candles=candles, as_of=utcnow(), source=SOURCE
        )

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Units: debt_to_equity is a plain ratio (0.46 = 46%), roe is a fraction
        (0.12 = 12%), market_cap is in INR, 52-week values are prices."""
        sym = normalise_symbol(symbol)
        t = yf.Ticker(sym)
        info = t.info or {}

        hi = _clean(info.get("fiftyTwoWeekHigh"))
        lo = _clean(info.get("fiftyTwoWeekLow"))
        if hi is None or lo is None:
            try:
                fi = t.fast_info
                hi = hi if hi is not None else _clean(fi["year_high"])
                lo = lo if lo is not None else _clean(fi["year_low"])
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                log.debug("fast_info unavailable for %s, falling back to history: %s", symbol, e)

        de = _clean(info.get("debtToEquity"))
        fields = {
            "pe": _clean(info.get("trailingPE")),
            "market_cap": _clean(info.get("marketCap")),
            "debt_to_equity": de / 100 if de is not None else None,  # yfinance gives percent
            "roe": _clean(info.get("returnOnEquity")),
            "week52_high": hi,
            "week52_low": lo,
        }
        if all(v is None for v in fields.values()):
            raise ProviderError("NO_DATA", f"No fundamentals for {sym}", retryable=True)

        missing = [k for k, v in fields.items() if v is None]
        return Fundamentals(
            symbol=sym, missing_fields=missing, as_of=utcnow(), source=SOURCE, **fields
        )
