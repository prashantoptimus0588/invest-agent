from abc import ABC, abstractmethod

from ..models import Fundamentals, History, Quote


class ProviderError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


def normalise_symbol(raw: str) -> str:
    """RELIANCE -> RELIANCE.NS ; keeps .NS/.BO and index tickers like ^NSEI."""
    s = raw.strip().upper()
    if not s:
        raise ProviderError("BAD_SYMBOL", "Empty symbol")
    if s.startswith("^") or s.endswith((".NS", ".BO")):
        return s
    return f"{s}.NS"


class MarketDataProvider(ABC):
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> History: ...

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Fundamentals: ...
