import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from typing import TypeVar

from .models import Fundamentals, History, Quote, ToolError
from .providers.base import MarketDataProvider, ProviderError, normalise_symbol

log = logging.getLogger(__name__)

T = TypeVar("T")

# Seconds. Move these into config.py later.
TTL = {"quote": 30, "history": 15 * 60, "fundamentals": 6 * 3600}


class TTLCache:
    """Tiny thread-safe in-memory cache. Redis replaces this in Phase 8."""

    def __init__(self):
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if time.monotonic() > expires:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value, ttl: float) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, value)


_pool = ThreadPoolExecutor(max_workers=8)


def call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    """yfinance has no reliable timeout, so we enforce one from outside."""
    fut = _pool.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except FutTimeout:
        raise ProviderError("TIMEOUT", f"No response within {timeout_s}s", retryable=True)


def with_retries(
    fn: Callable[[], T], attempts: int = 3, base_delay: float = 0.5, timeout_s: float = 10
) -> T:
    """Retry retryable failures with exponential backoff + jitter."""
    last: ProviderError | None = None
    for i in range(attempts):
        try:
            return call_with_timeout(fn, timeout_s)
        except ProviderError as e:
            if not e.retryable:
                raise  # e.g. BAD_SYMBOL: retrying is pointless
            last = e
        except Exception as e:
            log.warning("upstream failure (attempt %d/%d)", i + 1, attempts, exc_info=True)
            last = ProviderError("UPSTREAM_ERROR", f"{type(e).__name__}: {e}", retryable=True)
        if i < attempts - 1:
            time.sleep(base_delay * (2**i) + random.uniform(0, 0.2))
    assert last is not None
    raise last


class ResilientProvider:
    """Wraps any MarketDataProvider. Returns data OR a ToolError, never raises."""

    def __init__(
        self,
        inner: MarketDataProvider,
        cache: TTLCache | None = None,
        timeout_s: float = 10,
        attempts: int = 3,
        base_delay: float = 0.5,
    ):
        self.inner = inner
        self.cache = cache or TTLCache()
        self.timeout_s, self.attempts, self.base_delay = timeout_s, attempts, base_delay

    def _run(self, key: str, fn: Callable[[], T], ttl: float) -> T | ToolError:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        try:
            result = with_retries(fn, self.attempts, self.base_delay, self.timeout_s)
        except ProviderError as e:
            return ToolError(code=e.code, message=e.message, retryable=e.retryable)
        self.cache.set(key, result, ttl)  # errors are never cached
        return result

    def _sym(self, symbol: str) -> str | ToolError:
        try:
            return normalise_symbol(symbol)
        except ProviderError as e:
            return ToolError(code=e.code, message=e.message, retryable=False)

    def get_quote(self, symbol: str) -> Quote | ToolError:
        sym = self._sym(symbol)
        if isinstance(sym, ToolError):
            return sym
        return self._run(f"quote:{sym}", lambda: self.inner.get_quote(sym), TTL["quote"])

    def get_history(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> History | ToolError:
        sym = self._sym(symbol)
        if isinstance(sym, ToolError):
            return sym
        return self._run(
            f"history:{sym}:{period}:{interval}",
            lambda: self.inner.get_history(sym, period, interval),
            TTL["history"],
        )

    def get_fundamentals(self, symbol: str) -> Fundamentals | ToolError:
        sym = self._sym(symbol)
        if isinstance(sym, ToolError):
            return sym
        return self._run(
            f"fund:{sym}", lambda: self.inner.get_fundamentals(sym), TTL["fundamentals"]
        )
