import time

import pytest

from src.mcp_server.models import Quote, ToolError, utcnow
from src.mcp_server.providers.base import MarketDataProvider, ProviderError
from src.mcp_server.resilience import ResilientProvider, TTLCache, call_with_timeout, with_retries


class ScriptedProvider(MarketDataProvider):
    """Raises the scripted errors one by one, then returns a quote."""

    def __init__(self, errors=(), delay=0.0):
        self.errors, self.calls, self.delay = list(errors), 0, delay

    def get_quote(self, symbol):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.errors:
            raise self.errors.pop(0)
        return Quote(symbol=symbol, price=100.0, as_of=utcnow(), source="fake")

    def get_history(self, *a, **k):
        raise NotImplementedError

    def get_fundamentals(self, *a, **k):
        raise NotImplementedError


def retryable(code="NO_DATA"):
    return ProviderError(code, "simulated", retryable=True)


# ---- TTLCache --------------------------------------------------------------
def test_cache_returns_value_then_expires():
    c = TTLCache()
    c.set("k", "v", ttl=0.05)
    assert c.get("k") == "v"
    time.sleep(0.1)
    assert c.get("k") is None


def test_cache_miss_is_none():
    assert TTLCache().get("nope") is None


# ---- timeout and retries ---------------------------------------------------
def test_call_with_timeout_raises_retryable_timeout():
    with pytest.raises(ProviderError) as e:
        call_with_timeout(lambda: time.sleep(0.5), 0.05)
    assert e.value.code == "TIMEOUT" and e.value.retryable


def test_with_retries_recovers_after_transient_failures(no_sleep):
    p = ScriptedProvider([retryable(), retryable()])
    assert with_retries(lambda: p.get_quote("X"), attempts=3, base_delay=0).price == 100.0
    assert p.calls == 3


def test_with_retries_gives_up_after_all_attempts(no_sleep):
    p = ScriptedProvider([retryable()] * 5)
    with pytest.raises(ProviderError):
        with_retries(lambda: p.get_quote("X"), attempts=3, base_delay=0)
    assert p.calls == 3


def test_with_retries_does_not_retry_permanent_errors(no_sleep):
    p = ScriptedProvider([ProviderError("BAD_SYMBOL", "nope", retryable=False)] * 3)
    with pytest.raises(ProviderError) as e:
        with_retries(lambda: p.get_quote("X"), attempts=3, base_delay=0)
    assert e.value.code == "BAD_SYMBOL" and p.calls == 1


def test_with_retries_wraps_unknown_exceptions(no_sleep):
    p = ScriptedProvider([RuntimeError("boom")] * 3)
    with pytest.raises(ProviderError) as e:
        with_retries(lambda: p.get_quote("X"), attempts=2, base_delay=0)
    assert e.value.code == "UPSTREAM_ERROR" and e.value.retryable and p.calls == 2


# ---- ResilientProvider -----------------------------------------------------
def make(inner, **kw):
    return ResilientProvider(inner, timeout_s=2, attempts=kw.pop("attempts", 3), base_delay=0, **kw)


def test_success_is_cached(no_sleep):
    inner = ScriptedProvider()
    rp = make(inner)
    rp.get_quote("RELIANCE"), rp.get_quote("reliance")  # same normalised key
    assert inner.calls == 1


def test_retries_then_returns_data(no_sleep):
    inner = ScriptedProvider([retryable(), retryable()])
    assert isinstance(make(inner).get_quote("TCS"), Quote) and inner.calls == 3


def test_failure_becomes_toolerror_and_is_not_cached(no_sleep):
    inner = ScriptedProvider([retryable()])
    rp = make(inner, attempts=1)
    first = rp.get_quote("TCS")
    assert isinstance(first, ToolError) and first.code == "NO_DATA" and first.retryable
    assert isinstance(rp.get_quote("TCS"), Quote)  # not poisoned by the failure
    assert inner.calls == 2


def test_permanent_error_returned_without_retry(no_sleep):
    inner = ScriptedProvider([ProviderError("BAD_SYMBOL", "x", retryable=False)])
    res = make(inner).get_quote("TCS")
    assert isinstance(res, ToolError) and not res.retryable and inner.calls == 1


def test_empty_symbol_never_reaches_provider(no_sleep):
    inner = ScriptedProvider()
    res = make(inner).get_quote("   ")
    assert isinstance(res, ToolError) and res.code == "BAD_SYMBOL" and inner.calls == 0


def test_slow_provider_times_out_as_toolerror():
    inner = ScriptedProvider(delay=0.3)
    rp = ResilientProvider(inner, timeout_s=0.05, attempts=1, base_delay=0)
    res = rp.get_quote("TCS")
    assert isinstance(res, ToolError) and res.code == "TIMEOUT" and res.retryable
