import math
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta, timezone
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import OrderResult, ToolError, utcnow
from .providers.base import ProviderError, normalise_symbol
from .store import OrderBlocked, PortfolioStore

REPO_ROOT = Path(__file__).resolve().parents[2]
IST = timezone(timedelta(hours=5, minutes=30))  # India has no DST, so no tzdata needed
MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)

DEFAULT_ALLOWLIST = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,ITC.NS,SBIN.NS,LT.NS"


class BrokerSettings(BaseSettings):
    """Reads the same names as .env.example. Re-created per order, so edits to .env
    (for example TRADING_ENABLED=false) take effect on the next order."""

    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    trading_enabled: bool = True  # kill switch
    paper_mode: bool = True
    max_trade_inr: float = 20_000.0
    daily_cap_inr: float = 50_000.0
    price_band_pct: float = 2.0
    allowed_symbols: str = DEFAULT_ALLOWLIST

    @property
    def allowlist(self) -> set[str]:
        return {normalise_symbol(s) for s in self.allowed_symbols.split(",") if s.strip()}


def is_market_open(now: datetime) -> bool:
    t = now.astimezone(IST)
    return t.weekday() < 5 and MARKET_OPEN <= t.time() <= MARKET_CLOSE  # holidays not covered


def ist_day_window(now: datetime) -> tuple[str, str]:
    start = now.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()


class PaperBroker:
    def __init__(
        self,
        market,
        store: PortfolioStore,
        settings_factory: Callable[[], BrokerSettings] = BrokerSettings,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.market, self.store = market, store
        self.settings_factory, self.clock = settings_factory, clock

    # ---- helpers -----------------------------------------------------------
    def _block(
        self, code: str, message: str, req: dict, retryable: bool = False, **echo
    ) -> OrderResult:
        self.store.log_event(
            "ORDER_BLOCKED",
            allowed=False,
            reason=code,
            symbol=echo.get("symbol"),
            details={**req, "message": message},
        )
        return OrderResult(
            ok=False,
            status="BLOCKED",
            block_code=code,
            message=message,
            retryable=retryable,
            as_of=utcnow(),
            idempotency_key=req.get("idempotency_key")
            if isinstance(req.get("idempotency_key"), str)
            else None,
            **echo,
        )

    @staticmethod
    def _from_row(row: dict, replay: bool = False, cash_after: float | None = None) -> OrderResult:
        return OrderResult(
            ok=True,
            status=row["status"],
            order_id=row["order_id"],
            idempotency_key=row["idempotency_key"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=row["quantity"],
            fill_price=row["fill_price"],
            notional=round(row["quantity"] * row["fill_price"], 2),
            cash_after=cash_after,
            idempotent_replay=replay,
            as_of=datetime.fromisoformat(row["created_at"]),
        )

    # ---- the order pipeline ------------------------------------------------
    def place_order(
        self, symbol: str, side: str, quantity: int, limit_price: float, idempotency_key: str
    ) -> OrderResult:
        cfg, now = self.settings_factory(), self.clock()
        req = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "limit_price": limit_price,
            "idempotency_key": idempotency_key,
        }

        # 0. validate arguments
        try:
            sym = normalise_symbol(symbol)
        except ProviderError as e:
            return self._block("BAD_ARGUMENT", e.message, req)
        side_u = str(side).strip().upper()
        if side_u not in ("BUY", "SELL"):
            return self._block("BAD_ARGUMENT", "side must be BUY or SELL", req, symbol=sym)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            return self._block(
                "BAD_ARGUMENT", "quantity must be a positive integer", req, symbol=sym, side=side_u
            )
        try:
            lp = float(limit_price)
        except (TypeError, ValueError):
            lp = float("nan")
        if not math.isfinite(lp) or lp <= 0:
            return self._block(
                "BAD_ARGUMENT",
                "limit_price must be a positive number",
                req,
                symbol=sym,
                side=side_u,
                quantity=quantity,
            )
        key = (idempotency_key or "").strip() if isinstance(idempotency_key, str) else ""
        if not 8 <= len(key) <= 64:
            return self._block(
                "BAD_ARGUMENT",
                "idempotency_key must be 8 to 64 characters",
                req,
                symbol=sym,
                side=side_u,
                quantity=quantity,
            )
        echo = {"symbol": sym, "side": side_u, "quantity": quantity}

        # replay: an already-filled order with this key returns its original result
        prior = self.store.get_order_by_key(key)
        if prior:
            if (prior["symbol"], prior["side"], prior["quantity"]) != (sym, side_u, quantity):
                return self._block(
                    "IDEMPOTENCY_KEY_REUSED",
                    "This key was already used for a different order",
                    req,
                    **echo,
                )
            self.store.log_event(
                "ORDER_REPLAY",
                allowed=True,
                reason="idempotent replay",
                symbol=sym,
                details={"order_id": prior["order_id"], "key": key},
            )
            return self._from_row(prior, replay=True)

        # 1-4: cheap checks first, no network
        if not cfg.trading_enabled:
            return self._block(
                "TRADING_DISABLED", "Kill switch is on (TRADING_ENABLED=false)", req, **echo
            )
        if not cfg.paper_mode:
            return self._block(
                "LIVE_MODE_NOT_SUPPORTED", "Only paper trading is implemented", req, **echo
            )
        if sym not in cfg.allowlist:
            return self._block("SYMBOL_NOT_ALLOWED", f"{sym} is not on the allowlist", req, **echo)
        if not is_market_open(now):
            return self._block(
                "MARKET_CLOSED",
                "NSE hours are Mon-Fri 09:15-15:30 IST",
                req,
                retryable=True,
                **echo,
            )

        # live price (cached, retried, timed out via the resilience layer)
        q = self.market.get_quote(sym)
        if isinstance(q, ToolError):
            return self._block(
                "PRICE_UNAVAILABLE", f"No live price: {q.message}", req, retryable=True, **echo
            )
        live = q.price
        notional = round(quantity * live, 2)

        # 5-7: money limits and price band
        if notional > cfg.max_trade_inr:
            return self._block(
                "PER_TRADE_CAP",
                f"Order value Rs {notional:,.2f} exceeds per-trade cap Rs {cfg.max_trade_inr:,.2f}",
                req,
                **echo,
            )
        day_start, day_end = ist_day_window(now)
        spent = self.store.traded_today(day_start, day_end)
        if spent + notional > cfg.daily_cap_inr + 1e-9:
            return self._block(
                "DAILY_CAP",
                f"Daily cap Rs {cfg.daily_cap_inr:,.2f} exceeded: traded "
                f"Rs {spent:,.2f} today, this order adds Rs {notional:,.2f}",
                req,
                **echo,
            )
        drift = abs(lp - live) / live * 100
        if drift > cfg.price_band_pct:
            return self._block(
                "PRICE_BAND",
                f"Limit price {lp} is {drift:.2f}% from live price {live} "
                f"(band {cfg.price_band_pct}%)",
                req,
                **echo,
            )

        # state-dependent checks + fill, atomically
        try:
            res = self.store.fill_order(
                order_id="ORD-" + uuid.uuid4().hex[:12].upper(),
                key=key,
                symbol=sym,
                side=side_u,
                quantity=quantity,
                limit_price=lp,
                fill_price=live,
                daily_cap=cfg.daily_cap_inr,
                day_start=day_start,
                day_end=day_end,
                created_at=now.isoformat(),
            )
        except OrderBlocked as e:
            return self._block(e.code, e.message, req, **echo)
        except Exception as e:  # noqa: BLE001 (never raise into the LLM)
            return self._block("INTERNAL_ERROR", f"{type(e).__name__}: {e}", req, **echo)

        if res["replay"]:  # lost a race to an identical request
            self.store.log_event(
                "ORDER_REPLAY",
                allowed=True,
                reason="idempotent replay (race)",
                symbol=sym,
                details={"key": key},
            )
            return self._from_row(res["order"], replay=True, cash_after=res["cash_after"])
        self.store.log_event(
            "ORDER_FILLED",
            allowed=True,
            symbol=sym,
            details={**res["order"], "cash_after": res["cash_after"]},
        )
        return self._from_row(res["order"], cash_after=res["cash_after"])

    def get_order_status(self, order_id: str) -> OrderResult | ToolError:
        row = self.store.get_order((order_id or "").strip())
        if not row:
            return ToolError(
                code="ORDER_NOT_FOUND",
                message=f"No filled order {order_id}. Blocked attempts never "
                "create orders; they appear only in the audit log.",
            )
        return self._from_row(row)
