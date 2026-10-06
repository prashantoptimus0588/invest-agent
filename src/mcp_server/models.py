from datetime import UTC, datetime

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class ToolError(BaseModel):
    """Structured error returned to the LLM instead of raising."""

    ok: bool = False
    code: str  # e.g. "NO_DATA", "TIMEOUT", "UPSTREAM_ERROR", "BAD_SYMBOL"
    message: str
    retryable: bool = False


class Quote(BaseModel):
    symbol: str  # normalised, e.g. RELIANCE.NS
    price: float
    currency: str = "INR"
    prev_close: float | None = None
    as_of: datetime
    source: str


class Candle(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int | None = None


class History(BaseModel):
    symbol: str
    interval: str
    candles: list[Candle]
    as_of: datetime
    source: str


class Fundamentals(BaseModel):
    symbol: str
    pe: float | None = None
    market_cap: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    missing_fields: list[str] = Field(default_factory=list)
    as_of: datetime
    source: str


class NewsItem(BaseModel):
    title: str
    link: str
    published: datetime | None = None
    source: str  # e.g. "Google News", "ET Markets"
    summary: str = ""
    untrusted: bool = True  # always True: this text came from the outside world


class NewsResult(BaseModel):
    query: str
    items: list[NewsItem]
    feeds_failed: list[str] = Field(default_factory=list)
    as_of: datetime
    source: str = "rss"
    untrusted: bool = True


class Holding(BaseModel):
    symbol: str
    quantity: int
    avg_cost: float
    last_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    weight_pct: float | None = None  # share of total equity


class Portfolio(BaseModel):
    mode: str = "paper"
    cash: float
    holdings: list[Holding]
    cost_basis: float
    market_value: float | None  # holdings only; None if any price is missing
    total_equity: float | None  # cash + holdings; None if any price is missing
    valuation_complete: bool
    as_of: datetime
    source: str = "paper_portfolio"
