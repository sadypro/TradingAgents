"""Feed protocol and the candle type every feed returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class FeedError(RuntimeError):
    """Raised when a feed cannot supply data for a symbol.

    Distinct from a generic exception so the chain feed can tell "this venue
    failed, try the next" apart from a programming error, which should
    propagate.
    """


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar. ``ts`` is a UTC epoch in seconds, at the bar's open."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@runtime_checkable
class Feed(Protocol):
    """Minimum surface the desk needs from a market data source."""

    name: str

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[Candle]:
        """Return up to ``limit`` closed candles, oldest first."""
        ...

    def price(self, symbol: str) -> float:
        """Return the most recent traded price for ``symbol``."""
        ...


def split_symbol(symbol: str) -> tuple[str, str]:
    """Split ``BTC-USD`` into ``("BTC", "USD")``.

    Accepts the dashless broker form (``BTCUSD``) too, since users paste both
    and a silently wrong symbol is the worst possible failure mode for a feed.
    """
    s = symbol.strip().upper()
    if "-" in s:
        base, _, quote = s.partition("-")
        if base and quote:
            return base, quote
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)], quote
    raise FeedError(f"Cannot parse symbol {symbol!r}; expected a form like BTC-USD")
