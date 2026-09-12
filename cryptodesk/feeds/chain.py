"""Fallback chain over several feeds."""

from __future__ import annotations

import logging

from .base import Candle, Feed, FeedError

logger = logging.getLogger(__name__)


class ChainFeed:
    """Try each feed in order; return the first success.

    A venue outage is routine, and a desk that stops trading because one REST
    endpoint returned 502 is not a 24/7 desk. Every fallback is logged at
    WARNING so a feed that is *always* failing is visible rather than silently
    masked by the next one.
    """

    def __init__(self, feeds: list[Feed]):
        if not feeds:
            raise FeedError("ChainFeed requires at least one feed")
        self.feeds = feeds
        self.name = "chain(" + ",".join(f.name for f in feeds) + ")"
        self.last_used: str | None = None

    def _attempt(self, method: str, symbol: str, *args, **kwargs):
        errors = []
        for feed in self.feeds:
            try:
                result = getattr(feed, method)(symbol, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - any venue error means "try next"
                errors.append(f"{feed.name}: {exc}")
                logger.warning("Feed %s failed %s(%s): %s", feed.name, method, symbol, exc)
                continue
            self.last_used = feed.name
            return result
        raise FeedError(f"All feeds failed for {symbol}: {'; '.join(errors)}")

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[Candle]:
        return self._attempt("candles", symbol, interval, limit)

    def price(self, symbol: str) -> float:
        return self._attempt("price", symbol)
