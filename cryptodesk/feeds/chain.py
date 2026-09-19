"""Fallback chain over several feeds."""

from __future__ import annotations

import logging
import time

from .base import Candle, Feed, FeedError

logger = logging.getLogger(__name__)

# After this many consecutive failures a feed is skipped for the cooldown.
# Three is enough to separate a single 502 from an outage; 300 s keeps a
# black-holed venue from charging its timeouts to every tick for the whole
# outage while still re-probing within a handful of loop iterations.
_BREAKER_FAILURES = 3
_BREAKER_COOLDOWN_S = 300.0


class ChainFeed:
    """Try each feed in order; return the first success.

    A venue outage is routine, and a desk that stops trading because one REST
    endpoint returned 502 is not a 24/7 desk. Every fallback is logged at
    WARNING so a feed that is *always* failing is visible rather than silently
    masked by the next one, and ``health`` carries the same signal to the
    dashboard.

    Only ``FeedError`` is treated as "try the next venue". Anything else is a
    programming error (a parser that no longer matches the venue's schema, a
    typo) and masking it behind the fallback would let the primary be dead
    for days with nothing but a log line to show for it.
    """

    def __init__(self, feeds: list[Feed], clock=time.time):
        if not feeds:
            raise FeedError("ChainFeed requires at least one feed")
        self.feeds = feeds
        self.name = "chain(" + ",".join(f.name for f in feeds) + ")"
        self.last_used: str | None = None
        self._clock = clock
        self.health: dict[str, dict] = {
            f.name: {"failures": 0, "last_error": None, "last_ok_ts": None} for f in feeds
        }
        # Feed name -> clock time at which the breaker closes again.
        self._open_until: dict[str, float] = {}

    # ------------------------------------------------------------ breaker
    def _eligible(self) -> list[Feed]:
        now = self._clock()
        live = [f for f in self.feeds if self._open_until.get(f.name, 0.0) <= now]
        # The breaker exists to stop paying a dead venue's timeout while a
        # fallback works. When every venue is open there is nothing to protect
        # and probing is the only way to notice a recovery.
        return live or list(self.feeds)

    def _record_ok(self, feed: Feed) -> None:
        entry = self.health[feed.name]
        entry["failures"] = 0
        entry["last_ok_ts"] = int(self._clock())
        self._open_until.pop(feed.name, None)
        self.last_used = feed.name

    def _record_failure(self, feed: Feed, exc: FeedError) -> None:
        entry = self.health[feed.name]
        entry["failures"] += 1
        entry["last_error"] = str(exc)
        if entry["failures"] >= _BREAKER_FAILURES:
            self._open_until[feed.name] = self._clock() + _BREAKER_COOLDOWN_S

    # ------------------------------------------------------------ fetching
    def _attempt(self, method: str, symbol: str, *args, **kwargs):
        errors = []
        for feed in self._eligible():
            try:
                result = getattr(feed, method)(symbol, *args, **kwargs)
            except FeedError as exc:
                self._record_failure(feed, exc)
                errors.append(f"{feed.name}: {exc}")
                logger.warning("Feed %s failed %s(%s): %s", feed.name, method, symbol, exc)
                continue
            self._record_ok(feed)
            return result
        raise FeedError(f"All feeds failed for {symbol}: {'; '.join(errors)}")

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[Candle]:
        return self._attempt("candles", symbol, interval, limit)

    def price(self, symbol: str) -> float:
        return self._attempt("price", symbol)

    def market(self, symbol: str, interval: str = "5m",
               limit: int = 200) -> tuple[list[Candle], float, str]:
        """Candles and price from one venue, or fall through to the next.

        Falling back per call would let a transient ticker failure pair
        Crypto.com BTC_USD candles with a Binance BTCUSDT price; the basis
        between those is normally bps but has reached several percent during
        stablecoin dislocations — enough to trip a 5m ATR stop on its own.
        """
        errors = []
        for feed in self._eligible():
            try:
                candles = feed.candles(symbol, interval, limit)
                price = feed.price(symbol)
            except FeedError as exc:
                self._record_failure(feed, exc)
                errors.append(f"{feed.name}: {exc}")
                logger.warning("Feed %s failed market(%s): %s", feed.name, symbol, exc)
                continue
            self._record_ok(feed)
            return candles, price, feed.name
        raise FeedError(f"All feeds failed for {symbol}: {'; '.join(errors)}")
