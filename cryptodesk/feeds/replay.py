"""Replay a recorded CSV of candles.

Used to drive the desk over historical data (the backtest path) and to pin a
regression test to a real market episode — a flash crash, a squeeze — rather
than to synthetic noise.

CSV columns: ``ts,open,high,low,close,volume``, one file per symbol, named
``<SYMBOL>.csv`` (e.g. ``BTC-USD.csv``) inside ``directory``. ``ts`` may be
epoch seconds, epoch milliseconds, or an ISO-8601 timestamp; an ISO timestamp
without an offset is UTC.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from .base import Candle, FeedError, canonical_symbol


def _parse_ts(raw: str) -> int:
    raw = raw.strip()
    if raw.isdigit():
        value = int(raw)
        # Heuristic: anything past ~2001 in ms range is milliseconds.
        return value // 1000 if value > 100_000_000_000 else value
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FeedError(f"Unparseable timestamp {raw!r} in replay CSV") from exc
    # A naive timestamp would otherwise pick up the process's local zone, so
    # the same CSV replays at different times in Docker than on a laptop.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class ReplayFeed:
    """Serve candles from CSV, optionally advancing a cursor bar by bar.

    Once the cursor reaches the end of a file the feed is *exhausted*: the
    cursor stops moving, ``exhausted()`` reports it, and ``price()`` keeps
    returning the last close so a caller that overruns sees a frozen tape
    rather than an exception mid-tick.
    """

    name = "replay"

    def __init__(self, directory: str | Path = ".", cursor: int | None = None):
        self.directory = Path(directory).expanduser()
        # When set, only candles up to this index are visible — the mechanism
        # that makes a replay look-ahead-free.
        self.cursor = cursor
        self._exhausted = False
        self._cache: dict[str, list[Candle]] = {}

    def _load(self, symbol: str) -> list[Candle]:
        symbol = canonical_symbol(symbol)
        if symbol in self._cache:
            return self._cache[symbol]
        path = self.directory / f"{symbol}.csv"
        if not path.exists():
            raise FeedError(f"No replay CSV for {symbol} at {path}")
        rows: list[Candle] = []
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows.append(
                        Candle(
                            ts=_parse_ts(row["ts"]),
                            open=float(row["open"]), high=float(row["high"]),
                            low=float(row["low"]), close=float(row["close"]),
                            volume=float(row.get("volume") or 0.0),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise FeedError(f"Malformed row in {path}: {row} ({exc})") from exc
        if not rows:
            raise FeedError(f"Replay CSV {path} is empty")
        rows.sort(key=lambda c: c.ts)
        self._cache[symbol] = rows
        return rows

    def _visible(self, symbol: str) -> list[Candle]:
        rows = self._load(symbol)
        return rows if self.cursor is None else rows[: max(self.cursor, 1)]

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[Candle]:
        # Interval is ignored: the CSV's own bar size is authoritative. Passing
        # a mismatched interval silently would be worse than documenting it.
        return self._visible(symbol)[-limit:]

    def price(self, symbol: str) -> float:
        """Last visible close; once exhausted this is the file's final close."""
        return self._visible(symbol)[-1].close

    def market(self, symbol: str, interval: str = "5m",
               limit: int = 200) -> tuple[list[Candle], float, str]:
        return self.candles(symbol, interval, limit), self.price(symbol), self.name

    def total_bars(self, symbol: str) -> int:
        return len(self._load(symbol))

    def exhausted(self, symbol: str) -> bool:
        """True once the cursor sits at or past the end of ``symbol``'s file.

        A cursor-less feed already shows its whole history, so it has nothing
        left to reveal and counts as exhausted too.
        """
        if self._exhausted or self.cursor is None:
            return True
        return self.cursor >= self.total_bars(symbol)

    def advance(self, bars: int = 1) -> None:
        """Move the cursor forward, revealing more history; clamps at the end."""
        if self.cursor is None:
            raise FeedError("ReplayFeed.advance requires an initial cursor")
        target = self.cursor + bars
        # The end is only known for files already opened; a run that never
        # asked for candles has nothing to clamp against, and exhausted() still
        # compares the cursor to the file length when it is asked.
        end = max((len(rows) for rows in self._cache.values()), default=None)
        if end is not None and target >= end:
            target = end
            self._exhausted = True
        self.cursor = target
