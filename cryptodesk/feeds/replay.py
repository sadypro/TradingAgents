"""Replay a recorded CSV of candles.

Used to drive the desk over historical data (the backtest path) and to pin a
regression test to a real market episode — a flash crash, a squeeze — rather
than to synthetic noise.

CSV columns: ``ts,open,high,low,close,volume``, one file per symbol, named
``<SYMBOL>.csv`` (e.g. ``BTC-USD.csv``) inside ``directory``. ``ts`` may be
epoch seconds, epoch milliseconds, or an ISO-8601 timestamp.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .base import Candle, FeedError


def _parse_ts(raw: str) -> int:
    raw = raw.strip()
    if raw.isdigit():
        value = int(raw)
        # Heuristic: anything past ~2001 in ms range is milliseconds.
        return value // 1000 if value > 100_000_000_000 else value
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except ValueError as exc:
        raise FeedError(f"Unparseable timestamp {raw!r} in replay CSV") from exc


class ReplayFeed:
    """Serve candles from CSV, optionally advancing a cursor bar by bar."""

    name = "replay"

    def __init__(self, directory: str | Path = ".", cursor: int | None = None):
        self.directory = Path(directory).expanduser()
        # When set, only candles up to this index are visible — the mechanism
        # that makes a replay look-ahead-free.
        self.cursor = cursor
        self._cache: dict[str, list[Candle]] = {}

    def _load(self, symbol: str) -> list[Candle]:
        if symbol in self._cache:
            return self._cache[symbol]
        path = self.directory / f"{symbol.upper()}.csv"
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
        return self._visible(symbol)[-1].close

    def advance(self, bars: int = 1) -> None:
        """Move the cursor forward, revealing more history."""
        if self.cursor is None:
            raise FeedError("ReplayFeed.advance requires an initial cursor")
        self.cursor += bars
