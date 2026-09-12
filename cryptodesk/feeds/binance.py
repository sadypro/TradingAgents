"""Binance public market data (keyless), used as the fallback feed.

Endpoints:
  GET /api/v3/klines?symbol=BTCUSDT&interval=5m&limit=N
  GET /api/v3/ticker/price?symbol=BTCUSDT

Binance lists USDT pairs rather than USD, so a ``-USD`` request is served from
the USDT pair. For BTC/ETH the USD/USDT basis is a few basis points, well
inside the slippage assumption, but it is a real approximation and is noted
here so nobody mistakes it for an exact match.
"""

from __future__ import annotations

import requests

from .base import Candle, FeedError, split_symbol

_BASE = "https://api.binance.com/api/v3"

_INTERVALS = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


class BinanceFeed:
    name = "binance"

    def __init__(self, timeout: float = 10.0, session: requests.Session | None = None):
        self.timeout = timeout
        self._session = session or requests.Session()

    @staticmethod
    def venue_symbol(symbol: str) -> str:
        """``BTC-USD`` -> ``BTCUSDT`` (USD is served from the USDT pair)."""
        base, quote = split_symbol(symbol)
        if quote == "USD":
            quote = "USDT"
        return f"{base}{quote}"

    def _get(self, path: str, params: dict):
        try:
            resp = self._session.get(f"{_BASE}/{path}", params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise FeedError(f"binance request failed: {exc}") from exc
        except ValueError as exc:
            raise FeedError(f"binance returned non-JSON: {exc}") from exc

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[Candle]:
        venue_interval = _INTERVALS.get(interval)
        if venue_interval is None:
            raise FeedError(f"binance does not support interval {interval!r}")
        rows = self._get(
            "klines",
            {"symbol": self.venue_symbol(symbol), "interval": venue_interval,
             "limit": min(int(limit), 1000)},
        )
        if not rows:
            raise FeedError(f"binance returned no candles for {symbol}")
        return [
            Candle(
                ts=int(row[0]) // 1000,
                open=float(row[1]), high=float(row[2]),
                low=float(row[3]), close=float(row[4]), volume=float(row[5]),
            )
            for row in rows
        ]

    def price(self, symbol: str) -> float:
        payload = self._get("ticker/price", {"symbol": self.venue_symbol(symbol)})
        try:
            return float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FeedError(f"binance ticker for {symbol} carried no price: {payload}") from exc
