"""Crypto.com Exchange public market data (keyless).

Endpoints used (no authentication, no account needed):
  GET /exchange/v1/public/get-candlestick?instrument_name=BTC_USD&timeframe=5m&count=N
  GET /exchange/v1/public/get-tickers?instrument_name=BTC_USD

Chosen as the default because it needs no signup and its instrument naming maps
cleanly from the desk's canonical symbols.
"""

from __future__ import annotations

import requests

from .base import Candle, FeedError, split_symbol

_BASE = "https://api.crypto.com/exchange/v1/public"

# Desk interval -> venue timeframe token.
_TIMEFRAMES = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1D",
}


class CryptoComFeed:
    name = "cryptocom"

    def __init__(self, timeout: float = 10.0, session: requests.Session | None = None):
        self.timeout = timeout
        self._session = session or requests.Session()

    @staticmethod
    def venue_symbol(symbol: str) -> str:
        """``BTC-USD`` -> ``BTC_USD``; USDT/USDC quotes are kept as quoted."""
        base, quote = split_symbol(symbol)
        return f"{base}_{quote}"

    def _get(self, path: str, params: dict) -> dict:
        try:
            resp = self._session.get(f"{_BASE}/{path}", params=params, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise FeedError(f"crypto.com request failed: {exc}") from exc
        except ValueError as exc:
            raise FeedError(f"crypto.com returned non-JSON: {exc}") from exc
        # The venue signals application errors in-band with code != 0, which
        # would otherwise sail past raise_for_status as a 200.
        if payload.get("code") not in (0, None):
            raise FeedError(f"crypto.com error {payload.get('code')}: {payload.get('message')}")
        return payload.get("result") or {}

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[Candle]:
        timeframe = _TIMEFRAMES.get(interval)
        if timeframe is None:
            raise FeedError(f"crypto.com does not support interval {interval!r}")
        result = self._get(
            "get-candlestick",
            {"instrument_name": self.venue_symbol(symbol), "timeframe": timeframe,
             "count": min(int(limit), 1000)},
        )
        rows = result.get("data") or []
        if not rows:
            raise FeedError(f"crypto.com returned no candles for {symbol}")
        candles = [
            Candle(
                # Venue timestamps are milliseconds; the desk works in seconds.
                ts=int(row["t"]) // 1000,
                open=float(row["o"]), high=float(row["h"]),
                low=float(row["l"]), close=float(row["c"]),
                volume=float(row.get("v") or 0.0),
            )
            for row in rows
        ]
        candles.sort(key=lambda c: c.ts)
        return candles[-limit:]

    def price(self, symbol: str) -> float:
        result = self._get("get-tickers", {"instrument_name": self.venue_symbol(symbol)})
        rows = result.get("data") or []
        if not rows:
            raise FeedError(f"crypto.com returned no ticker for {symbol}")
        # "a" is the latest trade price in the v1 ticker payload; fall back to
        # the bid/ask midpoint when the venue omits it on a quiet instrument.
        row = rows[0]
        last = row.get("a") or row.get("l")
        if last is not None:
            return float(last)
        bid, ask = row.get("b"), row.get("k")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
        raise FeedError(f"crypto.com ticker for {symbol} carried no usable price")
