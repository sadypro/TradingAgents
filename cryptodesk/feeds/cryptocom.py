"""Crypto.com Exchange public market data (keyless).

Endpoints used (no authentication, no account needed):
  GET /exchange/v1/public/get-candlestick?instrument_name=BTC_USD&timeframe=5m&count=N
  GET /exchange/v1/public/get-tickers?instrument_name=BTC_USD

Chosen as the default because it needs no signup and its instrument naming maps
cleanly from the desk's canonical symbols.
"""

from __future__ import annotations

import time

import requests

from .base import Candle, FeedError, interval_seconds, split_symbol

_BASE = "https://api.crypto.com/exchange/v1/public"

# Desk interval -> venue timeframe token.
_TIMEFRAMES = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1D",
}

# The v1 get-candlestick endpoint serves at most this many rows per request.
_MAX_COUNT = 300

# Connect timeout is short and fixed: a black-holed venue should cost the
# chain a few seconds before it moves on, not a full read timeout.
_CONNECT_TIMEOUT = 3.0


class CryptoComFeed:
    name = "cryptocom"

    def __init__(self, timeout: float = 10.0, session: requests.Session | None = None,
                 clock=time.time):
        self.timeout = timeout
        self._session = session or requests.Session()
        self._clock = clock

    @staticmethod
    def venue_symbol(symbol: str) -> str:
        """``BTC-USD`` -> ``BTC_USD``; USDT/USDC quotes are kept as quoted."""
        base, quote = split_symbol(symbol)
        return f"{base}_{quote}"

    def _get(self, path: str, params: dict) -> dict:
        try:
            resp = self._session.get(f"{_BASE}/{path}", params=params,
                                     timeout=(_CONNECT_TIMEOUT, self.timeout))
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
        # Refuse rather than truncate: a desk configured for 500 bars of
        # history that quietly runs on 300 is a misconfiguration nobody sees.
        if limit > _MAX_COUNT:
            raise FeedError(
                f"crypto.com serves at most {_MAX_COUNT} candles per request; "
                f"{limit} requested (lower candle_lookback or use another feed)"
            )
        step = interval_seconds(interval)
        result = self._get(
            "get-candlestick",
            {"instrument_name": self.venue_symbol(symbol), "timeframe": timeframe,
             "count": int(limit)},
        )
        rows = result.get("data") or []
        if not rows:
            raise FeedError(f"crypto.com returned no candles for {symbol}")
        try:
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
        except (KeyError, TypeError, ValueError) as exc:
            raise FeedError(f"crypto.com candle payload for {symbol} is malformed: {exc}") from exc
        candles.sort(key=lambda c: c.ts)
        # The newest row is the bar still forming; indicators on a partial bar
        # deflate ATR at every bar open and ratchet trailing stops on noise.
        now = int(self._clock())
        candles = [c for c in candles if c.ts + step <= now]
        if not candles:
            raise FeedError(f"crypto.com returned no closed candles for {symbol}")
        return candles[-limit:]

    def price(self, symbol: str) -> float:
        result = self._get("get-tickers", {"instrument_name": self.venue_symbol(symbol)})
        rows = result.get("data") or []
        if not rows:
            raise FeedError(f"crypto.com returned no ticker for {symbol}")
        # "a" is the latest trade price in the v1 ticker payload and is null
        # when nothing has traded; then use the bid/ask midpoint. "l" is the
        # 24h *low*, never a price — marking the book at the day's low would
        # fabricate a gap-down that trips stops on a non-event.
        try:
            row = rows[0]
            last = row.get("a")
            if last is not None:
                return float(last)
            bid, ask = row.get("b"), row.get("k")
            if bid is not None and ask is not None:
                return (float(bid) + float(ask)) / 2.0
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise FeedError(f"crypto.com ticker payload for {symbol} is malformed: {exc}") from exc
        raise FeedError(f"crypto.com ticker for {symbol} carried no usable price")

    def market(self, symbol: str, interval: str = "5m",
               limit: int = 200) -> tuple[list[Candle], float, str]:
        return self.candles(symbol, interval, limit), self.price(symbol), self.name
