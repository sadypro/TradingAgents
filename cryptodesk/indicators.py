"""Cheap technical state, computed in pure Python.

Deliberately dependency-free: these run every minute for every symbol, forever,
and the desk should not carry pandas/numpy into a container for arithmetic this
simple. Every function takes oldest-first candles and returns ``None`` when
there is not enough history, rather than guessing — a fabricated indicator
value is worse than an absent one, because the risk engine sizes off ATR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .feeds.base import Candle


def sma(values: list[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: list[float], period: int) -> float | None:
    """Exponential moving average, seeded with the first ``period`` SMA."""
    if period <= 0 or len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    acc = sum(values[:period]) / period
    for value in values[period:]:
        acc = value * k + acc * (1.0 - k)
    return acc


def rsi(values: list[float], period: int = 14) -> float | None:
    """Wilder's RSI. Returns 0-100, or None without ``period + 1`` points."""
    if len(values) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
    if avg_loss == 0.0:
        # All-gains window: RSI is 100 by definition, not a divide-by-zero.
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles: list[Candle], period: int = 14) -> float | None:
    """Wilder's Average True Range, in quote currency.

    The desk's unit of risk. Position size, stop distance and the breakout
    trigger are all expressed in ATR multiples so that a 1% move in a quiet
    week and a 1% move in a violent one are not treated as the same event.
    """
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1].close
        c = candles[i]
        trs.append(max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close)))
    acc = sum(trs[:period]) / period
    for tr in trs[period:]:
        acc = (acc * (period - 1) + tr) / period
    return acc


def realized_vol(values: list[float], period: int = 48) -> float | None:
    """Stdev of log returns over ``period`` bars (per-bar, not annualised)."""
    if len(values) < period + 1:
        return None
    window = values[-(period + 1):]
    rets = [
        math.log(window[i] / window[i - 1])
        for i in range(1, len(window))
        if window[i] > 0 and window[i - 1] > 0
    ]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(variance)


@dataclass
class Snapshot:
    """Everything the fast loop knows about one symbol at one instant.

    Fields may be ``None`` when history is short; consumers must handle that
    (``risk.size_position`` refuses to size without an ATR, for instance).
    """

    symbol: str
    ts: int
    price: float
    ema_fast: float | None = None
    ema_slow: float | None = None
    rsi14: float | None = None
    atr14: float | None = None
    vol_short: float | None = None
    vol_long: float | None = None
    bars: int = 0

    @property
    def trend(self) -> str:
        """Coarse regime label: up / down / flat."""
        if self.ema_fast is None or self.ema_slow is None:
            return "unknown"
        if self.ema_fast > self.ema_slow * 1.001:
            return "up"
        if self.ema_fast < self.ema_slow * 0.999:
            return "down"
        return "flat"

    @property
    def atr_pct(self) -> float | None:
        """ATR as a fraction of price — comparable across symbols."""
        if self.atr14 is None or self.price <= 0:
            return None
        return self.atr14 / self.price

    @property
    def vol_ratio(self) -> float | None:
        """Short-window vol over long-window vol; >1 means vol is expanding."""
        if not self.vol_short or not self.vol_long:
            return None
        return self.vol_short / self.vol_long

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "ts": self.ts, "price": self.price,
            "ema_fast": self.ema_fast, "ema_slow": self.ema_slow,
            "rsi14": self.rsi14, "atr14": self.atr14,
            "atr_pct": self.atr_pct, "trend": self.trend,
            "vol_ratio": self.vol_ratio, "bars": self.bars,
        }


def snapshot(symbol: str, candles: list[Candle], price: float | None = None,
             fast: int = 12, slow: int = 48) -> Snapshot:
    """Compute the full indicator set for ``symbol`` from oldest-first candles."""
    if not candles:
        raise ValueError(f"snapshot({symbol}) requires at least one candle")
    closes = [c.close for c in candles]
    return Snapshot(
        symbol=symbol,
        ts=candles[-1].ts,
        price=float(price if price is not None else closes[-1]),
        ema_fast=ema(closes, fast),
        ema_slow=ema(closes, slow),
        rsi14=rsi(closes, 14),
        atr14=atr(candles, 14),
        vol_short=realized_vol(closes, 24),
        vol_long=realized_vol(closes, 96),
        bars=len(candles),
    )
