"""A deterministic synthetic price feed.

This exists for two reasons, both practical:

* **Tests.** Every downstream component (risk, broker, engine) needs price
  series that behave like markets — trends, reversals, volatility clustering —
  without a network call or a recorded fixture per case.
* **Offline demo.** A user (or a CI box, or a sandboxed container) with no
  access to exchange APIs can still start the desk, watch the loop run, and
  see the dashboard populate. Nothing about the desk's logic is stubbed in that
  mode; only the prices are simulated.

The series is generated once per (symbol, interval) and then *extended* as the
clock advances, never regenerated. That gives it the two properties a real feed
has and a naive generator does not: history never revises under you, and
``price()`` always agrees with the last close from ``candles()`` at the feed's
configured interval.

It is explicitly **not** a market model. Numbers produced here say nothing
about whether a strategy is profitable.
"""

from __future__ import annotations

import hashlib
import math
import random
import time

from .base import Candle, canonical_symbol, interval_seconds

# Starting levels chosen to be roughly plausible so dashboards look sane.
_ANCHORS = {
    "BTC": 60_000.0, "ETH": 3_000.0, "SOL": 150.0, "XRP": 0.60,
    "ADA": 0.45, "DOGE": 0.12, "LTC": 85.0, "AVAX": 30.0, "LINK": 15.0,
}

# How much history to materialise on first touch. Enough for a 200-bar
# indicator lookback with room to spare.
_PRIME_BARS = 400


class _Walk:
    """A growing candle series for one (symbol, interval) pair."""

    def __init__(self, rng: random.Random, start_index: int, price: float,
                 mu: float, base_sigma: float):
        self.rng = rng
        self.start_index = start_index
        self.candles: list[Candle] = []
        self.price = price
        self.mu = mu
        self.base_sigma = base_sigma
        self.sigma = base_sigma

    @property
    def next_index(self) -> int:
        return self.start_index + len(self.candles)

    def extend_to(self, target_index: int, step: int) -> None:
        """Append bars until the bar opening at ``target_index`` exists."""
        rng = self.rng
        while self.next_index <= target_index:
            index = self.next_index
            # Volatility clusters: sigma mean-reverts toward base with shocks.
            self.sigma = max(
                self.base_sigma * 0.3,
                self.sigma * 0.94 + self.base_sigma * 0.06 * rng.uniform(0.5, 2.5),
            )
            shock = rng.gauss(self.mu, self.sigma)
            # Rare regime jumps, the one feature that makes crypto crypto.
            if rng.random() < 0.004:
                shock += rng.choice([-1, 1]) * rng.uniform(3, 9) * self.sigma
            open_px = self.price
            close_px = max(open_px * (1.0 + shock), 1e-8)
            wick = abs(rng.gauss(0, self.sigma)) * open_px
            high = max(open_px, close_px) + wick
            low = max(min(open_px, close_px) - wick, 1e-9)
            volume = abs(rng.gauss(1000, 250)) * (1 + abs(shock) * 40)
            self.candles.append(
                Candle(ts=index * step, open=open_px, high=high,
                       low=low, close=close_px, volume=volume)
            )
            self.price = close_px


class SyntheticFeed:
    """Seeded simulated feed. Same seed + symbol + clock => same series."""

    name = "synthetic"

    def __init__(self, seed: int = 7, drift_per_year: float = 0.0,
                 annual_vol: float = 0.65, clock=time.time, interval: str = "5m"):
        self.seed = seed
        self.drift_per_year = drift_per_year
        self.annual_vol = annual_vol
        self._clock = clock
        # The interval price() marks on. Each interval is its own random walk,
        # so the mark must come from the same series the desk's candles do.
        interval_seconds(interval)
        self.interval = interval
        self._walks: dict[tuple[str, str], _Walk] = {}

    def _anchor(self, symbol: str) -> float:
        base = symbol.split("-")[0]
        if base in _ANCHORS:
            return _ANCHORS[base]
        # Stable pseudo-price for unknown symbols so tests stay deterministic.
        digest = hashlib.sha256(base.encode()).hexdigest()
        return 10.0 + (int(digest[:8], 16) % 5_000)

    def _last_closed_index(self, step: int) -> int:
        """Index of the most recently *closed* bar.

        The in-progress bar is withheld: handing the desk a partial bar is the
        single easiest way to bake look-ahead into a live loop.
        """
        return int(self._clock()) // step - 1

    def _walk(self, symbol: str, interval: str) -> tuple[_Walk, int]:
        step = interval_seconds(interval)
        # Canonicalise so BTCUSD and BTC-USD are the same asset here as they
        # are on the live venues, not a hashed pseudo-asset.
        symbol = canonical_symbol(symbol)
        key = (symbol, interval)
        walk = self._walks.get(key)
        last_index = self._last_closed_index(step)
        if walk is None:
            bars_per_year = (365 * 24 * 3600) / step
            walk = _Walk(
                rng=random.Random(f"{self.seed}:{symbol}:{interval}"),
                start_index=last_index - _PRIME_BARS + 1,
                price=self._anchor(symbol),
                mu=self.drift_per_year / bars_per_year,
                base_sigma=self.annual_vol / math.sqrt(bars_per_year),
            )
            self._walks[key] = walk
        walk.extend_to(last_index, step)
        return walk, step

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[Candle]:
        walk, _ = self._walk(symbol, interval)
        return walk.candles[-limit:]

    def price(self, symbol: str) -> float:
        """Last close at the configured interval — always consistent with candles()."""
        walk, _ = self._walk(symbol, self.interval)
        return walk.candles[-1].close

    def market(self, symbol: str, interval: str = "5m",
               limit: int = 200) -> tuple[list[Candle], float, str]:
        # Mark on the series just served rather than self.interval: the two
        # agree when the feed is configured to match, and when it is not the
        # candles' own last close is the only price that cannot disagree.
        walk, _ = self._walk(symbol, interval)
        return walk.candles[-limit:], walk.candles[-1].close, self.name
