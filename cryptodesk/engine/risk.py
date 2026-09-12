"""The risk engine: sizing, stops, and the limits that cannot be argued with.

This module is the reason the desk is allowed to run unattended. The committee
(LLM or heuristic) may only ever propose a *direction* and a *conviction* in
[0, 1]; every decision about how much money is at stake is made here, in
deterministic code, from volatility and the configured limits.

That split is deliberate. An LLM asked to size a position will happily talk
itself into "high conviction, 80% of the account" — and the one thing that
reliably ends a trading account is not being wrong, it is being wrong while
large. Here, conviction can only scale a position *within* a cap it can never
raise, and four independent brakes (per-symbol weight, gross exposure, daily
loss, peak drawdown) can each stop trading on their own.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from ..config import RiskLimits
from ..indicators import Snapshot

# Ratings the committee can return, mapped to a fraction of the maximum
# allowed weight for that symbol. "Hold" is deliberately absent: it means
# "change nothing", which is different from "target 50%".
RATING_WEIGHTS = {
    "Buy": 1.0,
    "Overweight": 0.6,
    "Underweight": 0.2,
    "Sell": 0.0,
}


@dataclass
class RiskState:
    """Mutable risk bookkeeping, persisted so a restart does not reset the brakes.

    Without persistence, a crash-loop would reset ``peak_equity`` and
    ``day_start_equity`` on every boot and the drawdown kill-switch would never
    fire — the failure mode where an unattended bot quietly bleeds out.
    """

    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day_key: str = ""            # UTC date, so the daily limit resets at 00:00Z
    halted: bool = False
    halt_reason: str = ""
    # symbol -> epoch seconds until which new entries are refused
    cooldowns: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> RiskState:
        raw = dict(raw or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def roll_day(self, equity: float, now: float | None = None) -> bool:
        """Start a new UTC day if needed. Returns True when the day rolled."""
        key = time.strftime("%Y-%m-%d", time.gmtime(now if now is not None else time.time()))
        if key != self.day_key:
            self.day_key = key
            self.day_start_equity = equity
            return True
        return False

    def observe(self, equity: float, now: float | None = None) -> None:
        """Record an equity mark: rolls the day and tracks the high-water mark."""
        if self.day_start_equity <= 0:
            self.day_start_equity = equity
        self.roll_day(equity, now)
        self.peak_equity = max(self.peak_equity, equity)

    def drawdown(self, equity: float) -> float:
        """Fractional drawdown from the high-water mark (0.0 when at a new high)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def day_pnl_pct(self, equity: float) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (equity - self.day_start_equity) / self.day_start_equity

    def cooling_down(self, symbol: str, now: float | None = None) -> bool:
        until = self.cooldowns.get(symbol)
        if until is None:
            return False
        if (now if now is not None else time.time()) >= until:
            # Expired: drop it so the dict does not grow without bound.
            self.cooldowns.pop(symbol, None)
            return False
        return True

    def start_cooldown(self, symbol: str, minutes: int, now: float | None = None) -> None:
        base = now if now is not None else time.time()
        self.cooldowns[symbol] = int(base + minutes * 60)


@dataclass
class EntryPlan:
    """The risk engine's verdict on a proposed entry."""

    allowed: bool
    reason: str
    qty: float = 0.0
    notional: float = 0.0
    stop_price: float | None = None
    target_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def size_position(equity: float, price: float, atr: float, limits: RiskLimits,
                  conviction: float = 1.0) -> float:
    """Volatility-targeted position size, in units of the base asset.

    The size is chosen so that a move to the stop costs ``risk_per_trade`` of
    equity. This is the mechanism that makes one configuration work across a
    quiet BTC week and a violent SOL one: when ATR doubles, size halves, and
    the dollar risk per trade is unchanged.

    ``conviction`` scales the result down only (it is clamped to [0, 1]), and
    the outcome is capped by ``max_symbol_weight`` regardless.
    """
    if equity <= 0 or price <= 0 or atr <= 0:
        return 0.0
    conviction = max(0.0, min(1.0, conviction))
    stop_distance = atr * limits.stop_atr_mult
    if stop_distance <= 0:
        return 0.0
    risk_qty = (equity * limits.risk_per_trade) / stop_distance
    weight_cap_qty = (equity * limits.max_symbol_weight) / price
    return max(0.0, min(risk_qty, weight_cap_qty) * conviction)


def plan_entry(symbol: str, snapshot: Snapshot, equity: float, cash: float,
               current_qty: float, gross_exposure: float, state: RiskState,
               limits: RiskLimits, conviction: float = 1.0,
               now: float | None = None) -> EntryPlan:
    """Decide whether, and how large, a long entry may be placed.

    Checks run cheapest-and-most-fatal first so the reason surfaced to the
    dashboard is the one that actually matters.
    """
    now = now if now is not None else time.time()

    if state.halted:
        return EntryPlan(False, f"desk halted: {state.halt_reason}")

    drawdown = state.drawdown(equity)
    if drawdown >= limits.max_drawdown_halt:
        return EntryPlan(False, f"drawdown {drawdown:.1%} at or past halt limit "
                                f"{limits.max_drawdown_halt:.1%}")

    day_pnl = state.day_pnl_pct(equity)
    if day_pnl <= -limits.daily_loss_limit:
        return EntryPlan(False, f"daily loss {day_pnl:.2%} past limit "
                                f"-{limits.daily_loss_limit:.2%}; no new entries today")

    if state.cooling_down(symbol, now):
        remaining = int((state.cooldowns[symbol] - now) / 60)
        return EntryPlan(False, f"cooldown active for {symbol} ({remaining}m remaining)")

    if snapshot.atr14 is None:
        return EntryPlan(False, f"no ATR for {symbol} yet ({snapshot.bars} bars); "
                                "refusing to size blind")

    if gross_exposure >= limits.max_gross_exposure:
        return EntryPlan(False, f"gross exposure {gross_exposure:.1%} at cap "
                                f"{limits.max_gross_exposure:.1%}")

    target_qty = size_position(equity, snapshot.price, snapshot.atr14, limits, conviction)
    # Only the incremental quantity is ordered; an existing position counts
    # toward the cap rather than being doubled.
    qty = target_qty - max(current_qty, 0.0)
    if qty <= 0:
        return EntryPlan(False, f"already at or above target size for {symbol}",
                         target_weight=target_qty * snapshot.price / equity if equity else 0.0)

    # Respect the remaining room under the gross cap.
    room = max(0.0, (limits.max_gross_exposure - gross_exposure) * equity)
    notional = qty * snapshot.price
    if notional > room:
        qty = room / snapshot.price
        notional = qty * snapshot.price

    # Never spend more cash than is held (paper or not, leverage is not modelled).
    if notional > cash:
        qty = max(0.0, cash / snapshot.price * 0.999)  # leave room for fees
        notional = qty * snapshot.price

    if notional < limits.min_order_notional:
        return EntryPlan(False, f"order notional ${notional:.2f} below minimum "
                                f"${limits.min_order_notional:.2f}")

    stop_price = snapshot.price - snapshot.atr14 * limits.stop_atr_mult
    if stop_price <= 0:
        return EntryPlan(False, f"computed stop for {symbol} is non-positive; ATR too wide")

    return EntryPlan(
        allowed=True,
        reason=f"sized from {limits.risk_per_trade:.2%} equity risk over "
               f"{limits.stop_atr_mult}x ATR ({snapshot.atr14:.4f}), conviction {conviction:.2f}",
        qty=qty, notional=notional, stop_price=stop_price,
        target_weight=notional / equity if equity > 0 else 0.0,
    )


def update_trailing_stop(entry_price: float, stop_price: float | None,
                         extreme_price: float, current_price: float, atr: float,
                         limits: RiskLimits) -> tuple[float | None, float]:
    """Ratchet a long stop upward as price makes new highs.

    Returns ``(stop_price, extreme_price)``. The stop only ever moves in the
    trader's favour — a stop that can loosen is not a stop.
    """
    extreme = max(extreme_price, current_price)
    if atr <= 0:
        return stop_price, extreme
    candidate = extreme - atr * limits.trail_atr_mult
    if stop_price is None:
        return candidate, extreme
    return max(stop_price, candidate), extreme


def stop_hit(stop_price: float | None, price: float, is_long: bool = True) -> bool:
    """True when price has traded through the stop."""
    if stop_price is None:
        return False
    return price <= stop_price if is_long else price >= stop_price
