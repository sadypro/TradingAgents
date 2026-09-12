"""When to wake the expensive committee.

A full TradingAgents run is roughly a dozen LLM calls over large contexts. Run
that on a one-minute loop across five symbols and it is ~7,000 runs a day: the
token bill would dwarf any plausible trading profit on a retail account. So the
committee is woken only when something has actually changed, plus a floor
cadence so a quiet market still gets reviewed.

Every trigger is expressed in ATR multiples rather than percentages, so "the
price moved a lot" means the same thing in a calm week and a violent one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import LLMBudget
from ..indicators import Snapshot


@dataclass
class Trigger:
    """Whether to run the committee, and why."""

    fired: bool
    reason: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.fired


@dataclass
class TriggerContext:
    """What the trigger logic needs to know about a symbol's recent history."""

    last_decision_ts: int | None = None
    last_decision_price: float | None = None
    last_trend: str | None = None
    has_position: bool = False
    spend_today: float = 0.0


# A move of this many ATRs since the last decision means the thesis is stale.
_ATR_MOVE_THRESHOLD = 1.5
# Short-vs-long realised vol ratio above which the regime is "expanding".
_VOL_EXPANSION_RATIO = 1.8
_RSI_OVERBOUGHT = 72.0
_RSI_OVERSOLD = 28.0


def evaluate(snapshot: Snapshot, ctx: TriggerContext, budget: LLMBudget,
             now: float | None = None) -> Trigger:
    """Decide whether the committee should run for this symbol right now.

    Budget and cooldown checks come first: they are the ones that protect real
    money, and a fired trigger that then gets refused wastes a log line and
    invites a retry loop.
    """
    now = now if now is not None else time.time()

    # ---- hard gates ---------------------------------------------------
    if ctx.spend_today >= budget.daily_usd_cap:
        return Trigger(False, "budget_exhausted",
                       f"${ctx.spend_today:.2f} spent today, cap ${budget.daily_usd_cap:.2f}")

    # Leave room for one more run before the cap, rather than overshooting it.
    if ctx.spend_today + budget.estimated_cost_per_run_usd > budget.daily_usd_cap:
        return Trigger(False, "budget_reserved",
                       f"next run (~${budget.estimated_cost_per_run_usd:.2f}) would exceed "
                       f"the ${budget.daily_usd_cap:.2f} daily cap")

    if snapshot.atr14 is None:
        return Trigger(False, "warming_up",
                       f"only {snapshot.bars} bars; need history before spending on analysis")

    if ctx.last_decision_ts is None:
        return Trigger(True, "cold_start", "no prior decision for this symbol")

    minutes_since = (now - ctx.last_decision_ts) / 60.0
    if minutes_since < budget.min_minutes_between_calls:
        return Trigger(False, "cooldown",
                       f"{minutes_since:.0f}m since last run, minimum "
                       f"{budget.min_minutes_between_calls}m")

    # ---- event triggers ------------------------------------------------
    if ctx.last_decision_price and snapshot.atr14 > 0:
        moved = abs(snapshot.price - ctx.last_decision_price) / snapshot.atr14
        if moved >= _ATR_MOVE_THRESHOLD:
            direction = "up" if snapshot.price > ctx.last_decision_price else "down"
            return Trigger(True, "atr_move",
                           f"price moved {moved:.1f} ATR {direction} since the last decision")

    if ctx.last_trend and snapshot.trend != "unknown" and snapshot.trend != ctx.last_trend:
        return Trigger(True, "trend_flip",
                       f"regime flipped {ctx.last_trend} -> {snapshot.trend}")

    vol_ratio = snapshot.vol_ratio
    if vol_ratio is not None and vol_ratio >= _VOL_EXPANSION_RATIO:
        return Trigger(True, "vol_expansion",
                       f"short-term vol {vol_ratio:.1f}x the longer window")

    if snapshot.rsi14 is not None:
        if snapshot.rsi14 >= _RSI_OVERBOUGHT and ctx.has_position:
            return Trigger(True, "rsi_overbought",
                           f"RSI {snapshot.rsi14:.0f} while holding; review the exit")
        if snapshot.rsi14 <= _RSI_OVERSOLD and not ctx.has_position:
            return Trigger(True, "rsi_oversold",
                           f"RSI {snapshot.rsi14:.0f} with no position; review the entry")

    # ---- floor cadence -------------------------------------------------
    if minutes_since >= budget.scheduled_interval_minutes:
        return Trigger(True, "scheduled",
                       f"{minutes_since:.0f}m since the last review")

    return Trigger(False, "no_trigger",
                   f"nothing material changed ({minutes_since:.0f}m since last run)")
