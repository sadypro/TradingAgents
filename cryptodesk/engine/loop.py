"""The desk: one deterministic ``tick()``, run forever on a timer.

All trading logic lives in :meth:`Desk.tick`, which takes a clock and returns a
summary of what it did. Nothing in it sleeps, spawns, or reads the wall clock
directly, so a test can drive a thousand ticks over a synthetic or replayed
market in milliseconds and assert on the outcome. :meth:`Desk.run_forever` is a
thin wrapper that calls ``tick`` on a timer and handles errors.

Order of operations inside a tick is deliberate: **mark, then protect, then
consider**. Stops and the kill-switch are evaluated before any new entry is
considered, so a tick that both breaches the drawdown limit and sees a buy
signal exits rather than buys.
"""

from __future__ import annotations

import logging
import threading
import time

from ..broker import OrderRejected, PaperBroker
from ..config import DeskConfig
from ..feeds import build_feed
from ..feeds.base import FeedError
from ..indicators import Snapshot, snapshot as compute_snapshot
from . import risk, triggers
from .committee import Proposal, build_committee
from .ledger import Ledger

logger = logging.getLogger(__name__)

_RISK_STATE_KEY = "risk_state"
_BROKER_STATE_KEY = "broker_state"


class Desk:
    """A 24/7 paper-trading desk over a set of crypto symbols."""

    def __init__(self, cfg: DeskConfig, feed=None, broker=None, ledger=None,
                 committee=None, clock=time.time):
        cfg.ensure_dirs()
        self.cfg = cfg
        self.clock = clock
        self.feed = feed or build_feed(cfg.feeds)
        self.ledger = ledger or Ledger(cfg.db_path)
        self.broker = broker or PaperBroker(
            starting_equity=cfg.starting_equity, fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps, allow_shorts=cfg.risk.allow_shorts,
        )
        self.committee = committee or build_committee(cfg)
        self.risk_state = risk.RiskState.from_dict(self.ledger.get_state(_RISK_STATE_KEY))
        # Restore the book before the first tick, so a restarted desk manages the
        # positions it actually holds instead of believing it is flat.
        self.restored = self.broker.load_state(self.ledger.get_state(_BROKER_STATE_KEY))
        if self.restored:
            self.ledger.record_event(
                "info", "restore",
                f"Restored {len(self.broker.positions())} position(s) and "
                f"${self.broker.cash():,.2f} cash from the ledger",
                ts=int(self.clock()))
        # Per-symbol memory the trigger logic needs between ticks.
        self._last_decision_price: dict[str, float] = {}
        self._last_trend: dict[str, str] = {}
        self._snapshots: dict[str, Snapshot] = {}
        self._prices: dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.started_ts = int(self.clock())
        self.last_tick_ts: int | None = None
        self.tick_count = 0
        self.last_error: str | None = None
        self.feed_errors: dict[str, str] = {}

    # ---- market data ---------------------------------------------------
    def _refresh(self) -> None:
        """Pull candles and prices for every symbol, tolerating per-symbol failure.

        One unreachable symbol must not stop the desk from managing the others —
        in particular it must not stop a stop-loss on a different symbol.
        """
        for symbol in self.cfg.symbols:
            try:
                candles = self.feed.candles(symbol, self.cfg.candle_interval,
                                            self.cfg.candle_lookback)
                price = self.feed.price(symbol)
                self._snapshots[symbol] = compute_snapshot(symbol, candles, price)
                self._prices[symbol] = price
                self.feed_errors.pop(symbol, None)
            except (FeedError, ValueError) as exc:
                self.feed_errors[symbol] = str(exc)
                logger.warning("Feed failure for %s: %s", symbol, exc)
                self.ledger.record_event("warning", "feed",
                                         f"{symbol}: {exc}", ts=int(self.clock()))

    # ---- the tick ------------------------------------------------------
    def tick(self, now: float | None = None) -> dict:
        """Run one full cycle: mark, protect, then consider new risk."""
        with self._lock:
            return self._tick_locked(now)

    def _tick_locked(self, now: float | None) -> dict:
        now = float(now if now is not None else self.clock())
        ts = int(now)
        self.tick_count += 1
        actions: list[dict] = []

        self._refresh()
        if not self._prices:
            self.last_tick_ts = ts
            return {"ts": ts, "error": "no market data for any symbol", "actions": []}

        # 1. Mark the book and update the risk high-water marks.
        equity = self.broker.equity(self._prices)
        self.risk_state.observe(equity, now)
        drawdown = self.risk_state.drawdown(equity)
        snap = self.broker.snapshot(self._prices)
        self.ledger.record_equity(
            ts=ts, equity=equity, cash=self.broker.cash(),
            gross_exposure=snap["gross_exposure"], drawdown=drawdown,
            realized_pnl=snap["total_realized_pnl"],
            unrealized_pnl=snap["unrealized_pnl"], fees=snap["total_fees"],
            benchmark_price=self._prices.get(self.cfg.benchmark_symbol),
        )

        # 2. Kill-switch, before anything else can add risk.
        if not self.risk_state.halted and drawdown >= self.cfg.risk.max_drawdown_halt:
            actions.extend(self._halt(
                f"drawdown {drawdown:.1%} breached the {self.cfg.risk.max_drawdown_halt:.1%} limit",
                ts))

        if self.risk_state.halted:
            self._persist(ts)
            self.last_tick_ts = ts
            return {"ts": ts, "equity": equity, "drawdown": drawdown,
                    "halted": True, "halt_reason": self.risk_state.halt_reason,
                    "actions": actions}

        # 3. Protect open positions: trail stops up, exit anything stopped out.
        actions.extend(self._manage_positions(ts))

        # 4. Consider new or changed exposure, symbol by symbol.
        for symbol in self.cfg.symbols:
            snapshot = self._snapshots.get(symbol)
            if snapshot is None:
                continue
            action = self._consider(symbol, snapshot, ts, now)
            if action:
                actions.append(action)

        self._persist(ts)
        self.last_tick_ts = ts
        self.last_error = None
        return {
            "ts": ts, "equity": equity, "drawdown": drawdown, "halted": False,
            "actions": actions, "prices": dict(self._prices),
        }

    # ---- position management -------------------------------------------
    def _manage_positions(self, ts: int) -> list[dict]:
        """Ratchet trailing stops and exit positions whose stop has been hit."""
        actions = []
        for symbol, pos in list(self.broker.positions().items()):
            price = self._prices.get(symbol)
            snapshot = self._snapshots.get(symbol)
            if price is None or snapshot is None or snapshot.atr14 is None:
                # Without a price or an ATR the stop cannot be evaluated
                # honestly; leave the position untouched rather than guess.
                continue
            new_stop, extreme = risk.update_trailing_stop(
                pos.avg_price, pos.stop_price, pos.extreme_price or pos.avg_price,
                price, snapshot.atr14, self.cfg.risk,
            )
            pos.stop_price, pos.extreme_price = new_stop, extreme

            if risk.stop_hit(pos.stop_price, price, pos.is_long):
                actions.append(self._exit(symbol, price, ts,
                                          f"stop hit at {pos.stop_price:.4f}",
                                          cooldown=True))
                continue

            trim = self._weight_drift_trim(symbol, pos, price, ts)
            if trim:
                actions.append(trim)
        return actions

    def _weight_drift_trim(self, symbol: str, pos, price: float, ts: int) -> dict | None:
        """Trim a position that mark-to-market has carried past its weight cap.

        The entry cap only constrains the order that opens a position. A winner
        that doubles would otherwise drift to twice its intended share of the
        account with no order ever breaching a limit — concentration arriving
        through the back door. The drift band makes the cap a real invariant
        while leaving room for a position to run.
        """
        limits = self.cfg.risk
        equity = self.broker.equity(self._prices)
        if equity <= 0:
            return None
        weight = abs(pos.market_value(price)) / equity
        ceiling = limits.max_symbol_weight * (1 + limits.max_weight_drift)
        if weight <= ceiling:
            return None

        target_qty = (equity * limits.max_symbol_weight) / price
        excess = abs(pos.qty) - target_qty
        if excess <= 0 or excess * price < limits.min_order_notional:
            return None
        try:
            fill = self.broker.market_order(
                symbol, "sell" if pos.is_long else "buy", excess, price, ts,
                f"weight drift {weight:.1%} over {limits.max_symbol_weight:.0%} cap")
        except OrderRejected as exc:
            logger.warning("Could not trim %s for weight drift: %s", symbol, exc)
            return None
        self.ledger.record_fill(fill.to_dict())
        self.ledger.record_event(
            "info", "trim",
            f"{symbol} trimmed to its {limits.max_symbol_weight:.0%} cap "
            f"(had drifted to {weight:.1%})", ts=ts)
        return {"symbol": symbol, "action": "reduced", "reason": "weight drift",
                "qty": fill.qty, "realized_pnl": fill.realized_pnl}

    def _exit(self, symbol: str, price: float, ts: int, reason: str,
              cooldown: bool = False) -> dict:
        """Flatten a symbol and record the fill."""
        try:
            fill = self.broker.close(symbol, price, ts, reason)
        except OrderRejected as exc:
            self.ledger.record_event("error", "exit", f"{symbol}: {exc}", ts=ts)
            return {"symbol": symbol, "action": "exit_rejected", "reason": str(exc)}
        if fill is None:
            return {"symbol": symbol, "action": "none", "reason": "already flat"}
        self.ledger.record_fill(fill.to_dict())
        if cooldown:
            self.risk_state.start_cooldown(symbol, self.cfg.risk.cooldown_minutes, ts)
        self.ledger.record_event(
            "info", "exit",
            f"{symbol} exited: {reason} (realised {fill.realized_pnl:+.2f})", ts=ts)
        return {"symbol": symbol, "action": "exited", "reason": reason,
                "qty": fill.qty, "price": fill.price,
                "realized_pnl": fill.realized_pnl}

    # ---- decision path --------------------------------------------------
    def _consider(self, symbol: str, snapshot: Snapshot, ts: int, now: float) -> dict | None:
        """Maybe wake the committee for ``symbol``, then act on its proposal."""
        pos = self.broker.position(symbol)
        ctx = triggers.TriggerContext(
            last_decision_ts=self.ledger.last_decision_ts(symbol),
            last_decision_price=self._last_decision_price.get(symbol),
            last_trend=self._last_trend.get(symbol),
            has_position=pos is not None,
            spend_today=self.ledger.spend_today(now),
        )
        trigger = triggers.evaluate(snapshot, ctx, self.cfg.llm, now)
        if not trigger.fired:
            return None

        proposal = self.committee.decide(symbol, snapshot, trigger.reason)
        if proposal.cost_usd:
            self.ledger.record_spend(proposal.cost_usd, symbol, self.committee.name,
                                     ok=proposal.error is None, ts=ts)
        self._last_decision_price[symbol] = snapshot.price
        self._last_trend[symbol] = snapshot.trend

        action = self._apply(symbol, snapshot, proposal, ts, now)
        self.ledger.record_decision(
            ts=ts, symbol=symbol, source=proposal.source, trigger=trigger.reason,
            rating=proposal.rating, conviction=proposal.conviction,
            action=action["action"], allowed=action.get("allowed", False),
            veto_reason=action.get("veto_reason"), qty=action.get("qty", 0.0),
            notional=action.get("notional", 0.0), stop_price=action.get("stop_price"),
            price=snapshot.price, cost_usd=proposal.cost_usd,
            latency_ms=proposal.latency_ms, summary=proposal.summary,
            detail={**proposal.detail, "trigger_detail": trigger.detail,
                    "cost_is_estimate": proposal.cost_is_estimate},
        )
        return {"symbol": symbol, "trigger": trigger.reason,
                "rating": proposal.rating, **action}

    def _apply(self, symbol: str, snapshot: Snapshot, proposal: Proposal,
               ts: int, now: float) -> dict:
        """Turn a rating into orders, subject to the risk engine."""
        pos = self.broker.position(symbol)
        current_qty = pos.qty if pos else 0.0

        if proposal.wants_exit:
            if current_qty <= 0:
                return {"action": "none", "allowed": True,
                        "veto_reason": "Sell rating but already flat"}
            exit_action = self._exit(symbol, snapshot.price, ts,
                                     f"committee rating Sell: {proposal.summary}")
            return {"action": exit_action["action"], "allowed": True,
                    "qty": exit_action.get("qty", 0.0),
                    "realized_pnl": exit_action.get("realized_pnl", 0.0)}

        if proposal.is_hold:
            return {"action": "none", "allowed": True, "veto_reason": "Hold: no change"}

        equity = self.broker.equity(self._prices)
        plan = risk.plan_entry(
            symbol=symbol, snapshot=snapshot, equity=equity, cash=self.broker.cash(),
            current_qty=current_qty, gross_exposure=self.broker.gross_exposure(self._prices),
            state=self.risk_state, limits=self.cfg.risk,
            conviction=proposal.conviction, now=now,
        )

        # An Underweight rating on an oversized position is a trim, not an entry.
        if not plan.allowed:
            target_qty = risk.size_position(equity, snapshot.price, snapshot.atr14 or 0.0,
                                            self.cfg.risk, proposal.conviction)
            if current_qty > 0 and target_qty < current_qty * 0.9:
                trim = current_qty - target_qty
                notional = trim * snapshot.price
                if notional >= self.cfg.risk.min_order_notional:
                    return self._trim(symbol, trim, snapshot.price, ts, proposal)
            return {"action": "none", "allowed": False, "veto_reason": plan.reason}

        try:
            fill = self.broker.market_order(symbol, "buy", plan.qty, snapshot.price,
                                           ts, f"{proposal.rating}: {proposal.summary}"[:200])
        except OrderRejected as exc:
            self.ledger.record_event("warning", "order", f"{symbol}: {exc}", ts=ts)
            return {"action": "none", "allowed": False, "veto_reason": str(exc)}

        self.ledger.record_fill(fill.to_dict())
        new_pos = self.broker.position(symbol)
        if new_pos is not None:
            # Set the protective stop immediately; an unstopped position is the
            # one that turns a bad tick into a bad month.
            new_pos.stop_price = (plan.stop_price if new_pos.stop_price is None
                                  else max(new_pos.stop_price, plan.stop_price))
            new_pos.extreme_price = max(new_pos.extreme_price, snapshot.price)
        self.ledger.record_event(
            "info", "entry",
            f"{symbol} {'added' if current_qty > 0 else 'entered'} "
            f"{fill.qty:.6f} @ {fill.price:.4f} ({proposal.rating})", ts=ts)
        return {"action": "added" if current_qty > 0 else "entered", "allowed": True,
                "qty": fill.qty, "notional": fill.notional,
                "stop_price": plan.stop_price, "reason": plan.reason}

    def _trim(self, symbol: str, qty: float, price: float, ts: int,
              proposal: Proposal) -> dict:
        """Reduce an oversized position toward its target weight."""
        try:
            fill = self.broker.market_order(symbol, "sell", qty, price, ts,
                                            f"trim to {proposal.rating}")
        except OrderRejected as exc:
            return {"action": "none", "allowed": False, "veto_reason": str(exc)}
        self.ledger.record_fill(fill.to_dict())
        self.ledger.record_event("info", "trim",
                                 f"{symbol} trimmed {fill.qty:.6f} @ {fill.price:.4f}", ts=ts)
        return {"action": "reduced", "allowed": True, "qty": fill.qty,
                "notional": fill.notional, "realized_pnl": fill.realized_pnl}

    # ---- halt / resume --------------------------------------------------
    def _halt(self, reason: str, ts: int) -> list[dict]:
        """Flatten everything and stop trading until a human resumes."""
        actions = []
        for fill in self.broker.close_all(self._prices, ts, "kill-switch"):
            self.ledger.record_fill(fill.to_dict())
            actions.append({"symbol": fill.symbol, "action": "exited",
                            "reason": "kill-switch", "qty": fill.qty,
                            "realized_pnl": fill.realized_pnl})
        self.risk_state.halted = True
        self.risk_state.halt_reason = reason
        logger.error("DESK HALTED: %s", reason)
        self.ledger.record_event("error", "halt", f"Desk halted: {reason}", ts=ts)
        return actions

    def resume(self, reset_peak: bool = True) -> None:
        """Clear a halt. Deliberately manual — the kill-switch is not self-clearing.

        ``reset_peak`` re-bases the high-water mark to current equity; without
        it the desk would re-halt on the next tick, since the drawdown that
        triggered the halt is still measured from the old peak.
        """
        with self._lock:
            self.risk_state.halted = False
            self.risk_state.halt_reason = ""
            if reset_peak:
                self.risk_state.peak_equity = self.broker.equity(self._prices)
            ts = int(self.clock())
            self.risk_state.day_start_equity = self.broker.equity(self._prices)
            self.ledger.record_event("info", "resume", "Desk resumed by operator", ts=ts)
            self._persist(ts)

    def halt(self, reason: str = "manual halt") -> None:
        """Operator-initiated halt (flattens the book)."""
        with self._lock:
            self._halt(reason, int(self.clock()))
            self._persist(int(self.clock()))

    def _persist(self, ts: int) -> None:
        self.ledger.set_state(_RISK_STATE_KEY, self.risk_state.to_dict())
        self.ledger.set_state(_BROKER_STATE_KEY, self.broker.state())

    # ---- runner ---------------------------------------------------------
    def run_forever(self) -> None:
        """Tick on a timer until :meth:`stop` is called.

        A failing tick is logged and retried on the next interval rather than
        killing the thread: a transient feed or database error must not silently
        leave positions unmanaged.
        """
        logger.info("Desk started: %s on %s (committee=%s, feed=%s)",
                    ", ".join(self.cfg.symbols), self.cfg.candle_interval,
                    self.committee.name, self.feed.name)
        self.ledger.record_event("info", "start",
                                 f"Desk started with {len(self.cfg.symbols)} symbols, "
                                 f"committee={self.committee.name}, feed={self.feed.name}")
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                self.last_error = str(exc)
                logger.exception("Tick failed")
                try:
                    self.ledger.record_event("error", "tick", f"Tick failed: {exc}")
                except Exception:  # pragma: no cover - ledger itself is broken
                    logger.exception("Could not record tick failure")
            elapsed = time.monotonic() - started
            self._stop.wait(max(1.0, self.cfg.fast_loop_seconds - elapsed))

    def stop(self) -> None:
        self._stop.set()

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="cryptodesk-engine",
                                  daemon=True)
        thread.start()
        return thread

    # ---- reporting ------------------------------------------------------
    def state(self) -> dict:
        """Everything the dashboard shows, as plain JSON-safe data."""
        with self._lock:
            prices = dict(self._prices)
            equity = self.broker.equity(prices) if prices else self.broker.starting_equity
            account = self.broker.snapshot(prices)
            positions = [
                pos.to_dict(prices.get(symbol, pos.avg_price))
                for symbol, pos in self.broker.positions().items()
            ]
            return {
                "now": int(self.clock()),
                "started_ts": self.started_ts,
                "last_tick_ts": self.last_tick_ts,
                "tick_count": self.tick_count,
                "committee": self.committee.name,
                "feed": self.feed.name,
                "feed_errors": dict(self.feed_errors),
                "last_error": self.last_error,
                "symbols": list(self.cfg.symbols),
                "account": account,
                "positions": positions,
                "risk": {
                    "halted": self.risk_state.halted,
                    "halt_reason": self.risk_state.halt_reason,
                    "drawdown": self.risk_state.drawdown(equity),
                    "peak_equity": self.risk_state.peak_equity,
                    "day_pnl_pct": self.risk_state.day_pnl_pct(equity),
                    "day_start_equity": self.risk_state.day_start_equity,
                    "cooldowns": dict(self.risk_state.cooldowns),
                    "limits": {
                        "max_symbol_weight": self.cfg.risk.max_symbol_weight,
                        "max_gross_exposure": self.cfg.risk.max_gross_exposure,
                        "risk_per_trade": self.cfg.risk.risk_per_trade,
                        "daily_loss_limit": self.cfg.risk.daily_loss_limit,
                        "max_drawdown_halt": self.cfg.risk.max_drawdown_halt,
                    },
                },
                "llm": {
                    "spend_today": self.ledger.spend_today(self.clock()),
                    "spend_total": self.ledger.spend_total(),
                    "daily_cap": self.cfg.llm.daily_usd_cap,
                },
                "indicators": {s: snap.to_dict() for s, snap in self._snapshots.items()},
                "trade_stats": self.ledger.trade_stats(),
            }
