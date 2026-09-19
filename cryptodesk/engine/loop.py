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

The committee is the one slow step (an LLM run is minutes), so the engine lock
is held in two short phases *around* it rather than across it: mark, protect
and collect the symbols whose triggers fired; release; run the committee for
each; re-acquire and act on the proposal against the price as it is *then*.
Stops, the kill-switch, ``halt()`` and the dashboard therefore never wait on a
model, and an order is never placed at a quote that is minutes old.

Every fill is written to the ledger in one transaction with the broker book
and risk state that resulted from it, so a crash at any instant restarts a
desk whose blotter and book agree.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from ..broker import OrderRejected, PaperBroker
from ..config import DeskConfig
from ..feeds import build_feed
from ..feeds.base import FeedError, interval_seconds
from ..indicators import Snapshot, snapshot as compute_snapshot
from . import risk, triggers
from .committee import Proposal, build_committee
from .ledger import Ledger

logger = logging.getLogger(__name__)

_RISK_STATE_KEY = "risk_state"
_BROKER_STATE_KEY = "broker_state"


@dataclass
class _Booking:
    """Ledger writes that must land together.

    A fill, the event or decision that explains it, and — added by
    :meth:`Desk._commit` — the broker book and risk state that resulted. One
    transaction for all of them means a restart can never see a fill its
    book does not hold, or a book holding a fill the blotter never saw.
    """

    fills: list[dict] = field(default_factory=list)
    events: list[tuple] = field(default_factory=list)
    decision: dict | None = None

    def fill(self, fill) -> None:
        self.fills.append(fill.to_dict())

    def event(self, level: str, kind: str, message: str, ts: int) -> None:
        self.events.append((level, kind, message, ts))

    def __bool__(self) -> bool:
        return bool(self.fills or self.events or self.decision is not None)


class Desk:
    """A 24/7 paper-trading desk over a set of crypto symbols."""

    def __init__(self, cfg: DeskConfig, feed=None, broker=None, ledger=None,
                 committee=None, clock=time.time):
        cfg.ensure_dirs()
        self.cfg = cfg
        self.clock = clock
        # The desk's clock and interval reach the feeds too, so the live feeds'
        # closed-bar rule, the chain's breaker and the synthetic mark all agree
        # with the engine on what "now" and "a bar" are.
        self.feed = feed or build_feed(cfg.feeds, interval=cfg.candle_interval, clock=clock)
        self._interval_s = interval_seconds(cfg.candle_interval)
        self.ledger = ledger or Ledger(cfg.db_path)
        self.broker = broker or PaperBroker(
            starting_equity=cfg.starting_equity, fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps, allow_shorts=cfg.risk.allow_shorts,
        )
        self.committee = committee or build_committee(cfg)
        self.risk_state = risk.RiskState.from_dict(self.ledger.get_state(_RISK_STATE_KEY))
        if self.risk_state.first_benchmark_price is None:
            # A ledger from before this field existed: anchor buy-and-hold at
            # its first recorded mark, not at whatever tick happens next.
            self.risk_state.first_benchmark_price = self.ledger.first_benchmark_price()
        # Restore the book before the first tick, so a restarted desk manages the
        # positions it actually holds instead of believing it is flat.
        self.restored = self.broker.load_state(self.ledger.get_state(_BROKER_STATE_KEY))
        if self.restored:
            self.ledger.record_event(
                "info", "restore",
                f"Restored {len(self.broker.positions())} position(s) and "
                f"${self.broker.cash():,.2f} cash from the ledger",
                ts=int(self.clock()))
            orphans = self.held_not_configured()
            if orphans:
                self.ledger.record_event(
                    "warning", "restore",
                    f"Held but no longer configured: {', '.join(orphans)}. "
                    "Priced and stop-managed, exit-only; never re-entered",
                    ts=int(self.clock()))
        # Per-symbol memory the trigger logic needs between ticks. Seeded from
        # the ledger so a restart does not silence the ATR-move and trend-flip
        # triggers until the next scheduled review.
        self._last_decision_price: dict[str, float] = {}
        self._last_trend: dict[str, str] = {}
        for symbol in cfg.symbols:
            self._seed_trigger_memory(symbol)
        self._snapshots: dict[str, Snapshot] = {}
        self._prices: dict[str, float] = {}
        # Freshness bookkeeping: when each symbol last had a good quote, which
        # venue served it, and which symbols are currently stale.
        self._last_good_ts: dict[str, int] = {}
        self._feed_venue: dict[str, str] = {}
        self._stale: set[str] = set()
        self._all_stale = False
        # Positions a halt could not close because their quote was stale.
        self._unflattened: set[str] = set()
        self._benchmark_price: float | None = None
        self._committee_switch_reported: str | None = None
        self._lock = threading.RLock()
        # Serialises committee runs (the graph is single-threaded) without
        # holding the engine lock across them.
        self._committee_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_ts = int(self.clock())
        # Liveness: bumped at every phase boundary so a health check can tell
        # "thinking for five minutes" from "dead", and ``busy`` says which.
        self.heartbeat_ts = self.started_ts
        self.busy: str | None = None
        self.last_tick_ts: int | None = None
        self.tick_count = 0
        self.last_error: str | None = None
        self.feed_errors: dict[str, str] = {}

    def _seed_trigger_memory(self, symbol: str) -> None:
        row = self.ledger.last_decision(symbol)
        if not row:
            return
        if row.get("price") is not None:
            self._last_decision_price[symbol] = float(row["price"])
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        trend = detail.get("trend") or (detail.get("indicators") or {}).get("trend")
        if trend:
            self._last_trend[symbol] = str(trend)

    def stale_symbols(self) -> list[str]:
        """Symbols whose quote is currently frozen. Cheap: no lock, no ledger.

        The health endpoint reads this from the API thread; ``state()`` would
        take the engine lock and run the trade statistics for the same answer.
        """
        return sorted(self._stale)

    def held_not_configured(self) -> list[str]:
        """Symbols the book holds that ``cfg.symbols`` no longer lists."""
        return sorted(s for s in self.broker.positions() if s not in self.cfg.symbols)

    # ---- market data ---------------------------------------------------
    def _symbols_to_price(self) -> list[str]:
        # Configured symbols plus anything held: a restored position whose
        # symbol left the config must still be priced and stop-managed, or it
        # sits at its entry mark with no exit until someone edits the database.
        return list(dict.fromkeys([*self.cfg.symbols, *self.broker.positions()]))

    def _refresh(self, now: float) -> None:
        """Pull candles and prices for every symbol, tolerating per-symbol failure.

        One unreachable symbol must not stop the desk from managing the others —
        in particular it must not stop a stop-loss on a different symbol.
        """
        wanted = self._symbols_to_price()
        for symbol in wanted:
            self._refresh_symbol(symbol, now)
        # A symbol that was only priced because it was held is forgotten once
        # it is flat; otherwise its frozen quote would count as stale forever.
        for symbol in [s for s in self._prices if s not in wanted]:
            for table in (self._prices, self._snapshots, self._last_good_ts,
                          self._feed_venue, self.feed_errors):
                table.pop(symbol, None)
        self._refresh_benchmark(now)

    def _refresh_symbol(self, symbol: str, now: float) -> None:
        ts = int(now)
        try:
            candles, price, venue = self.feed.market(
                symbol, self.cfg.candle_interval, self.cfg.candle_lookback)
            self._snapshots[symbol] = compute_snapshot(symbol, candles, price)
            self._prices[symbol] = price
            self._last_good_ts[symbol] = ts
        except (FeedError, ValueError) as exc:
            message = str(exc)
            # One event per distinct failure, not per tick: an outage would
            # otherwise write a row a minute for as long as it lasts.
            if self.feed_errors.get(symbol) != message:
                logger.warning("Feed failure for %s: %s", symbol, exc)
                self.ledger.record_event("warning", "feed", f"{symbol}: {message}", ts=ts)
            self.feed_errors[symbol] = message
            return
        self.feed_errors.pop(symbol, None)
        previous = self._feed_venue.get(symbol)
        if previous is not None and previous != venue:
            # The chain fell through. Worth a row: the basis between venues
            # can move a stop on its own, and a primary that is always failing
            # should not be invisible behind its fallback.
            self.ledger.record_event("warning", "feed",
                                     f"{symbol}: venue changed {previous} -> {venue}", ts=ts)
        self._feed_venue[symbol] = venue

    def _refresh_benchmark(self, now: float) -> None:
        bench = self.cfg.benchmark_symbol
        if bench in self._prices:
            self._benchmark_price = None
            return
        # Not traded, so not part of the symbol refresh: fetched on its own so
        # buy-and-hold is still measurable. A miss writes NULL for this mark
        # rather than repeating the last one.
        try:
            self._benchmark_price = float(self.feed.price(bench))
            self.feed_errors.pop(bench, None)
        except (FeedError, ValueError) as exc:
            self._benchmark_price = None
            if self.feed_errors.get(bench) != str(exc):
                logger.warning("Benchmark %s unavailable: %s", bench, exc)
            self.feed_errors[bench] = str(exc)

    def _benchmark_mark(self) -> float | None:
        bench = self.cfg.benchmark_symbol
        if bench in self._prices:
            return None if bench in self._stale else self._prices[bench]
        return self._benchmark_price

    # ---- freshness -----------------------------------------------------
    def _is_stale(self, symbol: str, now: float) -> bool:
        last_good = self._last_good_ts.get(symbol)
        snap = self._snapshots.get(symbol)
        if last_good is None or snap is None:
            return True
        if now - last_good > max(2 * self.cfg.fast_loop_seconds, self._interval_s):
            return True
        # The venue answered, but with a tape that stopped: the last closed
        # bar is more than two bars old.
        return now - (snap.ts + self._interval_s) > 2 * self._interval_s

    def _update_staleness(self, now: float) -> None:
        """Recompute the stale set, recording one event per transition."""
        ts = int(now)
        stale = {s for s in self._prices if self._is_stale(s, now)}
        for symbol in sorted(stale - self._stale):
            age = ts - self._last_good_ts.get(symbol, ts)
            self.ledger.record_event(
                "warning", "stale",
                f"{symbol}: market data stale (last good quote {age}s ago); "
                "stop exits and entries suspended until it is fresh again", ts=ts)
        for symbol in sorted(self._stale - stale):
            self.ledger.record_event("info", "stale", f"{symbol}: market data fresh again", ts=ts)
        self._stale = stale
        all_stale = bool(self._prices) and stale >= set(self._prices)
        if all_stale and not self._all_stale:
            self.ledger.record_event("warning", "stale",
                                     "Every symbol is stale; equity marks suspended", ts=ts)
        elif self._all_stale and not all_stale:
            self.ledger.record_event("info", "stale", "Equity marks resumed", ts=ts)
        self._all_stale = all_stale

    def _fresh_prices(self) -> dict[str, float]:
        return {s: p for s, p in self._prices.items() if s not in self._stale}

    # ---- the tick ------------------------------------------------------
    def tick(self, now: float | None = None) -> dict:
        """Run one full cycle: mark, protect, then consider new risk.

        Synchronous and deterministic: the committee runs inline, but with
        the engine lock released, so ``state()``/``halt()`` from another
        thread proceed while it thinks.
        """
        started = self.clock()
        now = float(now if now is not None else started)

        def later() -> float:
            # The tick's own time: the caller's ``now`` plus whatever the desk
            # clock has advanced since. A committee run is thereby charged to
            # the timestamps that follow it, not hidden behind the start stamp.
            return now + (self.clock() - started)

        with self._lock:
            result, to_consult = self._protect(now)

        for symbol, snapshot, trigger in to_consult:
            proposal = self._consult(symbol, snapshot, trigger, later)
            with self._lock:
                actions = self._decide(symbol, snapshot, trigger, proposal, now, later())
            result["consulted"].append(symbol)
            result["actions"].extend(actions)

        with self._lock:
            self._finish(result, later())
        return result

    def _protect(self, now: float) -> tuple[dict, list[tuple[str, Snapshot, triggers.Trigger]]]:
        """Phase one, under the lock: refresh, mark, brake, protect, and pick
        the symbols the committee should look at."""
        ts = int(now)
        self.tick_count += 1
        self.heartbeat_ts = ts
        actions: list[dict] = []

        self._refresh(now)
        self._update_staleness(now)
        if not self._prices:
            return {"ts": ts, "error": "no market data for any symbol",
                    "actions": [], "consulted": []}, []

        # 1. Mark the book and update the risk high-water marks. Stale symbols
        # are marked at their last quote: dropping them would make equity jump
        # and trip the kill-switch on a feed hiccup. A tick with *no* fresh
        # quote at all is not a mark, though, so it writes no equity point.
        equity = self.broker.equity(self._prices)
        self.risk_state.observe(equity, now)
        drawdown = self.risk_state.drawdown(equity)
        benchmark = self._benchmark_mark()
        if benchmark is not None and self.risk_state.first_benchmark_price is None:
            self.risk_state.first_benchmark_price = benchmark
        if not self._all_stale:
            snap = self.broker.snapshot(self._prices)
            self.ledger.record_equity(
                ts=ts, equity=equity, cash=self.broker.cash(),
                gross_exposure=snap["gross_exposure"], drawdown=drawdown,
                realized_pnl=snap["total_realized_pnl"],
                unrealized_pnl=snap["unrealized_pnl"], fees=snap["total_fees"],
                benchmark_price=benchmark,
            )

        # 2. Kill-switch, before anything else can add risk.
        if not self.risk_state.halted and drawdown >= self.cfg.risk.max_drawdown_halt:
            actions.extend(self._halt(
                f"drawdown {drawdown:.1%} breached the {self.cfg.risk.max_drawdown_halt:.1%} limit",
                ts))

        if self.risk_state.halted:
            actions.extend(self._retry_flatten(ts))
            self._persist(ts)
            return {"ts": ts, "equity": equity, "drawdown": drawdown,
                    "halted": True, "halt_reason": self.risk_state.halt_reason,
                    "unflattened": sorted(self.broker.positions()),
                    "stale_symbols": sorted(self._stale),
                    "actions": actions, "consulted": []}, []

        # 3. Protect open positions: trail stops up, exit anything stopped out,
        # trim what mark-to-market has carried past a cap.
        actions.extend(self._manage_positions(ts))
        actions.extend(self._gross_drift_trim(ts))
        self.heartbeat_ts = ts

        # 4. Decide who the committee should look at. Only configured symbols
        # can be entered; a held-but-unconfigured one is exit-only via stops.
        to_consult = []
        for symbol in self.cfg.symbols:
            snapshot = self._snapshots.get(symbol)
            if snapshot is None or symbol in self._stale:
                continue
            trigger = self._evaluate_trigger(symbol, snapshot, now)
            if trigger is not None:
                to_consult.append((symbol, snapshot, trigger))

        return {
            "ts": ts, "equity": equity, "drawdown": drawdown, "halted": False,
            "actions": actions, "prices": dict(self._prices),
            "stale_symbols": sorted(self._stale), "consulted": [],
        }, to_consult

    def _finish(self, result: dict, when: float) -> None:
        ts = int(when)
        if "error" not in result and not result.get("halted"):
            self._persist(ts)
            self.last_error = None
        # Stamped at completion: a health check compares this to the clock,
        # and a tick that took minutes is still a tick that finished.
        self.last_tick_ts = ts
        self.heartbeat_ts = ts

    # ---- position management -------------------------------------------
    def _manage_positions(self, ts: int) -> list[dict]:
        """Ratchet trailing stops and exit positions whose stop has been hit."""
        actions = []
        for symbol, pos in list(self.broker.positions().items()):
            actions.extend(self._manage_one(symbol, pos, ts))
        return actions

    def _manage_one(self, symbol: str, pos, ts: int) -> list[dict]:
        price = self._prices.get(symbol)
        snapshot = self._snapshots.get(symbol)
        if price is None or snapshot is None or snapshot.atr14 is None:
            # Without a price or an ATR the stop cannot be evaluated
            # honestly; leave the position untouched rather than guess.
            return []
        limits = self.cfg.risk
        if pos.stop_price is None:
            # A position that arrived without a stop (a book written by an
            # older version, an order placed straight on the broker) gets the
            # entry stop, not the wider trailing one the ratchet would seed.
            pos.stop_price = min(price, pos.avg_price) - snapshot.atr14 * limits.stop_atr_mult
        new_stop, extreme = risk.update_trailing_stop(
            pos.avg_price, pos.stop_price, pos.extreme_price or pos.avg_price,
            price, snapshot.atr14, limits,
        )
        pos.stop_price, pos.extreme_price = new_stop, extreme

        if risk.stop_hit(pos.stop_price, price, pos.is_long):
            if symbol in self._stale:
                # A stop-out on a frozen quote is not information. The stop
                # stays armed and fires on the first fresh price.
                return [{"symbol": symbol, "action": "stop_deferred",
                         "reason": f"stale: stop {pos.stop_price:.4f} not evaluated "
                                   "on a frozen quote"}]
            booking = _Booking()
            action = self._exit(symbol, price, ts, f"stop hit at {pos.stop_price:.4f}",
                                booking, cooldown=True)
            self._commit(booking)
            return [action]

        if symbol in self._stale:
            return []
        booking = _Booking()
        trim = self._weight_drift_trim(symbol, pos, price, ts, booking)
        self._commit(booking)
        return [trim] if trim else []

    def _weight_drift_trim(self, symbol: str, pos, price: float, ts: int,
                           booking: _Booking) -> dict | None:
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
        booking.fill(fill)
        booking.event(
            "info", "trim",
            f"{symbol} trimmed to its {limits.max_symbol_weight:.0%} cap "
            f"(had drifted to {weight:.1%})", ts)
        return {"symbol": symbol, "action": "reduced", "reason": "weight drift",
                "qty": fill.qty, "realized_pnl": fill.realized_pnl}

    def _gross_drift_trim(self, ts: int) -> list[dict]:
        """Trim the whole book when its gross exposure has drifted past the cap.

        Per-symbol trims bound each position but not their sum: four symbols
        at 24% each pass every per-symbol check while the account is 96%
        deployed. The gross cap exists to leave cash so a drawdown never
        forces a liquidation, so it gets the same drift band and the same
        treatment. The excess is spread pro-rata over the positions that can
        actually be sold (fresh quotes only), largest first.
        """
        limits = self.cfg.risk
        gross = self.broker.gross_exposure(self._prices)
        if gross <= limits.max_gross_exposure * (1 + limits.max_weight_drift):
            return []
        equity = self.broker.equity(self._prices)
        excess_notional = (gross - limits.max_gross_exposure) * equity
        fresh = self._fresh_prices()
        candidates = sorted(
            ((abs(pos.market_value(fresh[s])), s, pos)
             for s, pos in self.broker.positions().items() if s in fresh),
            key=lambda item: item[0], reverse=True)
        total = sum(mv for mv, _, _ in candidates)
        if total <= 0:
            return []

        actions = []
        for mv, symbol, pos in candidates:
            price = fresh[symbol]
            qty = min(excess_notional * mv / total / price, abs(pos.qty))
            if qty * price < limits.min_order_notional:
                continue
            reason = f"gross drift {gross:.1%} over {limits.max_gross_exposure:.0%} cap"
            try:
                fill = self.broker.market_order(
                    symbol, "sell" if pos.is_long else "buy", qty, price, ts, reason)
            except OrderRejected as exc:
                logger.warning("Could not trim %s for gross drift: %s", symbol, exc)
                continue
            booking = _Booking()
            booking.fill(fill)
            booking.event("info", "trim",
                          f"{symbol} trimmed {fill.qty:.6f} @ {fill.price:.4f}: {reason}", ts)
            self._commit(booking)
            actions.append({"symbol": symbol, "action": "reduced", "reason": "gross drift",
                            "qty": fill.qty, "realized_pnl": fill.realized_pnl})
        return actions

    def _exit(self, symbol: str, price: float, ts: int, reason: str,
              booking: _Booking, cooldown: bool = False) -> dict:
        """Flatten a symbol, queueing the fill on ``booking``."""
        try:
            fill = self.broker.close(symbol, price, ts, reason)
        except OrderRejected as exc:
            booking.event("error", "exit", f"{symbol}: {exc}", ts)
            return {"symbol": symbol, "action": "exit_rejected", "reason": str(exc)}
        if fill is None:
            return {"symbol": symbol, "action": "none", "reason": "already flat"}
        if cooldown:
            self.risk_state.start_cooldown(symbol, self.cfg.risk.cooldown_minutes, ts)
        booking.fill(fill)
        booking.event(
            "info", "exit",
            f"{symbol} exited: {reason} (realised {fill.realized_pnl:+.2f})", ts)
        return {"symbol": symbol, "action": "exited", "reason": reason,
                "qty": fill.qty, "price": fill.price,
                "realized_pnl": fill.realized_pnl}

    # ---- decision path --------------------------------------------------
    def _evaluate_trigger(self, symbol: str, snapshot: Snapshot,
                          now: float) -> triggers.Trigger | None:
        """The trigger that fired for ``symbol``, or None."""
        pos = self.broker.position(symbol)
        if pos is None:
            # Flat, and the risk state already forbids entries: the run could
            # not change the book whatever it said, so it is not worth a cent.
            # Nothing is recorded; triggers are cheap and this is routine.
            blocked = triggers.entries_blocked(
                self.risk_state, self.broker.equity(self._prices),
                self.broker.gross_exposure(self._prices), self.cfg.risk, symbol, now)
            if blocked:
                return None
        ctx = triggers.TriggerContext(
            last_decision_ts=self.ledger.last_decision_ts(symbol),
            last_decision_price=self._last_decision_price.get(symbol),
            last_trend=self._last_trend.get(symbol),
            has_position=pos is not None,
            spend_today=self.ledger.spend_today(now),
        )
        trigger = triggers.evaluate(snapshot, ctx, self.cfg.llm, now)
        return trigger if trigger.fired else None

    def _consult(self, symbol: str, snapshot: Snapshot, trigger: triggers.Trigger,
                 later) -> Proposal:
        """Phase two, with the engine lock released: ask the committee."""
        self.busy = f"committee {symbol}"
        self.heartbeat_ts = int(later())
        try:
            with self._committee_lock:
                return self.committee.decide(symbol, snapshot, trigger.reason, now=later())
        except Exception:
            # A committee that raises instead of returning an errored
            # Proposal is a bug, but a paid run may still have burned tokens
            # before it died; book the estimate so the daily cap sees it.
            if getattr(self.committee, "source", None) == "llm":
                self.ledger.record_spend(self.cfg.llm.estimated_cost_per_run_usd, symbol,
                                         self.committee.name, ok=False, ts=int(later()))
            raise
        finally:
            self.busy = None
            self.heartbeat_ts = int(later())

    def _decide(self, symbol: str, analysed: Snapshot, trigger: triggers.Trigger,
                proposal: Proposal, now: float, when: float) -> list[dict]:
        """Phase three, under the lock again: act on the proposal at today's price."""
        ts = int(when)
        if proposal.cost_usd:
            # Booked on its own, ahead of the decision: the daily cap must
            # count this run even if acting on it fails.
            self.ledger.record_spend(proposal.cost_usd, symbol, self.committee.name,
                                     ok=proposal.error is None, ts=ts)
        switched = getattr(self.committee, "switched_reason", None)
        if switched and switched != self._committee_switch_reported:
            self._committee_switch_reported = switched
            self.ledger.record_event("error", "committee",
                                     f"Committee switched to {self.committee.name}: {switched}",
                                     ts=ts)
        # The thesis was formed on ``analysed``; the triggers measure from it.
        self._last_decision_price[symbol] = analysed.price
        self._last_trend[symbol] = analysed.trend

        actions: list[dict] = []
        if when - now > self.cfg.fast_loop_seconds:
            # The committee outlived a loop period: the quote it was asked
            # about is history. Re-price, re-check the stop, then act on the
            # market as it is rather than as it was.
            self._refresh_symbol(symbol, when)
            self._update_staleness(when)
            pos = self.broker.position(symbol)
            if pos is not None:
                actions.extend(self._manage_one(symbol, pos, ts))

        current = self._snapshots.get(symbol)
        booking = _Booking()
        if current is None:
            action = {"action": "none", "allowed": False,
                      "veto_reason": f"no market data for {symbol}"}
        elif symbol in self._stale:
            action = {"action": "none", "allowed": False,
                      "veto_reason": f"stale: no fresh quote for {symbol}"}
        elif self.risk_state.halted:
            action = {"action": "none", "allowed": False,
                      "veto_reason": f"desk halted: {self.risk_state.halt_reason}"}
        else:
            action = self._apply(symbol, current, proposal, ts, when, booking)

        booking.decision = {
            "ts": ts, "symbol": symbol, "source": proposal.source,
            "trigger": trigger.reason, "rating": proposal.rating,
            "conviction": proposal.conviction, "action": action["action"],
            "allowed": action.get("allowed", False),
            "veto_reason": action.get("veto_reason"), "qty": action.get("qty", 0.0),
            "notional": action.get("notional", 0.0),
            "stop_price": action.get("stop_price"), "price": analysed.price,
            "cost_usd": proposal.cost_usd, "latency_ms": proposal.latency_ms,
            "summary": proposal.summary,
            "detail": {**proposal.detail, "trigger_detail": trigger.detail,
                       "cost_is_estimate": proposal.cost_is_estimate,
                       "trend": analysed.trend,
                       "applied_price": current.price if current else None,
                       "committee_seconds": round(when - now, 3)},
        }
        self._commit(booking)
        actions.append({"symbol": symbol, "trigger": trigger.reason,
                        "rating": proposal.rating, **action})
        return actions

    def _apply(self, symbol: str, snapshot: Snapshot, proposal: Proposal,
               ts: int, now: float, booking: _Booking) -> dict:
        """Turn a rating into orders, subject to the risk engine."""
        pos = self.broker.position(symbol)
        current_qty = pos.qty if pos else 0.0
        limits = self.cfg.risk
        price = snapshot.price

        if proposal.wants_exit:
            if current_qty <= 0:
                return {"action": "none", "allowed": True,
                        "veto_reason": "Sell rating but already flat"}
            exit_action = self._exit(symbol, price, ts,
                                     f"committee rating Sell: {proposal.summary}", booking)
            return {"action": exit_action["action"], "allowed": True,
                    "qty": exit_action.get("qty", 0.0),
                    "realized_pnl": exit_action.get("realized_pnl", 0.0)}

        if proposal.is_hold:
            return {"action": "none", "allowed": True, "veto_reason": "Hold: no change"}

        equity = self.broker.equity(self._prices)

        if proposal.rating == "Underweight":
            # The PM's own scale defines Underweight as "reduce exposure". It
            # never opens a position — with nothing held there is nothing to
            # reduce — and with one held it only ever sells toward its tier.
            if current_qty <= 0:
                return {"action": "none", "allowed": True,
                        "veto_reason": "Underweight with no position: not initiating"}
            tier = risk.RATING_WEIGHTS["Underweight"]
            if snapshot.atr14 is None:
                return {"action": "none", "allowed": False,
                        "veto_reason": f"no ATR for {symbol}; cannot size the Underweight target"}
            target_qty = risk.size_position(equity, price, snapshot.atr14, limits, tier)
            return self._trim_toward(symbol, current_qty, target_qty, price, ts,
                                     f"Underweight: trim toward {tier:.0%} of max", booking)

        plan = risk.plan_entry(
            symbol=symbol, snapshot=snapshot, equity=equity, cash=self.broker.cash(),
            current_qty=current_qty, gross_exposure=self.broker.gross_exposure(self._prices),
            state=self.risk_state, limits=limits,
            conviction=proposal.conviction, now=now,
            cost_rate=(self.cfg.fee_bps + self.cfg.slippage_bps) / 10_000,
        )

        if not plan.allowed:
            # A refused Buy/Overweight on a position the vol target now says
            # is oversized (ATR rose since entry) is rebalanced down. The
            # reason spells that out: a sell under a Buy rating must never
            # reach the decision log unexplained.
            if current_qty > 0 and snapshot.atr14 is not None:
                target_qty = risk.size_position(equity, price, snapshot.atr14, limits,
                                                proposal.conviction)
                if target_qty < current_qty * 0.9:
                    trimmed = self._trim_toward(
                        symbol, current_qty, target_qty, price, ts,
                        f"vol-target rebalance: target {target_qty:.6f} < 90% of held "
                        f"{current_qty:.6f} ({plan.reason})", booking)
                    if trimmed["action"] == "reduced":
                        return trimmed
            return {"action": "none", "allowed": False, "veto_reason": plan.reason}

        try:
            fill = self.broker.market_order(symbol, "buy", plan.qty, price,
                                           ts, f"{proposal.rating}: {proposal.summary}"[:200])
        except OrderRejected as exc:
            booking.event("warning", "order", f"{symbol}: {exc}", ts)
            return {"action": "none", "allowed": False, "veto_reason": str(exc)}

        new_pos = self.broker.position(symbol)
        if new_pos is not None:
            # Stop first, ledger second. An unstopped position is the one that
            # turns a bad tick into a bad month, and a write that fails after
            # the fill must not leave one behind.
            new_pos.stop_price = (plan.stop_price if new_pos.stop_price is None
                                  else max(new_pos.stop_price, plan.stop_price))
            new_pos.extreme_price = max(new_pos.extreme_price, price)
        booking.fill(fill)
        booking.event(
            "info", "entry",
            f"{symbol} {'added' if current_qty > 0 else 'entered'} "
            f"{fill.qty:.6f} @ {fill.price:.4f} ({proposal.rating})", ts)
        return {"action": "added" if current_qty > 0 else "entered", "allowed": True,
                "qty": fill.qty, "notional": fill.notional,
                "stop_price": plan.stop_price, "reason": plan.reason}

    def _trim_toward(self, symbol: str, current_qty: float, target_qty: float,
                     price: float, ts: int, reason: str, booking: _Booking) -> dict:
        """Sell down to ``target_qty`` if the excess is worth an order."""
        excess = current_qty - target_qty
        limits = self.cfg.risk
        if excess <= 0:
            return {"action": "none", "allowed": True,
                    "veto_reason": f"{reason}: already at or below target"}
        if excess * price < limits.min_order_notional:
            return {"action": "none", "allowed": True,
                    "veto_reason": f"{reason}: excess ${excess * price:.2f} below the "
                                   f"${limits.min_order_notional:.2f} minimum order"}
        return self._trim(symbol, excess, price, ts, reason, booking)

    def _trim(self, symbol: str, qty: float, price: float, ts: int,
              reason: str, booking: _Booking) -> dict:
        """Reduce an oversized position toward its target weight."""
        try:
            fill = self.broker.market_order(symbol, "sell", qty, price, ts, reason[:200])
        except OrderRejected as exc:
            return {"action": "none", "allowed": False, "veto_reason": str(exc)}
        booking.fill(fill)
        booking.event("info", "trim",
                      f"{symbol} trimmed {fill.qty:.6f} @ {fill.price:.4f}: {reason}", ts)
        # The reason rides in ``veto_reason`` too, so the decision row itself
        # explains why a non-Sell rating produced a sell.
        return {"action": "reduced", "allowed": True, "qty": fill.qty,
                "notional": fill.notional, "realized_pnl": fill.realized_pnl,
                "reason": reason, "veto_reason": reason}

    # ---- halt / resume --------------------------------------------------
    def _flatten(self, ts: int, reason: str, booking: _Booking) -> list[dict]:
        """Close every position that has a fresh quote."""
        actions = []
        for fill in self.broker.close_all(self._fresh_prices(), ts, reason):
            booking.fill(fill)
            actions.append({"symbol": fill.symbol, "action": "exited",
                            "reason": reason, "qty": fill.qty,
                            "realized_pnl": fill.realized_pnl})
        return actions

    def _halt(self, reason: str, ts: int) -> list[dict]:
        """Flatten everything and stop trading until a human resumes."""
        # Flags first, and committed with the fills: a crash between the two
        # would restart un-halted with the positions it just closed back on
        # the book, and close them again.
        self.risk_state.halted = True
        self.risk_state.halt_reason = reason
        booking = _Booking()
        actions = self._flatten(ts, "kill-switch", booking)
        logger.error("DESK HALTED: %s", reason)
        booking.event("error", "halt", f"Desk halted: {reason}", ts)
        left = sorted(self.broker.positions())
        if left:
            # A stale quote is no price to close at; the position stays, its
            # stop stays, and every tick retries until a fresh quote arrives.
            booking.event("error", "halt",
                          f"Could not flatten {', '.join(left)}: no fresh quote; "
                          "retrying every tick", ts)
        self._unflattened = set(left)
        self._commit(booking)
        return actions

    def _retry_flatten(self, ts: int) -> list[dict]:
        """While halted, keep closing whatever the halt could not price."""
        if not self.broker.positions():
            return []
        booking = _Booking()
        actions = self._flatten(ts, "kill-switch retry", booking)
        left = set(self.broker.positions())
        if left != self._unflattened:
            # Once per change, not per tick: a symbol that stays dark for an
            # hour is one fact, not sixty.
            if left:
                booking.event("error", "halt",
                              f"Still unable to flatten {', '.join(sorted(left))}: "
                              "no fresh quote", ts)
            else:
                booking.event("info", "halt", "Book flat after the halt", ts)
            self._unflattened = left
        self._commit(booking)
        return actions

    def resume(self, reset_peak: bool = True) -> None:
        """Clear a halt. Deliberately manual — the kill-switch is not self-clearing.

        Raises ``RuntimeError`` when the desk is not halted: on a running desk
        this would only re-base the brakes.

        ``reset_peak`` re-bases the high-water mark to current equity; without
        it the desk would re-halt on the next tick, since the drawdown that
        triggered the halt is still measured from the old peak.

        The day's starting equity is never re-based here. The daily-loss
        limit is a brake on the *day*, and a halt/resume cycle must not hand
        back the budget the day already spent; it rolls at 00:00 UTC as usual.
        """
        with self._lock:
            if not self.risk_state.halted:
                raise RuntimeError("desk is not halted")
            self.risk_state.halted = False
            self.risk_state.halt_reason = ""
            if reset_peak:
                self.risk_state.peak_equity = self.broker.equity(self._prices)
            self._unflattened = set()
            ts = int(self.clock())
            self.ledger.record_event(
                "info", "resume",
                "Desk resumed by operator" + (" (peak re-based)" if reset_peak else ""),
                ts=ts)
            self._persist(ts)

    def halt(self, reason: str = "manual halt") -> None:
        """Operator-initiated halt (flattens the book). Never waits on the committee."""
        with self._lock:
            self._halt(reason, int(self.clock()))

    # ---- persistence ----------------------------------------------------
    def _persist(self, ts: int) -> None:
        with self.ledger.transaction() as tx:
            tx.set_state(_RISK_STATE_KEY, self.risk_state.to_dict())
            tx.set_state(_BROKER_STATE_KEY, self.broker.state())

    def _commit(self, booking: _Booking) -> None:
        """Write a booking and the book that resulted from it in one transaction."""
        if not booking:
            return
        with self.ledger.transaction() as tx:
            for fill in booking.fills:
                tx.record_fill(fill)
            for level, kind, message, ts in booking.events:
                tx.record_event(level, kind, message, ts=ts)
            if booking.decision is not None:
                tx.record_decision(**booking.decision)
            tx.set_state(_BROKER_STATE_KEY, self.broker.state())
            tx.set_state(_RISK_STATE_KEY, self.risk_state.to_dict())

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
        try:
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
        finally:
            # However the loop ended, leave the ledger's book equal to the
            # broker's: every fill is already committed with its state, and
            # this is the last chance to write the marks that came after.
            try:
                with self._lock:
                    self._persist(int(self.clock()))
            except Exception:  # noqa: BLE001 - nothing left to do but say so
                logger.exception("Could not persist state on shutdown")

    def stop(self, timeout: float | None = None) -> None:
        """Ask the engine thread to stop and wait up to ``timeout`` seconds for it.

        A committee run in flight is not interrupted; whatever it filled
        before the deadline is on disk regardless (see :meth:`_commit`).
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="cryptodesk-engine",
                                  daemon=True)
        self._thread = thread
        thread.start()
        return thread

    # ---- reporting ------------------------------------------------------
    def state(self) -> dict:
        """Everything the dashboard shows, as plain JSON-safe data.

        Takes the engine lock, which the committee phase does not hold, so
        this returns promptly however long a model run takes.
        """
        with self._lock:
            prices = dict(self._prices)
            account = self.broker.snapshot(prices)
            # One mark for the account block and the risk meters alike; a
            # fallback to starting equity would show a green drawdown meter
            # on a desk restored at its halt threshold.
            equity = account["equity"]
            positions = [
                {**pos.to_dict(prices.get(symbol, pos.avg_price)),
                 "stale": symbol in self._stale or symbol not in prices}
                for symbol, pos in self.broker.positions().items()
            ]
            health = getattr(self.feed, "health", None) or {}
            return {
                "now": int(self.clock()),
                "started_ts": self.started_ts,
                "last_tick_ts": self.last_tick_ts,
                "heartbeat_ts": self.heartbeat_ts,
                "busy": self.busy,
                "tick_count": self.tick_count,
                "committee": self.committee.name,
                "feed": self.feed.name,
                "feed_errors": dict(self.feed_errors),
                "feed_venue": dict(self._feed_venue),
                "feed_health": {name: dict(entry) for name, entry in health.items()},
                "stale_symbols": sorted(self._stale),
                "benchmark_price": self._benchmark_mark(),
                "last_error": self.last_error,
                "symbols": list(self.cfg.symbols),
                "held_not_configured": self.held_not_configured(),
                "account": account,
                "positions": positions,
                "risk": {
                    "halted": self.risk_state.halted,
                    "halt_reason": self.risk_state.halt_reason,
                    "unflattened": sorted(self._unflattened),
                    "drawdown": self.risk_state.drawdown(equity),
                    "max_drawdown": self.risk_state.max_drawdown,
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
