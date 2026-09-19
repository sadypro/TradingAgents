"""A paper broker that fills at real market prices, minus costs.

Two modelling choices worth stating plainly, because they are the difference
between a paper record that means something and one that flatters you:

* **Costs are charged on every fill.** A taker fee (default 10 bps) and
  slippage (default 5 bps, against you) are applied. Ignoring these is the most
  common reason a "profitable" bot loses money live — a strategy that trades
  often can be comfortably positive before costs and clearly negative after.
* **Fills are immediate and complete at the marked price.** This is the model's
  main optimism: it assumes your size never moves the book and a stop always
  fills at its trigger. For BTC/ETH in normal conditions and retail size that
  is close enough; in a crash, a real stop fills *worse* than modelled, so
  drawdowns here are a floor, not a ceiling.

Both are encoded, not assumed: ``fee_bps`` and ``slippage_bps`` are config.
"""

from __future__ import annotations

import logging
import time

from .base import Fill, OrderRejected, Position, Side

logger = logging.getLogger(__name__)

# Quantities below this are treated as flat, so floating-point residue from
# repeated partial exits cannot leave a ghost position that blocks re-entry.
_DUST = 1e-12


class PaperBroker:
    """Simulated execution against real prices, with cash and position accounting."""

    def __init__(self, starting_equity: float = 10_000.0, fee_bps: float = 10.0,
                 slippage_bps: float = 5.0, allow_shorts: bool = False):
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        self.starting_equity = float(starting_equity)
        self._cash = float(starting_equity)
        self.fee_rate = fee_bps / 10_000.0
        self.slippage_rate = slippage_bps / 10_000.0
        self.allow_shorts = allow_shorts
        self._positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.total_fees = 0.0
        self.total_realized_pnl = 0.0
        # Counts lifecycles opened, so two positions opened in the same second
        # still get distinct trade ids. Persisted with the rest of the state.
        self._trade_seq = 0

    # ---- introspection ------------------------------------------------
    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus marked positions.

        A symbol missing from ``prices`` is marked at its average entry price
        rather than dropped: dropping it would make equity jump every time a
        feed hiccuped, and a jump in equity trips the drawdown kill-switch.
        """
        total = self._cash
        for symbol, pos in self._positions.items():
            price = prices.get(symbol)
            if price is None:
                logger.warning("No price for %s; marking at entry", symbol)
                price = pos.avg_price
            total += pos.market_value(price)
        return total

    def gross_exposure(self, prices: dict[str, float]) -> float:
        equity = self.equity(prices)
        if equity <= 0:
            return 0.0
        notional = sum(
            abs(pos.market_value(prices.get(symbol, pos.avg_price)))
            for symbol, pos in self._positions.items()
        )
        return notional / equity

    # ---- execution ----------------------------------------------------
    def fill_price(self, side: Side, price: float) -> float:
        """Apply slippage in the direction that costs the trader."""
        if price <= 0:
            raise OrderRejected(f"Refusing to fill at non-positive price {price}")
        return price * (1 + self.slippage_rate) if side == "buy" else price * (1 - self.slippage_rate)

    def market_order(self, symbol: str, side: Side, qty: float, price: float,
                     ts: int | None = None, reason: str = "") -> Fill:
        """Fill a market order, updating cash, position and realised P&L."""
        if side not in ("buy", "sell"):
            raise OrderRejected(f"Unknown side {side!r}")
        qty = float(qty)
        if qty <= _DUST:
            raise OrderRejected(f"Order qty must be positive, got {qty}")

        ts = int(ts if ts is not None else time.time())
        exec_price = self.fill_price(side, price)
        notional = qty * exec_price
        fee = notional * self.fee_rate
        existing = self._positions.get(symbol)

        # Decide whether this order opens/increases or reduces/closes.
        if side == "buy":
            reducing = existing is not None and existing.qty < 0
        else:
            reducing = existing is not None and existing.qty > 0

        if reducing:
            fill = self._reduce(existing, symbol, side, qty, exec_price, price, fee, ts, reason)
        else:
            if side == "sell" and existing is None and not self.allow_shorts:
                raise OrderRejected(
                    f"Short selling is disabled; cannot sell {symbol} with no position"
                )
            fill = self._increase(symbol, side, qty, exec_price, price, fee, ts, reason)

        # Accumulate from the fill, not the requested qty: a reduce recomputes
        # the fee on what actually closed.
        self.total_fees += fill.fee
        self.fills.append(fill)
        return fill

    def _increase(self, symbol: str, side: Side, qty: float, exec_price: float,
                  ref_price: float, fee: float, ts: int, reason: str) -> Fill:
        """Open or add to a position."""
        notional = qty * exec_price
        if side == "buy":
            if notional + fee > self._cash + 1e-9:
                raise OrderRejected(
                    f"Insufficient cash for {symbol}: need {notional + fee:.2f}, "
                    f"have {self._cash:.2f}"
                )
            self._cash -= notional + fee
            signed_qty = qty
        else:
            # Opening a short credits the proceeds; margin is not modelled, which
            # is part of why shorts are off by default.
            self._cash += notional - fee
            signed_qty = -qty

        pos = self._positions.get(symbol)
        if pos is None:
            self._trade_seq += 1
            pos = Position(
                symbol=symbol, qty=signed_qty, avg_price=exec_price, opened_ts=ts,
                extreme_price=exec_price, fees_paid=fee, entry_reason=reason,
                trade_id=f"{symbol}-{ts}-{self._trade_seq}", entry_fees=fee,
            )
            self._positions[symbol] = pos
        else:
            total_qty = pos.qty + signed_qty
            # Weighted-average entry over the combined absolute size.
            pos.avg_price = (
                (pos.avg_price * abs(pos.qty)) + (exec_price * qty)
            ) / abs(total_qty)
            pos.qty = total_qty
            pos.fees_paid += fee
            pos.entry_fees += fee
        return Fill(ts=ts, symbol=symbol, side=side, qty=qty, price=exec_price,
                    reference_price=ref_price, fee=fee, realized_pnl=0.0, reason=reason,
                    trade_id=pos.trade_id)

    def _reduce(self, pos: Position, symbol: str, side: Side, qty: float,
                exec_price: float, ref_price: float, fee: float, ts: int,
                reason: str) -> Fill:
        """Reduce or close a position, booking realised P&L."""
        closing = min(qty, abs(pos.qty))
        if closing < qty - _DUST:
            # An order larger than the position would reverse direction. Even
            # with shorts enabled that is refused rather than silently filled
            # as a plain close: the caller asked for a position it would not
            # get, and the book must never disagree with what was reported.
            raise OrderRejected(
                f"Cannot {side} {qty} {symbol}: only {abs(pos.qty)} held; "
                "reversing direction in one order is not supported"
            )
        notional = closing * exec_price
        # Fee was computed on the requested qty; recompute on what actually filled.
        fee = notional * self.fee_rate
        direction = 1 if pos.is_long else -1
        # Release the share of entry fees this exit closes out, so a round trip
        # at a flat price realises exactly its costs (see Position.entry_fees).
        entry_share = pos.entry_fees * closing / abs(pos.qty)
        realized = (exec_price - pos.avg_price) * closing * direction - fee - entry_share

        self._cash += notional - fee if pos.is_long else -(notional + fee)
        pos.qty -= closing * direction
        pos.entry_fees -= entry_share
        pos.realized_pnl += realized
        pos.fees_paid += fee
        self.total_realized_pnl += realized

        if abs(pos.qty) <= _DUST:
            del self._positions[symbol]

        return Fill(ts=ts, symbol=symbol, side=side, qty=closing, price=exec_price,
                    reference_price=ref_price, fee=fee, realized_pnl=realized,
                    reason=reason, trade_id=pos.trade_id)

    def close(self, symbol: str, price: float, ts: int | None = None,
              reason: str = "close") -> Fill | None:
        """Flatten ``symbol`` entirely. Returns None when already flat."""
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        side: Side = "sell" if pos.is_long else "buy"
        return self.market_order(symbol, side, abs(pos.qty), price, ts, reason)

    def close_all(self, prices: dict[str, float], ts: int | None = None,
                  reason: str = "flatten") -> list[Fill]:
        """Flatten every position for which a price is available."""
        fills = []
        for symbol in list(self._positions):
            price = prices.get(symbol)
            if price is None:
                logger.error("Cannot flatten %s: no price available", symbol)
                continue
            fill = self.close(symbol, price, ts, reason)
            if fill is not None:
                fills.append(fill)
        return fills

    # ---- persistence --------------------------------------------------
    def state(self) -> dict:
        """Serialise cash, positions and lifetime totals.

        A 24/7 desk gets restarted — containers are redeployed, hosts reboot,
        processes crash. Without this, a restart would resurrect the desk
        believing it were flat with its original cash, silently abandoning open
        positions (and their stops) while the equity curve jumped. The ledger
        persists this on every tick.
        """
        return {
            "starting_equity": self.starting_equity,
            "cash": self._cash,
            "total_fees": self.total_fees,
            "total_realized_pnl": self.total_realized_pnl,
            "trade_seq": self._trade_seq,
            "positions": [
                {
                    "symbol": pos.symbol, "qty": pos.qty, "avg_price": pos.avg_price,
                    "opened_ts": pos.opened_ts, "stop_price": pos.stop_price,
                    "extreme_price": pos.extreme_price, "realized_pnl": pos.realized_pnl,
                    "fees_paid": pos.fees_paid, "entry_reason": pos.entry_reason,
                    "trade_id": pos.trade_id, "entry_fees": pos.entry_fees,
                }
                for pos in self._positions.values()
            ],
        }

    def load_state(self, raw: dict | None) -> bool:
        """Restore from :meth:`state`. Returns True when state was applied.

        Malformed or partial state is refused rather than half-applied: a desk
        that restores three of four positions would place stops against a book
        that does not exist.
        """
        if not raw or raw.get("cash") is None:
            return False
        try:
            positions = {}
            for row in raw.get("positions") or []:
                pos = Position(
                    symbol=row["symbol"], qty=float(row["qty"]),
                    avg_price=float(row["avg_price"]), opened_ts=int(row["opened_ts"]),
                    stop_price=(None if row.get("stop_price") is None
                                else float(row["stop_price"])),
                    extreme_price=float(row.get("extreme_price") or 0.0),
                    realized_pnl=float(row.get("realized_pnl") or 0.0),
                    fees_paid=float(row.get("fees_paid") or 0.0),
                    entry_reason=row.get("entry_reason") or "",
                    # Ledgers written before trade ids existed: derive a stable
                    # id so the lifecycle can still be closed out under one key.
                    trade_id=row.get("trade_id") or f"{row['symbol']}-{int(row['opened_ts'])}",
                    entry_fees=float(row.get("entry_fees") or 0.0),
                )
                positions[pos.symbol] = pos
            cash = float(raw["cash"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Refusing to restore malformed broker state: %s", exc)
            return False

        stored_start = raw.get("starting_equity")
        if stored_start and abs(float(stored_start) - self.starting_equity) > 1e-6:
            # Keep the stored basis: returns must be measured from the capital
            # the track record actually started with, not a later edit.
            logger.warning(
                "Config starting_equity is %.2f but the ledger's run started at %.2f; "
                "keeping the stored figure so the track record stays comparable",
                self.starting_equity, float(stored_start))
            self.starting_equity = float(stored_start)

        self._cash = cash
        self._positions = positions
        self.total_fees = float(raw.get("total_fees") or 0.0)
        self.total_realized_pnl = float(raw.get("total_realized_pnl") or 0.0)
        self._trade_seq = int(raw.get("trade_seq") or 0)
        logger.info("Restored broker state: $%.2f cash, %d open position(s)",
                    self._cash, len(self._positions))
        return True

    def snapshot(self, prices: dict[str, float]) -> dict:
        """Account state for the ledger and the dashboard."""
        equity = self.equity(prices)
        return {
            "cash": self._cash,
            "equity": equity,
            "starting_equity": self.starting_equity,
            "total_return": (equity / self.starting_equity) - 1.0,
            "gross_exposure": self.gross_exposure(prices),
            "total_fees": self.total_fees,
            "total_realized_pnl": self.total_realized_pnl,
            "unrealized_pnl": sum(
                pos.unrealized_pnl(prices.get(s, pos.avg_price))
                for s, pos in self._positions.items()
            ),
            "open_positions": len(self._positions),
            "fill_count": len(self.fills),
        }
