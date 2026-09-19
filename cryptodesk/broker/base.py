"""Broker protocol and shared execution types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Side = Literal["buy", "sell"]


class OrderRejected(RuntimeError):
    """Raised when an order cannot be filled (no cash, no position, bad size).

    Rejections are normal control flow — the engine logs them and continues —
    so they carry a human-readable reason that goes straight to the dashboard.
    """


@dataclass
class Fill:
    """A completed execution, after fees and slippage."""

    ts: int
    symbol: str
    side: Side
    qty: float
    price: float           # the fill price actually paid, incl. slippage
    reference_price: float  # the market price before slippage
    fee: float
    realized_pnl: float = 0.0
    reason: str = ""
    # The position lifecycle this fill belongs to, so partial exits of one
    # trade can be scored as one trade rather than several.
    trade_id: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.price

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "symbol": self.symbol, "side": self.side,
            "qty": self.qty, "price": self.price,
            "reference_price": self.reference_price, "fee": self.fee,
            "realized_pnl": self.realized_pnl, "reason": self.reason,
            "notional": self.notional, "trade_id": self.trade_id,
        }


@dataclass
class Position:
    """An open position. ``qty`` is negative for a short."""

    symbol: str
    qty: float
    avg_price: float
    opened_ts: int
    stop_price: float | None = None
    # Highest (long) or lowest (short) price seen since entry, for trailing stops.
    extreme_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    entry_reason: str = ""
    tags: dict = field(default_factory=dict)
    # Unique per lifecycle (open -> flat); every fill of the trade carries it.
    trade_id: str = ""
    # Fees paid on opening/adding fills that have not yet been charged against
    # realised P&L. ``avg_price`` stays gross so the dashboard shows the true
    # entry; the entry cost is instead released pro-rata as the position is
    # reduced, so realised P&L reconciles with the change in equity.
    entry_fees: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.qty

    def unrealized_pct(self, price: float) -> float:
        if self.avg_price <= 0:
            return 0.0
        return (price - self.avg_price) / self.avg_price * (1 if self.is_long else -1)

    def to_dict(self, price: float | None = None) -> dict:
        data = {
            "symbol": self.symbol, "qty": self.qty, "avg_price": self.avg_price,
            "opened_ts": self.opened_ts, "stop_price": self.stop_price,
            "extreme_price": self.extreme_price, "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid, "entry_reason": self.entry_reason,
            "tags": dict(self.tags), "trade_id": self.trade_id,
            "entry_fees": self.entry_fees,
        }
        if price is not None:
            data.update({
                "price": price,
                "market_value": self.market_value(price),
                "unrealized_pnl": self.unrealized_pnl(price),
                "unrealized_pct": self.unrealized_pct(price),
            })
        return data


@runtime_checkable
class Broker(Protocol):
    """What the engine requires of any execution venue."""

    def market_order(self, symbol: str, side: Side, qty: float, price: float,
                     ts: int, reason: str = "") -> Fill:
        """Fill ``qty`` of ``symbol`` at approximately ``price``."""
        ...

    def positions(self) -> dict[str, Position]:
        ...

    def cash(self) -> float:
        ...

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus the marked value of all open positions."""
        ...

    def gross_exposure(self, prices: dict[str, float]) -> float:
        """Absolute notional of open positions, as a fraction of equity."""
        ...
