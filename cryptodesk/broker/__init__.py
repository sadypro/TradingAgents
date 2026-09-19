"""Execution layer.

``PaperBroker`` is a self-contained simulated ledger filled at real market
prices, so the desk needs no exchange account and no API keys to produce a
track record. The ``Broker`` protocol is the seam: a live or exchange-testnet
adapter implements the same five methods and the engine above it does not
change.
"""

from .base import Broker, Fill, OrderRejected, Position, Side
from .paper import PaperBroker

__all__ = ["Broker", "Fill", "OrderRejected", "Position", "Side", "PaperBroker"]
