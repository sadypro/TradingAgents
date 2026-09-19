"""CryptoDesk: a 24/7 paper-trading desk for crypto, driven by TradingAgents.

The design separates three concerns that are usually conflated in "AI trading
bot" projects, because conflating them is how people lose money:

1. **A cheap always-on loop** (``engine.loop``) polls prices, updates
   indicators, enforces stops and marks the book. It runs every minute, costs
   nothing, and is what actually makes the desk "24/7".
2. **An expensive committee** (``engine.committee``) — the TradingAgents
   multi-agent graph — is woken only on a trigger or a schedule, under a hard
   daily dollar budget. It proposes a *direction and conviction*, never a size.
3. **A deterministic risk engine** (``engine.risk``) decides size, places
   stops, and can veto the committee outright. It is code, not prose, so its
   behaviour is testable and cannot be argued out of a limit by an LLM.

Everything the desk does is written to a SQLite ledger so profitability is
measurable after the fact — including the LLM spend, which is booked against
P&L rather than ignored.
"""

__version__ = "0.1.0"
