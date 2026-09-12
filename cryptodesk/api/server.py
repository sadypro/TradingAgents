"""FastAPI app exposing the desk's state, history, and two control actions.

The API is read-mostly by design. The only mutating endpoints are ``halt`` and
``resume``, because those are the two decisions a human genuinely needs to make
about an unattended desk. There is deliberately no "place order" endpoint: a
manual order would corrupt the very thing the desk exists to produce — a clean,
attributable record of what the strategy did on its own.

Binds to 127.0.0.1 by default. There is no authentication, so exposing it to a
network means anyone who can reach it can halt your desk; put it behind a
reverse proxy with auth if you need remote access.
"""

from __future__ import annotations

import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

_STATIC = Path(__file__).parent / "static"

# Bars per year at a 5-minute cadence, used to annualise the equity-curve
# statistics. Crypto trades continuously, so unlike equities there is no
# trading-day adjustment: 24/7 is the whole point.
_MINUTES_PER_YEAR = 365 * 24 * 60


def performance(curve: list[dict], starting_equity: float, llm_spend: float = 0.0,
                bar_minutes: float = 5.0) -> dict:
    """Summarise an equity curve: return, drawdown, volatility, and benchmark alpha.

    The benchmark is buy-and-hold of the configured benchmark symbol over the
    same window, scaled to the same starting capital. For a crypto desk this is
    the only comparison that matters: beating cash is easy in a bull market, and
    "profitable" means nothing if simply holding BTC did better.

    ``llm_spend`` is subtracted to give a net figure, because tokens are a real
    cost of running the strategy.
    """
    if not curve:
        return {"points": 0}

    equities = [float(row["equity"]) for row in curve]
    final = equities[-1]
    peak = max(equities)

    # Max drawdown over the window, computed from the running peak.
    running_peak, max_dd = equities[0], 0.0
    for value in equities:
        running_peak = max(running_peak, value)
        if running_peak > 0:
            max_dd = max(max_dd, (running_peak - value) / running_peak)

    # Per-bar returns for volatility and a Sharpe-style ratio (zero risk-free
    # rate, which is a simplification worth naming).
    rets = [
        (equities[i] / equities[i - 1]) - 1.0
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]
    vol = sharpe = None
    if len(rets) > 2:
        mean = sum(rets) / len(rets)
        variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(variance)
        bars_per_year = _MINUTES_PER_YEAR / bar_minutes
        vol = sd * math.sqrt(bars_per_year)
        if sd > 0:
            sharpe = (mean * bars_per_year) / vol

    # Buy-and-hold of the benchmark over the same window.
    marks = [(row["ts"], row["benchmark_price"]) for row in curve
             if row.get("benchmark_price")]
    benchmark_return = None
    if len(marks) >= 2 and marks[0][1]:
        benchmark_return = (marks[-1][1] / marks[0][1]) - 1.0

    total_return = (final / starting_equity) - 1.0 if starting_equity else 0.0
    net_equity = final - llm_spend
    return {
        "points": len(curve),
        "start_ts": curve[0]["ts"],
        "end_ts": curve[-1]["ts"],
        "starting_equity": starting_equity,
        "final_equity": final,
        "peak_equity": peak,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "annualised_vol": vol,
        "sharpe": sharpe,
        "benchmark_return": benchmark_return,
        # The number that decides whether the desk earned its keep.
        "alpha_vs_benchmark": (total_return - benchmark_return)
                              if benchmark_return is not None else None,
        "llm_spend": llm_spend,
        "net_equity_after_llm": net_equity,
        "net_return_after_llm": (net_equity / starting_equity) - 1.0 if starting_equity else 0.0,
    }


def build_app(desk) -> FastAPI:
    """Build the FastAPI app bound to a ``Desk`` instance."""
    app = FastAPI(title="CryptoDesk", version="0.1.0",
                  description="24/7 crypto paper-trading desk")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        index = _STATIC / "index.html"
        if not index.exists():
            raise HTTPException(500, "dashboard asset missing")
        return HTMLResponse(index.read_text(encoding="utf-8"))

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(desk.state())

    @app.get("/api/equity")
    def equity(limit: int = 2000) -> JSONResponse:
        curve = desk.ledger.equity_curve(limit=limit)
        bar_minutes = max(desk.cfg.fast_loop_seconds / 60.0, 1 / 60.0)
        return JSONResponse({
            "curve": curve,
            "performance": performance(
                curve, desk.broker.starting_equity,
                llm_spend=desk.ledger.spend_total(), bar_minutes=bar_minutes,
            ),
        })

    @app.get("/api/trades")
    def trades(limit: int = 100) -> JSONResponse:
        return JSONResponse({"fills": desk.ledger.recent_fills(limit=limit),
                             "stats": desk.ledger.trade_stats()})

    @app.get("/api/decisions")
    def decisions(limit: int = 25) -> JSONResponse:
        return JSONResponse({"decisions": desk.ledger.recent_decisions(limit=limit)})

    @app.get("/api/events")
    def events(limit: int = 100) -> JSONResponse:
        return JSONResponse({"events": desk.ledger.recent_events(limit=limit)})

    @app.get("/api/health")
    def health() -> JSONResponse:
        """Liveness for a container healthcheck: is the loop still ticking?"""
        now = int(desk.clock())
        last = desk.last_tick_ts
        # Two missed intervals is a stall worth reporting as unhealthy.
        stale_after = desk.cfg.fast_loop_seconds * 2 + 30
        healthy = last is not None and (now - last) <= stale_after
        return JSONResponse(
            {"healthy": healthy, "last_tick_ts": last, "now": now,
             "seconds_since_tick": (now - last) if last else None,
             "tick_count": desk.tick_count, "last_error": desk.last_error,
             "halted": desk.risk_state.halted},
            status_code=200 if healthy else 503,
        )

    @app.post("/api/halt")
    def halt(reason: str = "manual halt via dashboard") -> JSONResponse:
        desk.halt(reason)
        return JSONResponse({"halted": True, "reason": reason})

    @app.post("/api/resume")
    def resume(reset_peak: bool = True) -> JSONResponse:
        desk.resume(reset_peak=reset_peak)
        return JSONResponse({"halted": False, "peak_equity": desk.risk_state.peak_equity})

    return app
