"""FastAPI app exposing the desk's state, history, and two control actions.

The API is read-mostly by design. The only mutating endpoints are ``halt`` and
``resume``, because those are the two decisions a human genuinely needs to make
about an unattended desk. There is deliberately no "place order" endpoint: a
manual order would corrupt the very thing the desk exists to produce — a clean,
attributable record of what the strategy did on its own.

Binds to 127.0.0.1 by default. There is no authentication, so exposing it to a
network means anyone who can reach it can halt your desk; put it behind a
reverse proxy with auth if you need remote access. A loopback bind does not by
itself protect the desk from the operator's own browser, so two checks are
always on: the Host header must be loopback, ``api_host`` or one of
``api_allowed_hosts`` (a DNS-rebinding page otherwise reads the state as if it
were same-origin), and the control endpoints require the
``X-CryptoDesk-Control: 1`` header and refuse ``Sec-Fetch-Site: cross-site``,
so a foreign page cannot POST them as a CORS "simple request".
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

_STATIC = Path(__file__).parent / "static"

# Bars per year at a 5-minute cadence, used to annualise the equity-curve
# statistics. Crypto trades continuously, so unlike equities there is no
# trading-day adjustment: 24/7 is the whole point.
_MINUTES_PER_YEAR = 365 * 24 * 60

# Header the dashboard sends on every control POST. Any custom header forces a
# browser to preflight a cross-origin request, and with no CORS middleware the
# preflight fails, so a foreign page can never drive halt/resume.
CONTROL_HEADER = "x-cryptodesk-control"

# Points the lifetime statistics are computed over. Enough that a thinned
# year-long curve still shows every drawdown of note; small enough that the
# query stays a single index scan.
_LIFETIME_POINTS = 4000


def _bar_minutes_from(curve: list[dict]) -> float | None:
    """Median spacing of the marks, in minutes; None when there is no spacing.

    Derived from the data rather than the configured loop period because a
    simulation ticks at its own step and a live desk skips ticks; annualising
    with the wrong bar length mis-scales vol and Sharpe by ``sqrt(ratio)``.
    """
    gaps = [curve[i]["ts"] - curve[i - 1]["ts"] for i in range(1, len(curve))
            if curve[i]["ts"] > curve[i - 1]["ts"]]
    return statistics.median(gaps) / 60.0 if gaps else None


def performance(curve: list[dict], starting_equity: float, llm_spend: float = 0.0,
                bar_minutes: float | None = None, *, max_drawdown: float | None = None,
                first_benchmark_price: float | None = None) -> dict:
    """Summarise an equity curve: return, drawdown, volatility, and benchmark alpha.

    The benchmark is buy-and-hold of the configured benchmark symbol over the
    same window, scaled to the same starting capital. For a crypto desk this is
    the only comparison that matters: beating cash is easy in a bull market, and
    "profitable" means nothing if simply holding BTC did better.

    Pass the *whole* history (``Ledger.equity_curve_sampled``) rather than a
    chart window: every figure here is meant to be lifetime, and a trailing
    window would quietly forget old drawdowns and re-base the benchmark.
    ``max_drawdown`` (the engine's exact lifetime figure) and
    ``first_benchmark_price`` (the benchmark at inception) override what a
    thinned curve can see. ``bar_minutes`` is derived from the marks' spacing
    unless given explicitly.

    ``llm_spend`` is subtracted to give a net figure, because tokens are a real
    cost of running the strategy.
    """
    if not curve:
        return {"points": 0}

    equities = [float(row["equity"]) for row in curve]
    final = equities[-1]
    peak = max(equities)

    # Max drawdown over the curve from the running peak; the engine's exact
    # figure wins when it is larger, since a thinned curve can miss a trough.
    running_peak, max_dd = equities[0], 0.0
    for value in equities:
        running_peak = max(running_peak, value)
        if running_peak > 0:
            max_dd = max(max_dd, (running_peak - value) / running_peak)
    if max_drawdown is not None:
        max_dd = max(max_dd, float(max_drawdown))

    # Per-bar returns for volatility and a Sharpe-style ratio (zero risk-free
    # rate, which is a simplification worth naming).
    rets = [
        (equities[i] / equities[i - 1]) - 1.0
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]
    if bar_minutes is None:
        bar_minutes = _bar_minutes_from(curve)
    vol = sharpe = None
    if len(rets) > 2 and bar_minutes:
        mean = sum(rets) / len(rets)
        variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(variance)
        bars_per_year = _MINUTES_PER_YEAR / bar_minutes
        vol = sd * math.sqrt(bars_per_year)
        if sd > 0:
            sharpe = (mean * bars_per_year) / vol

    # Buy-and-hold of the benchmark from inception to the latest mark.
    marks = [row["benchmark_price"] for row in curve if row.get("benchmark_price")]
    base = first_benchmark_price or (marks[0] if marks else None)
    benchmark_return = None
    if base and marks and (len(marks) >= 2 or first_benchmark_price):
        benchmark_return = (marks[-1] / base) - 1.0

    total_return = (final / starting_equity) - 1.0 if starting_equity else 0.0
    net_equity = final - llm_spend
    return {
        "points": len(curve),
        "start_ts": curve[0]["ts"],
        "end_ts": curve[-1]["ts"],
        "bar_minutes": bar_minutes,
        "starting_equity": starting_equity,
        "final_equity": final,
        "peak_equity": peak,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "annualised_vol": vol,
        "sharpe": sharpe,
        "benchmark_return": benchmark_return,
        # The price the benchmark line is anchored to, so a chart window that
        # starts after inception still draws buy-and-hold from day one.
        "benchmark_base_price": base,
        # The number that decides whether the desk earned its keep.
        "alpha_vs_benchmark": (total_return - benchmark_return)
                              if benchmark_return is not None else None,
        "llm_spend": llm_spend,
        "net_equity_after_llm": net_equity,
        "net_return_after_llm": (net_equity / starting_equity) - 1.0 if starting_equity else 0.0,
    }


def _host_only(value: str) -> str:
    """``desk.example.com:8787`` -> ``desk.example.com``.

    Starlette compares the Host header's host part only, so a configured
    ``host:port`` would never match unless the port is dropped here too.
    """
    value = value.strip()
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def allowed_hosts(cfg) -> list[str]:
    """Host header values the API answers to; everything else gets a 400."""
    hosts = ["127.0.0.1", "localhost", "::1", cfg.api_host, *cfg.api_allowed_hosts]
    out: list[str] = []
    for host in (_host_only(h) for h in hosts):
        if host and host not in out:
            out.append(host)
    return out


def _require_dashboard_origin(request: Request) -> None:
    """Control POSTs must come from the dashboard, not a page the operator visited."""
    if request.headers.get(CONTROL_HEADER) != "1":
        raise HTTPException(
            403, f"control requests must carry the {CONTROL_HEADER}: 1 header")
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        raise HTTPException(403, "control requests must not be cross-site")


def build_app(desk) -> FastAPI:
    """Build the FastAPI app bound to a ``Desk`` instance."""
    app = FastAPI(title="CryptoDesk", version="0.1.0",
                  description="24/7 crypto paper-trading desk")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts(desk.cfg))

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
    def equity(limit: int = Query(2000, ge=1, le=20_000)) -> JSONResponse:
        """The last ``limit`` marks for the chart, and lifetime statistics.

        The two are deliberately different windows: the chart shows recent
        history, the tiles judge the whole run.
        """
        curve = desk.ledger.equity_curve(limit=limit)
        lifetime = desk.ledger.equity_curve_sampled(max_points=_LIFETIME_POINTS)
        return JSONResponse({
            "curve": curve,
            "performance": performance(
                lifetime, desk.broker.starting_equity,
                llm_spend=desk.ledger.spend_total(),
                max_drawdown=desk.risk_state.max_drawdown,
                first_benchmark_price=desk.risk_state.first_benchmark_price,
            ),
        })

    @app.get("/api/trades")
    def trades(limit: int = Query(100, ge=1, le=1000)) -> JSONResponse:
        return JSONResponse({"fills": desk.ledger.recent_fills(limit=limit),
                             "stats": desk.ledger.trade_stats()})

    @app.get("/api/decisions")
    def decisions(limit: int = Query(25, ge=1, le=200)) -> JSONResponse:
        return JSONResponse({"decisions": desk.ledger.recent_decisions(limit=limit)})

    @app.get("/api/events")
    def events(limit: int = Query(100, ge=1, le=1000)) -> JSONResponse:
        return JSONResponse({"events": desk.ledger.recent_events(limit=limit)})

    @app.get("/api/health")
    def health() -> JSONResponse:
        """Liveness for a container healthcheck: is the engine alive?

        Alive means the engine thread exists and the heartbeat is recent, or
        the desk is inside a committee run that has not exceeded
        ``llm.max_run_seconds``. Tick completion is the wrong signal: a
        multi-minute model run would read as a stall and get the container
        restarted mid-run.
        """
        now = int(desk.clock())
        heartbeat = desk.heartbeat_ts
        last = desk.last_tick_ts
        since = now - heartbeat
        thread = getattr(desk, "_thread", None)
        # A desk driven synchronously (tests, ``simulate --serve``) has no
        # thread; only a thread that was started and died counts against it.
        thread_alive = thread.is_alive() if thread is not None else None
        alive = thread_alive is not False
        # Two missed intervals is a stall worth reporting as unhealthy.
        stale_after = desk.cfg.fast_loop_seconds * 2 + 30
        ticking = alive and since <= stale_after
        thinking = alive and desk.busy is not None and since < desk.cfg.llm.max_run_seconds
        healthy = ticking or thinking
        return JSONResponse(
            {"healthy": healthy, "busy": desk.busy, "heartbeat_ts": heartbeat,
             "seconds_since_heartbeat": since, "thread_alive": thread_alive,
             "last_tick_ts": last, "now": now,
             "seconds_since_tick": (now - last) if last else None,
             "tick_count": desk.tick_count, "last_error": desk.last_error,
             "halted": desk.risk_state.halted,
             "stale_symbols": desk.stale_symbols()},
            status_code=200 if healthy else 503,
        )

    @app.post("/api/halt", dependencies=[Depends(_require_dashboard_origin)])
    def halt(reason: str = Query("manual halt via dashboard", max_length=200)) -> JSONResponse:
        desk.halt(reason)
        return JSONResponse({"halted": True, "reason": reason})

    @app.post("/api/resume", dependencies=[Depends(_require_dashboard_origin)])
    def resume(reset_peak: bool = True) -> JSONResponse:
        try:
            desk.resume(reset_peak=reset_peak)
        except RuntimeError as exc:
            # Not halted: resuming would only re-base the brakes.
            raise HTTPException(409, str(exc)) from exc
        return JSONResponse({"halted": False, "peak_equity": desk.risk_state.peak_equity})

    return app
