"""The HTTP surface and the performance arithmetic behind the dashboard."""

import pytest
from fastapi.testclient import TestClient

from cryptodesk.api.server import build_app, performance
from cryptodesk.config import DeskConfig
from cryptodesk.engine.committee import HeuristicCommittee
from cryptodesk.engine.loop import Desk
from cryptodesk.feeds import SyntheticFeed


class Clock:
    def __init__(self, start=1_760_000_000):
        self.t = float(start)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def client(tmp_path):
    clock = Clock()
    cfg = DeskConfig.from_dict({
        "symbols": ["BTC-USD"], "starting_equity": 10_000, "home": str(tmp_path),
        "committee": "heuristic",
        "llm": {"min_minutes_between_calls": 20, "scheduled_interval_minutes": 60},
    })
    desk = Desk(cfg, feed=SyntheticFeed(seed=4, clock=clock),
                committee=HeuristicCommittee(), clock=clock)
    for _ in range(300):
        desk.tick()
        clock.advance(300)
    # Tick once more without advancing, so the desk is "fresh" at the current
    # clock — otherwise health would (correctly) report a stalled loop.
    desk.tick()
    return make_client(desk), desk


# The dashboard's own POSTs carry this; without it the API refuses controls.
CONTROL = {"X-CryptoDesk-Control": "1"}


def make_client(desk) -> TestClient:
    # The API only answers to loopback Host headers (and the configured ones),
    # so the client must present itself as 127.0.0.1 rather than "testserver".
    return TestClient(build_app(desk), base_url="http://127.0.0.1")


# ---------------------------------------------------------------- performance
def test_performance_reports_alpha_against_buy_and_hold():
    curve = [{"ts": i * 300, "equity": 10_000 * (1 + i * 0.001),
              "benchmark_price": 60_000 * (1 + i * 0.002)} for i in range(100)]
    perf = performance(curve, 10_000)
    assert perf["total_return"] == pytest.approx(0.099)
    assert perf["benchmark_return"] == pytest.approx(0.198)
    # Up 9.9% while simply holding made 19.8% is a loss in the only comparison
    # that matters.
    assert perf["alpha_vs_benchmark"] == pytest.approx(-0.099)


def test_performance_books_llm_spend_against_the_result():
    curve = [{"ts": 0, "equity": 10_000, "benchmark_price": 1},
             {"ts": 300, "equity": 10_100, "benchmark_price": 1}]
    perf = performance(curve, 10_000, llm_spend=150.0)
    assert perf["total_return"] == pytest.approx(0.01)
    assert perf["net_equity_after_llm"] == pytest.approx(9_950.0)
    assert perf["net_return_after_llm"] < 0, "tokens cost more than the desk made"


def test_performance_measures_drawdown_from_the_running_peak():
    curve = [{"ts": i, "equity": e} for i, e in enumerate([100, 120, 60, 90])]
    assert performance(curve, 100)["max_drawdown"] == pytest.approx(0.5)


@pytest.mark.parametrize("curve", [[], [{"ts": 0, "equity": 10_000}]])
def test_performance_degrades_gracefully_on_thin_history(curve):
    perf = performance(curve, 10_000)
    assert perf["points"] == len(curve)
    if curve:
        assert perf["sharpe"] is None and perf["benchmark_return"] is None


def test_performance_without_benchmark_marks_reports_no_alpha():
    curve = [{"ts": i * 300, "equity": 10_000 + i} for i in range(10)]
    perf = performance(curve, 10_000)
    assert perf["benchmark_return"] is None and perf["alpha_vs_benchmark"] is None


# ---------------------------------------------------------------- endpoints
def test_dashboard_is_served(client):
    api, _ = client
    response = api.get("/")
    assert response.status_code == 200
    assert "CryptoDesk" in response.text


@pytest.mark.parametrize("path, key", [
    ("/api/state", "account"), ("/api/equity", "curve"), ("/api/trades", "fills"),
    ("/api/decisions", "decisions"), ("/api/events", "events"),
])
def test_every_endpoint_answers_with_its_payload(client, path, key):
    api, _ = client
    response = api.get(path)
    assert response.status_code == 200
    assert key in response.json()


def test_state_carries_everything_the_dashboard_renders(client):
    api, _ = client
    state = api.get("/api/state").json()
    for key in ("account", "positions", "risk", "llm", "indicators", "trade_stats",
                "committee", "feed", "tick_count"):
        assert key in state
    for key in ("max_symbol_weight", "max_drawdown_halt", "daily_loss_limit"):
        assert key in state["risk"]["limits"]


def test_health_is_green_while_the_loop_is_ticking(client):
    api, _ = client
    response = api.get("/api/health")
    assert response.status_code == 200
    assert response.json()["healthy"] is True


def test_health_turns_red_when_the_loop_stalls(client):
    api, desk = client
    # Jump the clock well past two loop intervals without ticking.
    desk.clock.advance(desk.cfg.fast_loop_seconds * 10 + 60)
    response = api.get("/api/health")
    assert response.status_code == 503
    assert response.json()["healthy"] is False


def test_halt_and_resume_are_reachable_from_the_dashboard(client):
    api, desk = client
    assert api.post("/api/halt", headers=CONTROL).json()["halted"] is True
    assert desk.risk_state.halted and desk.broker.positions() == {}

    assert api.post("/api/resume", headers=CONTROL).json()["halted"] is False
    assert not desk.risk_state.halted


def test_there_is_no_manual_order_endpoint(client):
    """A hand-placed order would corrupt the track record the desk exists to build."""
    api, _ = client
    paths = {route.path for route in api.app.routes}
    assert not any("order" in path or "trade" in path.replace("/api/trades", "")
                   for path in paths)


# ---------------------------------------------------------------- lifetime statistics (item 1, 3)
def test_performance_statistics_are_lifetime_while_the_chart_is_a_window(client):
    """The tiles judge the whole run; the chart shows the tail. Mixing the two
    made alpha 'lifetime return minus trailing-25h benchmark' after day one."""
    api, desk = client
    full = desk.ledger.equity_curve_sampled(max_points=4000)
    assert len(full) > 50, "the fixture must have more history than the chart limit"

    payload = api.get("/api/equity?limit=50").json()
    assert len(payload["curve"]) == 50
    perf = payload["performance"]
    assert perf["points"] == len(full)
    assert perf["start_ts"] == full[0]["ts"] and perf["end_ts"] == full[-1]["ts"]
    # Exactly what a full-history computation gives, not the window's figures.
    reference = performance(full, desk.broker.starting_equity,
                            llm_spend=desk.ledger.spend_total(),
                            max_drawdown=desk.risk_state.max_drawdown,
                            first_benchmark_price=desk.risk_state.first_benchmark_price)
    windowed = performance(payload["curve"], desk.broker.starting_equity)
    for key in ("benchmark_return", "alpha_vs_benchmark", "max_drawdown", "annualised_vol"):
        assert perf[key] == pytest.approx(reference[key])
    assert perf["benchmark_base_price"] == desk.risk_state.first_benchmark_price
    assert perf["benchmark_return"] != pytest.approx(windowed["benchmark_return"])
    assert perf["max_drawdown"] >= desk.risk_state.max_drawdown


def test_performance_takes_the_lifetime_drawdown_and_the_inception_benchmark():
    # A window that starts after the trough and after the benchmark's move.
    window = [{"ts": 1000 + i * 300, "equity": 10_500, "benchmark_price": 120.0} for i in range(5)]
    perf = performance(window, 10_000, max_drawdown=0.30, first_benchmark_price=100.0)
    assert perf["max_drawdown"] == pytest.approx(0.30)
    assert perf["benchmark_return"] == pytest.approx(0.20)
    assert perf["alpha_vs_benchmark"] == pytest.approx(0.05 - 0.20)
    assert perf["benchmark_base_price"] == 100.0


def test_annualised_vol_and_sharpe_do_not_depend_on_how_often_the_curve_is_sampled():
    """The bar length comes from the marks' spacing: the same path sampled every
    minute and every five minutes must annualise to the same figures (item 3)."""
    import random
    rng = random.Random(11)
    equity, path = 10_000.0, []
    for i in range(30 * 24 * 60):
        equity *= 1 + rng.gauss(0.00002, 0.0006)
        path.append({"ts": 1_700_000_000 + i * 60, "equity": equity})
    one_minute = performance(path, 10_000)
    five_minute = performance(path[::5], 10_000)
    assert one_minute["bar_minutes"] == 1 and five_minute["bar_minutes"] == 5
    assert five_minute["annualised_vol"] == pytest.approx(one_minute["annualised_vol"], rel=0.05)
    assert five_minute["sharpe"] == pytest.approx(one_minute["sharpe"], rel=0.05)
    # The old code annualised every curve as one-minute bars.
    assert performance(path[::5], 10_000, bar_minutes=1.0)["annualised_vol"] == pytest.approx(
        five_minute["annualised_vol"] * 5 ** 0.5, rel=0.01)


# ---------------------------------------------------------------- browser-facing hardening (items 45, 46, 51)
@pytest.mark.parametrize("path", ["/api/halt", "/api/resume"])
def test_controls_refuse_requests_without_the_dashboard_header(client, path):
    api, desk = client
    response = api.post(path)
    assert response.status_code == 403
    assert not desk.risk_state.halted and desk.broker.positions() != {}


def test_controls_refuse_cross_site_requests_even_with_the_header(client):
    """A page on another origin cannot halt (and then resume, re-basing the
    brakes) an operator's desk through the operator's own browser."""
    api, desk = client
    evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site", **CONTROL}
    assert api.post("/api/halt?reason=" + "x" * 50, headers=evil).status_code == 403
    assert not desk.risk_state.halted and desk.broker.positions() != {}
    # The same request from the dashboard itself is fine.
    same = {"Sec-Fetch-Site": "same-origin", **CONTROL}
    assert api.post("/api/halt", headers=same).status_code == 200
    assert desk.risk_state.halted


def test_halt_reason_is_bounded(client):
    api, desk = client
    assert api.post("/api/halt?reason=" + "x" * 5000, headers=CONTROL).status_code == 422
    assert not desk.risk_state.halted


def test_resume_on_a_running_desk_is_a_conflict_not_a_rebase(client):
    api, desk = client
    peak = desk.risk_state.peak_equity
    response = api.post("/api/resume", headers=CONTROL)
    assert response.status_code == 409
    assert desk.risk_state.peak_equity == peak


@pytest.mark.parametrize("path, bad", [
    ("/api/equity", -1), ("/api/equity", 10 ** 20), ("/api/equity", 0),
    ("/api/trades", -1), ("/api/decisions", 10 ** 20), ("/api/events", -5),
])
def test_limit_parameters_are_bounded(client, path, bad):
    api, _ = client
    assert api.get(f"{path}?limit={bad}").status_code == 422


def test_a_foreign_host_header_is_refused(client):
    """DNS rebinding: a page on attacker.example that later resolves to
    127.0.0.1 would otherwise read the state as a same-origin fetch."""
    api, _ = client
    assert api.get("/api/state", headers={"Host": "attacker.example"}).status_code == 400
    assert api.get("/api/state", headers={"Host": "attacker.example:8787"}).status_code == 400
    for ok in ("127.0.0.1", "localhost", "localhost:8787"):
        assert api.get("/api/state", headers={"Host": ok}).status_code == 200


def test_configured_hosts_are_allowed_with_or_without_a_port(tmp_path):
    from cryptodesk.api.server import allowed_hosts
    cfg = DeskConfig.from_dict({
        "symbols": ["BTC-USD"], "home": str(tmp_path), "committee": "heuristic",
        "api_host": "0.0.0.0", "api_allowed_hosts": ["desk.example.com:8787", "proxy.internal"],
    })
    # Starlette matches on the host part of the header only, so a configured
    # host:port must be reduced to its host or it would never match.
    assert allowed_hosts(cfg) == ["127.0.0.1", "localhost", "::1", "0.0.0.0",
                                  "desk.example.com", "proxy.internal"]
    clock = Clock()
    desk = Desk(cfg, feed=SyntheticFeed(seed=4, clock=clock),
                committee=HeuristicCommittee(), clock=clock)
    api = make_client(desk)
    assert api.get("/api/state", headers={"Host": "desk.example.com:9000"}).status_code == 200
    assert api.get("/api/state", headers={"Host": "proxy.internal"}).status_code == 200
    assert api.get("/api/state", headers={"Host": "other.example.com"}).status_code == 400


# ---------------------------------------------------------------- health during a committee run (items 30, 60)
def test_health_stays_green_during_a_long_committee_run(client):
    """A model run takes minutes; the heartbeat is old but the desk is busy,
    which is alive until llm.max_run_seconds, not a stall to restart."""
    api, desk = client
    stale_after = desk.cfg.fast_loop_seconds * 2 + 30
    desk.busy = "committee BTC-USD"
    desk.clock.advance(stale_after + 200)
    response = api.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["healthy"] is True and body["busy"] == "committee BTC-USD"
    for key in ("heartbeat_ts", "stale_symbols", "seconds_since_heartbeat"):
        assert key in body

    # Past the run budget, "busy" is a hang.
    desk.clock.advance(desk.cfg.llm.max_run_seconds)
    assert api.get("/api/health").status_code == 503

    # Not busy and no heartbeat inside two loop periods is a stall.
    desk.busy = None
    desk.heartbeat_ts = int(desk.clock()) - stale_after - 1
    assert api.get("/api/health").status_code == 503
    desk.heartbeat_ts = int(desk.clock())
    assert api.get("/api/health").status_code == 200


def test_health_is_red_when_the_engine_thread_has_died(client):
    import threading
    api, desk = client
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    desk._thread = dead
    response = api.get("/api/health")
    assert response.status_code == 503 and response.json()["thread_alive"] is False


def test_state_exposes_venue_and_staleness_per_symbol(client):
    api, desk = client
    state = api.get("/api/state").json()
    assert state["feed_venue"] == {"BTC-USD": "synthetic"}
    assert state["stale_symbols"] == [] and "feed_health" in state
    assert "busy" in state and "heartbeat_ts" in state and "held_not_configured" in state
    assert all("stale" in p for p in state["positions"])
    assert "unflattened" in state["risk"] and "max_drawdown" in state["risk"]
