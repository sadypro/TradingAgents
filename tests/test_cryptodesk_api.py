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
    return TestClient(build_app(desk)), desk


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
    assert api.post("/api/halt").json()["halted"] is True
    assert desk.risk_state.halted and desk.broker.positions() == {}

    assert api.post("/api/resume").json()["halted"] is False
    assert not desk.risk_state.halted


def test_there_is_no_manual_order_endpoint(client):
    """A hand-placed order would corrupt the track record the desk exists to build."""
    api, _ = client
    paths = {route.path for route in api.app.routes}
    assert not any("order" in path or "trade" in path.replace("/api/trades", "")
                   for path in paths)
