"""The command line: report, simulate --replay, init, doctor and run."""

import json

import pytest

from cryptodesk.__main__ import build_parser, main
from cryptodesk.config import _ENV_OVERRIDES
from cryptodesk.engine.ledger import Ledger


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # Never touch the developer's real ~/.cryptodesk, and never read their env.
    for name in list(_ENV_OVERRIDES) + ["CRYPTODESK_CONFIG"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRYPTODESK_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _write_series(directory, name="BTC-USD", bars=80, start=1_700_000_000, step=300):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.csv").write_text(
        "ts,open,high,low,close,volume\n" +
        "".join(f"{start + i * step},{100 + i * 0.1},{101 + i * 0.1},{99 + i * 0.1},"
                f"{100.5 + i * 0.1},10\n" for i in range(bars)),
        encoding="utf-8")


# ---------------------------------------------------------------- report (item 5)
def test_report_measures_from_the_ledgers_own_starting_capital(tmp_path, capsys):
    """The running desk keeps the ledger's starting figure when the config
    disagrees, so the report must too, or the two disagree on total return."""
    db = tmp_path / "old.db"
    ledger = Ledger(db)
    ledger.set_state("broker_state", {"starting_equity": 8_000.0, "cash": 8_800.0,
                                      "positions": {}})
    for i in range(10):
        ledger.record_equity(ts=1_700_000_000 + i * 300, equity=8_000 + i * 80, cash=8_800,
                             gross_exposure=0.0, drawdown=0.0, benchmark_price=100 + i)
    ledger.close()

    assert main(["report", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "$8,000.00 → $8,720.00" in out
    assert "+9.00%" in out, "9% from the ledger's 8,000, not -12.8% from the config's 10,000"


def test_report_falls_back_to_the_config_when_the_ledger_has_no_book(tmp_path, capsys, monkeypatch):
    db = tmp_path / "bare.db"
    ledger = Ledger(db)
    ledger.record_equity(ts=1_700_000_000, equity=12_000, cash=12_000, gross_exposure=0, drawdown=0)
    ledger.close()
    monkeypatch.setenv("CRYPTODESK_STARTING_EQUITY", "12000")
    assert main(["report", "--db", str(db)]) == 0
    assert "$12,000.00 → $12,000.00" in capsys.readouterr().out


def test_report_without_a_ledger_fails_cleanly(tmp_path, capsys):
    assert main(["report", "--db", str(tmp_path / "none.db")]) == 1
    assert "No ledger" in capsys.readouterr().err


# ---------------------------------------------------------------- simulate --replay (item 37)
def test_replay_stops_at_the_end_of_the_data_and_stamps_the_ledger_with_bar_times(tmp_path, capsys):
    data = tmp_path / "episode"
    _write_series(data, bars=80)
    home = tmp_path / "sim"
    # --days asks for far more ticks than the file holds.
    assert main(["simulate", "--replay", str(data), "--warmup", "50", "--days", "30",
                 "--symbols", "BTC-USD", "--home", str(home)]) == 0
    out = capsys.readouterr().out
    assert "30 ticks" in out

    ledger = Ledger(home / "desk.db")
    curve = ledger.equity_curve_sampled()
    ledger.close()
    # One mark per bar after the warm-up, each stamped at that bar's close —
    # the episode's own dates, not "now minus 30 days".
    assert [row["ts"] for row in curve] == [1_700_000_000 + (50 + i) * 300 for i in range(30)]


def test_replay_honours_a_shorter_days_budget(tmp_path):
    data = tmp_path / "episode"
    _write_series(data, bars=80)
    home = tmp_path / "sim"
    # 0.01 days of 5-minute steps is 2 ticks.
    assert main(["simulate", "--replay", str(data), "--warmup", "50", "--days", "0.01",
                 "--symbols", "BTC-USD", "--home", str(home)]) == 0
    ledger = Ledger(home / "desk.db")
    assert [row["ts"] for row in ledger.equity_curve_sampled()] == [
        1_700_000_000 + 50 * 300, 1_700_000_000 + 51 * 300]
    ledger.close()


def test_replay_warns_when_the_csv_bar_size_disagrees_with_the_config(tmp_path, capsys):
    data = tmp_path / "hourly"
    _write_series(data, bars=60, step=3600)
    assert main(["simulate", "--replay", str(data), "--warmup", "50", "--days", "1",
                 "--symbols", "BTC-USD", "--home", str(tmp_path / "sim")]) == 0
    assert "candle_interval is 5m" in capsys.readouterr().out


def test_synthetic_simulation_runs_on_the_configured_interval(tmp_path, capsys):
    home = tmp_path / "sim"
    assert main(["simulate", "--days", "0.5", "--symbols", "BTC-USD", "--home", str(home)]) == 0
    assert "144 ticks" in capsys.readouterr().out
    ledger = Ledger(home / "desk.db")
    assert len(ledger.equity_curve_sampled()) == 144
    ledger.close()


# ---------------------------------------------------------------- init / doctor (item 49)
def test_init_writes_once_and_refuses_to_overwrite(tmp_path, capsys):
    path = tmp_path / "cryptodesk.json"
    assert main(["init", str(path)]) == 0
    written = json.loads(path.read_text())
    assert written["symbols"] == ["BTC-USD", "ETH-USD"]
    assert written["llm"]["max_run_seconds"] == 900 and written["api_allowed_hosts"] == []

    assert main(["init", str(path)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["init", str(path), "--force"]) == 0


def test_doctor_reports_a_missing_config_file_instead_of_running_on_defaults(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.json"), "doctor"]) == 1
    out = capsys.readouterr().out
    assert "nope.json (from --config)" in out
    assert "[FAIL]" in out and "not found" in out


def test_doctor_reports_an_invalid_config_value(tmp_path, capsys, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"risk": {"max_drawdown_halt": "nan"}}))
    monkeypatch.setenv("CRYPTODESK_CONFIG", str(path))
    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "(from CRYPTODESK_CONFIG)" in out and "max_drawdown_halt" in out


def test_doctor_passes_on_a_valid_offline_config(tmp_path, capsys):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"feeds": ["synthetic"], "symbols": ["eth-usd"],
                                "committee": "heuristic"}))
    assert main(["--config", str(path), "doctor"]) == 0
    out = capsys.readouterr().out
    assert "[ok]   synthetic" in out and "via synthetic" in out
    assert "BTC-USD is not in symbols; fetched for comparison only" in out
    assert "No blocking problems." in out


# ---------------------------------------------------------------- run
def test_run_accepts_host_and_port_overrides():
    args = build_parser().parse_args(["run", "--port", "9999", "--host", "0.0.0.0"])
    assert args.port == 9999 and args.host == "0.0.0.0"
    assert build_parser().parse_args(["run"]).port is None


def test_run_serves_on_the_overridden_port_and_stops_the_engine(tmp_path, monkeypatch, capsys):
    import cryptodesk.__main__ as cli

    served, desks = {}, []

    def fake_uvicorn_run(app, **kwargs):
        served.update(kwargs)

    real_build_app = cli.build_app

    def spy_build_app(desk):
        desks.append(desk)
        return real_build_app(desk)

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    monkeypatch.setattr(cli, "build_app", spy_build_app)

    assert main(["run", "--feed", "synthetic", "--committee", "heuristic",
                 "--symbols", "BTC-USD", "--port", "9999", "--host", "127.0.0.2"]) == 0
    assert served["port"] == 9999 and served["host"] == "127.0.0.2"
    assert "http://127.0.0.2:9999" in capsys.readouterr().out
    desk = desks[0]
    # stop() joined the engine thread, so the book on disk is final.
    assert desk._thread is not None and not desk._thread.is_alive()
    assert desk.ledger.get_state("broker_state") is not None
