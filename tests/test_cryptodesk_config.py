"""Config loading: validation, canonicalisation, explicit paths, env overrides."""

import json
import math

import pytest

from cryptodesk.config import (
    _ENV_OVERRIDES,
    DeskConfig,
    LLMBudget,
    RiskLimits,
    config_source,
)


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    # A developer's own CRYPTODESK_* environment must not leak into the matrix.
    for name in list(_ENV_OVERRIDES) + ["CRYPTODESK_CONFIG"]:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- validation matrix (item 48)
@pytest.mark.parametrize("overrides, field", [
    ({"fee_bps": -50}, "fee_bps"),
    ({"slippage_bps": -5}, "slippage_bps"),
    ({"starting_equity": 0}, "starting_equity"),
    ({"fast_loop_seconds": 4}, "fast_loop_seconds"),
    ({"candle_lookback": 49}, "candle_lookback"),
    ({"candle_lookback": 301, "feeds": ["cryptocom"]}, "candle_lookback"),
    ({"candle_interval": "7m"}, "candle_interval"),
    ({"symbols": []}, "symbols"),
    ({"symbols": ["???"]}, "symbols"),
    ({"benchmark_symbol": "!!"}, "benchmark_symbol"),
    ({"feeds": ["nope"]}, "feeds"),
    ({"feeds": []}, "feeds"),
    ({"committee": "oracle"}, "committee"),
    ({"api_port": 0}, "api_port"),
    ({"api_port": 70000}, "api_port"),
    ({"risk": {"risk_per_trade": -1.0}}, "risk.risk_per_trade"),
    ({"risk": {"risk_per_trade": 0.5}}, "risk.risk_per_trade"),
    ({"risk": {"max_symbol_weight": 3.0}}, "risk.max_symbol_weight"),
    ({"risk": {"max_gross_exposure": 0}}, "risk.max_gross_exposure"),
    ({"risk": {"max_symbol_weight": 0.7, "max_gross_exposure": 0.6}}, "risk.max_symbol_weight"),
    ({"risk": {"max_weight_drift": -0.1}}, "risk.max_weight_drift"),
    ({"risk": {"daily_loss_limit": 1.0}}, "risk.daily_loss_limit"),
    ({"risk": {"daily_loss_limit": 0}}, "risk.daily_loss_limit"),
    ({"risk": {"max_drawdown_halt": 1.5}}, "risk.max_drawdown_halt"),
    ({"risk": {"stop_atr_mult": 0}}, "risk.stop_atr_mult"),
    ({"risk": {"trail_atr_mult": -1}}, "risk.trail_atr_mult"),
    ({"risk": {"cooldown_minutes": -1}}, "risk.cooldown_minutes"),
    ({"risk": {"min_order_notional": 0}}, "risk.min_order_notional"),
    ({"llm": {"daily_usd_cap": 0}}, "llm.daily_usd_cap"),
    ({"llm": {"estimated_cost_per_run_usd": 0}}, "llm.estimated_cost_per_run_usd"),
    ({"llm": {"scheduled_interval_minutes": 0}}, "llm.scheduled_interval_minutes"),
    ({"llm": {"min_minutes_between_calls": -1}}, "llm.min_minutes_between_calls"),
    ({"llm": {"max_run_seconds": 0}}, "llm.max_run_seconds"),
    ({"llm": {"price_per_million_input_usd": 3.0}}, "llm.price_per_million_input_usd"),
    ({"llm": {"price_per_million_input_usd": -3.0, "price_per_million_output_usd": 1}},
     "llm.price_per_million_input_usd"),
])
def test_out_of_range_values_are_refused_naming_the_field(overrides, field):
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        DeskConfig.from_dict(overrides)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("path", ["risk.max_drawdown_halt", "fee_bps", "llm.daily_usd_cap"])
def test_non_finite_numbers_are_refused(path, value):
    """``0.99 >= nan`` is False, so a nan limit is a kill-switch that never fires."""
    section, _, name = path.rpartition(".")
    raw = {section: {name: value}} if section else {name: value}
    with pytest.raises(ValueError, match=name):
        DeskConfig.from_dict(raw)


@pytest.mark.parametrize("raw, field", [
    ({"fee_bps": [1]}, "fee_bps"),
    ({"risk": {"allow_shorts": "maybe"}}, "risk.allow_shorts"),
    ({"risk": {"allow_shorts": 1}}, "risk.allow_shorts"),
    ({"risk": {"max_drawdown_halt": "fifteen"}}, "risk.max_drawdown_halt"),
    ({"symbols": [1, 2]}, "symbols"),
    ({"candle_lookback": 1.5}, "candle_lookback"),
])
def test_wrongly_typed_values_are_refused(raw, field):
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        DeskConfig.from_dict(raw)


def test_json_strings_are_coerced_like_env_strings():
    """A quoted number or boolean in the file must not smuggle a str into the
    engine: 'false' is truthy and '0.15' raises on the first comparison."""
    cfg = DeskConfig.from_dict({
        "risk": {"allow_shorts": "false", "max_drawdown_halt": "0.15", "cooldown_minutes": "30"},
        "starting_equity": "5000", "symbols": "btc-usd, eth-usd",
        "llm": {"price_per_million_input_usd": "3", "price_per_million_output_usd": "15"},
    })
    assert cfg.risk.allow_shorts is False
    assert cfg.risk.max_drawdown_halt == 0.15 and isinstance(cfg.risk.max_drawdown_halt, float)
    assert cfg.risk.cooldown_minutes == 30 and isinstance(cfg.risk.cooldown_minutes, int)
    assert cfg.starting_equity == 5000.0
    assert cfg.symbols == ["BTC-USD", "ETH-USD"]
    assert cfg.llm.price_per_million_output_usd == 15.0


def test_defaults_validate_and_integers_are_accepted_for_floats():
    cfg = DeskConfig.from_dict({"fee_bps": 3, "risk": {"cooldown_minutes": 60.0}})
    assert cfg.fee_bps == 3.0 and isinstance(cfg.fee_bps, float)
    assert cfg.risk.cooldown_minutes == 60 and isinstance(cfg.risk.cooldown_minutes, int)
    assert DeskConfig().validate() is not None


# ---------------------------------------------------------------- canonicalisation (item 50)
def test_symbols_feeds_and_committee_are_canonicalised():
    cfg = DeskConfig.from_dict({
        "symbols": ["btc-usd", "ETHUSD", " sol-usdt ", "BTC-USD"],
        "benchmark_symbol": "btcusd", "feeds": [" Synthetic "], "committee": "Heuristic",
    })
    assert cfg.symbols == ["BTC-USD", "ETH-USD", "SOL-USDT"]
    assert cfg.benchmark_symbol == "BTC-USD" and cfg.benchmark_symbol in cfg.symbols
    assert cfg.feeds == ["synthetic"] and cfg.committee == "heuristic"


def test_env_symbols_are_canonicalised_too(monkeypatch):
    monkeypatch.setenv("CRYPTODESK_SYMBOLS", "eth-usd,solusd")
    monkeypatch.setenv("CRYPTODESK_BENCHMARK", "btc-usd")
    cfg = DeskConfig.load()
    assert cfg.symbols == ["ETH-USD", "SOL-USD"] and cfg.benchmark_symbol == "BTC-USD"


# ---------------------------------------------------------------- explicit paths (item 49)
def test_an_explicit_missing_config_path_raises(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match=r"--config"):
        DeskConfig.load(tmp_path / "nope.json")
    monkeypatch.setenv("CRYPTODESK_CONFIG", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError, match="CRYPTODESK_CONFIG"):
        DeskConfig.load()


def test_defaults_apply_only_when_no_path_was_asked_for():
    assert config_source() == (None, None)
    assert DeskConfig.load().symbols == ["BTC-USD", "ETH-USD"]


def test_config_source_names_where_the_file_came_from(tmp_path, monkeypatch):
    path, origin = config_source(tmp_path / "a.json")
    assert path == tmp_path / "a.json" and origin == "--config"
    monkeypatch.setenv("CRYPTODESK_CONFIG", str(tmp_path / "b.json"))
    path, origin = config_source()
    assert path == tmp_path / "b.json" and origin == "CRYPTODESK_CONFIG"
    # An explicit argument beats the env var.
    assert config_source(tmp_path / "a.json")[1] == "--config"


def test_a_file_is_loaded_validated_and_env_still_wins(tmp_path, monkeypatch):
    path = tmp_path / "desk.json"
    path.write_text(json.dumps({"symbols": ["sol-usd"], "fee_bps": "7", "unknown_key": 1}))
    cfg = DeskConfig.load(path)
    assert cfg.symbols == ["SOL-USD"] and cfg.fee_bps == 7.0

    monkeypatch.setenv("CRYPTODESK_FEE_BPS", "12")
    assert DeskConfig.load(path).fee_bps == 12.0

    path.write_text(json.dumps({"risk": {"max_drawdown_halt": 5}}))
    with pytest.raises(ValueError, match="max_drawdown_halt"):
        DeskConfig.load(path)

    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        DeskConfig.load(path)


def test_a_file_that_relies_on_env_for_its_symbols_is_judged_after_env(tmp_path, monkeypatch):
    path = tmp_path / "desk.json"
    path.write_text(json.dumps({"symbols": []}))
    monkeypatch.setenv("CRYPTODESK_SYMBOLS", "BTC-USD")
    assert DeskConfig.load(path).symbols == ["BTC-USD"]


def test_round_trip_through_save_and_load(tmp_path):
    path = DeskConfig.from_dict({"symbols": ["btc-usd"], "api_allowed_hosts": ["desk.example.com"]}).save(
        tmp_path / "out.json")
    cfg = DeskConfig.load(path)
    assert cfg.symbols == ["BTC-USD"] and cfg.api_allowed_hosts == ["desk.example.com"]
    assert cfg.llm.max_run_seconds == 900 and cfg.llm.price_per_million_input_usd is None


# ---------------------------------------------------------------- env rows (item 52)
@pytest.mark.parametrize("env_var, value, path, expected", [
    ("CRYPTODESK_LLM_COST_PER_RUN", "0.80", "llm.estimated_cost_per_run_usd", 0.80),
    ("CRYPTODESK_MAX_WEIGHT_DRIFT", "0.5", "risk.max_weight_drift", 0.5),
    ("CRYPTODESK_MIN_ORDER_NOTIONAL", "40", "risk.min_order_notional", 40.0),
    ("CRYPTODESK_CANDLE_LOOKBACK", "120", "candle_lookback", 120),
    ("CRYPTODESK_API_ALLOWED_HOSTS", "desk.example.com, proxy.internal", "api_allowed_hosts",
     ["desk.example.com", "proxy.internal"]),
    ("CRYPTODESK_LLM_MAX_RUN_SECONDS", "1200", "llm.max_run_seconds", 1200),
])
def test_new_env_rows_reach_their_fields(monkeypatch, env_var, value, path, expected):
    monkeypatch.setenv(env_var, value)
    cfg = DeskConfig.load()
    section, _, name = path.rpartition(".")
    assert getattr(getattr(cfg, section) if section else cfg, name) == expected


def test_llm_prices_come_from_env_as_a_pair(monkeypatch):
    monkeypatch.setenv("CRYPTODESK_LLM_PRICE_IN", "3")
    with pytest.raises(ValueError, match="price_per_million"):
        DeskConfig.load()
    monkeypatch.setenv("CRYPTODESK_LLM_PRICE_OUT", "15")
    cfg = DeskConfig.load()
    assert (cfg.llm.price_per_million_input_usd, cfg.llm.price_per_million_output_usd) == (3.0, 15.0)
    monkeypatch.setenv("CRYPTODESK_LLM_PRICE_IN", "none")
    monkeypatch.setenv("CRYPTODESK_LLM_PRICE_OUT", "none")
    assert DeskConfig.load().llm.price_per_million_input_usd is None


def test_invalid_env_values_name_the_variable(monkeypatch):
    monkeypatch.setenv("CRYPTODESK_ALLOW_SHORTS", "maybe")
    with pytest.raises(ValueError, match="CRYPTODESK_ALLOW_SHORTS"):
        DeskConfig.load()
    monkeypatch.setenv("CRYPTODESK_ALLOW_SHORTS", "yes")
    monkeypatch.setenv("CRYPTODESK_MAX_DRAWDOWN_HALT", "nan")
    with pytest.raises(ValueError, match="CRYPTODESK_MAX_DRAWDOWN_HALT"):
        DeskConfig.load()


def test_every_field_has_an_env_override():
    """A container deployment is env-only; a knob without a row is unreachable."""
    covered = set(_ENV_OVERRIDES.values())
    expected = {f.name for f in DeskConfig.__dataclass_fields__.values()
                if f.name not in ("risk", "llm")}
    expected |= {f"risk.{name}" for name in RiskLimits.__dataclass_fields__}
    expected |= {f"llm.{name}" for name in LLMBudget.__dataclass_fields__}
    assert expected <= covered, sorted(expected - covered)
