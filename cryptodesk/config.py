"""Configuration for the desk: symbols, cadence, risk limits, LLM budget.

Config is a plain JSON file plus ``CRYPTODESK_*`` environment overrides, so a
container deployment can be reconfigured without editing files and no YAML
dependency is pulled in. Every default here is deliberately conservative: the
desk is meant to survive long enough to produce a statistically meaningful
track record, and a blown-up account produces no data at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_HOME = Path(os.path.expanduser("~")) / ".cryptodesk"


@dataclass
class RiskLimits:
    """Hard limits enforced in code on every order, before it reaches the book.

    These are not suggestions to the LLM — the committee never sees them and
    cannot widen them. ``engine.risk`` applies them to whatever the committee
    proposes.
    """

    # Largest fraction of equity a single symbol may hold at entry.
    max_symbol_weight: float = 0.25
    # How far a winning position may drift above that cap through
    # mark-to-market before it is trimmed back. Without a band, every tick
    # would trim a rising position by a few basis points and bleed fees; with
    # no band at all, a position that triples would quietly become most of the
    # account. Set very high to let winners run unbounded.
    max_weight_drift: float = 0.20
    # Largest fraction of equity deployed across all symbols at once. Leaves
    # cash so a drawdown does not force liquidation at the worst price.
    max_gross_exposure: float = 0.60
    # Fraction of equity risked per trade, i.e. the loss taken if the stop
    # fills exactly. Position size is derived from this and ATR, never from
    # the LLM's conviction alone.
    risk_per_trade: float = 0.01
    # Initial stop distance, in ATR multiples.
    stop_atr_mult: float = 2.5
    # Trailing stop distance once a position is in profit, in ATR multiples.
    trail_atr_mult: float = 3.0
    # Realised loss over a UTC day that halts new entries (existing positions
    # keep their stops).
    daily_loss_limit: float = 0.03
    # Drawdown from peak equity that flattens the book and halts the desk.
    # Recovery requires a human explicitly resuming it.
    max_drawdown_halt: float = 0.15
    # After a stop-out, how long to refuse re-entry on that symbol. Prevents
    # the classic loop of re-buying into the same falling knife.
    cooldown_minutes: int = 120
    # Shorting is off by default: it is unbounded-loss, and paper fills for
    # borrow/funding are not modelled here honestly enough to trust.
    allow_shorts: bool = False
    # Minimum notional for an order to be worth placing, in quote currency.
    min_order_notional: float = 25.0


@dataclass
class LLMBudget:
    """Spending and cadence controls for the committee.

    A full TradingAgents run is roughly a dozen LLM calls over large contexts.
    Running that every minute on five symbols would cost more than most
    accounts make, so the committee is rate-limited three ways: a daily dollar
    cap, a per-symbol cooldown, and a scheduled interval that acts as the
    floor cadence when no trigger fires.
    """

    daily_usd_cap: float = 5.0
    min_minutes_between_calls: int = 45
    scheduled_interval_minutes: int = 240
    # Rough per-run cost estimate used for pre-flight budget checks when the
    # provider does not report usage. Override to match your own measured cost.
    estimated_cost_per_run_usd: float = 0.35


@dataclass
class DeskConfig:
    """Top-level desk configuration."""

    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    starting_equity: float = 10_000.0
    # Paper-fill realism. 10 bps taker fee and 5 bps slippage is a reasonable
    # retail crypto assumption; raise it if you trade illiquid pairs.
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    # Fast-loop period. 60s is plenty: the committee thinks in hours, and a
    # tighter loop only burns rate limit.
    fast_loop_seconds: int = 60
    # Feed chain, tried in order until one returns data.
    feeds: list[str] = field(default_factory=lambda: ["cryptocom", "binance"])
    candle_interval: str = "5m"
    candle_lookback: int = 200
    # "llm" runs the TradingAgents committee; "heuristic" runs a free
    # trend/RSI rule that serves as the control group; "auto" uses the LLM
    # when credentials exist and falls back to the heuristic otherwise.
    committee: str = "auto"
    benchmark_symbol: str = "BTC-USD"
    home: str = str(DEFAULT_HOME)
    api_host: str = "127.0.0.1"
    api_port: int = 8787
    risk: RiskLimits = field(default_factory=RiskLimits)
    llm: LLMBudget = field(default_factory=LLMBudget)

    # ---- paths -------------------------------------------------------
    @property
    def home_path(self) -> Path:
        return Path(self.home).expanduser()

    @property
    def db_path(self) -> Path:
        return self.home_path / "desk.db"

    @property
    def reports_path(self) -> Path:
        return self.home_path / "reports"

    def ensure_dirs(self) -> None:
        self.home_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)

    # ---- serialisation -----------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> DeskConfig:
        """Build a config from a dict, ignoring unknown keys.

        Unknown keys are ignored rather than raising so a config file written
        by a newer version still loads; invalid *values* still raise, because
        a typo'd limit must not silently become a default.
        """
        raw = dict(raw or {})
        risk = RiskLimits(**_known(RiskLimits, raw.pop("risk", {}) or {}))
        llm = LLMBudget(**_known(LLMBudget, raw.pop("llm", {}) or {}))
        return cls(risk=risk, llm=llm, **_known(cls, raw))

    @classmethod
    def load(cls, path: str | Path | None = None) -> DeskConfig:
        """Load config from ``path`` (or ``$CRYPTODESK_CONFIG``), then env vars."""
        path = path or os.environ.get("CRYPTODESK_CONFIG")
        raw: dict = {}
        if path:
            p = Path(path).expanduser()
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
        cfg = cls.from_dict(raw)
        return _apply_env(cfg)

    def save(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p


def _known(cls, raw: dict) -> dict:
    """Filter ``raw`` down to the dataclass's own field names."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


# Env var -> (dotted attribute path). Kept as one table so adding an override
# is a single row, mirroring how tradingagents/default_config.py does it.
_ENV_OVERRIDES = {
    "CRYPTODESK_SYMBOLS": "symbols",
    "CRYPTODESK_STARTING_EQUITY": "starting_equity",
    "CRYPTODESK_FEEDS": "feeds",
    "CRYPTODESK_CANDLE_INTERVAL": "candle_interval",
    "CRYPTODESK_FAST_LOOP_SECONDS": "fast_loop_seconds",
    "CRYPTODESK_COMMITTEE": "committee",
    "CRYPTODESK_HOME": "home",
    "CRYPTODESK_API_HOST": "api_host",
    "CRYPTODESK_API_PORT": "api_port",
    "CRYPTODESK_FEE_BPS": "fee_bps",
    "CRYPTODESK_SLIPPAGE_BPS": "slippage_bps",
    "CRYPTODESK_BENCHMARK": "benchmark_symbol",
    "CRYPTODESK_MAX_SYMBOL_WEIGHT": "risk.max_symbol_weight",
    "CRYPTODESK_MAX_GROSS_EXPOSURE": "risk.max_gross_exposure",
    "CRYPTODESK_RISK_PER_TRADE": "risk.risk_per_trade",
    "CRYPTODESK_STOP_ATR_MULT": "risk.stop_atr_mult",
    "CRYPTODESK_TRAIL_ATR_MULT": "risk.trail_atr_mult",
    "CRYPTODESK_DAILY_LOSS_LIMIT": "risk.daily_loss_limit",
    "CRYPTODESK_MAX_DRAWDOWN_HALT": "risk.max_drawdown_halt",
    "CRYPTODESK_COOLDOWN_MINUTES": "risk.cooldown_minutes",
    "CRYPTODESK_ALLOW_SHORTS": "risk.allow_shorts",
    "CRYPTODESK_LLM_DAILY_CAP": "llm.daily_usd_cap",
    "CRYPTODESK_LLM_MIN_MINUTES": "llm.min_minutes_between_calls",
    "CRYPTODESK_LLM_INTERVAL_MINUTES": "llm.scheduled_interval_minutes",
}

_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce an env string to the type of the current value.

    Invalid values raise instead of falling back to the default: a misspelled
    risk limit must fail loudly at startup, not quietly run at 25% weight.
    """
    if isinstance(reference, bool):
        low = value.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if isinstance(reference, list):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env(cfg: DeskConfig) -> DeskConfig:
    for env_var, path in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        target, _, attr = path.rpartition(".")
        obj = getattr(cfg, target) if target else cfg
        try:
            setattr(obj, attr, _coerce(raw, getattr(obj, attr)))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return cfg
