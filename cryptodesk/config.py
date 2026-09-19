"""Configuration for the desk: symbols, cadence, risk limits, LLM budget.

Config is a plain JSON file plus ``CRYPTODESK_*`` environment overrides, so a
container deployment can be reconfigured without editing files and no YAML
dependency is pulled in. Every default here is deliberately conservative: the
desk is meant to survive long enough to produce a statistically meaningful
track record, and a blown-up account produces no data at all.

Loading validates. A limit that is negative, ``nan``, a string, or simply out
of the range the engine can act on is refused at startup with the field named,
because the alternative — a kill-switch comparing against ``nan`` and never
firing — is exactly the quiet failure an unattended desk cannot afford.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .feeds import _REGISTRY as _FEED_REGISTRY
from .feeds.base import INTERVAL_SECONDS, FeedError, canonical_symbol

DEFAULT_HOME = Path(os.path.expanduser("~")) / ".cryptodesk"

COMMITTEE_MODES = ("auto", "llm", "heuristic")

# crypto.com serves at most this many candles per request, and the feed
# refuses larger lookbacks rather than truncating; catch it here so doctor
# reports it instead of every tick failing.
_CRYPTOCOM_MAX_CANDLES = 300


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
    # Mark-to-market loss (realised plus unrealised) over a UTC day that halts
    # new entries; existing positions keep their stops. Measured from the
    # equity at 00:00 UTC, and a halt/resume cycle does not reset it.
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
    # Rough per-run cost estimate used for pre-flight budget checks and booked
    # as spend when the provider's usage cannot be priced. Override to match
    # your own measured cost.
    estimated_cost_per_run_usd: float = 0.35
    # Longest a single committee run may take before the health check stops
    # treating "busy" as alive. A cold TradingAgents run with news tools can
    # take several minutes; a run past this is a hang, not thinking.
    max_run_seconds: int = 900
    # Provider list prices per million tokens. When both are set the committee
    # prices each run from the tokens it actually used and labels the cost as
    # measured; when unset it books the estimate above and says so.
    price_per_million_input_usd: float | None = None
    price_per_million_output_usd: float | None = None


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
    # Buy-and-hold comparison. It need not be traded: when it is not in
    # ``symbols`` the desk fetches its price separately, for the comparison only.
    benchmark_symbol: str = "BTC-USD"
    home: str = str(DEFAULT_HOME)
    api_host: str = "127.0.0.1"
    api_port: int = 8787
    # Extra Host header values the API answers to, for a reverse proxy in
    # front of it (``desk.example.com``). Loopback names and ``api_host`` are
    # always allowed; everything else is refused with 400 so a DNS-rebinding
    # page cannot read the desk's state as if it were same-origin.
    api_allowed_hosts: list[str] = field(default_factory=list)
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
    def from_dict(cls, raw: dict, *, validate: bool = True) -> DeskConfig:
        """Build a config from a dict, ignoring unknown keys.

        Unknown keys are ignored rather than raising so a config file written
        by a newer version still loads; invalid *values* still raise, because
        a typo'd limit must not silently become a default. Strings are coerced
        by the field's declared type (``"0.15"`` -> 0.15, ``"false"`` -> False),
        so a JSON file that quoted a number does not smuggle a str into the
        engine.
        """
        raw = dict(raw or {})
        risk = RiskLimits(**_typed(RiskLimits, raw.pop("risk", {}) or {}, "risk."))
        llm = LLMBudget(**_typed(LLMBudget, raw.pop("llm", {}) or {}, "llm."))
        cfg = cls(risk=risk, llm=llm, **_typed(cls, raw, ""))
        return cfg.validate() if validate else cfg

    @classmethod
    def load(cls, path: str | Path | None = None) -> DeskConfig:
        """Load config from ``path`` (or ``$CRYPTODESK_CONFIG``), then env vars.

        A path that was asked for but does not exist raises: silently running
        on defaults because of a typo in ``CRYPTODESK_CONFIG`` is the opposite
        of what an explicit path means. Defaults apply only when neither the
        argument nor the env var is set.
        """
        p, origin = config_source(path)
        raw: dict = {}
        if p is not None:
            if not p.is_file():
                raise FileNotFoundError(f"Config file {p} not found (from {origin})")
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Config file {p} is not valid JSON: {exc}") from exc
        # Validate once, after the env overrides, so a file that relies on an
        # env var for e.g. its symbols is judged as the desk will actually run.
        cfg = cls.from_dict(raw, validate=False)
        return _apply_env(cfg).validate()

    def save(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    # ---- validation ----------------------------------------------------
    def validate(self) -> DeskConfig:
        """Refuse anything the engine would silently misbehave on.

        Raises ``ValueError`` naming the field. Also canonicalises what it
        checks: symbols become ``BASE-QUOTE`` upper-case, feeds and the
        committee mode lower-case, so the rest of the desk can key on them.
        """
        _check_types(self, "")
        _check_types(self.risk, "risk.")
        _check_types(self.llm, "llm.")

        self.symbols = _canonical_symbols(self.symbols, "symbols")
        if not self.symbols:
            raise ValueError("symbols: at least one symbol is required")
        self.benchmark_symbol = _canonical_symbols([self.benchmark_symbol], "benchmark_symbol")[0]

        self.feeds = [name.strip().lower() for name in self.feeds if name.strip()]
        if not self.feeds:
            raise ValueError("feeds: at least one feed is required")
        unknown = [name for name in self.feeds if name not in _FEED_REGISTRY]
        if unknown:
            raise ValueError(f"feeds: unknown feed(s) {', '.join(unknown)}; "
                             f"expected one of {', '.join(sorted(_FEED_REGISTRY))}")

        self.committee = self.committee.strip().lower()
        if self.committee not in COMMITTEE_MODES:
            raise ValueError(f"committee: expected one of {', '.join(COMMITTEE_MODES)}, "
                             f"got {self.committee!r}")
        if self.candle_interval not in INTERVAL_SECONDS:
            raise ValueError(f"candle_interval: expected one of "
                             f"{', '.join(INTERVAL_SECONDS)}, got {self.candle_interval!r}")
        self.api_allowed_hosts = [h.strip() for h in self.api_allowed_hosts if h.strip()]
        if not self.home.strip():
            raise ValueError("home: must not be empty")
        if not self.api_host.strip():
            raise ValueError("api_host: must not be empty")

        _require("starting_equity", self.starting_equity, gt=0)
        _require("fee_bps", self.fee_bps, ge=0)
        _require("slippage_bps", self.slippage_bps, ge=0)
        _require("fast_loop_seconds", self.fast_loop_seconds, ge=5)
        _require("candle_lookback", self.candle_lookback, ge=50)
        if "cryptocom" in self.feeds:
            _require("candle_lookback", self.candle_lookback, le=_CRYPTOCOM_MAX_CANDLES,
                     why="crypto.com serves at most 300 candles per request")
        _require("api_port", self.api_port, ge=1, le=65535)

        r = self.risk
        _require("risk.max_symbol_weight", r.max_symbol_weight, gt=0, le=1)
        _require("risk.max_gross_exposure", r.max_gross_exposure, gt=0, le=1)
        if r.max_symbol_weight > r.max_gross_exposure:
            raise ValueError(f"risk.max_symbol_weight: {r.max_symbol_weight} exceeds "
                             f"risk.max_gross_exposure {r.max_gross_exposure}")
        _require("risk.max_weight_drift", r.max_weight_drift, ge=0)
        _require("risk.risk_per_trade", r.risk_per_trade, gt=0, le=0.1)
        _require("risk.stop_atr_mult", r.stop_atr_mult, gt=0)
        _require("risk.trail_atr_mult", r.trail_atr_mult, gt=0)
        _require("risk.daily_loss_limit", r.daily_loss_limit, gt=0, lt=1)
        _require("risk.max_drawdown_halt", r.max_drawdown_halt, gt=0, lt=1)
        _require("risk.cooldown_minutes", r.cooldown_minutes, ge=0)
        _require("risk.min_order_notional", r.min_order_notional, gt=0)

        b = self.llm
        _require("llm.daily_usd_cap", b.daily_usd_cap, gt=0)
        _require("llm.estimated_cost_per_run_usd", b.estimated_cost_per_run_usd, gt=0)
        _require("llm.min_minutes_between_calls", b.min_minutes_between_calls, ge=0)
        _require("llm.scheduled_interval_minutes", b.scheduled_interval_minutes, gt=0)
        _require("llm.max_run_seconds", b.max_run_seconds, gt=0)
        prices = (b.price_per_million_input_usd, b.price_per_million_output_usd)
        if (prices[0] is None) != (prices[1] is None):
            # Measured cost needs both sides; one alone would price runs as
            # if the other side were free.
            raise ValueError("llm.price_per_million_input_usd and "
                             "llm.price_per_million_output_usd must be set together")
        if prices[0] is not None:
            _require("llm.price_per_million_input_usd", prices[0], ge=0)
            _require("llm.price_per_million_output_usd", prices[1], ge=0)
        return self


def config_source(path: str | Path | None = None) -> tuple[Path | None, str | None]:
    """Resolve where config comes from: ``(path, origin)``.

    ``origin`` is ``"--config"`` for an explicit path, ``"CRYPTODESK_CONFIG"``
    for the env var, and ``None`` (with no path) when defaults apply.
    """
    if path:
        return Path(path).expanduser(), "--config"
    env = os.environ.get("CRYPTODESK_CONFIG")
    if env:
        return Path(env).expanduser(), "CRYPTODESK_CONFIG"
    return None, None


def _canonical_symbols(raw: list[str], name: str) -> list[str]:
    out: list[str] = []
    for symbol in raw:
        try:
            canonical = canonical_symbol(symbol)
        except FeedError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        if canonical not in out:
            out.append(canonical)
    return out


def _require(name: str, value, gt=None, ge=None, lt=None, le=None, why: str = "") -> None:
    ok = ((gt is None or value > gt) and (ge is None or value >= ge)
          and (lt is None or value < lt) and (le is None or value <= le))
    if ok:
        return
    bounds = " and ".join(text for flag, text in (
        (gt is not None, f"> {gt}"), (ge is not None, f">= {ge}"),
        (lt is not None, f"< {lt}"), (le is not None, f"<= {le}")) if flag)
    raise ValueError(f"{name}: must be {bounds}, got {value!r}"
                     + (f" ({why})" if why else ""))


# ---- typing --------------------------------------------------------------
# Field annotations are strings (``from __future__ import annotations``), so
# the declared type is read from the text rather than from the current value:
# a field whose default is None still knows it wants a float.
_KINDS = {
    "bool": "bool", "int": "int", "float": "float", "str": "str",
    "list[str]": "list", "float | None": "optional_float",
}


def _kind(cls, name: str) -> str | None:
    """The coercion kind of a dataclass field, or None for nested dataclasses."""
    annotation = cls.__dataclass_fields__[name].type
    return _KINDS.get(str(annotation).replace("Optional[float]", "float | None"))


def _known(cls, raw: dict) -> dict:
    """Filter ``raw`` down to the dataclass's own field names."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def _typed(cls, raw: dict, prefix: str) -> dict:
    """``_known`` plus coercion of string values to the field's declared type."""
    out = {}
    for key, value in _known(cls, raw).items():
        kind = _kind(cls, key)
        if isinstance(value, str) and kind not in (None, "str"):
            try:
                value = _coerce(value, kind)
            except ValueError as exc:
                raise ValueError(f"{prefix}{key}: {exc}") from exc
        out[key] = value
    return out


def _check_types(obj, prefix: str) -> None:
    """Every field holds a value of its declared type; numbers are finite."""
    for f in fields(obj):
        kind = _kind(type(obj), f.name)
        if kind is None:
            continue
        value = getattr(obj, f.name)
        label = f"{prefix}{f.name}"
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{label}: expected a boolean, got {value!r}")
        elif kind == "int":
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label}: expected an integer, got {value!r}")
        elif kind in ("float", "optional_float"):
            if kind == "optional_float" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label}: expected a number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{label}: must be a finite number, got {value!r}")
            value = float(value)
        elif kind == "list":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{label}: expected a list of strings, got {value!r}")
        elif kind == "str" and not isinstance(value, str):
            raise ValueError(f"{label}: expected a string, got {value!r}")
        setattr(obj, f.name, value)


# Env var -> (dotted attribute path). Kept as one table so adding an override
# is a single row, mirroring how tradingagents/default_config.py does it.
_ENV_OVERRIDES = {
    "CRYPTODESK_SYMBOLS": "symbols",
    "CRYPTODESK_STARTING_EQUITY": "starting_equity",
    "CRYPTODESK_FEEDS": "feeds",
    "CRYPTODESK_CANDLE_INTERVAL": "candle_interval",
    "CRYPTODESK_CANDLE_LOOKBACK": "candle_lookback",
    "CRYPTODESK_FAST_LOOP_SECONDS": "fast_loop_seconds",
    "CRYPTODESK_COMMITTEE": "committee",
    "CRYPTODESK_HOME": "home",
    "CRYPTODESK_API_HOST": "api_host",
    "CRYPTODESK_API_PORT": "api_port",
    "CRYPTODESK_API_ALLOWED_HOSTS": "api_allowed_hosts",
    "CRYPTODESK_FEE_BPS": "fee_bps",
    "CRYPTODESK_SLIPPAGE_BPS": "slippage_bps",
    "CRYPTODESK_BENCHMARK": "benchmark_symbol",
    "CRYPTODESK_MAX_SYMBOL_WEIGHT": "risk.max_symbol_weight",
    "CRYPTODESK_MAX_WEIGHT_DRIFT": "risk.max_weight_drift",
    "CRYPTODESK_MAX_GROSS_EXPOSURE": "risk.max_gross_exposure",
    "CRYPTODESK_RISK_PER_TRADE": "risk.risk_per_trade",
    "CRYPTODESK_STOP_ATR_MULT": "risk.stop_atr_mult",
    "CRYPTODESK_TRAIL_ATR_MULT": "risk.trail_atr_mult",
    "CRYPTODESK_DAILY_LOSS_LIMIT": "risk.daily_loss_limit",
    "CRYPTODESK_MAX_DRAWDOWN_HALT": "risk.max_drawdown_halt",
    "CRYPTODESK_COOLDOWN_MINUTES": "risk.cooldown_minutes",
    "CRYPTODESK_ALLOW_SHORTS": "risk.allow_shorts",
    "CRYPTODESK_MIN_ORDER_NOTIONAL": "risk.min_order_notional",
    "CRYPTODESK_LLM_DAILY_CAP": "llm.daily_usd_cap",
    "CRYPTODESK_LLM_MIN_MINUTES": "llm.min_minutes_between_calls",
    "CRYPTODESK_LLM_INTERVAL_MINUTES": "llm.scheduled_interval_minutes",
    "CRYPTODESK_LLM_COST_PER_RUN": "llm.estimated_cost_per_run_usd",
    "CRYPTODESK_LLM_MAX_RUN_SECONDS": "llm.max_run_seconds",
    "CRYPTODESK_LLM_PRICE_IN": "llm.price_per_million_input_usd",
    "CRYPTODESK_LLM_PRICE_OUT": "llm.price_per_million_output_usd",
}

_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")
_NONE = ("none", "null")


def _coerce(value: str, kind: str):
    """Coerce a string to a field's declared kind (see ``_KINDS``).

    Invalid values raise instead of falling back to the default: a misspelled
    risk limit must fail loudly at startup, not quietly run at 25% weight.
    """
    low = value.strip().lower()
    if kind == "bool":
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if kind == "list":
        return [part.strip() for part in value.split(",") if part.strip()]
    if kind == "int":
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"expected an integer, got {value!r}") from None
    if kind in ("float", "optional_float"):
        if kind == "optional_float" and low in _NONE:
            return None
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"expected a number, got {value!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"expected a finite number, got {value!r}")
        return number
    return value


def _apply_env(cfg: DeskConfig) -> DeskConfig:
    for env_var, path in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        target, _, attr = path.rpartition(".")
        obj = getattr(cfg, target) if target else cfg
        try:
            setattr(obj, attr, _coerce(raw, _kind(type(obj), attr) or "str"))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return cfg
