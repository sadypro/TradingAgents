"""SQLite persistence for the desk.

Everything the desk does is written here, for three reasons:

* **Restart safety.** The risk brakes (peak equity, the day's starting equity,
  per-symbol cooldowns) live in the ``state`` table. A desk that forgot its
  high-water mark on reboot would never fire its drawdown kill-switch.
* **Measurability.** The equity curve, every fill, and every decision — with
  the reasoning that produced it — are retained so the question "did this
  actually make money, and was it the committee or the market?" is answerable
  from data rather than memory.
* **Honest accounting.** LLM spend is a row in the same database as trading
  P&L. A desk that makes $40 a week while spending $60 on tokens is losing
  money, and that should be impossible to miss.

The engine thread writes and the API thread reads, so a single connection is
shared under a lock with WAL enabled.

Writes come in two flavours. The plain ``record_*`` / ``set_state`` methods
each commit on their own. :meth:`Ledger.transaction` groups several of them
into one commit, which is how the engine keeps a fill and the broker book that
resulted from it together: a crash between the two would otherwise restart the
desk with a blotter that shows a trade its book does not hold.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_curve (
    ts INTEGER PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    gross_exposure REAL NOT NULL,
    drawdown REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    fees REAL NOT NULL DEFAULT 0,
    benchmark_price REAL
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    reference_price REAL NOT NULL,
    fee REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    reason TEXT,
    trade_id TEXT                  -- position lifecycle; NULL in pre-migration rows
);
CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,          -- llm | heuristic
    trigger TEXT,                  -- why the committee was woken
    rating TEXT,                   -- Buy/Overweight/Hold/Underweight/Sell
    conviction REAL,
    action TEXT,                   -- entered | added | exited | reduced | none
    allowed INTEGER NOT NULL DEFAULT 0,
    veto_reason TEXT,              -- why the risk engine refused, if it did
    qty REAL DEFAULT 0,
    notional REAL DEFAULT 0,
    stop_price REAL,
    price REAL,
    cost_usd REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    summary TEXT,                  -- short human-readable rationale
    detail TEXT                    -- JSON: per-agent reports, indicator snapshot
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);

CREATE TABLE IF NOT EXISTS llm_spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    day TEXT NOT NULL,             -- UTC date, for the daily cap
    symbol TEXT,
    model TEXT,
    cost_usd REAL NOT NULL,
    ok INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_spend_day ON llm_spend(day);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    level TEXT NOT NULL,           -- info | warning | error
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_ts INTEGER NOT NULL
);
"""


# Columns added after the first release, applied by ``ALTER TABLE`` on open so
# an existing ledger keeps working without a manual migration step.
_MIGRATIONS = [
    ("fills", "trade_id", "TEXT"),
]


class _Writes:
    """The write statements, shared by the auto-commit and transactional paths.

    Subclasses supply ``_exec``; everything else is the same SQL either way,
    so a fill recorded inside a transaction cannot drift from one recorded
    outside it.
    """

    def _exec(self, sql: str, params: tuple = ()) -> int:
        raise NotImplementedError

    def set_state(self, key: str, value) -> None:
        self._exec(
            "INSERT INTO state(key, value, updated_ts) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts",
            (key, json.dumps(value), int(time.time())),
        )

    def record_equity(self, ts: int, equity: float, cash: float, gross_exposure: float,
                      drawdown: float, realized_pnl: float = 0.0,
                      unrealized_pnl: float = 0.0, fees: float = 0.0,
                      benchmark_price: float | None = None) -> None:
        # One row per timestamp; a re-mark inside the same second overwrites.
        self._exec(
            "INSERT INTO equity_curve(ts, equity, cash, gross_exposure, drawdown, "
            "realized_pnl, unrealized_pnl, fees, benchmark_price) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(ts) DO UPDATE SET equity=excluded.equity, cash=excluded.cash, "
            "gross_exposure=excluded.gross_exposure, drawdown=excluded.drawdown, "
            "realized_pnl=excluded.realized_pnl, unrealized_pnl=excluded.unrealized_pnl, "
            "fees=excluded.fees, benchmark_price=excluded.benchmark_price",
            (int(ts), equity, cash, gross_exposure, drawdown, realized_pnl,
             unrealized_pnl, fees, benchmark_price),
        )

    def record_fill(self, fill_dict: dict) -> int:
        return self._exec(
            "INSERT INTO fills(ts, symbol, side, qty, price, reference_price, fee, "
            "realized_pnl, reason, trade_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (int(fill_dict["ts"]), fill_dict["symbol"], fill_dict["side"],
             fill_dict["qty"], fill_dict["price"], fill_dict["reference_price"],
             fill_dict["fee"], fill_dict.get("realized_pnl", 0.0),
             fill_dict.get("reason", ""), fill_dict.get("trade_id") or None),
        )

    def record_decision(self, **kw) -> int:
        detail = kw.get("detail")
        return self._exec(
            "INSERT INTO decisions(ts, symbol, source, trigger, rating, conviction, "
            "action, allowed, veto_reason, qty, notional, stop_price, price, cost_usd, "
            "latency_ms, summary, detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (int(kw.get("ts") or time.time()), kw["symbol"], kw.get("source", "heuristic"),
             kw.get("trigger"), kw.get("rating"), kw.get("conviction"),
             kw.get("action", "none"), 1 if kw.get("allowed") else 0,
             kw.get("veto_reason"), kw.get("qty", 0.0), kw.get("notional", 0.0),
             kw.get("stop_price"), kw.get("price"), kw.get("cost_usd", 0.0),
             kw.get("latency_ms", 0), kw.get("summary"),
             json.dumps(detail) if detail is not None else None),
        )

    def record_spend(self, cost_usd: float, symbol: str | None = None,
                     model: str | None = None, ok: bool = True,
                     ts: int | None = None) -> None:
        ts = int(ts or time.time())
        day = time.strftime("%Y-%m-%d", time.gmtime(ts))
        self._exec(
            "INSERT INTO llm_spend(ts, day, symbol, model, cost_usd, ok) VALUES(?,?,?,?,?,?)",
            (ts, day, symbol, model, float(cost_usd), 1 if ok else 0),
        )

    def record_event(self, level: str, kind: str, message: str,
                     detail: dict | None = None, ts: int | None = None) -> None:
        self._exec(
            "INSERT INTO events(ts, level, kind, message, detail) VALUES(?,?,?,?,?)",
            (int(ts or time.time()), level, kind, message,
             json.dumps(detail) if detail else None),
        )


class Tx(_Writes):
    """Writes executed on the ledger's connection without committing.

    Only obtainable from :meth:`Ledger.transaction`, which holds the lock for
    the duration; do not call the ledger's own methods from inside the block
    (the lock is not re-entrant, and they would commit early).
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def _exec(self, sql: str, params: tuple = ()) -> int:
        return self._conn.execute(sql, params).lastrowid


class Ledger(_Writes):
    """Thread-safe SQLite store for desk state and history."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL lets the API thread read while the engine thread writes.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced since the ledger was created (idempotent)."""
        for table, column, decl in _MIGRATIONS:
            present = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in present:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- low level ----------------------------------------------------
    def _write(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    _exec = _write

    def _read(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    @contextmanager
    def transaction(self):
        """Group several writes into one commit.

        Yields a :class:`Tx` with the same ``record_*``/``set_state`` methods
        as the ledger. Everything written in the block lands together on a
        clean exit and is rolled back if the block raises, so a restart sees
        either the whole update or none of it.
        """
        with self._lock:
            try:
                yield Tx(self._conn)
            except BaseException:
                self._conn.rollback()
                raise
            self._conn.commit()

    # ---- state --------------------------------------------------------
    def get_state(self, key: str, default=None):
        rows = self._read("SELECT value FROM state WHERE key = ?", (key,))
        if not rows:
            return default
        try:
            return json.loads(rows[0]["value"])
        except json.JSONDecodeError:
            return default

    # ---- reads --------------------------------------------------------
    def spend_today(self, now: float | None = None) -> float:
        day = time.strftime("%Y-%m-%d", time.gmtime(now if now is not None else time.time()))
        rows = self._read("SELECT COALESCE(SUM(cost_usd), 0) AS total FROM llm_spend WHERE day = ?",
                          (day,))
        return float(rows[0]["total"]) if rows else 0.0

    def spend_total(self) -> float:
        rows = self._read("SELECT COALESCE(SUM(cost_usd), 0) AS total FROM llm_spend")
        return float(rows[0]["total"]) if rows else 0.0

    def equity_curve(self, limit: int = 2000) -> list[dict]:
        """Most recent points, oldest first (the order a chart wants)."""
        rows = self._read(
            "SELECT * FROM (SELECT * FROM equity_curve ORDER BY ts DESC LIMIT ?) "
            "ORDER BY ts ASC", (limit,))
        return rows

    def first_benchmark_price(self) -> float | None:
        """The earliest recorded benchmark mark, or None.

        Buy-and-hold is measured from inception. A ledger written before the
        risk state carried ``first_benchmark_price`` still has that mark in
        its first equity rows, and anchoring there rather than at the first
        post-upgrade tick keeps the comparison honest across an upgrade.
        """
        rows = self._read(
            "SELECT benchmark_price FROM equity_curve WHERE benchmark_price IS NOT NULL "
            "ORDER BY ts ASC LIMIT 1")
        return float(rows[0]["benchmark_price"]) if rows else None

    def equity_curve_sampled(self, max_points: int = 4000) -> list[dict]:
        """The whole history, thinned evenly by time to at most ~max_points.

        Lifetime statistics (return vs buy-and-hold, max drawdown) must see the
        first mark and every era in between, which :meth:`equity_curve`'s
        trailing window cannot give once a desk has run for a day. Points are
        bucketed by ``ts`` (the primary key, so this is a single index scan),
        keeping the first mark of each bucket plus the final mark.
        """
        bounds = self._read("SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM equity_curve")
        if not bounds or bounds[0]["lo"] is None:
            return []
        lo, hi = int(bounds[0]["lo"]), int(bounds[0]["hi"])
        bucket = max(1, math.ceil((hi - lo + 1) / max(1, max_points)))
        return self._read(
            "SELECT * FROM equity_curve WHERE ts IN ("
            "  SELECT MIN(ts) FROM equity_curve GROUP BY (ts - ?) / ?"
            ") OR ts = ? ORDER BY ts ASC",
            (lo, bucket, hi))

    def recent_fills(self, limit: int = 50) -> list[dict]:
        return self._read("SELECT * FROM fills ORDER BY ts DESC, id DESC LIMIT ?", (limit,))

    @staticmethod
    def _parse_detail(rows: list[dict]) -> list[dict]:
        for row in rows:
            if row.get("detail"):
                try:
                    row["detail"] = json.loads(row["detail"])
                except json.JSONDecodeError:
                    row["detail"] = None
        return rows

    def recent_decisions(self, limit: int = 25) -> list[dict]:
        rows = self._read("SELECT * FROM decisions ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
        return self._parse_detail(rows)

    def last_decision(self, symbol: str) -> dict | None:
        """The newest decision for ``symbol`` (detail JSON-parsed), or None."""
        rows = self._read(
            "SELECT * FROM decisions WHERE symbol = ? ORDER BY ts DESC, id DESC LIMIT 1",
            (symbol,))
        return self._parse_detail(rows)[0] if rows else None

    def recent_events(self, limit: int = 50) -> list[dict]:
        return self._read("SELECT * FROM events ORDER BY ts DESC, id DESC LIMIT ?", (limit,))

    def last_decision_ts(self, symbol: str) -> int | None:
        rows = self._read("SELECT MAX(ts) AS ts FROM decisions WHERE symbol = ?", (symbol,))
        ts = rows[0]["ts"] if rows else None
        return int(ts) if ts else None

    def _open_trade_ids(self) -> set[str]:
        """Trade ids of positions the broker still holds, per its persisted book."""
        raw = self.get_state("broker_state") or {}
        ids = set()
        for row in raw.get("positions") or []:
            try:
                # Same fallback the broker uses for books written before trade ids.
                ids.add(row.get("trade_id") or f"{row['symbol']}-{int(row['opened_ts'])}")
            except (KeyError, TypeError, ValueError):
                continue
        return ids

    def trade_stats(self) -> dict:
        """Aggregate stats per trade, not per fill.

        A trade is one position lifecycle (open -> flat), identified by
        ``trade_id``; its realised P&L is the sum over all its reducing fills,
        so a trimmed winner that later stops out scores as one loss rather than
        one win and one loss. A trade is closed once it has booked P&L and the
        broker's persisted book no longer holds it. Fills from before trade ids
        existed (``trade_id`` NULL) keep the old per-fill treatment.
        """
        rows = self._read(
            "SELECT trade_id, side, realized_pnl FROM fills ORDER BY ts ASC, id ASC")
        open_ids = self._open_trade_ids()

        pnls: list[float] = []
        partial_fills = 0
        by_trade: dict[str, dict] = {}
        for row in rows:
            trade_id = row["trade_id"]
            pnl = float(row["realized_pnl"])
            if trade_id is None:
                if pnl != 0:
                    pnls.append(pnl)
                continue
            trade = by_trade.setdefault(trade_id, {"side": row["side"], "pnl": 0.0, "reduces": 0})
            # A reversal is refused by the broker, so any fill on the opposite
            # side from the opening fill is a reduction.
            if row["side"] != trade["side"]:
                trade["pnl"] += pnl
                trade["reduces"] += 1
        for trade_id, trade in by_trade.items():
            if not trade["reduces"]:
                continue
            if trade_id in open_ids:
                partial_fills += trade["reduces"]
            else:
                pnls.append(trade["pnl"])

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_win, gross_loss = sum(wins), abs(sum(losses))
        return {
            "closed_trades": len(pnls),
            "partial_fills": partial_fills,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(pnls)) if pnls else 0.0,
            "avg_win": (gross_win / len(wins)) if wins else 0.0,
            "avg_loss": (gross_loss / len(losses)) if losses else 0.0,
            # Profit factor: gross profit / gross loss. Infinite with no losses
            # yet, which is reported as None rather than a misleading number.
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
            "gross_profit": gross_win,
            "gross_loss": gross_loss,
            "net_realized": sum(pnls),
        }
