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
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
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
    reason TEXT
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


class Ledger:
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
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- low level ----------------------------------------------------
    def _write(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    def _read(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    # ---- state --------------------------------------------------------
    def get_state(self, key: str, default=None):
        rows = self._read("SELECT value FROM state WHERE key = ?", (key,))
        if not rows:
            return default
        try:
            return json.loads(rows[0]["value"])
        except json.JSONDecodeError:
            return default

    def set_state(self, key: str, value) -> None:
        self._write(
            "INSERT INTO state(key, value, updated_ts) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts",
            (key, json.dumps(value), int(time.time())),
        )

    # ---- writes -------------------------------------------------------
    def record_equity(self, ts: int, equity: float, cash: float, gross_exposure: float,
                      drawdown: float, realized_pnl: float = 0.0,
                      unrealized_pnl: float = 0.0, fees: float = 0.0,
                      benchmark_price: float | None = None) -> None:
        # One row per timestamp; a re-mark inside the same second overwrites.
        self._write(
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
        return self._write(
            "INSERT INTO fills(ts, symbol, side, qty, price, reference_price, fee, "
            "realized_pnl, reason) VALUES(?,?,?,?,?,?,?,?,?)",
            (int(fill_dict["ts"]), fill_dict["symbol"], fill_dict["side"],
             fill_dict["qty"], fill_dict["price"], fill_dict["reference_price"],
             fill_dict["fee"], fill_dict.get("realized_pnl", 0.0),
             fill_dict.get("reason", "")),
        )

    def record_decision(self, **kw) -> int:
        detail = kw.get("detail")
        return self._write(
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
        self._write(
            "INSERT INTO llm_spend(ts, day, symbol, model, cost_usd, ok) VALUES(?,?,?,?,?,?)",
            (ts, day, symbol, model, float(cost_usd), 1 if ok else 0),
        )

    def record_event(self, level: str, kind: str, message: str,
                     detail: dict | None = None, ts: int | None = None) -> None:
        self._write(
            "INSERT INTO events(ts, level, kind, message, detail) VALUES(?,?,?,?,?)",
            (int(ts or time.time()), level, kind, message,
             json.dumps(detail) if detail else None),
        )

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

    def recent_fills(self, limit: int = 50) -> list[dict]:
        return self._read("SELECT * FROM fills ORDER BY ts DESC, id DESC LIMIT ?", (limit,))

    def recent_decisions(self, limit: int = 25) -> list[dict]:
        rows = self._read("SELECT * FROM decisions ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
        for row in rows:
            if row.get("detail"):
                try:
                    row["detail"] = json.loads(row["detail"])
                except json.JSONDecodeError:
                    row["detail"] = None
        return rows

    def recent_events(self, limit: int = 50) -> list[dict]:
        return self._read("SELECT * FROM events ORDER BY ts DESC, id DESC LIMIT ?", (limit,))

    def last_decision_ts(self, symbol: str) -> int | None:
        rows = self._read("SELECT MAX(ts) AS ts FROM decisions WHERE symbol = ?", (symbol,))
        ts = rows[0]["ts"] if rows else None
        return int(ts) if ts else None

    def trade_stats(self) -> dict:
        """Aggregate stats over closing fills (the ones that book P&L)."""
        rows = self._read(
            "SELECT realized_pnl FROM fills WHERE realized_pnl != 0 ORDER BY ts ASC")
        pnls = [float(r["realized_pnl"]) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_win, gross_loss = sum(wins), abs(sum(losses))
        return {
            "closed_trades": len(pnls),
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
