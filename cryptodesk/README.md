# CryptoDesk

A 24/7 crypto **paper-trading** desk built on TradingAgents: an always-on loop
that manages positions every minute, an LLM committee woken only when something
changes, a deterministic risk engine that sizes every trade, and a dashboard
that shows you what it did and why.

No real orders are placed. No exchange account is required.

---

## What it is for

The honest purpose of this system is **to find out whether the agents have any
edge, without risking money to do it**. It is built so that:

- every decision is logged with the reasoning that produced it,
- the result is compared against buy-and-hold, which is the only benchmark that
  matters in crypto,
- the LLM bill is booked against P&L, because a desk that makes $40 a week
  while spending $60 on tokens is losing money,
- and the downside is bounded by code rather than by good intentions.

It will not reliably make money, and nothing here should be read as a claim
that it will. What it *will* do is give you a real track record to judge.

---

## Quick start

```bash
pip install ".[desk]"

python -m cryptodesk doctor        # check config, feeds, credentials
python -m cryptodesk simulate --days 30 --serve   # fast-forward a month, then look at it
python -m cryptodesk run           # start the live paper desk + dashboard
```

Then open <http://127.0.0.1:8787>.

With Docker:

```bash
cp .env.example .env               # optional: add an LLM provider key
docker compose -f docker-compose.desk.yml up -d
```

Without an LLM key, the desk runs the free heuristic committee and says so on
the dashboard. Nothing else is stubbed.

---

## How it works

Three layers, deliberately separated:

```
        every 60s, free                on a trigger, paid           every order
   ┌──────────────────────┐      ┌────────────────────────┐   ┌──────────────────┐
   │  fast loop           │      │  committee             │   │  risk engine     │
   │  prices, indicators, │─────▶│  TradingAgents graph   │──▶│  sizes, stops,   │
   │  stops, marks, the   │ wake │  or heuristic baseline │   │  caps, vetoes    │
   │  kill-switch         │      │  → rating + conviction │   │  → qty + stop    │
   └──────────────────────┘      └────────────────────────┘   └──────────────────┘
```

**The fast loop** (`engine/loop.py`) is what makes the desk 24/7. It polls
prices, recomputes indicators, ratchets trailing stops, exits anything stopped
out, marks the book, and writes an equity point. It costs nothing and runs
forever.

**The committee** (`engine/committee.py`) is the expensive part, so it is woken
only when something material happened — a move of 1.5 ATR, a trend flip, a
volatility expansion, an RSI extreme — plus a floor cadence so a quiet market
still gets reviewed. Three independent limits gate it: a daily dollar cap, a
per-symbol cooldown, and the trigger logic itself.

It returns a five-tier rating and a conviction in [0, 1]. **It never returns a
quantity.**

**The risk engine** (`engine/risk.py`) turns that into an order. Size is
volatility-targeted: chosen so that a move to the stop costs a fixed fraction
of equity, which means size halves when volatility doubles and the dollar risk
per trade stays constant. Conviction can only scale a position *down* within a
cap it cannot raise.

### The brakes

Four independent limits, each able to stop trading on its own:

| Limit | Default | What it does |
|---|---|---|
| `max_symbol_weight` | 25% | Most of equity one symbol may hold at entry |
| `max_weight_drift` | +20% | How far a winner may drift above that before being trimmed |
| `max_gross_exposure` | 60% | Total deployed across all symbols |
| `risk_per_trade` | 1% | Equity lost if the stop fills exactly |
| `daily_loss_limit` | 3% | Halts *new entries* for the rest of the UTC day |
| `max_drawdown_halt` | 15% | Flattens the book and halts until a human resumes |
| `cooldown_minutes` | 120 | No re-entry on a symbol after it stops out |

Plus a hard stop and a trailing stop on every position, set at entry.

The kill-switch does not clear itself. `python -m cryptodesk run` will start
back up halted; resuming is a deliberate act (the Resume button, or
`POST /api/resume`).

---

## The dashboard

- **Stat tiles** — equity, return, return *net of LLM spend*, alpha vs
  buy-and-hold, max drawdown, win rate, open risk, today's token bill.
- **Equity curve** — the desk against buy-and-hold of the benchmark, on one
  dollar axis, with a dashed line at starting capital.
- **Open positions** — entry, mark, stop, distance to stop, open P&L.
- **Risk limits** — a meter per brake, so you can see how close each is.
- **Decisions** — every committee run: the trigger, the rating, what it argued,
  what the risk engine did with it, and the reason if it was refused. Expand
  one to read the bull case, bear case and risk-manager verdict verbatim.
- **Trade blotter** and **event log**.

Controls are limited to Halt and Resume on purpose. There is no manual-order
button: a hand-placed trade would corrupt the track record the desk exists to
produce.

---

## Configuration

JSON file plus `CRYPTODESK_*` environment overrides:

```bash
python -m cryptodesk init cryptodesk.json     # write the defaults, then edit
CRYPTODESK_CONFIG=cryptodesk.json python -m cryptodesk run
```

```bash
# or configure entirely by environment
CRYPTODESK_SYMBOLS=BTC-USD,ETH-USD,SOL-USD \
CRYPTODESK_RISK_PER_TRADE=0.005 \
CRYPTODESK_LLM_DAILY_CAP=2.00 \
python -m cryptodesk run
```

Every key in `config.py` has a comment explaining what it does and why the
default is what it is.

---

## Commands

| Command | What it does |
|---|---|
| `run` | Start the desk and the dashboard; trade until stopped |
| `simulate` | Fast-forward days or weeks in seconds (`--serve` to view the result) |
| `doctor` | Check config, feed reachability and LLM credentials |
| `report` | Print a performance summary from a ledger |
| `init` | Write a config file to edit |

`simulate` defaults to the **free heuristic** committee. Passing
`--committee llm` makes real, paid API calls on every trigger — thousands over
a long window.

---

## Feeds

`cryptocom` (default, keyless) → `binance` (fallback, keyless), then:

- `synthetic` — a seeded random walk. Runs with no network at all, so the desk
  and dashboard work offline. **Results on synthetic prices test the machinery,
  not the strategy.**
- `replay` — a directory of `<SYMBOL>.csv` candle files, for driving the desk
  over real historical episodes.

Binance lists USDT rather than USD, so a `-USD` request is served from the USDT
pair — a few basis points of difference, inside the slippage assumption.

---

## What the paper fills do and do not model

Modelled: taker fees (10 bps default), slippage against you (5 bps default),
cash constraints, and position accounting.

**Not** modelled: market impact, partial fills, funding on perpetuals, and —
most importantly — **stops are assumed to fill at their trigger price.** In a
real crash they fill worse. Treat the drawdowns here as a floor, not a ceiling.

Shorting is off by default: it is unbounded-loss and the borrow/funding side is
not modelled honestly enough to trust.

---

## Cost control

A full TradingAgents run is roughly a dozen LLM calls over large contexts. Run
that every minute across five symbols and it is ~7,000 runs a day.

The desk defaults to a **$5/day cap**, a 45-minute per-symbol cooldown, and a
4-hour floor cadence. The reported cost is the configured *estimate*
(`llm.estimated_cost_per_run_usd`), because the graph does not report provider
usage back to callers — the dashboard labels it as an estimate. Measure one run
against your provider's billing dashboard and set that number to the real one;
the cap is only as honest as that figure.

---

## Testing

```bash
python -m pytest tests/test_cryptodesk_*.py -q
```

The engine tests drive the real broker, ledger and risk logic over a
deterministic synthetic market — thousands of ticks in seconds — and assert the
invariants that matter: exposure never exceeds its cap, cash never goes
negative, every open position carries a stop, the kill-switch halts and
flattens, the daily LLM cap holds, and the whole book survives a restart.

---

## Limitations worth knowing

- **Daily-ish bars, not microstructure.** No order book, funding rate, open
  interest, or on-chain flow. The committee reasons in paragraphs over candles.
- **The LLM committee is unproven.** That is the entire point of running this.
  The heuristic baseline is included as the control group — if the LLM cannot
  beat it net of cost, it is not earning its keep.
- **Synthetic results prove nothing about edge.** They exercise the machinery.
- **The dashboard has no authentication.** Bind it to localhost or put a
  reverse proxy with auth in front of it.
