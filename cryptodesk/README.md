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
| `daily_loss_limit` | 3% | Mark-to-market loss (realised plus unrealised) since 00:00 UTC that halts *new entries* for the rest of the day |
| `max_drawdown_halt` | 15% | Flattens the book and halts until a human resumes |
| `cooldown_minutes` | 120 | No re-entry on a symbol after it stops out |

Plus a hard stop and a trailing stop on every position, set at entry.

The kill-switch does not clear itself. `python -m cryptodesk run` will start
back up halted; resuming is a deliberate act (the Resume button, or
`POST /api/resume`). Resuming re-bases the drawdown peak to current equity
(otherwise the same drawdown would halt the desk again on the next tick) but
**keeps the day's loss baseline**: a halt/resume cycle never hands back the
budget the day already spent. Resume on a desk that is not halted is refused
(`409`) rather than quietly re-basing the brakes.

### Stale feeds

A symbol whose venue stops answering, or whose last closed candle is more than
two intervals old, is *stale*. A stale symbol gets no new entries, no stop
exits (a stop cannot fill on a frozen quote), and the committee is not woken
for it; its trailing stop still ratchets. A halt leaves a stale position open
with its stop armed and retries the flatten every tick until a fresh quote
arrives. When *every* symbol is stale no equity point is written, so an outage
does not paint a flat line into the track record. The dashboard header says
`feed stale: SOL-USD` and the positions table marks the row.

Positions the ledger restored for symbols that are no longer in `symbols` are
priced, stop-managed and **exit-only**: managed out, never re-entered, and
listed on the dashboard as such.

Fills and decisions are stamped at the time the committee *finished*, and the
decision row keeps the price the committee analysed alongside the price the
order actually filled at.

---

## The dashboard

- **Stat tiles** — equity, return, return *net of LLM spend*, alpha vs
  buy-and-hold, max drawdown, win rate, open risk, today's token bill. These
  are **lifetime** figures, computed over the whole ledger (thinned to ~4,000
  marks) with the desk's exact lifetime max drawdown and the benchmark's price
  at inception.
- **Equity curve** — the desk against buy-and-hold of the benchmark, on one
  dollar axis, with a dashed line at starting capital. The chart shows only
  the **last 1,500 marks** (about a day at the default cadence); the buy-and-hold
  line is anchored at the first-ever benchmark price, so it still reads from
  day one. The footer under the chart says which window is which.
- **Open positions** — entry, mark, stop, distance to stop, open P&L, the
  venue that served the quote, and a *stale* / *exit-only* flag where it applies.
- **Risk limits** — a meter per brake, so you can see how close each is.
- **Decisions** — every committee run: the trigger, the rating, what it argued,
  what the risk engine did with it, and the reason if it was refused. Expand
  one to read the bull case, bear case and risk-manager verdict verbatim.
- **Trade blotter** and **event log**.

Controls are limited to Halt and Resume on purpose. There is no manual-order
button: a hand-placed trade would corrupt the track record the desk exists to
produce. Resume is only enabled while halted, and asks for confirmation
because it re-bases the drawdown peak.

### Scripting the API

The two control endpoints (`POST /api/halt`, `POST /api/resume`) require the
header `X-CryptoDesk-Control: 1` and refuse `Sec-Fetch-Site: cross-site`.
A custom header forces a browser to preflight a cross-origin request, and the
API answers no preflight, so a page you happen to visit cannot halt your desk
(or resume it and re-base the brakes) through your own browser. Any script
just adds the header:

```bash
curl -X POST -H 'X-CryptoDesk-Control: 1' 'http://127.0.0.1:8787/api/halt?reason=maintenance'
```

The API also only answers to a `Host` header of `127.0.0.1`, `localhost`,
`::1`, `api_host`, or an entry in `api_allowed_hosts`; anything else is a
`400`. That is the DNS-rebinding guard: without it a page on
`attacker.example` that later resolves to `127.0.0.1` could read the state,
the decisions (with the verbatim agent reports) and drive the controls as a
same-origin request. Behind a reverse proxy, set
`CRYPTODESK_API_ALLOWED_HOSTS=desk.example.com` (a port may be included; the
host part is what is matched).

`GET /api/health` is green while the engine thread is alive and has
heartbeated within two loop periods, **or** while it is inside a committee
run shorter than `llm.max_run_seconds` — a multi-minute model run is `busy`,
not dead. The payload carries `busy`, `heartbeat_ts` and `stale_symbols`.

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
default is what it is. Loading validates: negative fees, a `nan` limit, a
weight above 100%, a quoted number or boolean in the JSON, an unknown feed or
interval, or a symbol that will not parse all fail at startup with the field
named. Symbols are stored canonically (`btcusd` -> `BTC-USD`). An explicit
config path (`--config` or `CRYPTODESK_CONFIG`) that does not exist is an
error, not a silent fall-back to defaults; `doctor` prints which file it
loaded. The benchmark need not be traded — when it is not in `symbols` its
price is fetched separately, for the comparison only.

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
4-hour floor cadence.

Tokens are counted per run. When `llm.price_per_million_input_usd` and
`llm.price_per_million_output_usd` are set (`CRYPTODESK_LLM_PRICE_IN` /
`CRYPTODESK_LLM_PRICE_OUT`), each run is priced from the tokens it actually
used and the dashboard labels the cost **(measured)**. Otherwise the configured
*estimate* (`llm.estimated_cost_per_run_usd`, `CRYPTODESK_LLM_COST_PER_RUN`)
is booked and labelled **(estimate)**; the cap is only as honest as that
figure, so measure one run against your provider's billing and set it. A run
that dies before its first model call costs nothing; one that dies mid-way
still pays for what it burned.

`TRADINGAGENTS_LLM_PROVIDER` and the model env vars pick the provider, and the
key checked at startup is the one *that* provider needs. The graph's results,
data cache and decision memory live under `<home>/tradingagents/`.

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
  reverse proxy with auth in front of it. A localhost bind is not by itself
  protection from your own browser; the control header and Host allow-list
  above are what provide that.
