# ictbot — ICT 9:30 A.M. Open Model (automated trading system)

An automated, back-testable implementation of the **ICT "9:30 A.M. Open
Model"** described in the strategy notes. The whole system is built on one
principle:

> **Liquidity → Manipulation → Delivery**

It runs on the Python standard library only (Python 3.10+) — no third-party
packages to install — so you can back-test, paper-trade, and study the logic
out of the box.

> ⚠️ **Risk disclaimer.** This is an educational/research tool. It ships with
> **synthetic** data that is *shaped* to contain the pattern so the pipeline can
> be demonstrated offline — those numbers say **nothing** about real-world
> profitability. Trading futures involves substantial risk of loss. Validate on
> real historical data, paper-trade first, and never run the live path with real
> money until you have done your own due diligence. Nothing here is financial
> advice.

---

## The model, straight from the notes

The notes describe a repeatable intraday framework with two daily variants:

| Variant | What happens |
|---|---|
| **Reversal** | Manipulation *at* 9:30 — liquidity is taken, then price reverses |
| **Continuation** | Manipulation *before* 9:30 — liquidity already cleared, price displaces at the open toward the higher-time-frame (HTF) objective |

The flowchart from the notes maps directly onto the code:

```
Define Liquidity ─▶ Manipulation ─▶ Wait for Alignment w/ HTF Bias & Draw ─▶ Entry Model ─▶ Targets
```

| Note concept | Where it lives |
|---|---|
| **Liquidity** — PMH/PML, previous session H/L, previous daily H/L | `ictbot/liquidity.py` |
| **Manipulation** — the raid/sweep of a resting pool | `ictbot/liquidity.py` (`detect_sweep`) + `strategy.py` |
| **HTF bias** — H1 order flow + supporting H1 FVG | `ictbot/bias.py` |
| **Delivery / PD Arrays** — Fair Value Gaps, the "1st PFVG", the IOFED entry | `ictbot/indicators.py` + `strategy.py` |
| **Entry model** — anchored to the first PFVG, framed by **macro time windows** | `ictbot/strategy.py` |
| **Targets** — the draw on liquidity (opposite pool) | `ictbot/strategy.py` + `backtest.py` |

The notes' best-practice rules are enforced in code:

- **Best applied to NQ/ES** — both are first-class instruments in `config.py`.
- **Stack confluence (AM draw + HTF bias + order flow)** — a setup requires a
  directional HTF bias *and* a liquidity raid *and* a displacement FVG.
- **Wait for manipulation before entering** — the state machine cannot reach the
  entry state until a sweep is registered.
- **Anchor to a PD Array (the 1st PFVG) and frame around macro windows** —
  entries only trigger on a rebalance into the first post-displacement FVG while
  price is inside a macro window (the "9:50 macro" and friends).
- **One clean shot** — `max_trades_per_day = 1` by default.

---

## Quick start

```bash
cd trading-system

# 1) Generate synthetic data AND back-test it in one step
python -m ictbot demo --symbol NQ --days 120

# 2) Save synthetic 1-minute data to a CSV
python -m ictbot gen --symbol NQ --days 120 --out data/sample_NQ_1m.csv

# 3) Back-test against your OWN data (CSV columns: time,open,high,low,close,volume)
python -m ictbot backtest --symbol NQ --csv data/my_data.csv

# 4) Paper-trade by streaming a CSV bar-by-bar (prints live entries/exits)
python -m ictbot paper --symbol NQ --csv data/sample_NQ_1m.csv

# Run the test suite
python -m unittest discover -s tests
```

Example back-test summary (synthetic NQ, 120 sessions, seed 7):

```
 ICT 9:30 Open Model - Backtest Summary
 Instrument        : NQ
 Trades            : 14
 Win rate          :  42.9%
 Total R           : +12.77R
 Expectancy        : +0.91R / trade
 Profit factor     : 2.60
 Max drawdown      : 3.00R
```

(Win rate below 50% with positive expectancy is the classic ICT profile — small,
tight stops below the manipulation, large draws to the opposite liquidity.)

---

## How a trade is formed (state machine)

For each session, `NineThirtyOpenModel` walks one **closed** candle at a time
(no look-ahead) through these states:

1. **WAIT_OPEN** — accumulate the pre-market range. If a counter-bias pool is
   raided *before* 09:30 → label the day **Continuation**.
2. **WAIT_MANIPULATION** — after 09:30, wait for the raid that runs *against* the
   HTF bias (long bias → sweep a sellside low). Record the manipulation extreme
   (the stop reference). This is the **Reversal** path.
3. **WAIT_DISPLACEMENT** — wait for the first expansion leg back in the bias
   direction that prints an FVG — the **1st PFVG**.
4. **WAIT_ENTRY** — inside a **macro window**, wait for price to rebalance into
   that FVG (the **IOFED** entry). Stop goes beyond the manipulation extreme;
   the target is the nearest draw-on-liquidity pool that still gives ≥ `min_rr`.
5. **DONE** — one setup per session.

Risk is sized so a stop-out loses ~`risk_per_trade` (default 0.5%) of the
account, capped at `max_contracts`.

---

## Project layout

```
trading-system/
├── ictbot/
│   ├── config.py      session clock, macro windows, instruments, risk
│   ├── models.py      Candle, FVG, LiquidityPool, Signal, Trade
│   ├── data.py        CSV loader + synthetic session generator
│   ├── indicators.py  FVG / swing / ATR / displacement detection
│   ├── liquidity.py   liquidity pools + sweep detection + target selection
│   ├── bias.py        H1 order-flow bias
│   ├── strategy.py    the 9:30 Open Model state machine
│   ├── broker.py      Broker interface + PaperBroker + position sizing
│   ├── backtest.py    event-driven backtester + metrics
│   ├── live.py        paper/live trading loop (session rollover)
│   └── cli.py         command-line interface
├── tests/             unit + end-to-end tests
├── data/              sample CSV(s)
└── requirements.txt
```

---

## Tuning

Edit `ictbot/config.py`:

- `StrategyConfig.fvg_entry_depth` — how deep into the FVG to enter (0.5 = the
  50% / consequent-encroachment level).
- `StrategyConfig.displacement_atr_mult` — how energetic a candle must be to
  count as displacement.
- `SessionConfig.macro_windows` — the intraday windows entries are allowed in.
- `RiskConfig.min_rr` / `risk_per_trade` / `max_trades_per_day`.

---

## Going live

`live.py` already drives the exact strategy + sizing used in back-testing. To
trade for real you would:

1. Replace the CSV/`generate_sessions` feed with your broker/data-feed stream.
2. Implement the `Broker` interface (`broker.py`) against your broker's order
   API and swap it in for `PaperBroker`.

Everything else — signal logic, risk sizing, session rollover — stays identical.
**Do this deliberately, with your own credentials and risk controls.**
