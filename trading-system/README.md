# ictbot — ICT automated trading systems

A back-testable implementation of **five** trading systems distilled from the
ICT (Inner Circle Trader) strategy notes. Every system is built on one
principle:

> **Liquidity → Manipulation → Delivery**

It runs on the Python standard library only (Python 3.10+) — no third-party
packages to install — so you can back-test, paper-trade and study the logic out
of the box.

> ⚠️ **Risk disclaimer.** This is an educational/research tool. It ships with
> **synthetic** data that is *shaped* to contain the patterns so the pipeline
> can be demonstrated offline — those numbers say **nothing** about real-world
> profitability. Trading futures involves substantial risk of loss. Validate on
> real historical data, paper-trade first, and never run a live path with real
> money until you have done your own due diligence. Nothing here is financial
> advice.

---

## The five systems

The notes were grouped into five categories; each became its own system. They
all share the same primitives (Fair Value Gaps, liquidity sweeps, displacement,
HTF bias) and the same risk engine, and they all run as **no-look-ahead** state
machines fed one *closed* candle at a time.

| # | System | File | Session (ET) | The setup, from the notes |
|---|---|---|---|---|
| **Cat 1** | **9:30 Open Model** | `strategy.py` | 09:30 RTH | Reversal/Continuation: sweep resting liquidity, realign with HTF bias, enter the 1st PFVG inside a macro window |
| **Cat 2** | **Setup For Life** (ICT 2024) | `model_2024.py` | RTH macros | Stop-hunt at a premium/discount PD array → displacement → FVG (1st entry) / Breaker (2nd entry) |
| **Cat 3** | **PD Arrays toolkit** | `pd_arrays.py` | — | Order Block, Rejection Block & Breaker Block detectors — the shared entry library, not a standalone signal emitter |
| **Cat 4** | **Opening Ranges ×5** | `opening_range.py` | Midnight / London / NY Kill Zone / AM / PM | Each 30-min OR sets Open/High/Low; sweep one side, deliver the other to standard-deviation targets (0.5/1.0/1.5/2.5×) |
| **Cat 5** | **Asia Killzone** | `asia_killzone.py` | 18:00–21:30 | NDOG/NWOG gap → 7:50 PM Judas Swing → 8:50 PM displacement & delivery to the opposite extreme |

The **PD Arrays** of Cat 3 are how the notes say to "stack confluence": the
shared `best_pd_entry()` ranks candidate entries **FVG → Order Block → Breaker →
Rejection Block** and is reused by the other systems.

### How a trade is formed (shared shape)

For each session the relevant model walks one **closed** candle at a time
through: define **liquidity** → wait for **manipulation** (the sweep) → confirm
**displacement** (an FVG in the delivery direction) → enter on the **rebalance**
into that FVG *inside a macro window* → target the **draw on liquidity**. The
stop sits beyond the manipulation extreme; risk is sized so a stop-out loses
~`risk_per_trade` (default 0.5%) of the account, capped at `max_contracts`.
Degenerate setups (a stop only a couple of ticks from entry) are rejected.

---

## Quick start

```bash
cd trading-system

# ── All five systems at once ───────────────────────────────────────────────
# Generate full-day synthetic data (overnight + RTH + Asia) and back-test
# every system on it:
python -m ictbot multidemo --symbol NQ --days 120

# Back-test all five against your own full-day CSV:
python -m ictbot multibacktest --symbol NQ --csv data/sample_NQ_1m.csv

# ── Cat 1 only (the original 9:30 model) ──────────────────────────────────
python -m ictbot demo      --symbol NQ --days 120
python -m ictbot backtest  --symbol NQ --csv data/sample_NQ_1m.csv
python -m ictbot paper     --symbol NQ --csv data/sample_NQ_1m.csv   # streams bar-by-bar

# ── Utilities ─────────────────────────────────────────────────────────────
python -m ictbot gen --symbol NQ --days 60 --full-day --out data/my_data.csv

# Run the test suite (35 tests)
python -m unittest discover -s tests
```

Example `multidemo` summary (synthetic NQ, 120 sessions, seed 42):

```
 ICT Multi-System Backtest Summary
 Instrument: NQ
  Cat1_930Open        :  17 trades  WR   65%  +47.90R  E=+2.82R
  Cat2_SetupForLife   :  25 trades  WR   12%  -3.18R  E=-0.13R
  Cat4_OR_Midnight    :  66 trades  WR   35%  +52.83R  E=+0.80R
  Cat4_OR_London      :  61 trades  WR   25%  +31.16R  E=+0.51R
  Cat4_OR_NYKillZone  :  39 trades  WR   54%  +98.13R  E=+2.52R
  Cat4_OR_AM          :   3 trades  WR    0%  -3.00R  E=-1.00R
  Cat4_OR_PM          :  57 trades  WR   23%  +13.79R  E=+0.24R
  Cat5_AsiaKZ         :  43 trades  WR   84%  +109.17R  E=+2.54R
  TOTAL               : 311 trades combined
```

(Low win rate with positive expectancy is the classic ICT profile — small,
tight stops below the manipulation, large draws to the opposite liquidity.)

### A note on the synthetic data

`generate_sessions(..., full_day=True)` builds a complete 24-hour tape so every
system gets data in its own window. It **engineers designed setups** for the
9:30 model, the overnight Opening Ranges (Midnight / London / NY Kill Zone) and
the Asia Killzone, so those systems trade their intended patterns. The **AM/PM
Opening Ranges** and **Setup-For-Life** trade off whatever *incidental*
structure the RTH tape produces, so their synthetic results are noisier and
often negative. This is deliberate and honest: it shows the wiring works without
pretending every system has an edge. **Re-validate everything on real data.**

---

## Design: no look-ahead

Every model is fed bars that have already **closed**, one at a time, via
`on_candle(...)`. The multi-system runner additionally:

- feeds each calendar day's full tape to all systems and lets each one
  **self-gate** on its own session/macro clock;
- computes the HTF bias for each window **as of that window's close**, using
  only prior candles (a `bisect` over the sorted tape keeps it O(log n));
- **flattens** open positions at every session boundary (early → RTH → evening)
  so a trade never bleeds across an overnight price discontinuity;
- tags every trade with its originating system for per-system statistics.

The same `PaperBroker` and sizing run in back-test, paper and (with a real
broker adapter) live trading.

---

## Project layout

```
trading-system/
├── ictbot/
│   ├── config.py         session clock, macro windows, instruments, risk
│   ├── models.py         Candle, FVG, OrderBlock, RejectionBlock, BreakerBlock,
│   │                     OpeningRange, LiquidityPool, Signal, Trade
│   ├── data.py           CSV loader + synthetic generator (RTH + full-day)
│   ├── indicators.py     FVG / swing / ATR / displacement detection
│   ├── liquidity.py      liquidity pools + sweep detection + target selection
│   ├── bias.py           H1 order-flow bias
│   ├── pd_arrays.py      Cat 3 — Order/Rejection/Breaker block toolkit
│   ├── strategy.py       Cat 1 — 9:30 Open Model state machine
│   ├── model_2024.py     Cat 2 — Setup For Life state machine
│   ├── opening_range.py  Cat 4 — the 5 Opening Range state machines
│   ├── asia_killzone.py  Cat 5 — Asia Killzone NDOG/NWOG state machine
│   ├── runner.py         multi-system runner (all 5 in parallel)
│   ├── broker.py         Broker interface + PaperBroker + position sizing
│   ├── backtest.py       event-driven backtester + metrics (Cat 1)
│   ├── live.py           paper/live loop with session rollover (Cat 1)
│   └── cli.py            command-line interface
├── tests/                unit + end-to-end tests (35)
├── data/                 sample full-day CSV
└── requirements.txt
```

---

## Tuning

Edit `ictbot/config.py`: `StrategyConfig.fvg_entry_depth` (how deep into the FVG
to enter — 0.5 = consequent encroachment), `displacement_atr_mult` (how
energetic a candle must be to count as displacement), `SessionConfig`
macro/clock windows, and `RiskConfig.min_rr` / `risk_per_trade` /
`max_trades_per_day`. The five Opening-Range windows and their macro times live
in `OR_DEFINITIONS` (`opening_range.py`); the Asia macros in `asia_killzone.py`.

---

## Going live

`live.py` already drives the exact Cat 1 strategy + sizing used in
back-testing. To trade for real you would (1) replace the CSV/`generate_sessions`
feed with your broker/data-feed stream, and (2) implement the `Broker` interface
(`broker.py`) against your broker's order API and swap it in for `PaperBroker`.
Everything else — signal logic, risk sizing, session rollover — stays identical.
**Do this deliberately, with your own credentials and risk controls.**
