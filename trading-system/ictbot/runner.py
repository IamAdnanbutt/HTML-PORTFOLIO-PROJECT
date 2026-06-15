"""Unified multi-system runner.

Feeds every closed candle through all active trading systems in parallel and
collects signals/trades from each.  This lets you compare systems on the same
data or run them together in paper / live mode.

Systems:
  Cat1: NineThirtyOpenModel         (9:30 AM Open Model)
  Cat2: SetupForLife                (ICT 2024 "The Setup For Life")
  Cat4: OpeningRangeModel × 5      (Midnight / London / NYKillZone / AM / PM)
  Cat5: AsiaKillzoneModel           (Asia KZ NDOG / NWOG)

(Cat3 PD Arrays is the shared entry toolkit, not a standalone signal emitter.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Dict, List, Optional

from .asia_killzone import AsiaKillzoneModel
from .backtest import BacktestResult, Backtester
from .bias import htf_bias
from .broker import PaperBroker, position_size
from .config import Config
from .liquidity import build_liquidity_pools, group_by_day
from .model_2024 import SetupForLife
from .models import Candle, Direction, ExitReason, LiquidityPool, Signal, Trade
from .opening_range import OpeningRangeModel, build_all_or_models
from .strategy import NineThirtyOpenModel


@dataclass
class SystemResult:
    """Per-system trade log and summary stats."""

    name: str
    trades: List[Trade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        wins = [t for t in self.trades if t.r_multiple > 0]
        return len(wins) / self.n_trades if self.n_trades else 0.0

    @property
    def total_r(self) -> float:
        return sum(t.r_multiple for t in self.trades)

    @property
    def expectancy(self) -> float:
        return self.total_r / self.n_trades if self.n_trades else 0.0

    def summary_line(self) -> str:
        return (f"  {self.name:<20}: {self.n_trades:3d} trades  "
                f"WR {self.win_rate * 100:4.0f}%  "
                f"{self.total_r:+.2f}R  "
                f"E={self.expectancy:+.2f}R")


@dataclass
class MultiSystemResult:
    systems: Dict[str, SystemResult] = field(default_factory=dict)
    config: Optional[Config] = None

    def summary(self) -> str:
        lines = [
            "=" * 66,
            " ICT Multi-System Backtest Summary",
            f" Instrument: {self.config.instrument.symbol if self.config else '?'}",
            "=" * 66,
        ]
        total_trades = sum(s.n_trades for s in self.systems.values())
        for name, sys_res in self.systems.items():
            if sys_res.n_trades:
                lines.append(sys_res.summary_line())
        lines.append("-" * 66)
        lines.append(f"  {'TOTAL':<20}: {total_trades:3d} trades combined")
        lines.append("=" * 66)
        return "\n".join(lines)


class MultiSystemRunner:
    """Run all systems in parallel on the same candle stream."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cat1 = NineThirtyOpenModel(config)
        self.cat2 = SetupForLife(config)
        self.cat4 = build_all_or_models(config)
        self.cat5 = AsiaKillzoneModel(config)
        self.broker = PaperBroker(config.instrument)
        self.result = MultiSystemResult(config=config)
        for name in (["Cat1_930Open", "Cat2_SetupForLife"] +
                     [f"Cat4_OR_{n}" for n in self.cat4] +
                     ["Cat5_AsiaKZ"]):
            self.result.systems[name] = SystemResult(name)

    def run(self, candles: List[Candle]) -> MultiSystemResult:
        candles = sorted(candles, key=lambda c: c.time)
        by_day = group_by_day(candles)
        days = sorted(by_day)

        for i, day in enumerate(days):
            day_candles = by_day[day]
            s = self.cfg.session
            session = [c for c in day_candles
                       if s.open_time <= c.time.time() <= s.close_time]
            premarket = [c for c in day_candles
                         if s.premarket_start <= c.time.time() < s.open_time]
            prev_day = by_day[days[i - 1]] if i >= 1 else []
            prev_session = [c for c in prev_day
                            if s.open_time <= c.time.time() <= s.close_time] if prev_day else []

            if not session:
                continue

            pools = build_liquidity_pools(premarket, prev_session, prev_day)

            # Compute HTF bias as of the open.
            open_dt = next((c.time for c in session
                            if c.time.time() >= s.open_time), session[0].time)
            history = [c for c in candles if c.time < open_dt]
            bias = htf_bias(history, open_dt,
                            swing_lookback=self.cfg.strategy.swing_lookback)

            # Arm each system for the new session.
            self.cat1.start_session(day, pools, bias)
            self.cat2.start_session(pools, bias)
            for or_model in self.cat4.values():
                or_model._reset()

            # Asia KZ: use prior day's 4 PM close.
            prev_4pm = (prev_day[-1].close if prev_day else
                        day_candles[0].open)
            self.cat5.start_session(prev_4pm_close=prev_4pm,
                                    htf_bias=bias)

            session_high = float("-inf")
            session_low = float("inf")
            trades_today: Dict[str, int] = {k: 0 for k in self.result.systems}
            full = [c for c in day_candles
                    if s.premarket_start <= c.time.time() <= s.close_time]

            for candle in full:
                session_high = max(session_high, candle.high)
                session_low = min(session_low, candle.low)

                # Manage open positions.
                for trade in self.broker.update(candle):
                    tag = trade.tags.get("system", "unknown")
                    if tag in self.result.systems:
                        self.result.systems[tag].trades.append(trade)

                max_t = self.cfg.risk.max_trades_per_day

                # Cat1: 9:30 Open Model.
                if trades_today["Cat1_930Open"] < max_t:
                    sig = self.cat1.on_candle(candle)
                    if sig:
                        self._submit(sig, "Cat1_930Open", trades_today)

                # Cat2: Setup For Life.
                if trades_today["Cat2_SetupForLife"] < max_t:
                    sig = self.cat2.on_candle(candle, session_high, session_low)
                    if sig:
                        self._submit(sig, "Cat2_SetupForLife", trades_today)

                # Cat4: Opening Ranges.
                for name, or_model in self.cat4.items():
                    sys_key = f"Cat4_OR_{name}"
                    if trades_today[sys_key] < max_t:
                        sig = or_model.on_candle(candle, bias)
                        if sig:
                            self._submit(sig, sys_key, trades_today)

                # Cat5: Asia KZ (only processes overnight bars).
                if trades_today["Cat5_AsiaKZ"] < max_t:
                    sig = self.cat5.on_candle(candle)
                    if sig:
                        self._submit(sig, "Cat5_AsiaKZ", trades_today)

            # Flatten all open trades at end of session.
            if full:
                for trade in self.broker.flatten(full[-1], ExitReason.SESSION_CLOSE):
                    tag = trade.tags.get("system", "unknown")
                    if tag in self.result.systems:
                        self.result.systems[tag].trades.append(trade)

        return self.result

    def _submit(self, sig: Signal, sys_key: str,
                trades_today: Dict[str, int]) -> None:
        size = position_size(sig, self.cfg.instrument, self.cfg.risk)
        if size > 0:
            trade = self.broker.submit(sig, size)
            trade.tags["system"] = sys_key
            trades_today[sys_key] += 1
