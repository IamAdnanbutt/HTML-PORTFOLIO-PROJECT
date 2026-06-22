"""Event-driven backtester and performance metrics.

The engine walks the candle stream in time order, day by day. For each session
it builds the liquidity pools from prior data, computes the HTF bias, then
feeds bars to :class:`NineThirtyOpenModel`. Signals are filled by the
:class:`PaperBroker`; open positions are flattened at the session close.

No bar is ever revealed to the strategy before it has 'closed', so the results
contain no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from .bias import htf_bias
from .broker import PaperBroker, position_size
from .config import Config
from .liquidity import build_liquidity_pools, group_by_day
from .models import Candle, ExitReason, Trade
from .strategy import NineThirtyOpenModel


def _slice_session(day_candles: List[Candle], cfg: Config) -> List[Candle]:
    """Candles from pre-market start through the close, for one day."""
    s = cfg.session
    return [c for c in day_candles
            if s.premarket_start <= c.time.time() <= s.close_time]


def _premarket(day_candles: List[Candle], cfg: Config) -> List[Candle]:
    s = cfg.session
    return [c for c in day_candles
            if s.premarket_start <= c.time.time() < s.open_time]


def _regular_session(day_candles: List[Candle], cfg: Config) -> List[Candle]:
    s = cfg.session
    return [c for c in day_candles
            if s.open_time <= c.time.time() <= s.close_time]


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[tuple] = field(default_factory=list)  # (datetime, equity)
    config: Optional[Config] = None

    # --------------------------------------------------------------- metrics
    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> List[Trade]:
        return [t for t in self.trades if t.r_multiple > 0]

    @property
    def losses(self) -> List[Trade]:
        return [t for t in self.trades if t.r_multiple <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / self.n_trades if self.n_trades else 0.0

    @property
    def total_r(self) -> float:
        return sum(t.r_multiple for t in self.trades)

    @property
    def expectancy_r(self) -> float:
        return self.total_r / self.n_trades if self.n_trades else 0.0

    def net_pnl(self) -> float:
        inst = self.config.instrument
        total = 0.0
        for t in self.trades:
            total += t.pnl_points * inst.point_value * t.size - t.fees
        return total

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.r_multiple for t in self.wins)
        gross_loss = abs(sum(t.r_multiple for t in self.losses))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def max_drawdown_r(self) -> float:
        peak = 0.0
        cum = 0.0
        max_dd = 0.0
        for t in self.trades:
            cum += t.r_multiple
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        return max_dd

    def summary(self) -> str:
        lines = [
            "=" * 56,
            " ICT 9:30 Open Model - Backtest Summary",
            "=" * 56,
            f" Instrument        : {self.config.instrument.symbol}",
            f" Trades            : {self.n_trades}",
            f" Win rate          : {self.win_rate * 100:5.1f}%",
            f" Total R           : {self.total_r:+.2f}R",
            f" Expectancy        : {self.expectancy_r:+.2f}R / trade",
            f" Profit factor     : {self.profit_factor:.2f}",
            f" Max drawdown      : {self.max_drawdown_r:.2f}R",
            f" Net P&L           : ${self.net_pnl():,.2f}",
            "=" * 56,
        ]
        # Per-model breakdown (Reversal vs Continuation).
        models: Dict[str, List[Trade]] = {}
        for t in self.trades:
            models.setdefault(t.signal.model, []).append(t)
        for name, ts in sorted(models.items()):
            r = sum(t.r_multiple for t in ts)
            wr = len([t for t in ts if t.r_multiple > 0]) / len(ts) * 100
            lines.append(f" {name:<16}: {len(ts):2d} trades  "
                         f"{wr:4.0f}% WR  {r:+.2f}R")
        lines.append("=" * 56)
        return "\n".join(lines)


class Backtester:
    def __init__(self, config: Config):
        self.cfg = config

    def run(self, candles: List[Candle]) -> BacktestResult:
        candles = sorted(candles, key=lambda c: c.time)
        by_day = group_by_day(candles)
        days: List[date] = sorted(by_day)

        model = NineThirtyOpenModel(self.cfg)
        broker = PaperBroker(self.cfg.instrument)
        result = BacktestResult(config=self.cfg)

        equity = self.cfg.risk.account_size
        result.equity_curve.append((days[0] if days else datetime.now(), equity))

        for i, day in enumerate(days):
            day_candles = by_day[day]
            session = _regular_session(day_candles, self.cfg)
            if not session:
                continue

            # Liquidity context from prior days.
            prev_day = by_day[days[i - 1]] if i >= 1 else []
            prev_session = _regular_session(prev_day, self.cfg) if prev_day else []
            premarket = _premarket(day_candles, self.cfg)
            pools = build_liquidity_pools(premarket, prev_session, prev_day)

            # HTF bias as of the open, using all history up to today's open.
            open_dt = self._open_datetime(day, session)
            history = [c for c in candles if c.time < open_dt]
            bias = htf_bias(history, open_dt,
                            swing_lookback=self.cfg.strategy.swing_lookback)

            model.start_session(day, pools, bias)
            trades_today = 0

            full = _slice_session(day_candles, self.cfg)
            for candle in full:
                # 1) manage any open position on this bar
                for closed in broker.update(candle):
                    equity += self._trade_pnl(closed)
                    result.trades.append(closed)
                    result.equity_curve.append((candle.time, equity))

                # 2) ask the strategy for a new signal
                if trades_today < self.cfg.risk.max_trades_per_day:
                    signal = model.on_candle(candle)
                    if signal is not None:
                        size = position_size(signal, self.cfg.instrument,
                                             self.cfg.risk)
                        if size > 0:
                            broker.submit(signal, size)
                            trades_today += 1

            # Flatten anything still open at the close.
            if full:
                for closed in broker.flatten(full[-1], ExitReason.SESSION_CLOSE):
                    equity += self._trade_pnl(closed)
                    result.trades.append(closed)
                    result.equity_curve.append((full[-1].time, equity))

        return result

    def _open_datetime(self, day: date, session: List[Candle]) -> datetime:
        for c in session:
            if c.time.time() >= self.cfg.session.open_time:
                return c.time
        return session[0].time

    def _trade_pnl(self, trade: Trade) -> float:
        inst = self.cfg.instrument
        return trade.pnl_points * inst.point_value * trade.size - trade.fees
