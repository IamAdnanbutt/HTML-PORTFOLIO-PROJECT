"""Tests for the 9:30 Open Model state machine and the broker."""

import unittest
from datetime import datetime, time, timedelta

from ictbot.broker import PaperBroker, position_size
from ictbot.config import Config
from ictbot.models import (Candle, Direction, ExitReason, LiquidityPool,
                           Signal)
from ictbot.strategy import NineThirtyOpenModel


def C(minute, o, h, l, c):
    return Candle(datetime(2026, 1, 5, 9, 30) + timedelta(minutes=minute),
                  o, h, l, c)


class TestStrategySignal(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.for_symbol("NQ")
        # Widen the macro window so the constructed entry triggers immediately.
        self.cfg.session.macro_windows = [(time(9, 30), time(11, 0))]

    def test_long_reversal_signal(self):
        pools = [
            LiquidityPool("PML", 99.5, Direction.SHORT),   # to be swept
            LiquidityPool("PMH", 101.0, Direction.LONG),   # below entry, ignored
            LiquidityPool("PSH", 112.0, Direction.LONG),   # the draw / target
        ]
        model = NineThirtyOpenModel(self.cfg)
        model.start_session(datetime(2026, 1, 5).date(), pools, Direction.LONG)

        candles = [
            C(0, 100.0, 100.5, 99.0, 99.5),     # 09:30 raid on PML (manipulation)
            C(1, 99.5, 104.0, 99.4, 103.8),     # displacement up
            C(2, 103.8, 106.0, 103.6, 105.8),   # completes bullish FVG 100.5..103.6
            C(3, 105.8, 106.0, 101.9, 103.0),   # retrace into FVG -> entry
        ]
        signal = None
        for c in candles:
            out = model.on_candle(c)
            if out is not None:
                signal = out

        self.assertIsNotNone(signal, "expected a long signal")
        self.assertIs(signal.direction, Direction.LONG)
        self.assertEqual(signal.model, "Reversal")
        self.assertLess(signal.stop, signal.entry)
        self.assertGreater(signal.target, signal.entry)
        self.assertGreaterEqual(signal.rr, self.cfg.risk.min_rr)
        self.assertEqual(signal.target_pool.name, "PSH")
        # stop sits below the manipulation low (99) minus the tick buffer
        self.assertLess(signal.stop, 99.0)

    def test_no_signal_without_bias(self):
        model = NineThirtyOpenModel(self.cfg)
        model.start_session(datetime(2026, 1, 5).date(), [], None)
        self.assertIsNone(model.on_candle(C(0, 100, 101, 99, 100)))


class TestBroker(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.for_symbol("NQ")

    def test_position_size_respects_risk(self):
        sig = Signal(time=datetime(2026, 1, 5, 9, 50), direction=Direction.LONG,
                     entry=18000.0, stop=17980.0, target=18060.0, model="Reversal")
        # risk budget = 100k * 0.5% = $500; stop = 20 pts * $20 = $400/contract
        size = position_size(sig, self.cfg.instrument, self.cfg.risk)
        self.assertEqual(size, 1)

    def test_position_size_equity_shares(self):
        cfg = Config.for_symbol("QQQ")            # equity: $1/share, $0.01 tick
        sig = Signal(time=datetime(2026, 1, 5, 9, 50), direction=Direction.LONG,
                     entry=100.0, stop=98.0, target=104.0, model="OR_AM")
        # $500 budget / ($2 stop * $1/share) = 250 shares (under both caps)
        self.assertEqual(position_size(sig, cfg.instrument, cfg.risk), 250)

    def test_position_size_crypto_fractional(self):
        cfg = Config.for_symbol("BTC/USD")        # crypto: fractional units
        sig = Signal(time=datetime(2026, 1, 5, 2, 50), direction=Direction.LONG,
                     entry=60000.0, stop=59000.0, target=63000.0, model="OR_Midnight")
        # $500 budget / ($1000 stop * $1/unit) = 0.5 BTC (notional cap 1.66 BTC)
        self.assertAlmostEqual(position_size(sig, cfg.instrument, cfg.risk), 0.5)

    def test_long_hits_target(self):
        broker = PaperBroker(self.cfg.instrument)
        sig = Signal(time=datetime(2026, 1, 5, 9, 30), direction=Direction.LONG,
                     entry=100.0, stop=95.0, target=110.0, model="Reversal")
        broker.submit(sig, 1)
        closed = broker.update(C(1, 101, 111, 100, 109))  # high 111 >= target
        self.assertEqual(len(closed), 1)
        self.assertIs(closed[0].exit_reason, ExitReason.TARGET)
        self.assertAlmostEqual(closed[0].r_multiple, 2.0, places=6)

    def test_short_hits_stop_first_when_ambiguous(self):
        broker = PaperBroker(self.cfg.instrument)
        sig = Signal(time=datetime(2026, 1, 5, 9, 30), direction=Direction.SHORT,
                     entry=100.0, stop=105.0, target=90.0, model="Reversal")
        broker.submit(sig, 1)
        # bar straddles both stop and target -> stop assumed first (conservative)
        closed = broker.update(C(1, 100, 106, 89, 95))
        self.assertEqual(len(closed), 1)
        self.assertIs(closed[0].exit_reason, ExitReason.STOP)


class TestBacktestSmoke(unittest.TestCase):
    def test_end_to_end_produces_valid_trades(self):
        from ictbot.backtest import Backtester
        from ictbot.data import generate_sessions
        cfg = Config.for_symbol("NQ")
        candles = generate_sessions(60, seed=7, tick=cfg.instrument.tick_size)
        result = Backtester(cfg).run(candles)
        self.assertGreater(result.n_trades, 0)
        for t in result.trades:
            sig = t.signal
            if sig.direction is Direction.LONG:
                self.assertLess(sig.stop, sig.entry)
                self.assertGreater(sig.target, sig.entry)
            else:
                self.assertGreater(sig.stop, sig.entry)
                self.assertLess(sig.target, sig.entry)
            self.assertGreaterEqual(sig.rr, cfg.risk.min_rr - 1e-9)


if __name__ == "__main__":
    unittest.main()
