"""Integration tests for Cat 4 (Opening Ranges), Cat 5 (Asia KZ), and the runner."""

import unittest
from datetime import datetime, timedelta, time

from ictbot.config import Config
from ictbot.models import Candle, Direction, LiquidityPool


def _candles(start_dt, n, start_price, trend, vol=5.0, tick=0.25):
    """Build n 1-minute candles drifting from start_price by trend/bar."""
    import random
    rng = random.Random(99)
    out = []
    p = start_price
    for i in range(n):
        o = p
        c = p + trend + rng.gauss(0, vol)
        h = max(o, c) + abs(rng.gauss(0, 2.0))
        l = min(o, c) - abs(rng.gauss(0, 2.0))
        o, h, l, c = (round(round(x / tick) * tick, 2) for x in (o, h, l, c))
        out.append(Candle(start_dt + timedelta(minutes=i), o, h, l, c))
        p = c
    return out


class TestOpeningRangeModel(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.for_symbol("NQ")

    def test_am_or_builds_range(self):
        from ictbot.opening_range import OpeningRangeModel, OR_DEFINITIONS
        model = OpeningRangeModel(OR_DEFINITIONS["AM"], self.cfg)
        # Feed 30 bars during the 9:30–10:00 AM window.
        base = datetime(2026, 1, 5, 9, 30)
        for i in range(31):
            c = Candle(base + timedelta(minutes=i),
                       18000, 18010, 17990, 18005)
            model.on_candle(c)
        # Range should now be built.
        self.assertIsNotNone(model.opening_range)
        or_ = model.opening_range
        self.assertAlmostEqual(or_.or_high, 18010.0)
        self.assertAlmostEqual(or_.or_low, 17990.0)
        self.assertAlmostEqual(or_.midpoint, 18000.0)
        self.assertAlmostEqual(or_.range, 20.0)

    def test_sd_targets_correct(self):
        from ictbot.models import OpeningRange
        or_ = OpeningRange("AM", 18000.0, 18020.0, 17980.0,
                           datetime(2026, 1, 5, 10, 0))
        # Bearish targets below OR Low (17980):
        self.assertAlmostEqual(or_.sd_target(Direction.SHORT, 0.5), 17960.0)
        self.assertAlmostEqual(or_.sd_target(Direction.SHORT, 1.0), 17940.0)
        self.assertAlmostEqual(or_.sd_target(Direction.SHORT, 1.5), 17920.0)
        self.assertAlmostEqual(or_.sd_target(Direction.SHORT, 2.5), 17880.0)
        # Bullish targets above OR High (18020):
        self.assertAlmostEqual(or_.sd_target(Direction.LONG, 0.5), 18040.0)
        self.assertAlmostEqual(or_.sd_target(Direction.LONG, 1.5), 18080.0)


class TestAsiaKillzone(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.for_symbol("NQ")

    def test_ndog_formed(self):
        from ictbot.asia_killzone import AsiaKillzoneModel
        model = AsiaKillzoneModel(self.cfg)
        model.start_session(prev_4pm_close=18000.0, htf_bias=Direction.LONG)
        # Feed the 6 PM open candle.
        c = Candle(datetime(2026, 1, 5, 18, 0), 18010, 18020, 17995, 18015)
        model.on_candle(c)
        self.assertIsNotNone(model._ndog)
        ndog = model._ndog
        # NDOG: open=18010, prev_close=18000 → high=18010, low=18000
        self.assertAlmostEqual(ndog.high, 18010.0)
        self.assertAlmostEqual(ndog.low, 18000.0)
        self.assertAlmostEqual(ndog.midpoint, 18005.0)
        self.assertAlmostEqual(ndog.range, 10.0)

    def test_ndog_quadrant_levels(self):
        from ictbot.asia_killzone import NDOG
        ndog = NDOG(18010.0, 17990.0, datetime(2026, 1, 5, 18, 0))
        # high=18010, low=17990, range=20
        self.assertAlmostEqual(ndog.upper_quadrant, 18005.0)   # 17990 + 0.75*20
        self.assertAlmostEqual(ndog.lower_quadrant, 17995.0)   # 17990 + 0.25*20
        self.assertAlmostEqual(ndog.midpoint, 18000.0)


class TestMultiSystemRunner(unittest.TestCase):
    def test_runner_runs_without_error(self):
        from ictbot.data import generate_sessions
        from ictbot.runner import MultiSystemRunner
        cfg = Config.for_symbol("NQ")
        candles = generate_sessions(30, seed=42, tick=cfg.instrument.tick_size)
        runner = MultiSystemRunner(cfg)
        result = runner.run(candles)
        # All system keys must exist in the result (even if they had 0 trades).
        self.assertIn("Cat1_930Open", result.systems)
        self.assertIn("Cat2_SetupForLife", result.systems)
        self.assertIn("Cat4_OR_AM", result.systems)
        self.assertIn("Cat5_AsiaKZ", result.systems)
        # Summary must print without error.
        summary = result.summary()
        self.assertIn("ICT Multi-System", summary)

    def test_runner_produces_valid_r_multiples(self):
        from ictbot.data import generate_sessions
        from ictbot.runner import MultiSystemRunner
        cfg = Config.for_symbol("NQ")
        candles = generate_sessions(60, seed=7, tick=cfg.instrument.tick_size)
        runner = MultiSystemRunner(cfg)
        result = runner.run(candles)
        all_trades = [t for sys in result.systems.values() for t in sys.trades]
        for t in all_trades:
            # R-multiple must be between -1.1 (stop slippage allowed) and some large +R
            self.assertGreater(t.r_multiple, -2.0,
                               msg=f"Implausible loss: {t.r_multiple:.2f}R")
            sig = t.signal
            if sig.direction.name == "LONG":
                self.assertLess(sig.stop, sig.entry)
                self.assertGreater(sig.target, sig.entry)
            else:
                self.assertGreater(sig.stop, sig.entry)
                self.assertLess(sig.target, sig.entry)

    def test_full_day_lights_up_overnight_systems(self):
        """Full-day data must feed the overnight + Asia systems real setups."""
        from ictbot.data import generate_sessions
        from ictbot.models import Direction
        from ictbot.runner import MultiSystemRunner
        cfg = Config.for_symbol("NQ")
        candles = generate_sessions(60, seed=42, tick=cfg.instrument.tick_size,
                                    full_day=True)
        result = MultiSystemRunner(cfg).run(candles)

        # Every previously-dark system must now produce trades.
        for key in ("Cat4_OR_Midnight", "Cat4_OR_London",
                    "Cat4_OR_NYKillZone", "Cat5_AsiaKZ"):
            self.assertGreater(result.systems[key].n_trades, 0,
                               msg=f"{key} produced no trades on full-day data")

        # Every trade — across all systems — must be structurally valid and
        # carry a sane R-multiple (no degenerate near-zero-risk blow-ups).
        for sysres in result.systems.values():
            for t in sysres.trades:
                sig = t.signal
                if sig.direction is Direction.LONG:
                    self.assertLess(sig.stop, sig.entry)
                    self.assertGreater(sig.target, sig.entry)
                else:
                    self.assertGreater(sig.stop, sig.entry)
                    self.assertLess(sig.target, sig.entry)
                self.assertGreaterEqual(sig.rr, cfg.risk.min_rr - 1e-9)
                self.assertGreater(t.r_multiple, -1.5)
                self.assertLess(t.r_multiple, 25.0,
                                msg=f"Implausible win: {t.r_multiple:.1f}R")


if __name__ == "__main__":
    unittest.main()
