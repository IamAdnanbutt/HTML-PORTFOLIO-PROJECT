"""Unit tests for the price-action primitives."""

import unittest
from datetime import datetime, timedelta

from ictbot.indicators import atr, find_fvg, swing_highs, swing_lows
from ictbot.models import Candle, Direction


def C(i, o, h, l, c):
    return Candle(datetime(2026, 1, 5, 9, 30) + timedelta(minutes=i), o, h, l, c)


class TestFVG(unittest.TestCase):
    def test_bullish_fvg_detected(self):
        # candle3.low (105) > candle1.high (101) -> bullish gap 101..105
        candles = [C(0, 99, 101, 98, 100),
                   C(1, 100, 110, 100, 109),   # displacement up
                   C(2, 106, 112, 105, 111)]
        fvg = find_fvg(candles, 2)
        self.assertIsNotNone(fvg)
        self.assertIs(fvg.direction, Direction.LONG)
        self.assertEqual(fvg.bottom, 101)
        self.assertEqual(fvg.top, 105)
        self.assertEqual(fvg.mid, 103)

    def test_bearish_fvg_detected(self):
        candles = [C(0, 110, 112, 109, 110),
                   C(1, 110, 110, 100, 101),   # displacement down
                   C(2, 100, 104, 98, 99)]     # high (104) < candle1.low (109)
        fvg = find_fvg(candles, 2)
        self.assertIsNotNone(fvg)
        self.assertIs(fvg.direction, Direction.SHORT)
        self.assertEqual(fvg.top, 109)
        self.assertEqual(fvg.bottom, 104)

    def test_no_fvg_when_overlap(self):
        candles = [C(0, 100, 105, 99, 102),
                   C(1, 102, 106, 101, 104),
                   C(2, 104, 107, 103, 105)]   # low 103 < candle1.high 105
        self.assertIsNone(find_fvg(candles, 2))

    def test_contains_and_size(self):
        candles = [C(0, 99, 101, 98, 100),
                   C(1, 100, 110, 100, 109),
                   C(2, 106, 112, 105, 111)]
        fvg = find_fvg(candles, 2)
        self.assertTrue(fvg.contains(103))
        self.assertFalse(fvg.contains(120))
        self.assertEqual(fvg.size, 4)


class TestATR(unittest.TestCase):
    def test_atr_constant_range(self):
        # each bar has a true range of 10
        candles = [C(i, 100, 105, 95, 100) for i in range(20)]
        self.assertAlmostEqual(atr(candles, 14), 10.0, places=6)

    def test_atr_insufficient_data(self):
        self.assertEqual(atr([C(0, 1, 2, 0, 1)], 14), 0.0)


class TestSwings(unittest.TestCase):
    def test_swing_high_low(self):
        # A clean zig-zag: peak at index 3, trough at index 6.
        highs = [100, 102, 104, 106, 104, 102, 100, 102, 104, 106, 104, 102, 100]
        lows = [h - 3 for h in highs]
        candles = [C(i, highs[i], highs[i], lows[i], highs[i])
                   for i in range(len(highs))]
        self.assertIn(3, swing_highs(candles, 2))   # the 106 peak
        self.assertIn(6, swing_lows(candles, 2))    # the 97 trough


if __name__ == "__main__":
    unittest.main()
