"""Unit tests for Cat 3: PD Arrays (Order Blocks, Rejection Blocks, Breakers)."""

import unittest
from datetime import datetime, timedelta

from ictbot.models import Candle, Direction
from ictbot.pd_arrays import (find_order_block, find_rejection_block,
                              scan_breaker_blocks, nearest_order_block)


def C(i, o, h, l, c):
    return Candle(datetime(2026, 1, 5, 9, 30) + timedelta(minutes=i), o, h, l, c)


class TestOrderBlock(unittest.TestCase):
    def _make_bullish_ob_series(self):
        # A down-close candle followed by an energetic up-displacement.
        return [
            C(0, 105, 106, 104, 105),   # prior context
            C(1, 105, 106, 103, 104),   # prior context
            C(2, 104, 105, 100, 101),   # prior
            C(3, 101, 102, 99, 100),    # prior
            C(4, 100, 101, 98, 99),     # prior - sets ATR reference
            C(5, 99, 99.5, 97, 98),     # THE down-close OB candidate
            C(6, 98, 120, 97, 118),     # displacement UP (large range, up-close)
        ]

    def test_bullish_ob_found(self):
        s = self._make_bullish_ob_series()
        ob = find_order_block(s, 6, atr_period=5, atr_mult=1.5)
        self.assertIsNotNone(ob)
        self.assertIs(ob.direction, Direction.LONG)
        # The OB is the last down-close candle before index 6 = C(5)
        # C5 = C(5, 99, 99.5, 97, 98): open=99, high=99.5, low=97, close=98
        # down-close: close(98) < open(99) ✓
        # body_high = max(99, 98) = 99, body_low = min(99, 98) = 98
        self.assertAlmostEqual(ob.top, 99.0)
        self.assertAlmostEqual(ob.bottom, 98.0)
        self.assertAlmostEqual(ob.mth, 98.5)

    def test_bearish_ob_found(self):
        # Last up-close candle before a down-displacement.
        s = [
            C(0, 95, 96, 94, 95), C(1, 95, 97, 94, 96),
            C(2, 96, 98, 95, 97), C(3, 97, 99, 96, 98),
            C(4, 98, 100, 97, 99),    # prior
            C(5, 100, 102, 99, 101),  # THE up-close OB candidate
            C(6, 101, 102, 80, 82),   # displacement DOWN
        ]
        ob = find_order_block(s, 6, atr_period=5, atr_mult=1.5)
        self.assertIsNotNone(ob)
        self.assertIs(ob.direction, Direction.SHORT)
        self.assertEqual(ob.top, 101)    # body_high = max(100, 101) = 101
        self.assertEqual(ob.bottom, 100) # body_low = min(100, 101) = 100
        self.assertAlmostEqual(ob.mth, 100.5)

    def test_no_ob_without_displacement(self):
        s = [C(i, 100, 101, 99, 100) for i in range(8)]
        ob = find_order_block(s, 7, atr_period=5, atr_mult=1.5)
        self.assertIsNone(ob)


class TestRejectionBlock(unittest.TestCase):
    def test_bullish_rb_long_lower_wick(self):
        # Down-close candle with big lower wick: open=100, high=100, low=90, close=99
        # lower_wick = body_low(99) - low(90) = 9; range = 10; ratio = 0.9 > 0.4
        c = C(0, 100, 100, 90, 99)
        rb = find_rejection_block(c, 0, min_wick_ratio=0.4)
        self.assertIsNotNone(rb)
        self.assertIs(rb.direction, Direction.LONG)
        self.assertEqual(rb.wick_tip, 90)        # the low (wick tip)
        self.assertEqual(rb.body_edge, 99)       # body_low = min(100,99) = 99
        self.assertAlmostEqual(rb.ce, 94.5)      # (90+99)/2

    def test_bearish_rb_long_upper_wick(self):
        # Up-close candle with big upper wick: open=100, high=110, low=100, close=101
        # upper_wick = high(110) - body_high(101) = 9; range = 10; ratio = 0.9 > 0.4
        c = C(0, 100, 110, 100, 101)
        rb = find_rejection_block(c, 0, min_wick_ratio=0.4)
        self.assertIsNotNone(rb)
        self.assertIs(rb.direction, Direction.SHORT)
        self.assertEqual(rb.wick_tip, 110)
        self.assertEqual(rb.body_edge, 101)
        self.assertAlmostEqual(rb.ce, 105.5)

    def test_no_rb_small_wick(self):
        # Balanced candle: no dominant wick.
        c = C(0, 100, 102, 98, 101)
        rb = find_rejection_block(c, 0, min_wick_ratio=0.4)
        self.assertIsNone(rb)


class TestBreakerBlock(unittest.TestCase):
    def test_bullish_breaker_detected(self):
        # Pattern: bearish displacement → bearish OB → price closes above OB → Breaker
        # C0-C5: context for ATR; C5=up-close (OB candidate from below),
        # C6=displacement down (bearish disp, so we need a SHORT OB for a LONG breaker)
        # Then C7 closes above the OB → bullish breaker
        series = [
            C(0, 110, 111, 109, 110),
            C(1, 110, 112, 109, 111),
            C(2, 111, 113, 110, 112),
            C(3, 112, 113, 111, 112.5),
            C(4, 112, 114, 111, 113),    # prior context
            C(5, 113, 115, 112, 114),    # up-close → potential bearish OB
            C(6, 114, 115, 94, 96),      # bearish displacement → creates SHORT OB at C5
            C(7, 96, 120, 95, 118),      # closes ABOVE C5 OB top (115) → LONG breaker
        ]
        bbs = scan_breaker_blocks(series, Direction.LONG, atr_period=5, atr_mult=1.5)
        self.assertGreater(len(bbs), 0)
        bb = bbs[0]
        self.assertIs(bb.direction, Direction.LONG)
        self.assertGreater(bb.top, bb.bottom)
        self.assertAlmostEqual(bb.ce, (bb.top + bb.bottom) / 2.0)


class TestOpeningRangeModel(unittest.TestCase):
    def test_opening_range_properties(self):
        from ictbot.models import OpeningRange, Direction
        or_ = OpeningRange("AM", 100.0, 110.0, 90.0, datetime(2026, 1, 5, 10, 0))
        self.assertAlmostEqual(or_.midpoint, 100.0)
        self.assertAlmostEqual(or_.range, 20.0)
        self.assertAlmostEqual(or_.upper_quadrant, 105.0)
        self.assertAlmostEqual(or_.lower_quadrant, 95.0)
        # SD targets
        self.assertAlmostEqual(or_.sd_target(Direction.LONG, 1.0), 130.0)   # 110+20
        self.assertAlmostEqual(or_.sd_target(Direction.SHORT, 0.5), 80.0)   # 90-10
        self.assertAlmostEqual(or_.sd_target(Direction.SHORT, 1.5), 60.0)   # 90-30

    def test_normalized_level(self):
        from ictbot.models import OpeningRange
        or_ = OpeningRange("AM", 100.0, 110.0, 90.0, datetime(2026, 1, 5, 10, 0))
        self.assertAlmostEqual(or_.normalized_level(90.0), 0.0)
        self.assertAlmostEqual(or_.normalized_level(100.0), 0.5)
        self.assertAlmostEqual(or_.normalized_level(110.0), 1.0)
        self.assertAlmostEqual(or_.normalized_level(120.0), 1.5)  # SD extension


if __name__ == "__main__":
    unittest.main()
