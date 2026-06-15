"""Unit tests for liquidity pools, sweeps and target selection."""

import unittest
from datetime import datetime, timedelta

from ictbot.liquidity import (build_liquidity_pools, detect_sweep,
                              nearest_target_pool)
from ictbot.models import Candle, Direction, LiquidityPool


def C(i, o, h, l, c):
    return Candle(datetime(2026, 1, 5, 9, 30) + timedelta(minutes=i), o, h, l, c)


class TestSweep(unittest.TestCase):
    def test_sellside_sweep(self):
        pool = LiquidityPool("PML", 100.0, Direction.SHORT)
        self.assertTrue(detect_sweep(C(0, 101, 102, 99, 101), pool))   # low < 100
        self.assertFalse(detect_sweep(C(0, 101, 102, 100.5, 101), pool))

    def test_buyside_sweep(self):
        pool = LiquidityPool("PMH", 100.0, Direction.LONG)
        self.assertTrue(detect_sweep(C(0, 99, 101, 98, 99), pool))     # high > 100
        self.assertFalse(detect_sweep(C(0, 99, 99.5, 98, 99), pool))


class TestPools(unittest.TestCase):
    def test_build_pools(self):
        pm = [C(0, 100, 110, 90, 105)]
        prev_session = [C(0, 100, 120, 80, 110)]
        prev_day = [C(0, 100, 130, 70, 110)]
        pools = build_liquidity_pools(pm, prev_session, prev_day)
        names = {p.name: p.price for p in pools}
        self.assertEqual(names["PMH"], 110)
        self.assertEqual(names["PML"], 90)
        self.assertEqual(names["PSH"], 120)
        self.assertEqual(names["PDL"], 70)


class TestTargetSelection(unittest.TestCase):
    def test_nearest_far_enough(self):
        pools = [
            LiquidityPool("PMH", 105, Direction.LONG),   # only 5 away
            LiquidityPool("PSH", 130, Direction.LONG),   # 30 away
            LiquidityPool("PDH", 160, Direction.LONG),   # 60 away
        ]
        # With a 20-point minimum, PMH is skipped and PSH (nearest >= 20) chosen.
        target = nearest_target_pool(pools, Direction.LONG, 100, min_distance=20)
        self.assertEqual(target.name, "PSH")

    def test_fallback_to_furthest(self):
        pools = [LiquidityPool("PMH", 105, Direction.LONG)]
        # Nothing is far enough -> fall back to the furthest available.
        target = nearest_target_pool(pools, Direction.LONG, 100, min_distance=50)
        self.assertEqual(target.name, "PMH")

    def test_no_pool_on_correct_side(self):
        pools = [LiquidityPool("PML", 90, Direction.SHORT)]
        self.assertIsNone(nearest_target_pool(pools, Direction.LONG, 100))


if __name__ == "__main__":
    unittest.main()
