"""Offline tests for the forward-tester's pure processing core (no network)."""

import unittest
from datetime import datetime, timedelta

from ictbot.forwardtest import ForwardTester
from ictbot.models import Candle


def _candles(n, start_min):
    base = datetime(2026, 6, 18, 9, 30)
    return [Candle(base + timedelta(minutes=start_min + i),
                   100.0, 101.0, 99.0, 100.0, 1000.0) for i in range(n)]


class TestIncrementalFeed(unittest.TestCase):
    def test_only_new_bars_are_fed(self):
        ft = ForwardTester(client=None, top_k=3)
        bars = _candles(5, 0)                       # 09:30..09:34

        ft.process(datetime(2026, 6, 18, 9, 35), {"QQQ": bars})
        self.assertEqual(len(ft._runners["QQQ"].history), 5)

        # Re-fetching the same window must not replay bars already seen.
        ft.process(datetime(2026, 6, 18, 9, 36), {"QQQ": bars})
        self.assertEqual(len(ft._runners["QQQ"].history), 5)

        # A genuinely new bar is fed exactly once.
        ft.process(datetime(2026, 6, 18, 9, 37), {"QQQ": bars + _candles(1, 5)})
        self.assertEqual(len(ft._runners["QQQ"].history), 6)

    def test_universe_header(self):
        ft = ForwardTester(client=None)
        out = ft.process(datetime(2026, 6, 18, 9, 35), {"QQQ": _candles(3, 0)})
        self.assertTrue(out[0].startswith("[2026-06-18 09:35]"))
        self.assertIn("equity", out[0])
        self.assertIn("QQQ", out[0])


if __name__ == "__main__":
    unittest.main()
