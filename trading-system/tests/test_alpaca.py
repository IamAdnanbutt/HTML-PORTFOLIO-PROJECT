"""Unit tests for the pure Alpaca <-> ictbot bridges (no network)."""

import unittest
from datetime import datetime

from ictbot.alpaca import (bars_to_candles, bracket_payload, screener_symbols,
                           stats_from_bars)
from ictbot.models import Direction, Signal
from ictbot.universe import EQUITY


class TestBarParsing(unittest.TestCase):
    def test_utc_bars_convert_to_eastern(self):
        # 13:30Z in June is 09:30 EDT — the RTH open.
        bars = [{"t": "2026-06-18T13:30:00Z", "o": 100, "h": 101,
                 "l": 99.5, "c": 100.5, "v": 1200}]
        candles = bars_to_candles(bars)
        self.assertEqual(len(candles), 1)
        c = candles[0]
        self.assertEqual(c.time, datetime(2026, 6, 18, 9, 30))
        self.assertEqual((c.open, c.high, c.low, c.close, c.volume),
                         (100.0, 101.0, 99.5, 100.5, 1200.0))

    def test_nanosecond_fraction_is_tolerated(self):
        bars = [{"t": "2026-06-18T13:30:00.123456789Z",
                 "o": 1, "h": 2, "l": 1, "c": 2, "v": 5}]
        # Should parse without error and still land on the ET 09:30 minute.
        self.assertEqual(bars_to_candles(bars)[0].time.hour, 9)


class TestScreener(unittest.TestCase):
    def test_extracts_symbols(self):
        payload = {"most_actives": [{"symbol": "AAPL", "volume": 9},
                                    {"symbol": "TSLA", "volume": 8},
                                    {"volume": 7}]}   # malformed row dropped
        self.assertEqual(screener_symbols(payload), ["AAPL", "TSLA"])

    def test_empty(self):
        self.assertEqual(screener_symbols({}), [])


class TestStatsFromBars(unittest.TestCase):
    def test_dollar_volume_and_atr_pct(self):
        bars = [{"t": f"2026-06-18T13:{30+i:02d}:00Z", "o": 100, "h": 102,
                 "l": 98, "c": 100, "v": 1000} for i in range(20)]
        candles = bars_to_candles(bars)
        st = stats_from_bars("QQQ", candles, EQUITY, atr_period=14)
        self.assertEqual(st.symbol, "QQQ")
        self.assertEqual(st.asset_class, EQUITY)
        # 20 bars * (100 close * 1000 vol) = 2,000,000 dollar volume
        self.assertAlmostEqual(st.dollar_volume, 2_000_000.0)
        self.assertGreater(st.atr_pct, 0.0)      # range = 4 on a 100 close

    def test_empty_returns_none(self):
        self.assertIsNone(stats_from_bars("QQQ", [], EQUITY))


class TestBracketPayload(unittest.TestCase):
    def test_long_is_buy_with_oco_exits(self):
        sig = Signal(time=datetime(2026, 6, 18, 9, 50), direction=Direction.LONG,
                     entry=100.0, stop=98.0, target=104.0, model="OR_AM")
        p = bracket_payload(sig, 250, "QQQ")
        self.assertEqual(p["side"], "buy")
        self.assertEqual(p["order_class"], "bracket")
        self.assertEqual(p["qty"], "250")
        self.assertEqual(p["limit_price"], "100.0")
        self.assertEqual(p["take_profit"]["limit_price"], "104.0")
        self.assertEqual(p["stop_loss"]["stop_price"], "98.0")

    def test_short_is_sell(self):
        sig = Signal(time=datetime(2026, 6, 18, 9, 50), direction=Direction.SHORT,
                     entry=100.0, stop=102.0, target=96.0, model="OR_AM")
        self.assertEqual(bracket_payload(sig, 10, "SPY")["side"], "sell")


if __name__ == "__main__":
    unittest.main()
