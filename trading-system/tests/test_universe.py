"""Tests for the dynamic forward-test universe selector."""

import unittest
from datetime import datetime, time

from ictbot.universe import (CRYPTO, EQUITY, SymbolStats, active_asset_class,
                             rank_universe, score_candidates, select_universe)


def S(sym, ac, vol, atr):
    return SymbolStats(sym, ac, vol, atr)


class TestAssetRouting(unittest.TestCase):
    def test_equities_during_rth(self):
        self.assertEqual(active_asset_class(time(9, 30)), EQUITY)
        self.assertEqual(active_asset_class(time(12, 0)), EQUITY)
        self.assertEqual(active_asset_class(time(16, 0)), EQUITY)

    def test_crypto_outside_rth(self):
        self.assertEqual(active_asset_class(time(16, 1)), CRYPTO)  # just after close
        self.assertEqual(active_asset_class(time(2, 50)), CRYPTO)  # Midnight macro
        self.assertEqual(active_asset_class(time(20, 50)), CRYPTO)  # Asia delivery
        self.assertEqual(active_asset_class(time(9, 29)), CRYPTO)  # just before open


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.cands = [
            S("A", EQUITY, 1_000_000_000, 0.05),   # high volume + high vol
            S("B", EQUITY, 100_000_000, 0.01),     # low volume + low vol
            S("C", EQUITY, 500_000_000, 0.03),     # middling on both
            S("D", CRYPTO, 2_000_000_000, 0.06),   # other asset class
        ]

    def test_combined_score_orders_equities(self):
        ranked = rank_universe(self.cands, EQUITY, top_k=3)
        self.assertEqual(ranked, ["A", "C", "B"])

    def test_top_k_truncates(self):
        self.assertEqual(rank_universe(self.cands, EQUITY, top_k=1), ["A"])

    def test_asset_class_filter(self):
        # Only the crypto name is eligible when ranking crypto.
        self.assertEqual(rank_universe(self.cands, CRYPTO, top_k=5), ["D"])

    def test_liquidity_floor_drops_illiquid(self):
        ranked = rank_universe(self.cands, EQUITY, top_k=5,
                               min_dollar_volume=200_000_000)
        self.assertNotIn("B", ranked)          # 100M < 200M floor
        self.assertEqual(ranked, ["A", "C"])

    def test_volatility_floor_drops_quiet(self):
        ranked = rank_universe(self.cands, EQUITY, top_k=5, min_atr_pct=0.02)
        self.assertNotIn("B", ranked)          # 0.01 ATR% < 0.02 floor

    def test_vol_weight_pure_volatility(self):
        # An illiquid-but-volatile name can lead when weight is all volatility.
        cands = [S("X", EQUITY, 10, 0.20), S("Y", EQUITY, 1_000_000_000, 0.02)]
        self.assertEqual(rank_universe(cands, EQUITY, top_k=1, vol_weight=0.0),
                         ["X"])
        self.assertEqual(rank_universe(cands, EQUITY, top_k=1, vol_weight=1.0),
                         ["Y"])

    def test_empty_candidates(self):
        self.assertEqual(score_candidates([]), [])
        self.assertEqual(rank_universe([], EQUITY), [])


class TestSelectUniverse(unittest.TestCase):
    def setUp(self):
        self.cands = [
            S("QQQ", EQUITY, 1_000_000_000, 0.02),
            S("TQQQ", EQUITY, 800_000_000, 0.05),
            S("BTC/USD", CRYPTO, 3_000_000_000, 0.04),
            S("ETH/USD", CRYPTO, 1_500_000_000, 0.05),
        ]

    def test_rth_selects_equities(self):
        picks = select_universe(self.cands, datetime(2026, 6, 18, 10, 30),
                                top_k=5)
        self.assertCountEqual(picks, ["QQQ", "TQQQ"])

    def test_overnight_selects_crypto(self):
        picks = select_universe(self.cands, datetime(2026, 6, 18, 2, 50),
                                top_k=5)
        self.assertCountEqual(picks, ["BTC/USD", "ETH/USD"])


if __name__ == "__main__":
    unittest.main()
