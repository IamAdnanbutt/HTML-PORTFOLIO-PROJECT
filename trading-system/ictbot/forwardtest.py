"""Live (paper) forward-test runner against Alpaca.

DRY RUN by default: it pulls live bars for the dynamically-selected universe,
runs the strategy bar-by-bar, and prints the signals/fills it *would* take —
**no orders are placed**. This proves the whole data -> selector -> model
pipeline against live data with zero execution risk; wiring real Alpaca bracket
orders is the next step (to be verified against the live API).

Run it where there is network access to Alpaca — your own machine, or a sandbox
with the Alpaca hosts allowlisted — with credentials in the environment::

    export ALPACA_API_KEY=...      # never commit these
    export ALPACA_API_SECRET=...
    python -m ictbot forwardtest --top-k 3 --cycles 1     # one pass, then exit
"""

from __future__ import annotations

import time as _time
from datetime import datetime
from typing import Dict, List, Optional

from .alpaca import (bars_to_candles, screener_symbols, stats_from_bars)
from .config import Config, INSTRUMENTS, Instrument
from .live import LiveRunner
from .models import Candle
from .universe import CRYPTO, EQUITY, active_asset_class, select_universe

# A small liquid crypto set for the overnight window (24/7 markets).
DEFAULT_CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD"]


def _config_for(symbol: str, asset_class: str) -> Config:
    """Config for a symbol, falling back to a generic spec for screener names."""
    inst = INSTRUMENTS.get(symbol)
    if inst is None:
        tick = 0.01
        inst = Instrument(symbol, tick, tick, 1.0, commission=0.0,
                          asset_class=asset_class)
    return Config(instrument=inst)


class ForwardTester:
    """Dynamic, time-of-day-routed paper forward test (dry run).

    ``process`` is pure (no network) and unit-testable; ``select`` / ``cycle`` /
    ``run`` add the Alpaca data fetch and the poll loop.
    """

    def __init__(self, client, top_k: int = 3, lookback: int = 300):
        self.client = client
        self.top_k = top_k
        self.lookback = lookback
        self._runners: Dict[str, LiveRunner] = {}
        self._last_ts: Dict[str, datetime] = {}

    # ------------------------------------------------------------- network
    def candidate_symbols(self, asset_class: str) -> List[str]:
        if asset_class == EQUITY:
            return screener_symbols(self.client.most_actives(top=40))
        return list(DEFAULT_CRYPTO)

    def fetch_bars(self, symbol: str, asset_class: str) -> List[Candle]:
        raw = (self.client.crypto_bars(symbol, limit=self.lookback)
               if asset_class == CRYPTO
               else self.client.stock_bars(symbol, limit=self.lookback))
        return bars_to_candles(raw)

    def select(self, now: datetime, asset_class: str) -> Dict[str, List[Candle]]:
        """Fetch candidate bars, rank by volume × volatility, keep the top-k."""
        bars: Dict[str, List[Candle]] = {}
        stats = []
        for sym in self.candidate_symbols(asset_class):
            try:
                candles = self.fetch_bars(sym, asset_class)
            except Exception:
                continue                      # skip a symbol that fails to fetch
            if not candles:
                continue
            bars[sym] = candles
            st = stats_from_bars(sym, candles, asset_class)
            if st:
                stats.append(st)
        picks = select_universe(stats, now, top_k=self.top_k)
        return {s: bars[s] for s in picks if s in bars}

    # ------------------------------------------------- pure processing core
    def process(self, now: datetime,
                bars_by_symbol: Dict[str, List[Candle]]) -> List[str]:
        """Feed only *new* bars per symbol to that symbol's runner.

        Each cycle re-fetches a window of bars; we replay only those newer than
        the last one already seen, so a bar is never processed (or signalled)
        twice.
        """
        ac = active_asset_class(now.time())
        out = [f"[{now:%Y-%m-%d %H:%M}] {ac} universe -> {list(bars_by_symbol)}"]
        for sym, candles in bars_by_symbol.items():
            runner = self._runners.get(sym)
            if runner is None:
                inst_ac = (INSTRUMENTS[sym].asset_class
                           if sym in INSTRUMENTS else ac)
                runner = LiveRunner(_config_for(sym, inst_ac))
                self._runners[sym] = runner
            last = self._last_ts.get(sym)
            fresh = [c for c in candles if last is None or c.time > last]
            for c in fresh:
                for msg in runner.on_candle(c):
                    out.append(f"  {sym}: {msg}")
            if candles:
                self._last_ts[sym] = candles[-1].time
        return out

    def cycle(self, now: Optional[datetime] = None) -> List[str]:
        now = now or datetime.now()
        return self.process(now, self.select(now, active_asset_class(now.time())))

    def run(self, cycles: Optional[int] = None, poll_seconds: int = 60) -> None:
        n = 0
        while cycles is None or n < cycles:
            for line in self.cycle():
                print(line, flush=True)
            n += 1
            if cycles is not None and n >= cycles:
                break
            _time.sleep(poll_seconds)
