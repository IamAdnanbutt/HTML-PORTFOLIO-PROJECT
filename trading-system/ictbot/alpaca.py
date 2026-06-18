"""Alpaca adapter: REST client + the pure data/order bridges to ictbot.

Splits cleanly into two layers:

* :class:`AlpacaClient` — thin REST wrapper (account, bars, the most-actives
  screener, order submission). It reads credentials from ``ALPACA_API_KEY`` /
  ``ALPACA_API_SECRET`` (never hard-coded) and is the only part that touches the
  network, so it is exercised live, not in unit tests.
* Pure helpers — convert Alpaca JSON to/from ictbot types: bars → ``Candle``
  (converting Alpaca's UTC timestamps to the Eastern clock the session logic
  uses), bars → ``SymbolStats`` for the universe selector, and a ``Signal`` →
  Alpaca bracket-order payload. These are deterministic and fully unit-tested.

NOTE: Alpaca offers no futures. Equity bracket orders are well supported;
crypto order types are more limited (bracket/OCO support must be confirmed
against the live API before the crypto exit path is finalised).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from .indicators import atr
from .models import Candle, Direction, Signal
from .universe import SymbolStats

_ET = ZoneInfo("America/New_York")
_FRAC = re.compile(r"\.(\d+)")


# --------------------------------------------------------------- pure helpers
def _parse_ts(s: str) -> datetime:
    """Parse an Alpaca RFC3339 timestamp (UTC) into a naive **Eastern** datetime.

    The rest of ictbot treats candle times as naive ET, so a 13:30Z bar in June
    becomes 09:30 (the RTH open). Handles 'Z', explicit offsets, and
    nanosecond fractions that ``fromisoformat`` would otherwise reject.
    """
    s = s.strip().replace("Z", "+00:00")
    m = _FRAC.search(s)
    if m and len(m.group(1)) > 6:                      # trim ns -> us
        s = s[:m.start()] + "." + m.group(1)[:6] + s[m.end():]
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_ET).replace(tzinfo=None)


def bars_to_candles(bars: List[dict]) -> List[Candle]:
    """Alpaca bar dicts (``t/o/h/l/c/v``) → ictbot ``Candle`` list, ET-clocked."""
    return [Candle(_parse_ts(b["t"]), float(b["o"]), float(b["h"]),
                   float(b["l"]), float(b["c"]), float(b.get("v", 0)))
            for b in bars]


def screener_symbols(payload: dict) -> List[str]:
    """Extract the candidate symbol list from a most-actives screener response."""
    return [row["symbol"] for row in payload.get("most_actives", [])
            if row.get("symbol")]


def stats_from_bars(symbol: str, candles: List[Candle], asset_class: str,
                    atr_period: int = 14) -> Optional[SymbolStats]:
    """Build a :class:`SymbolStats` (dollar volume + ATR%) from recent bars."""
    if not candles:
        return None
    last = candles[-1].close
    dollar_volume = sum(c.close * c.volume for c in candles)
    atr_pct = (atr(candles, atr_period) / last) if last else 0.0
    return SymbolStats(symbol, asset_class, dollar_volume, atr_pct)


def bracket_payload(signal: Signal, qty, symbol: str,
                    time_in_force: str = "day") -> dict:
    """Build an Alpaca bracket-order payload from a Signal.

    A bracket places the entry plus an OCO take-profit / stop-loss in one shot,
    so the broker manages the exit — mirroring the strategy's stop/target.
    """
    side = "buy" if signal.direction is Direction.LONG else "sell"
    return {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "limit",
        "limit_price": f"{signal.entry}",
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{signal.target}"},
        "stop_loss": {"stop_price": f"{signal.stop}"},
    }


# ------------------------------------------------------------- network client
class AlpacaClient:
    """Minimal Alpaca REST client (paper by default). Network-only.

    Credentials are read from the environment if not passed explicitly; they are
    never written to disk or committed.
    """

    def __init__(self, key: Optional[str] = None, secret: Optional[str] = None,
                 paper: bool = True):
        self.key = key or os.environ.get("ALPACA_API_KEY")
        self.secret = secret or os.environ.get("ALPACA_API_SECRET")
        if not (self.key and self.secret):
            raise RuntimeError(
                "Set ALPACA_API_KEY and ALPACA_API_SECRET in the environment.")
        self.trade_base = ("https://paper-api.alpaca.markets" if paper
                           else "https://api.alpaca.markets")
        self.data_base = "https://data.alpaca.markets"

    # -- low-level ---------------------------------------------------------
    def _req(self, method: str, url: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    # -- account / orders --------------------------------------------------
    def account(self) -> dict:
        return self._req("GET", f"{self.trade_base}/v2/account")

    def submit_order(self, payload: dict) -> dict:
        return self._req("POST", f"{self.trade_base}/v2/orders", payload)

    def get_order(self, order_id: str) -> dict:
        return self._req("GET", f"{self.trade_base}/v2/orders/{order_id}")

    def positions(self) -> List[dict]:
        return self._req("GET", f"{self.trade_base}/v2/positions")

    # -- market data -------------------------------------------------------
    def stock_bars(self, symbol: str, timeframe: str = "1Min",
                   limit: int = 200) -> List[dict]:
        url = (f"{self.data_base}/v2/stocks/{symbol}/bars"
               f"?timeframe={timeframe}&limit={limit}")
        return self._req("GET", url).get("bars") or []

    def crypto_bars(self, symbol: str, timeframe: str = "1Min",
                    limit: int = 200) -> List[dict]:
        url = (f"{self.data_base}/v1beta3/crypto/us/bars"
               f"?symbols={urllib.parse.quote(symbol)}"
               f"&timeframe={timeframe}&limit={limit}")
        return (self._req("GET", url).get("bars") or {}).get(symbol) or []

    def most_actives(self, top: int = 50) -> dict:
        url = (f"{self.data_base}/v1beta1/screener/stocks/most-actives"
               f"?top={top}")
        return self._req("GET", url)
