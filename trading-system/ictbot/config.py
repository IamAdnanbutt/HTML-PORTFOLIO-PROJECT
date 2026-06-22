"""Configuration: session clock, macro windows, instruments and risk.

All times are New York / Eastern (the reference clock for the 9:30 model).
Times are stored as ``datetime.time`` so they can be compared against the
(naive, ET-interpreted) timestamps on each :class:`~ictbot.models.Candle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import List, Tuple


# A macro window is a (start, end) tuple of ET clock times. These are the
# recurring intraday windows when algorithmic delivery tends to occur; the
# strategy only takes entries while price is inside one of them. The notes
# explicitly reference the "9:50 macro".
MacroWindow = Tuple[time, time]


@dataclass
class SessionConfig:
    """Defines the trading session in Eastern time."""

    premarket_start: time = time(8, 0)     # start accumulating the AM range
    open_time: time = time(9, 30)          # the 9:30 open
    # latest time we will still arm a setup (avoid late-day chop)
    setup_cutoff: time = time(11, 30)
    close_time: time = time(16, 0)         # flatten everything by the close

    macro_windows: List[MacroWindow] = field(default_factory=lambda: [
        (time(9, 50), time(10, 10)),       # the "9:50 macro" from the notes
        (time(10, 50), time(11, 10)),
        (time(13, 10), time(13, 40)),
        (time(14, 50), time(15, 10)),
    ])

    def in_macro(self, t: time) -> bool:
        return any(start <= t <= end for start, end in self.macro_windows)


@dataclass
class Instrument:
    """Contract / symbol specification.

    Futures use contract economics (``point_value`` = $ per 1.00 move per
    contract). For Alpaca-tradeable proxies there are no futures, so equities
    and crypto are modelled as ``point_value = 1.0`` (1 unit moves $1 per $1 of
    price) and sized in shares / units rather than contracts.
    """

    symbol: str
    tick_size: float          # minimum price increment
    tick_value: float         # $ per tick per contract
    point_value: float        # $ per 1.00 price move per contract
    commission: float = 2.5   # $ per side per contract
    asset_class: str = "futures"   # "futures" | "equity" | "crypto"


# The notes recommend NQ and ES "where liquidity delivery is consistent".
INSTRUMENTS = {
    "NQ": Instrument("NQ", tick_size=0.25, tick_value=5.0, point_value=20.0),
    "ES": Instrument("ES", tick_size=0.25, tick_value=12.5, point_value=50.0),
    "MNQ": Instrument("MNQ", tick_size=0.25, tick_value=0.5, point_value=2.0),
    "MES": Instrument("MES", tick_size=0.25, tick_value=1.25, point_value=5.0),
    # --- Alpaca-tradeable proxies (Alpaca offers no futures) ---------------
    # Index ETFs (RTH) — the natural NQ/ES proxies for the 9:30 model …
    "QQQ": Instrument("QQQ", 0.01, 0.01, 1.0, commission=0.0, asset_class="equity"),
    "SPY": Instrument("SPY", 0.01, 0.01, 1.0, commission=0.0, asset_class="equity"),
    # … plus leveraged ETFs that carry the day's volatility.
    "TQQQ": Instrument("TQQQ", 0.01, 0.01, 1.0, commission=0.0, asset_class="equity"),
    "SQQQ": Instrument("SQQQ", 0.01, 0.01, 1.0, commission=0.0, asset_class="equity"),
    "SOXL": Instrument("SOXL", 0.01, 0.01, 1.0, commission=0.0, asset_class="equity"),
    # Crypto (24/7) — keeps the overnight ORs and Asia Killzone fed.
    "BTC/USD": Instrument("BTC/USD", 1.0, 1.0, 1.0, commission=0.0, asset_class="crypto"),
    "ETH/USD": Instrument("ETH/USD", 0.1, 0.1, 1.0, commission=0.0, asset_class="crypto"),
    "SOL/USD": Instrument("SOL/USD", 0.01, 0.01, 1.0, commission=0.0, asset_class="crypto"),
}



@dataclass
class RiskConfig:
    """Risk and money-management settings."""

    account_size: float = 100_000.0
    risk_per_trade: float = 0.005      # 0.5% of account risked per trade
    max_trades_per_day: int = 1        # notes: anchor on the *first* PFVG
    min_rr: float = 2.0                # skip setups that cannot reach >= 2R
    max_contracts: int = 20            # hard cap (futures contracts)
    max_shares: int = 1000             # hard cap (equity shares)
    max_notional: float = 100_000.0    # per-position notional ceiling (equity/crypto)


@dataclass
class StrategyConfig:
    """Tunables for the 9:30 Open Model detection logic."""

    # swing pivot lookback used for structure / HTF bias (in bars)
    swing_lookback: int = 2
    # a displacement candle must travel at least this multiple of the recent
    # average bar range to count as a genuine expansion leg
    displacement_atr_mult: float = 1.5
    atr_period: int = 14
    # how many bars after the open we still accept the manipulation sweep
    manipulation_window_bars: int = 30
    # entry triggers when price trades back to this fraction into the FVG
    # (0.0 = edge, 0.5 = consequent encroachment / mid)
    fvg_entry_depth: float = 0.5
    # stop is placed this many ticks beyond the manipulation extreme
    stop_buffer_ticks: int = 4


@dataclass
class Config:
    """Top-level configuration bundle."""

    instrument: Instrument = field(default_factory=lambda: INSTRUMENTS["NQ"])
    session: SessionConfig = field(default_factory=SessionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @classmethod
    def for_symbol(cls, symbol: str) -> "Config":
        symbol = symbol.upper()
        if symbol not in INSTRUMENTS:
            raise ValueError(
                f"Unknown instrument {symbol!r}. "
                f"Known: {', '.join(INSTRUMENTS)}"
            )
        return cls(instrument=INSTRUMENTS[symbol])
