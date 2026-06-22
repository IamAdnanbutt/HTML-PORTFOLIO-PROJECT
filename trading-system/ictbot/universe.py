"""Dynamic instrument selection for forward testing.

The forward tester does not trade one fixed symbol. At each decision point it
picks the most liquid *and* most volatile instruments to trade, and routes by
asset class on the clock so there is always something live:

  * US regular session (09:30-16:00 ET): US equities / ETFs, including the
    leveraged ETFs that carry the day's volatility.
  * Outside the equity session: crypto, which trades 24/7 — this is what keeps
    the overnight Opening Ranges and the Asia Killzone fed.

The selection logic here is pure and unit-tested. Fetching the candidate stats
(recent dollar volume and ATR) from the data feed is the caller's job, so this
module needs no network and no API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import List, Sequence, Tuple

EQUITY = "equity"
CRYPTO = "crypto"


@dataclass(frozen=True)
class SymbolStats:
    """A candidate symbol with its two ranking axes over a recent window."""

    symbol: str
    asset_class: str        # EQUITY | CRYPTO
    dollar_volume: float    # recent traded notional — the *liquidity* axis
    atr_pct: float          # ATR / price — the *volatility* axis


def active_asset_class(t: time,
                       rth_open: time = time(9, 30),
                       rth_close: time = time(16, 0)) -> str:
    """Which asset class is in play at clock time ``t`` (ET).

    Equities during the regular session, crypto otherwise — so the overnight
    and Asia systems always have a live, volatile market to trade.
    """
    return EQUITY if rth_open <= t <= rth_close else CRYPTO


def _minmax(values: Sequence[float]) -> List[float]:
    """Min-max normalise to [0, 1]; a flat list maps to all-0.5 (no signal)."""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def score_candidates(cands: Sequence[SymbolStats],
                     vol_weight: float = 0.5) -> List[Tuple[str, float]]:
    """Score each candidate by normalised liquidity and volatility.

    ``vol_weight`` blends the two axes: 0.5 weighs traded volume and volatility
    equally, 1.0 ranks purely on volume, 0.0 purely on volatility.
    """
    if not cands:
        return []
    nv = _minmax([c.dollar_volume for c in cands])
    na = _minmax([c.atr_pct for c in cands])
    w = vol_weight
    return [(c.symbol, w * v + (1.0 - w) * a)
            for c, v, a in zip(cands, nv, na)]


def rank_universe(cands: Sequence[SymbolStats], asset_class: str,
                  top_k: int = 5, min_dollar_volume: float = 0.0,
                  min_atr_pct: float = 0.0,
                  vol_weight: float = 0.5) -> List[str]:
    """Top ``top_k`` symbols of ``asset_class`` by combined volume × volatility.

    Candidates below the liquidity (``min_dollar_volume``) or volatility
    (``min_atr_pct``) floors are dropped before ranking, so an illiquid name
    can't win on volatility alone (and vice versa).
    """
    pool = [c for c in cands
            if c.asset_class == asset_class
            and c.dollar_volume >= min_dollar_volume
            and c.atr_pct >= min_atr_pct]
    scored = score_candidates(pool, vol_weight)
    scored.sort(key=lambda sc: sc[1], reverse=True)
    return [sym for sym, _ in scored[:top_k]]


def select_universe(cands: Sequence[SymbolStats], now,
                    top_k: int = 5, min_dollar_volume: float = 0.0,
                    min_atr_pct: float = 0.0,
                    vol_weight: float = 0.5) -> List[str]:
    """The symbols to trade right now: the active asset class, top-ranked.

    ``now`` is a ``datetime`` (ET); the asset class is chosen from its clock
    time, then ranked among same-class candidates.
    """
    asset_class = active_asset_class(now.time())
    return rank_universe(cands, asset_class, top_k=top_k,
                         min_dollar_volume=min_dollar_volume,
                         min_atr_pct=min_atr_pct, vol_weight=vol_weight)
