"""
ictbot - An automated trading system implementing the ICT "9:30 A.M. Open Model".

Core principle (from the strategy notes):

    Liquidity  ->  Manipulation  ->  Delivery

The package is organised into small, testable modules:

    config      - session times, macro windows, instruments, risk settings
    models      - plain data classes (Candle, FVG, LiquidityPool, Signal, Trade)
    data        - CSV loader + synthetic intraday data generator
    indicators  - Fair Value Gaps, swing points, displacement
    liquidity   - liquidity pools (PMH/PML, prev session/daily H/L) + sweeps
    bias        - higher-time-frame (H1) order-flow bias
    strategy    - the 9:30 Open Model state machine (Reversal / Continuation)
    backtest    - event-driven backtester + performance metrics
    broker      - broker adapter interface + a paper broker
    live        - paper/live trading loop skeleton
    cli         - command line entry point

Everything here uses only the Python standard library so it runs anywhere
Python 3.10+ is installed - no third-party packages required.
"""

__version__ = "1.0.0"
