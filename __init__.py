"""
BTC-ETH Hedged Long/Short Bot for Pacifica DEX

Ultra-minimal configuration market-neutral hedge bot.
"""

__version__ = "1.0.0"
__author__ = "Pacifica Trading Team"

from .hedge_ratio import HedgeRatioCalculator, calculate_hedge_ratio
from .pacifica_client import PacificaClient

__all__ = [
    "HedgeRatioCalculator",
    "calculate_hedge_ratio",
    "PacificaClient"
]
