"""
Hedge ratio calculator for BTC-ETH pairs.
Uses simple beta calculation based on historical price movements.
"""
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from ewma_hedge_ratio import load_prices_from_api
except ModuleNotFoundError:  # pragma: no cover - fallback for package-relative imports
    from .ewma_hedge_ratio import load_prices_from_api


class HedgeRatioCalculator:
    """
    Calculates optimal hedge ratio (h) for BTC-ETH long/short strategy.

    The hedge ratio determines how much ETH to short for each $1 of BTC long:
    - h = 0.5 means for $100 BTC long, short $50 ETH
    - h = 1.0 means for $100 BTC long, short $100 ETH

    Uses beta calculation: h = Cov(BTC, ETH) / Var(ETH)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        fallback_ratio: float = 0.85,
        interval: str = "5m"
    ):
        """
        Initialize calculator.

        Args:
            data_dir: Legacy parameter (unused, kept for backward compatibility)
            fallback_ratio: Default ratio if calculation fails (default 0.85)
            interval: Kline interval to request from Pacifica API
        """
        # data_dir kept for backward compatibility but no longer used
        self.data_dir = data_dir or Path(__file__).parent.parent / "PACIFICA_data"
        self.fallback_ratio = fallback_ratio
        self.interval = interval
        self._cached_ratio: Optional[float] = None
        self._last_calc_time: float = 0

    def estimate_ratio(self, window_hours: int = 24, force_refresh: bool = False) -> float:
        """
        Estimate hedge ratio using historical price data.

        Args:
            window_hours: Hours of historical data to use (default 24)
            force_refresh: Force recalculation even if cached (default False)

        Returns:
            Hedge ratio (h)
        """
        import time

        # Use cached value if recent (within 1 hour)
        if not force_refresh and self._cached_ratio is not None:
            if time.time() - self._last_calc_time < 3600:
                return self._cached_ratio

        try:
            # Load price data from API
            prices_btc, prices_eth, metadata = load_prices_from_api(
                symbol_btc="BTC",
                symbol_eth="ETH",
                window_hours=window_hours,
                interval=self.interval
            )

            samples = len(prices_btc)
            if samples < 100:
                print(f"Warning: Insufficient API data ({samples} samples), using fallback")
                return self.fallback_ratio

            prices_btc = np.asarray(prices_btc, dtype=float)
            prices_eth = np.asarray(prices_eth, dtype=float)

            # Calculate log returns
            btc_returns = np.diff(np.log(prices_btc))
            eth_returns = np.diff(np.log(prices_eth))

            if len(btc_returns) < 50 or len(eth_returns) < 50:
                print(f"Warning: Insufficient return data ({len(btc_returns)} rows), using fallback")
                return self.fallback_ratio

            # Calculate beta (hedge ratio)
            covariance = np.cov(btc_returns, eth_returns)[0, 1]
            eth_variance = np.var(eth_returns)

            if eth_variance == 0:
                print("Warning: ETH variance is zero, using fallback")
                return self.fallback_ratio

            ratio = covariance / eth_variance

            # Sanity check: ratio should be between 0.3 and 2.0
            if ratio < 0.3 or ratio > 2.0:
                print(f"Warning: Calculated ratio {ratio:.3f} outside bounds [0.3, 2.0], using fallback")
                return self.fallback_ratio

            # Cache result
            self._cached_ratio = ratio
            self._last_calc_time = time.time()

            print(f"Calculated hedge ratio: {ratio:.3f} (using {samples} samples from API)")
            return ratio

        except Exception as e:
            print(f"Warning: Hedge ratio calculation failed: {e}, using fallback")
            return self.fallback_ratio

    def get_ratio(self) -> float:
        """Get current hedge ratio (uses cache if available)."""
        return self.estimate_ratio(window_hours=24, force_refresh=False)


# Standalone function for easy import
def calculate_hedge_ratio(
    data_dir: Optional[Path] = None,
    window_hours: int = 24,
    fallback: float = 0.85,
    interval: str = "5m"
) -> float:
    """
    Calculate hedge ratio for BTC-ETH pair.

    Args:
        data_dir: Legacy parameter (ignored; API data is always used)
        window_hours: Hours of historical data to use
        fallback: Fallback ratio if calculation fails
        interval: Kline interval to request from Pacifica API

    Returns:
        Hedge ratio (h)
    """
    calc = HedgeRatioCalculator(
        data_dir=data_dir,
        fallback_ratio=fallback,
        interval=interval
    )
    return calc.estimate_ratio(window_hours=window_hours)


if __name__ == "__main__":
    # Test calculation
    ratio = calculate_hedge_ratio()
    print(f"\nFinal hedge ratio: {ratio:.4f}")
    print(f"Example: For $1000 BTC long, short ${ratio * 1000:.2f} ETH")
