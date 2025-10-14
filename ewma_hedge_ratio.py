"""
EWMA-based hedge ratio calculator for BTC-ETH pairs.

Uses exponentially weighted moving average of returns covariance/variance
to compute beta (hedge ratio). More reactive and efficient than simple
correlation-based methods.

Method: h = EWMA_Cov(BTC, ETH) / EWMA_Var(ETH)
"""
import sys
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from math import log, exp, isfinite


API_BASE_URL = "https://api.extended.exchange/api"
KLINE_ENDPOINT = f"{API_BASE_URL}/kline"
_PRICE_CACHE: Dict[Tuple[str, str, int, str], Tuple[List[float], List[float], dict]] = {}

# Thresholds used to detect suspiciously low variance/covariance
LOW_VARIANCE_THRESHOLD = 1e-8
LOW_COVARIANCE_THRESHOLD = 1e-9
MAX_WINDOW_HOURS = 168

# Mapping of supported kline intervals to seconds
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
}


class EWMAHedgeCalculator:
    """
    Streaming EWMA hedge ratio calculator.

    Updates hedge ratio incrementally as new price data arrives.
    Perfect for production daemons.
    """

    def __init__(
        self,
        half_life_bars: int = 288,  # ~1 day at 5-minute bars
        h_min: float = 0.3,
        h_max: float = 2.0,
        outlier_threshold: float = 0.02,  # ±2% return clamp
        min_var: float = 1e-12
    ):
        """
        Initialize EWMA calculator.

        Args:
            half_life_bars: Half-life in bars for EWMA decay (default 288 = 1 day at 5min)
            h_min: Minimum allowed hedge ratio
            h_max: Maximum allowed hedge ratio
            outlier_threshold: Winsorize returns beyond this threshold (default ±2%)
            min_var: Minimum variance threshold (numerical stability)
        """
        self.half_life_bars = half_life_bars
        self.lam = exp(-log(2) / half_life_bars)  # Decay factor
        self.alpha = 1.0 - self.lam  # Update weight

        self.h_min = h_min
        self.h_max = h_max
        self.outlier_threshold = outlier_threshold
        self.min_var = min_var

        # State
        self.cov = 0.0  # EWMA covariance
        self.var = 0.0  # EWMA variance (ETH)
        self.h = 1.0    # Current hedge ratio
        self.prev_price_btc: Optional[float] = None
        self.prev_price_eth: Optional[float] = None
        self.update_count = 0

    def update(self, price_btc: float, price_eth: float) -> float:
        """
        Update hedge ratio with new prices.

        Args:
            price_btc: Current BTC price
            price_eth: Current ETH price

        Returns:
            Current hedge ratio h
        """
        if self.prev_price_btc is not None and self.prev_price_eth is not None:
            # Calculate log returns
            rb = log(price_btc / self.prev_price_btc)
            re = log(price_eth / self.prev_price_eth)

            # Winsorize (outlier clamp)
            rb = max(min(rb, self.outlier_threshold), -self.outlier_threshold)
            re = max(min(re, self.outlier_threshold), -self.outlier_threshold)

            # Update EWMA statistics
            self.cov = self.lam * self.cov + self.alpha * (rb * re)
            self.var = self.lam * self.var + self.alpha * (re * re)

            # Calculate hedge ratio if variance is sufficient
            if self.var > self.min_var and isfinite(self.cov / self.var):
                raw_h = self.cov / self.var
                self.h = max(self.h_min, min(raw_h, self.h_max))

            self.update_count += 1

        # Store prices for next update
        self.prev_price_btc = price_btc
        self.prev_price_eth = price_eth

        return self.h

    def is_warmed_up(self, min_bars: int = 500) -> bool:
        """Check if calculator has enough data to be reliable."""
        return self.update_count >= min_bars

    def get_stats(self) -> dict:
        """Get current internal statistics."""
        return {
            "h": self.h,
            "cov": self.cov,
            "var": self.var,
            "updates": self.update_count,
            "warmed_up": self.is_warmed_up()
        }


def calculate_ewma_hedge_ratio_batch(
    prices_btc: List[float],
    prices_eth: List[float],
    half_life_bars: int = 288,
    h_min: float = 0.3,
    h_max: float = 2.0,
    outlier_threshold: float = 0.02,
    prev_h: float = 1.0
) -> Tuple[float, dict]:
    """
    Calculate hedge ratio from historical price arrays (batch mode).

    Args:
        prices_btc: List of BTC prices (chronological)
        prices_eth: List of ETH prices (chronological)
        half_life_bars: Half-life for EWMA decay
        h_min: Minimum allowed h
        h_max: Maximum allowed h
        outlier_threshold: Return winsorization threshold
        prev_h: Fallback h if calculation fails

    Returns:
        (hedge_ratio, stats_dict)
    """
    if len(prices_btc) != len(prices_eth):
        raise ValueError(f"Price arrays must be same length: BTC={len(prices_btc)}, ETH={len(prices_eth)}")

    if len(prices_btc) < 2:
        return prev_h, {"error": "Insufficient data", "samples": len(prices_btc)}

    # Initialize
    lam = exp(-log(2) / half_life_bars)
    alpha = 1.0 - lam
    cov = 0.0
    var = 0.0

    # Process all returns
    valid_samples = 0
    for i in range(1, len(prices_btc)):
        # Log returns
        rb = log(prices_btc[i] / prices_btc[i-1])
        re = log(prices_eth[i] / prices_eth[i-1])

        # Winsorize
        rb = max(min(rb, outlier_threshold), -outlier_threshold)
        re = max(min(re, outlier_threshold), -outlier_threshold)

        # Update EWMA
        cov = lam * cov + alpha * (rb * re)
        var = lam * var + alpha * (re * re)
        valid_samples += 1

    # Calculate h
    eps = 1e-12
    if var <= eps or not isfinite(cov / var):
        h = prev_h if isfinite(prev_h) else 1.0
        stats = {
            "h": h,
            "cov": cov,
            "var": var,
            "samples": valid_samples,
            "status": "fallback",
            "reason": "insufficient_variance" if var <= eps else "nan_detected"
        }
    else:
        raw_h = cov / var
        h = max(h_min, min(raw_h, h_max))
        clamped = (raw_h != h)

        stats = {
            "h": h,
            "raw_h": raw_h,
            "cov": cov,
            "var": var,
            "samples": valid_samples,
            "status": "clamped" if clamped else "ok",
            "half_life_bars": half_life_bars
        }

    return h, stats


def calculate_rolling_ols_hedge_ratio(
    prices_btc: List[float],
    prices_eth: List[float],
    window: int = 1000,
    h_min: float = 0.3,
    h_max: float = 2.0
) -> Tuple[float, dict]:
    """
    Calculate hedge ratio using rolling window OLS (fallback method).

    h = Cov(BTC, ETH) / Var(ETH) over last N returns

    Args:
        prices_btc: BTC prices
        prices_eth: ETH prices
        window: Number of bars to use
        h_min: Min h
        h_max: Max h

    Returns:
        (hedge_ratio, stats_dict)
    """
    if len(prices_btc) != len(prices_eth) or len(prices_btc) < 2:
        return 1.0, {"error": "Insufficient data"}

    # Calculate returns
    returns_btc = np.diff(np.log(prices_btc))
    returns_eth = np.diff(np.log(prices_eth))

    # Use last N returns
    if len(returns_btc) > window:
        returns_btc = returns_btc[-window:]
        returns_eth = returns_eth[-window:]

    # Calculate covariance and variance
    cov = np.cov(returns_btc, returns_eth)[0, 1]
    var = np.var(returns_eth)

    if var < 1e-12:
        return 1.0, {"error": "Zero variance", "cov": cov, "var": var}

    raw_h = cov / var
    h = max(h_min, min(raw_h, h_max))

    stats = {
        "h": h,
        "raw_h": raw_h,
        "cov": cov,
        "var": var,
        "samples": len(returns_btc),
        "method": "rolling_ols"
    }

    return h, stats


def calculate_vol_ratio_hedge_ratio(
    prices_btc: List[float],
    prices_eth: List[float],
    half_life_bars: int = 288,
    h_min: float = 0.3,
    h_max: float = 2.0
) -> Tuple[float, dict]:
    """
    Calculate hedge ratio using volatility ratio (crude fallback).

    h ≈ σ(BTC) / σ(ETH)

    Only use if correlation is very high or data is sparse.

    Args:
        prices_btc: BTC prices
        prices_eth: ETH prices
        half_life_bars: Half-life for EWMA volatility
        h_min: Min h
        h_max: Max h

    Returns:
        (hedge_ratio, stats_dict)
    """
    if len(prices_btc) != len(prices_eth) or len(prices_btc) < 2:
        return 1.0, {"error": "Insufficient data"}

    # Calculate returns
    returns_btc = np.diff(np.log(prices_btc))
    returns_eth = np.diff(np.log(prices_eth))

    # EWMA volatility
    lam = exp(-log(2) / half_life_bars)
    alpha = 1.0 - lam

    var_btc = 0.0
    var_eth = 0.0

    for rb, re in zip(returns_btc, returns_eth):
        var_btc = lam * var_btc + alpha * (rb * rb)
        var_eth = lam * var_eth + alpha * (re * re)

    if var_eth < 1e-12:
        return 1.0, {"error": "Zero ETH variance"}

    raw_h = (var_btc ** 0.5) / (var_eth ** 0.5)
    h = max(h_min, min(raw_h, h_max))

    stats = {
        "h": h,
        "raw_h": raw_h,
        "vol_btc": var_btc ** 0.5,
        "vol_eth": var_eth ** 0.5,
        "method": "vol_ratio"
    }

    return h, stats


def _fetch_klines_data(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int
) -> pd.DataFrame:
    """
    Fetch kline data from Extended REST API for a single symbol.

    Splits requests into batches if the requested range exceeds the API
    per-request limit. Returns a DataFrame sorted by timestamp.
    """
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(INTERVAL_SECONDS.keys())}")

    interval_ms = INTERVAL_SECONDS[interval] * 1000
    max_klines = 3000
    klines: List[dict] = []
    current_start = start_time_ms
    attempts = 0

    while current_start < end_time_ms and attempts < 20:
        current_end = min(current_start + max_klines * interval_ms, end_time_ms)
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_time": current_start,
            "end_time": current_end,
            "limit": max_klines,
        }

        try:
            response = requests.get(KLINE_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch klines for {symbol}: {exc}") from exc

        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(
                f"API error while fetching klines for {symbol}: {data.get('error', 'unknown error')}"
            )

        if isinstance(data, dict):
            batch = data.get("data") or data.get("result") or data.get("klines") or []
        elif isinstance(data, list):
            batch = data
        else:
            batch = []
        if not batch:
            break

        klines.extend(batch)

        # Advance start pointer. Add interval to avoid fetching the last candle again.
        last_open_time = batch[-1].get("t")
        if last_open_time is None:
            break

        current_start = int(last_open_time) + interval_ms
        attempts += 1

        if current_start >= end_time_ms or len(batch) < max_klines:
            break

    if not klines:
        raise ValueError(f"No kline data returned for {symbol} between {start_time_ms} and {end_time_ms}")

    df = pd.DataFrame(klines)

    # Ensure required columns exist
    if "t" not in df or "c" not in df:
        raise ValueError(f"Kline response for {symbol} missing required fields.")

    # Convert to numeric types and construct timestamp column
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    numeric_columns = ["o", "h", "l", "c"]
    for col in numeric_columns:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "c"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    df = df[["timestamp", "c"]].rename(columns={"c": "price"})

    return df


def load_prices_from_api(
    symbol_btc: str = "BTC",
    symbol_eth: str = "ETH",
    window_hours: int = 24,
    interval: str = "5m"
) -> Tuple[List[float], List[float], dict]:
    """
    Load BTC and ETH prices directly from Extended kline API.

    Args:
        symbol_btc: BTC symbol (e.g., 'BTC')
        symbol_eth: ETH symbol (e.g., 'ETH')
        window_hours: Number of hours of data to fetch
        interval: Kline interval (default '5m')

    Returns:
        (btc_prices, eth_prices, metadata)
    """
    cache_key = (symbol_btc, symbol_eth, window_hours, interval)
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]

    end_time = int(time.time() * 1000)
    start_time = end_time - int(window_hours * 60 * 60 * 1000)

    btc_df = _fetch_klines_data(symbol_btc, interval, start_time, end_time)
    eth_df = _fetch_klines_data(symbol_eth, interval, start_time, end_time)

    # Merge on timestamp with tolerance equal to half the interval
    tolerance_seconds = INTERVAL_SECONDS[interval]
    merged = pd.merge_asof(
        btc_df.rename(columns={"price": "btc_price"}),
        eth_df.rename(columns={"price": "eth_price"}),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds)
    )

    merged = merged.dropna()

    if len(merged) < 50:
        raise ValueError(f"Insufficient API data: {len(merged)} rows for interval {interval}")

    prices_btc = merged["btc_price"].astype(float).tolist()
    prices_eth = merged["eth_price"].astype(float).tolist()

    metadata = {
        "samples": len(prices_btc),
        "btc_mean": float(np.mean(prices_btc)),
        "eth_mean": float(np.mean(prices_eth)),
        "window_hours": window_hours,
        "interval": interval,
        "symbol_btc": symbol_btc,
        "symbol_eth": symbol_eth,
        "source": "api",
        "start_timestamp": merged["timestamp"].iloc[0],
        "end_timestamp": merged["timestamp"].iloc[-1],
    }

    _PRICE_CACHE[cache_key] = (prices_btc, prices_eth, metadata)
    return prices_btc, prices_eth, metadata


def load_prices_from_csv(
    data_dir: Path,
    symbol_btc: str = "BTC",
    symbol_eth: str = "ETH",
    window_hours: Optional[int] = None
) -> Tuple[List[float], List[float], dict]:
    """
    Load BTC and ETH prices from CSV files.

    Args:
        data_dir: Path to PACIFICA_data directory
        symbol_btc: BTC symbol (default "BTC")
        symbol_eth: ETH symbol (default "ETH")
        window_hours: Only load last N hours (None = all data)

    Returns:
        (btc_prices, eth_prices, metadata)
    """
    btc_file = data_dir / f"prices_{symbol_btc}.csv"
    eth_file = data_dir / f"prices_{symbol_eth}.csv"

    if not btc_file.exists() or not eth_file.exists():
        raise FileNotFoundError(f"Missing price files in {data_dir}")

    # Load data
    btc_df = pd.read_csv(btc_file)
    eth_df = pd.read_csv(eth_file)

    # Convert timestamp (handle both 'timestamp' and 'unix_timestamp' columns)
    timestamp_col = 'unix_timestamp' if 'unix_timestamp' in btc_df.columns else 'timestamp'
    btc_df['timestamp'] = pd.to_datetime(btc_df[timestamp_col], unit='s')

    timestamp_col = 'unix_timestamp' if 'unix_timestamp' in eth_df.columns else 'timestamp'
    eth_df['timestamp'] = pd.to_datetime(eth_df[timestamp_col], unit='s')

    # Filter by window
    if window_hours is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=window_hours)
        btc_df = btc_df[btc_df['timestamp'] >= cutoff]
        eth_df = eth_df[eth_df['timestamp'] >= cutoff]

    # Use mid price
    btc_df['mid'] = (btc_df['bid'] + btc_df['ask']) / 2.0
    eth_df['mid'] = (eth_df['bid'] + eth_df['ask']) / 2.0

    # Sort before merge_asof (required by pandas)
    btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)
    eth_df = eth_df.sort_values('timestamp').reset_index(drop=True)

    # Merge on timestamp
    merged = pd.merge_asof(
        btc_df[['timestamp', 'mid']].rename(columns={'mid': 'btc_price'}),
        eth_df[['timestamp', 'mid']].rename(columns={'mid': 'eth_price'}),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=10)
    )

    # Drop NaN
    merged = merged.dropna()

    if len(merged) < 100:
        raise ValueError(f"Insufficient merged data: {len(merged)} rows")

    prices_btc = merged['btc_price'].tolist()
    prices_eth = merged['eth_price'].tolist()

    metadata = {
        "samples": len(prices_btc),
        "btc_mean": float(np.mean(prices_btc)),
        "eth_mean": float(np.mean(prices_eth)),
        "window_hours": window_hours,
        "source": "csv",
        "symbol_btc": symbol_btc,
        "symbol_eth": symbol_eth
    }

    return prices_btc, prices_eth, metadata


def _stats_are_stable(stats: dict, method: str) -> bool:
    """Determine if the calculated statistics look healthy."""
    if method in ("ewma", "rolling_ols"):
        var = float(abs(stats.get("var", 0.0)))
        cov = float(abs(stats.get("cov", 0.0)))
        if var < LOW_VARIANCE_THRESHOLD or cov < LOW_COVARIANCE_THRESHOLD:
            return False
    elif method == "vol_ratio":
        vol_eth = stats.get("vol_eth")
        if vol_eth is not None and float(vol_eth) ** 2 < LOW_VARIANCE_THRESHOLD:
            return False
    return True


def _build_data_fetch_plan(initial_window: int, initial_interval: str) -> List[Tuple[int, str]]:
    """Generate (window_hours, interval) combinations to try."""
    initial_window = max(int(initial_window), 1)
    window_candidates: List[int] = [initial_window]

    # Expand window progressively up to the max cap
    doubled = min(MAX_WINDOW_HOURS, max(initial_window * 2, initial_window + 6))
    quad = min(MAX_WINDOW_HOURS, max(initial_window * 4, initial_window + 24))

    for candidate in (doubled, quad):
        if candidate not in window_candidates:
            window_candidates.append(candidate)

    # Preserve order: initial window first, then larger horizons
    interval_order: List[str] = []
    for value in [initial_interval, "5m", "15m", "1h"]:
        if value not in interval_order:
            interval_order.append(value)

    combos: List[Tuple[int, str]] = []
    seen: set[Tuple[int, str]] = set()

    def add_combo(window: int, interval: str):
        key = (window, interval)
        if key not in seen:
            combos.append(key)
            seen.add(key)

    # Start with the original parameters
    add_combo(initial_window, initial_interval)

    # Try longer windows with the same interval first
    for window in window_candidates[1:]:
        add_combo(window, initial_interval)

    # Explore alternative intervals, coupled with each window size
    for interval in interval_order[1:]:
        add_combo(initial_window, interval)
        for window in window_candidates[1:]:
            add_combo(window, interval)

    return combos


def calculate_hedge_ratio_auto(
    data_dir: Optional[Path] = None,
    window_hours: int = 24,
    method: str = "ewma",
    fallback_h: float = 0.85,
    verbose: bool = False,
    use_api: bool = True,
    symbol_btc: str = "BTC",
    symbol_eth: str = "ETH",
    interval: str = "5m"
) -> Tuple[float, dict]:
    """
    Automatically calculate hedge ratio with best available method.

    Args:
        data_dir: Path to PACIFICA_data (None = auto-detect)
        window_hours: Hours of data to use
        method: "ewma", "rolling_ols", or "vol_ratio"
        fallback_h: Fallback if all methods fail
        verbose: Print detailed info
        use_api: When True (default) and data_dir is None, fetch data from Extended API
        symbol_btc: BTC symbol to fetch
        symbol_eth: ETH symbol to fetch
        interval: Kline interval to request when using API

    Returns:
        (hedge_ratio, stats_dict)
    """
    plans: List[Tuple[int, str]]
    if data_dir is None and use_api:
        plans = _build_data_fetch_plan(window_hours, interval)
    else:
        plans = [(window_hours, interval)]

    low_variance_attempts: List[Tuple[float, dict]] = []
    last_error: Optional[Exception] = None

    for window_candidate, interval_candidate in plans:
        try:
            if data_dir is None and use_api:
                prices_btc, prices_eth, metadata = load_prices_from_api(
                    symbol_btc=symbol_btc,
                    symbol_eth=symbol_eth,
                    window_hours=window_candidate,
                    interval=interval_candidate
                )
            else:
                resolved_dir = data_dir or Path(__file__).parent.parent / "PACIFICA_data"
                prices_btc, prices_eth, metadata = load_prices_from_csv(
                    resolved_dir,
                    symbol_btc=symbol_btc,
                    symbol_eth=symbol_eth,
                    window_hours=window_candidate
                )

            if verbose:
                print(f"Loaded {metadata['samples']} price samples "
                      f"(window={window_candidate}h interval={interval_candidate})")
                print(f"BTC mean: ${metadata['btc_mean']:.2f}")
                print(f"ETH mean: ${metadata['eth_mean']:.2f}")

            # Calculate based on method
            if method == "ewma":
                h, stats = calculate_ewma_hedge_ratio_batch(
                    prices_btc, prices_eth, half_life_bars=288
                )
            elif method == "rolling_ols":
                h, stats = calculate_rolling_ols_hedge_ratio(
                    prices_btc, prices_eth, window=min(1000, len(prices_btc))
                )
            elif method == "vol_ratio":
                h, stats = calculate_vol_ratio_hedge_ratio(
                    prices_btc, prices_eth, half_life_bars=288
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            stats.update(metadata)
            stats["window_hours_used"] = window_candidate
            stats["interval_used"] = interval_candidate

            if _stats_are_stable(stats, method):
                stats["status"] = stats.get("status", "ok")
                if verbose:
                    print(f"\nMethod: {method}")
                    print(f"Hedge ratio h: {h:.4f}")
                    if "raw_h" in stats:
                        print(f"Raw h (unclamped): {stats['raw_h']:.4f}")
                    print(f"Status: {stats.get('status', 'ok')}")
                return h, stats

            # Record low-variance attempt and keep searching
            stats["status"] = "low_variance"
            stats["low_variance_flag"] = True
            low_variance_attempts.append((h, stats))

            if verbose:
                print(f"Variance/covariance too low at window {window_candidate}h {interval_candidate}, "
                      f"trying broader data...")

        except Exception as exc:
            last_error = exc
            if verbose:
                print(f"Attempt failed for window={window_candidate}h "
                      f"interval={interval_candidate}: {exc}")
            continue

    # If we reach this point, either all attempts were low variance or errors occurred
    if low_variance_attempts:
        low_h, low_stats = low_variance_attempts[-1]
        low_stats = low_stats.copy()
        low_stats.update({
            "h": low_h,
            "status": "low_variance_fallback",
            "method": method
        })
        if verbose:
            print("All data combinations produced low variance; returning the last computed value.")
        return low_h, low_stats

    if last_error is not None:
        if verbose:
            print(f"Warning: Failed to calculate hedge ratio: {last_error}")
            print(f"Using fallback: h = {fallback_h}")

        return fallback_h, {
            "h": fallback_h,
            "status": "fallback",
            "error": str(last_error),
            "method": method
        }

    # Unexpected path – return fallback for safety
    if verbose:
        print("Warning: No valid data combinations succeeded; using fallback.")
    return fallback_h, {
        "h": fallback_h,
        "status": "fallback",
        "error": "no_valid_data_combinations",
        "method": method
    }


if __name__ == "__main__":
    """Test hedge ratio calculation."""
    print("=" * 60)
    print("EWMA HEDGE RATIO CALCULATOR TEST")
    print("=" * 60)

    # Test all methods
    methods = ["ewma", "rolling_ols", "vol_ratio"]

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Testing method: {method.upper()}")
        print("=" * 60)

        h, stats = calculate_hedge_ratio_auto(
            window_hours=24,
            method=method,
            verbose=True
        )

        # Validate h is reasonable
        print(f"\n[Validation]")
        if 0.5 <= h <= 1.5:
            print(f"  [OK] h = {h:.4f} is reasonable (0.5-1.5 range)")
        elif 0.3 <= h < 0.5:
            print(f"  [WARN] h = {h:.4f} is low (30/70 BTC/ETH split)")
        elif 1.5 < h <= 2.0:
            print(f"  [WARN] h = {h:.4f} is high (67/33 BTC/ETH split)")
        else:
            print(f"  [ERROR] h = {h:.4f} is outside expected range!")

        # Show split
        btc_pct = 100 / (1 + h)
        eth_pct = 100 * h / (1 + h)
        print(f"  Portfolio split: {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
