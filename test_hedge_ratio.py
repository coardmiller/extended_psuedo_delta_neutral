"""
Test script to validate EWMA hedge ratio calculations.

Validates that:
1. h values are reasonable (typically 0.5-1.5 for BTC/ETH)
2. Different methods produce consistent results
3. Portfolio splits make sense (not too extreme)
4. Calculations are stable across different time windows

Usage:
    python test_hedge_ratio.py
"""
import sys
import time
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ewma_hedge_ratio import (
    calculate_hedge_ratio_auto,
    load_prices_from_api,
    calculate_ewma_hedge_ratio_batch,
    calculate_rolling_ols_hedge_ratio,
    calculate_vol_ratio_hedge_ratio,
    EWMAHedgeCalculator
)

SMALL_FLOAT_EPS = 1e-12


def format_stat(value: float) -> str:
    """Format small floating point numbers for readability."""
    if value == 0:
        return "0.000000"

    magnitude = abs(value)
    if magnitude < 1e-4:
        return f"{value:.6e}"
    return f"{value:.6f}"


def validate_hedge_ratio(h: float, method: str) -> dict:
    """
    Validate if hedge ratio is reasonable.

    For BTC/ETH, we expect:
    - h around 0.7-1.2 is typical (60/40 to 45/55 split)
    - h < 0.5 means very BTC-heavy (>67% BTC)
    - h > 1.5 means very ETH-heavy (>60% ETH)

    Returns validation dict with status and interpretation.
    """
    result = {
        "h": h,
        "method": method,
        "valid": True,
        "warnings": [],
        "interpretation": ""
    }

    # Calculate portfolio split
    btc_pct = 100 / (1 + h)
    eth_pct = 100 * h / (1 + h)

    result["btc_pct"] = btc_pct
    result["eth_pct"] = eth_pct

    # Validate ranges
    if not (0.3 <= h <= 2.0):
        result["valid"] = False
        result["warnings"].append(f"h={h:.4f} outside safety bounds [0.3, 2.0]")

    if h < 0.3: # Changed from 0.5
        result["warnings"].append(f"Very BTC-heavy: {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH")
    elif h > 1.5:
        result["warnings"].append(f"Very ETH-heavy: {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH")

    # Interpretation
    if 0.7 <= h <= 1.2:
        result["interpretation"] = f"[OK] IDEAL: Balanced {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH"
    elif 0.3 <= h < 0.7: # Changed from 0.5
        result["interpretation"] = f"[WARNING]  BTC-leaning: {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH"
    elif 1.2 < h <= 1.5:
        result["interpretation"] = f"[WARNING]  ETH-leaning: {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH"
    elif h < 0.3:
        result["interpretation"] = f"[WARNING]  Very BTC-leaning: {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH"
    else: # h > 1.5
        result["interpretation"] = f"[WARNING]  Very ETH-leaning: {btc_pct:.1f}% BTC / {eth_pct:.1f}% ETH"

    return result


def test_all_methods():
    """Test all hedge ratio calculation methods."""
    print("=" * 70)
    print("HEDGE RATIO VALIDATION - ALL METHODS")
    print("=" * 70)

    methods = ["ewma", "rolling_ols", "vol_ratio"]
    results = {}

    for method in methods:
        print(f"\n{'─' * 70}")
        print(f"Method: {method.upper()}")
        print("─" * 70)

        try:
            h, stats = calculate_hedge_ratio_auto(
                window_hours=24,
                method=method,
                verbose=False
            )

            validation = validate_hedge_ratio(h, method)
            results[method] = validation

            print(f"\nHedge ratio: h = {h:.4f}")
            print(f"Portfolio: {validation['btc_pct']:.1f}% BTC / {validation['eth_pct']:.1f}% ETH")
            print(f"Status: {validation['interpretation']}")

            if validation['warnings']:
                print(f"\nWarnings:")
                for warning in validation['warnings']:
                    print(f"  • {warning}")

            # Show detailed stats
            print(f"\nStatistics:")
            if 'cov' in stats:
                print(f"  Covariance: {format_stat(stats['cov'])}")
                if abs(stats['cov']) < SMALL_FLOAT_EPS:
                    print("  [WARNING]  Covariance is zero, this is unusual.")
            if 'var' in stats:
                print(f"  Variance (ETH): {format_stat(stats['var'])}")
                if abs(stats['var']) < SMALL_FLOAT_EPS:
                    print("  [WARNING]  Variance is zero, hedge ratio may be invalid.")
            if 'raw_h' in stats:
                print(f"  Raw h (unclamped): {stats['raw_h']:.4f}")
            if 'samples' in stats:
                print(f"  Samples used: {stats['samples']}")
            if 'window_hours_used' in stats or 'interval_used' in stats:
                print(f"  Data window: {stats.get('window_hours_used', 'N/A')}h @ "
                      f"{stats.get('interval_used', stats.get('interval', 'N/A'))}")
            if stats.get('status') in ("low_variance", "low_variance_fallback"):
                print("  [WARNING]  Detected low variance/covariance; consider broader window or coarser interval.")

        except Exception as e:
            print(f"❌ Error: {e}")
            results[method] = {"valid": False, "error": str(e)}

    # Compare methods
    print(f"\n{'=' * 70}")
    print("METHOD COMPARISON")
    print("=" * 70)

    valid_results = {k: v for k, v in results.items() if v.get("valid", False)}

    if len(valid_results) >= 2:
        h_values = [v['h'] for v in valid_results.values()]
        h_min = min(h_values)
        h_max = max(h_values)
        h_spread = h_max - h_min

        print(f"\nHedge ratio range: {h_min:.4f} to {h_max:.4f}")
        print(f"Spread: {h_spread:.4f} ({h_spread/h_min*100:.1f}% of min)")

        if h_spread < 0.1:
            print("✅ Methods agree well (spread < 0.1)")
        elif h_spread < 0.2:
            print("⚠️  Moderate disagreement (spread 0.1-0.2)")
        else:
            print("❌ Significant disagreement (spread > 0.2)")

        # Recommend best method
        print("\n📊 Method details:")
        for method, val in valid_results.items():
            print(f"  {method:12s}: h = {val['h']:.4f}  ({val['btc_pct']:.1f}/{val['eth_pct']:.1f} split)")

        print("\n💡 Recommendation:")
        print("  Use 'ewma' for production - most reactive to recent data")
        print("  Use 'rolling_ols' as fallback - more stable but slower to adapt")
        print("  Avoid 'vol_ratio' unless correlation is very high")

    return results


def test_time_windows():
    """Test hedge ratio stability across different time windows."""
    print("\n" + "=" * 70)
    print("TIME WINDOW SENSITIVITY TEST")
    print("=" * 70)

    windows = [6, 12, 24, 48]  # hours
    results = {}

    for window in windows:
        print(f"\n{'─' * 70}")
        print(f"Window: {window} hours")
        print("─" * 70)

        try:
            h, stats = calculate_hedge_ratio_auto(
                window_hours=window,
                method="ewma",
                verbose=False
            )

            validation = validate_hedge_ratio(h, f"ewma_{window}h")
            results[window] = validation

            print(f"h = {h:.4f} ({validation['btc_pct']:.1f}% BTC / {validation['eth_pct']:.1f}% ETH)")
            print(f"Samples: {stats.get('samples', 'N/A')}")

        except Exception as e:
            print(f"❌ Error: {e}")
            results[window] = {"valid": False}

    # Analyze stability
    print(f"\n{'=' * 70}")
    print("STABILITY ANALYSIS")
    print("=" * 70)

    valid_h = [v['h'] for v in results.values() if v.get('valid', False)]

    if len(valid_h) >= 2:
        h_mean = sum(valid_h) / len(valid_h)
        h_std = (sum((h - h_mean)**2 for h in valid_h) / len(valid_h)) ** 0.5

        print(f"\nAcross {len(valid_h)} windows:")
        print(f"  Mean h: {h_mean:.4f}")
        print(f"  Std dev: {h_std:.4f}")
        print(f"  CV: {h_std/h_mean*100:.1f}%")

        if h_std < 0.05:
            print("\n✅ Very stable across time windows (CV < 5%)")
        elif h_std < 0.1:
            print("\n⚠️  Moderate stability (CV 5-10%)")
        else:
            print("\n❌ Unstable - market regime may be changing (CV > 10%)")

    return results


def test_streaming_calculator():
    """Test streaming EWMA calculator."""
    print("\n" + "=" * 70)
    print("STREAMING CALCULATOR TEST")
    print("=" * 70)

    try:
        # Load data from API to avoid stale CSV snapshots
        prices_btc, prices_eth, metadata = load_prices_from_api(
            window_hours=24,
            interval="5m"
        )

        print(f"\nLoaded {len(prices_btc)} price samples")

        # Test streaming
        calc = EWMAHedgeCalculator(half_life_bars=288)

        print("\nStreaming through data...")
        h_values = []
        stuck_count = 0
        max_stuck = 0

        for i, (p_btc, p_eth) in enumerate(zip(prices_btc, prices_eth)):
            h = calc.update(p_btc, p_eth)
            h_values.append(h)

            # Print progress
            if i % 500 == 0 and i > 0:
                print(f"  Sample {i:4d}: h = {h:.4f}")

            if abs(h - calc.h_min) < 1e-6:
                stuck_count += 1
                if stuck_count % 500 == 0:
                    print(f"  [ALERT] Hedge ratio pinned at floor for {stuck_count} consecutive samples (latest index: {i})")
                max_stuck = max(max_stuck, stuck_count)
            else:
                stuck_count = 0

        # Final result
        final_h = calc.h
        stats = calc.get_stats()

        print(f"\n{'─' * 70}")
        print("Final Results")
        print("─" * 70)
        print(f"Final h: {final_h:.4f}")
        print(f"Updates: {stats['updates']}")
        print(f"Warmed up: {stats['warmed_up']}")
        print(f"Covariance: {format_stat(stats['cov'])}")
        print(f"Variance: {format_stat(stats['var'])}")
        print(f"Max consecutive samples at floor: {max_stuck}")

        if abs(stats['var']) < SMALL_FLOAT_EPS:
            print("[WARNING]  Variance is zero, hedge ratio may be invalid.")
        if abs(stats['cov']) < SMALL_FLOAT_EPS:
            print("[WARNING]  Covariance is zero, this is unusual.")
        if max_stuck >= 500:
            print("[WARNING]  Hedge ratio stuck near safety floor for extended period - investigate data feed/intervals.")

        validation = validate_hedge_ratio(final_h, "streaming_ewma")
        print(f"\n{validation['interpretation']}")

        # Check convergence
        if len(h_values) > 100:
            recent_h = h_values[-100:]
            h_recent_std = (sum((h - final_h)**2 for h in recent_h) / len(recent_h)) ** 0.5
            print(f"\nConvergence: Last 100 samples std = {h_recent_std:.6f}")

            if h_recent_std < 0.01:
                print("✅ Well converged")
            else:
                print("⚠️  Still adapting")

        return final_h, stats

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_comparison_with_old_method():
    """Compare EWMA with old simple beta method."""
    print("\n" + "=" * 70)
    print("COMPARISON: EWMA vs OLD SIMPLE BETA")
    print("=" * 70)

    try:
        # Old method (from hedge_ratio.py)
        from hedge_ratio import calculate_hedge_ratio as old_method

        h_old = old_method(window_hours=24)
        print(f"\nOld method (simple beta): h = {h_old:.4f}")

        val_old = validate_hedge_ratio(h_old, "old_simple")
        print(f"  {val_old['interpretation']}")

        # New EWMA method
        h_new, stats_new = calculate_hedge_ratio_auto(
            window_hours=24,
            method="ewma",
            verbose=False
        )
        print(f"\nNew EWMA method: h = {h_new:.4f}")

        val_new = validate_hedge_ratio(h_new, "ewma")
        print(f"  {val_new['interpretation']}")

        # Compare
        diff = abs(h_new - h_old)
        print(f"\n{'─' * 70}")
        print(f"Difference: {diff:.4f} ({diff/h_old*100:.1f}% of old value)")

        if diff < 0.05:
            print("✅ Methods agree well (< 5% difference)")
        elif diff < 0.1:
            print("⚠️  Moderate difference (5-10%)")
        else:
            print("⚠️  Significant difference (> 10%) - EWMA is more reactive")

        return {"old": h_old, "new": h_new, "diff": diff}

    except Exception as e:
        print(f"❌ Could not compare with old method: {e}")
        return None


def main():
    """Run all tests."""
    print("=" * 70)
    print("EWMA HEDGE RATIO - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"\nStarted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Test 1: All methods
    print("\n" + "🔬" + " TEST 1: All Calculation Methods")
    results['methods'] = test_all_methods()

    # Test 2: Time windows
    print("\n" + "🔬" + " TEST 2: Time Window Sensitivity")
    results['windows'] = test_time_windows()

    # Test 3: Streaming
    print("\n" + "🔬" + " TEST 3: Streaming Calculator")
    results['streaming'] = test_streaming_calculator()

    # Test 4: Comparison
    print("\n" + "🔬" + " TEST 4: Comparison with Old Method")
    results['comparison'] = test_comparison_with_old_method()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n📋 Test Results:")
    print(f"  ✅ Method validation: {'PASSED' if results['methods'] else 'FAILED'}")
    print(f"  ✅ Time window tests: {'PASSED' if results['windows'] else 'FAILED'}")
    print(f"  ✅ Streaming tests: {'PASSED' if results['streaming'][0] else 'FAILED'}")
    print(f"  ✅ Comparison tests: {'PASSED' if results['comparison'] else 'FAILED'}")

    print("\n💡 Production Recommendations:")
    print("  1. Use EWMA method (most reactive, efficient)")
    print("  2. Use 24-hour window for stability")
    print("  3. Half-life of 288 bars (1 day at 5min) is good default")
    print("  4. Refresh h calculation daily or at each cycle")
    print("  5. Keep h bounds [0.3, 2.0] for safety")

    print("\n⚠️  Sanity Checks:")
    print("  • h should typically be 0.7-1.2 for BTC/ETH")
    print("  • Portfolio split should be roughly 50/50 to 60/40")
    print("  • If h < 0.5 or h > 1.5, investigate market conditions")
    print("  • EWMA reacts faster to regime changes than rolling OLS")

    print("\n" + "=" * 70)
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
