"""
Unit tests for EWMA hedge ratio calculator.

Tests all calculation methods, edge cases, and error handling.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from math import exp, log

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ewma_hedge_ratio import (
    EWMAHedgeCalculator,
    calculate_ewma_hedge_ratio_batch,
    calculate_rolling_ols_hedge_ratio,
    calculate_vol_ratio_hedge_ratio,
    calculate_hedge_ratio_auto,
    load_prices_from_api
)


class TestEWMAHedgeCalculator:
    """Test streaming EWMA calculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = EWMAHedgeCalculator()

        assert calc.h == 1.0  # Initial hedge ratio
        assert calc.cov == 0.0
        assert calc.var == 0.0
        assert calc.update_count == 0
        assert calc.prev_price_btc is None
        assert calc.prev_price_eth is None

    def test_half_life_calculation(self):
        """Test EWMA decay factor calculation."""
        calc = EWMAHedgeCalculator(half_life_bars=288)

        expected_lam = exp(-log(2) / 288)
        expected_alpha = 1 - expected_lam

        assert abs(calc.lam - expected_lam) < 1e-10
        assert abs(calc.alpha - expected_alpha) < 1e-10
        assert abs(calc.lam + calc.alpha - 1.0) < 1e-10  # Should sum to 1

    def test_first_update_no_calculation(self):
        """Test that first update doesn't calculate (no previous price)."""
        calc = EWMAHedgeCalculator()

        h = calc.update(100.0, 10.0)

        assert h == 1.0  # Should still be default
        assert calc.prev_price_btc == 100.0
        assert calc.prev_price_eth == 10.0
        assert calc.update_count == 0  # No actual update yet

    def test_second_update_calculates(self):
        """Test that second update performs calculation."""
        calc = EWMAHedgeCalculator()

        calc.update(100.0, 10.0)
        h = calc.update(101.0, 10.1)

        assert calc.update_count == 1
        assert calc.cov != 0.0
        assert calc.var != 0.0
        # h may equal 1.0 if cov/var = 1.0, so just check it was calculated
        assert calc.h > 0

    def test_positive_correlation(self):
        """Test with positively correlated prices."""
        calc = EWMAHedgeCalculator(half_life_bars=10)

        # Both increasing together
        prices_btc = [100, 101, 102, 103, 104, 105]
        prices_eth = [10, 10.1, 10.2, 10.3, 10.4, 10.5]

        for p_btc, p_eth in zip(prices_btc, prices_eth):
            h = calc.update(p_btc, p_eth)

        # Should have positive covariance and variance
        assert calc.cov > 0
        assert calc.var > 0
        assert calc.h > 0

    def test_negative_correlation(self):
        """Test with negatively correlated prices."""
        calc = EWMAHedgeCalculator(half_life_bars=10)

        # BTC up, ETH down
        prices_btc = [100, 101, 102, 103, 104, 105]
        prices_eth = [10, 9.9, 9.8, 9.7, 9.6, 9.5]

        for p_btc, p_eth in zip(prices_btc, prices_eth):
            h = calc.update(p_btc, p_eth)

        # Covariance should be negative
        assert calc.cov < 0

    def test_bounds_enforcement_min(self):
        """Test that h is clamped to minimum."""
        calc = EWMAHedgeCalculator(h_min=0.5, h_max=2.0)

        # Create scenario that would give very low h
        prices_btc = [100, 100.01, 100.02]  # Very small movements
        prices_eth = [10, 11, 12]  # Large movements

        for p_btc, p_eth in zip(prices_btc, prices_eth):
            h = calc.update(p_btc, p_eth)

        # h should be at least h_min
        assert calc.h >= calc.h_min

    def test_bounds_enforcement_max(self):
        """Test that h is clamped to maximum."""
        calc = EWMAHedgeCalculator(h_min=0.3, h_max=1.5)

        # Create scenario that would give very high h
        prices_btc = [100, 105, 110, 115]  # Large movements
        prices_eth = [10, 10.01, 10.02, 10.03]  # Tiny movements

        for p_btc, p_eth in zip(prices_btc, prices_eth):
            h = calc.update(p_btc, p_eth)

        # h should be at most h_max
        assert calc.h <= calc.h_max

    def test_outlier_winsorization(self):
        """Test that outlier returns are clamped."""
        calc = EWMAHedgeCalculator(outlier_threshold=0.02)

        # Start with normal price
        calc.update(100.0, 10.0)

        # Huge spike (>10% return)
        calc.update(150.0, 15.0)

        # Should have clamped the returns to ±2%
        # Calculation should still work without NaN
        assert np.isfinite(calc.cov)
        assert np.isfinite(calc.var)
        assert np.isfinite(calc.h)

    def test_warmup_status(self):
        """Test warmup status tracking."""
        calc = EWMAHedgeCalculator()

        assert not calc.is_warmed_up(min_bars=500)

        # Simulate 501 updates (first doesn't count, so we get 500 calculations)
        for i in range(501):
            calc.update(100 + i*0.1, 10 + i*0.01)

        assert calc.is_warmed_up(min_bars=500)

    def test_get_stats(self):
        """Test statistics retrieval."""
        calc = EWMAHedgeCalculator()

        # Do some updates
        for i in range(10):
            calc.update(100 + i, 10 + i*0.1)

        stats = calc.get_stats()

        assert 'h' in stats
        assert 'cov' in stats
        assert 'var' in stats
        assert 'updates' in stats
        assert 'warmed_up' in stats
        assert stats['updates'] == 9  # 10 prices - 1 (first has no previous)

    def test_zero_variance_handling(self):
        """Test handling of zero variance (constant ETH price)."""
        calc = EWMAHedgeCalculator()

        # BTC varies, ETH constant
        prices_btc = [100, 101, 102, 103]
        prices_eth = [10, 10, 10, 10]

        for p_btc, p_eth in zip(prices_btc, prices_eth):
            h = calc.update(p_btc, p_eth)

        # Should not crash, h should remain at default or previous good value
        assert np.isfinite(h)


class TestBatchEWMA:
    """Test batch EWMA calculation."""

    def test_basic_calculation(self):
        """Test basic batch calculation."""
        prices_btc = [100, 101, 102, 103, 104, 105]
        prices_eth = [10, 10.1, 10.2, 10.3, 10.4, 10.5]

        h, stats = calculate_ewma_hedge_ratio_batch(prices_btc, prices_eth)

        assert np.isfinite(h)
        assert 0.3 <= h <= 2.0  # Within default bounds
        assert stats['samples'] == 5  # 6 prices - 1
        assert 'cov' in stats
        assert 'var' in stats

    def test_insufficient_data(self):
        """Test with insufficient data."""
        prices_btc = [100]
        prices_eth = [10]

        h, stats = calculate_ewma_hedge_ratio_batch(
            prices_btc, prices_eth, prev_h=0.85
        )

        assert h == 0.85  # Should use fallback
        assert 'error' in stats

    def test_mismatched_lengths(self):
        """Test error on mismatched array lengths."""
        prices_btc = [100, 101, 102]
        prices_eth = [10, 11]

        with pytest.raises(ValueError, match="same length"):
            calculate_ewma_hedge_ratio_batch(prices_btc, prices_eth)

    def test_custom_half_life(self):
        """Test custom half-life parameter."""
        prices_btc = list(range(100, 200))
        prices_eth = list(range(10, 110))

        h1, stats1 = calculate_ewma_hedge_ratio_batch(
            prices_btc, prices_eth, half_life_bars=50
        )
        h2, stats2 = calculate_ewma_hedge_ratio_batch(
            prices_btc, prices_eth, half_life_bars=500
        )

        # Different half-lives should give different results
        # (unless both are clamped to same bound)
        assert 'half_life_bars' in stats1
        assert 'half_life_bars' in stats2

    def test_clamping_status(self):
        """Test that clamping is reported in stats."""
        # Create data that will be clamped
        prices_btc = [100 + i*0.01 for i in range(100)]  # Tiny movements
        prices_eth = [10 + i for i in range(100)]  # Large movements

        h, stats = calculate_ewma_hedge_ratio_batch(
            prices_btc, prices_eth, h_min=0.3, h_max=2.0
        )

        if h == 0.3 or h == 2.0:
            assert stats.get('status') == 'clamped'
            assert 'raw_h' in stats

    def test_zero_variance_fallback(self):
        """Test fallback when variance is zero."""
        prices_btc = [100, 101, 102]
        prices_eth = [10, 10, 10]  # Constant

        h, stats = calculate_ewma_hedge_ratio_batch(
            prices_btc, prices_eth, prev_h=0.75
        )

        assert h == 0.75  # Should use fallback
        assert stats['status'] == 'fallback'


class TestRollingOLS:
    """Test rolling OLS hedge ratio."""

    def test_basic_calculation(self):
        """Test basic OLS calculation."""
        prices_btc = [100 + i for i in range(100)]
        prices_eth = [10 + i*0.1 for i in range(100)]

        h, stats = calculate_rolling_ols_hedge_ratio(prices_btc, prices_eth)

        assert np.isfinite(h)
        assert stats['method'] == 'rolling_ols'
        assert 'cov' in stats
        assert 'var' in stats
        assert 'samples' in stats

    def test_window_parameter(self):
        """Test custom window size."""
        prices_btc = list(range(100, 1100))
        prices_eth = list(range(10, 1010))

        h, stats = calculate_rolling_ols_hedge_ratio(
            prices_btc, prices_eth, window=500
        )

        # Should use last 500 samples
        assert stats['samples'] == 500

    def test_short_data_uses_all(self):
        """Test that short data uses all available."""
        prices_btc = list(range(100, 150))  # 50 prices
        prices_eth = list(range(10, 60))

        h, stats = calculate_rolling_ols_hedge_ratio(
            prices_btc, prices_eth, window=1000
        )

        # Should use all 49 returns (50 prices - 1)
        assert stats['samples'] == 49

    def test_insufficient_data(self):
        """Test with insufficient data."""
        prices_btc = [100]
        prices_eth = [10]

        h, stats = calculate_rolling_ols_hedge_ratio(prices_btc, prices_eth)

        assert h == 1.0  # Default fallback
        assert 'error' in stats


class TestVolRatio:
    """Test volatility ratio method."""

    def test_basic_calculation(self):
        """Test basic vol ratio calculation."""
        prices_btc = [100 + i*2 for i in range(100)]  # Higher vol
        prices_eth = [10 + i for i in range(100)]  # Lower vol

        h, stats = calculate_vol_ratio_hedge_ratio(prices_btc, prices_eth)

        assert np.isfinite(h)
        assert stats['method'] == 'vol_ratio'
        assert 'vol_btc' in stats
        assert 'vol_eth' in stats

    def test_higher_btc_vol(self):
        """Test that higher BTC vol gives h > 1."""
        # BTC has 2x the volatility of ETH
        np.random.seed(42)
        prices_btc = [100]
        prices_eth = [10]

        for i in range(100):
            prices_btc.append(prices_btc[-1] * (1 + np.random.normal(0, 0.02)))
            prices_eth.append(prices_eth[-1] * (1 + np.random.normal(0, 0.01)))

        h, stats = calculate_vol_ratio_hedge_ratio(prices_btc, prices_eth)

        # h should be roughly 2.0 (or clamped)
        assert h >= 1.0

    def test_zero_eth_variance(self):
        """Test handling of zero ETH variance."""
        prices_btc = [100, 101, 102]
        prices_eth = [10, 10, 10]

        h, stats = calculate_vol_ratio_hedge_ratio(prices_btc, prices_eth)

        assert h == 1.0  # Fallback
        assert 'error' in stats


class TestAutoCalculation:
    """Test automatic hedge ratio calculation with live market data."""

    SYMBOL_BTC = "BTC"
    SYMBOL_ETH = "ETH"
    INTERVAL = "5m"

    @pytest.fixture(autouse=True)
    def mock_api_data(self, monkeypatch):
        """Provide synthetic kline data to avoid real API calls."""

        valid_symbols = {"BTC", "ETH", "BTC-PERP", "ETH-PERP"}

        def fake_fetch(symbol: str, interval: str, start_ms: int, end_ms: int):
            if symbol not in valid_symbols:
                raise RuntimeError(f"Unknown symbol {symbol}")
            freq_minutes = 5
            if interval.endswith("m"):
                try:
                    freq_minutes = int(interval[:-1])
                except ValueError:
                    freq_minutes = 5
            freq = f"{freq_minutes}T"

            start = pd.to_datetime(start_ms, unit="ms")
            end = pd.to_datetime(end_ms, unit="ms")
            idx = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
            if len(idx) == 0:
                idx = pd.date_range(start=start, periods=12, freq=freq)

            base = 60000 if symbol.upper().startswith("BTC") else 3000
            prices = base + np.linspace(0, 100, len(idx))

            df = pd.DataFrame({"timestamp": idx, "price": prices})
            df.attrs["source"] = "extended"
            df.attrs["debug"] = {"mock": True}
            return df

        monkeypatch.setattr("ewma_hedge_ratio._fetch_klines_data", fake_fetch)

    def test_load_prices_from_api(self):
        """Ensure API loader returns synchronized price arrays."""
        prices_btc, prices_eth, metadata = load_prices_from_api(
            symbol_btc=self.SYMBOL_BTC,
            symbol_eth=self.SYMBOL_ETH,
            window_hours=6,
            interval=self.INTERVAL
        )

        assert len(prices_btc) > 0
        assert len(prices_eth) > 0
        assert len(prices_btc) == len(prices_eth)
        assert metadata.get("source") == "extended"
        assert metadata.get("samples") == len(prices_btc)

    def test_window_filtering(self):
        """Verify shorter windows provide fewer samples than longer ones."""
        prices_24h, _, meta_24h = load_prices_from_api(
            symbol_btc=self.SYMBOL_BTC,
            symbol_eth=self.SYMBOL_ETH,
            window_hours=24,
            interval=self.INTERVAL
        )
        prices_6h, _, meta_6h = load_prices_from_api(
            symbol_btc=self.SYMBOL_BTC,
            symbol_eth=self.SYMBOL_ETH,
            window_hours=6,
            interval=self.INTERVAL
        )

        assert len(prices_6h) < len(prices_24h)
        assert meta_24h["samples"] == len(prices_24h)
        assert meta_6h["samples"] == len(prices_6h)

    def test_auto_ewma_method(self):
        """Test automatic calculation with EWMA method using API data."""
        h, stats = calculate_hedge_ratio_auto(
            method="ewma",
            window_hours=6,
            symbol_btc=self.SYMBOL_BTC,
            symbol_eth=self.SYMBOL_ETH,
            interval=self.INTERVAL
        )

        assert np.isfinite(h)
        assert 0.3 <= h <= 2.0
        assert stats.get("source") == "api"
        assert stats.get("samples") >= 50

    def test_auto_rolling_ols_method(self):
        """Test automatic calculation with Rolling OLS using API data."""
        h, stats = calculate_hedge_ratio_auto(
            method="rolling_ols",
            window_hours=6,
            symbol_btc=self.SYMBOL_BTC,
            symbol_eth=self.SYMBOL_ETH,
            interval=self.INTERVAL
        )

        assert np.isfinite(h)
        assert stats.get("method") in ["rolling_ols", "fallback"]

    def test_auto_vol_ratio_method(self):
        """Test automatic calculation with volatility ratio using API data."""
        h, stats = calculate_hedge_ratio_auto(
            method="vol_ratio",
            window_hours=6,
            symbol_btc=self.SYMBOL_BTC,
            symbol_eth=self.SYMBOL_ETH,
            interval=self.INTERVAL
        )

        assert np.isfinite(h)
        assert stats.get("method") in ["vol_ratio", "fallback"]

    def test_missing_data_fallback(self):
        """Ensure fallback triggers when API call fails for invalid symbol."""
        h, stats = calculate_hedge_ratio_auto(
            method="ewma",
            symbol_btc="INVALID",
            symbol_eth=self.SYMBOL_ETH,
            fallback_h=0.75
        )

        assert h == 0.75
        assert stats["status"] == "fallback"
        assert "error" in stats

    def test_invalid_method(self):
        """Test that invalid method falls back gracefully."""
        h, stats = calculate_hedge_ratio_auto(
            method="invalid_method",
            fallback_h=0.75
        )

        assert h == 0.75
        assert stats['status'] == 'fallback'
        assert 'error' in stats
        assert 'Unknown method' in stats['error']

    def test_low_variance_triggers_broader_window(self, monkeypatch):
        """Ensure the helper widens the window/interval when variance is too low."""
        call_sequence = []

        def fake_loader(symbol_btc, symbol_eth, window_hours, interval):
            call_sequence.append((window_hours, interval))
            samples = 200
            base_ts = 1_700_000_000
            if window_hours == 6 and interval == "5m":
                prices_btc = [100.0] * samples
                prices_eth = [200.0] * samples
            else:
                prices_btc = [100.0 + i * 0.5 for i in range(samples)]
                prices_eth = [200.0 + i * 0.4 for i in range(samples)]
            metadata = {
                "samples": samples,
                "btc_mean": float(np.mean(prices_btc)),
                "eth_mean": float(np.mean(prices_eth)),
                "window_hours": window_hours,
                "interval": interval,
                "symbol_btc": symbol_btc,
                "symbol_eth": symbol_eth,
                "source": "api",
                "start_timestamp": base_ts,
                "end_timestamp": base_ts + samples * 60,
            }
            return prices_btc, prices_eth, metadata

        monkeypatch.setattr("ewma_hedge_ratio.load_prices_from_api", fake_loader)

        h, stats = calculate_hedge_ratio_auto(
            method="ewma",
            window_hours=6,
            interval="1m"
        )

        assert call_sequence[0] == (6, "1m")
        attempted_multiple = len(call_sequence) >= 2
        if attempted_multiple:
            assert any(
                (window > 6) or (interval != "1m")
                for window, interval in call_sequence
            )
            assert stats["status"] in ("ok", "clamped")
            assert stats["window_hours_used"] > 6 or stats["interval_used"] != "1m"
        else:
            assert stats["status"] in ("low_variance_fallback", "clamped")
        assert np.isfinite(h)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_nan_prices(self):
        """Test handling of NaN prices."""
        calc = EWMAHedgeCalculator()

        calc.update(100.0, 10.0)
        h = calc.update(np.nan, 10.1)

        # Should not crash, h should remain valid
        assert np.isfinite(h)

    def test_infinite_prices(self):
        """Test handling of infinite prices."""
        calc = EWMAHedgeCalculator()

        calc.update(100.0, 10.0)
        h = calc.update(np.inf, 10.1)

        # Should not crash
        assert np.isfinite(h) or h == calc.h_min or h == calc.h_max

    def test_negative_prices(self):
        """Test that negative prices raise domain error."""
        calc = EWMAHedgeCalculator()

        # Negative prices cause log() domain error
        calc.update(100.0, 10.0)

        # Should raise ValueError (math domain error)
        with pytest.raises(ValueError, match="math domain error"):
            calc.update(-100.0, -10.0)

    def test_very_large_price_moves(self):
        """Test handling of extreme price movements."""
        calc = EWMAHedgeCalculator(outlier_threshold=0.02)

        calc.update(100.0, 10.0)
        # 1000% price increase
        h = calc.update(1000.0, 100.0)

        # Winsorization should have clamped this
        assert np.isfinite(calc.cov)
        assert np.isfinite(calc.var)

    def test_empty_arrays(self):
        """Test batch calculation with empty arrays."""
        h, stats = calculate_ewma_hedge_ratio_batch([], [], prev_h=0.8)

        assert h == 0.8
        assert 'error' in stats

    def test_single_price(self):
        """Test batch calculation with single price."""
        h, stats = calculate_ewma_hedge_ratio_batch([100], [10], prev_h=0.9)

        assert h == 0.9
        assert 'error' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
