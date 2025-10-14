"""
Unit tests for main bot logic.

Tests bot initialization, position sizing, stop-loss, and lifecycle.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import HedgeBot


class TestHedgeBotInitialization:
    """Test bot initialization."""

    @pytest.fixture
    def config_file(self):
        """Create temporary config file."""
        temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        config = {
            "capital_pct": 95,
            "leverage": 4,
            "refresh_hours": 8,
            "stoploss_pct": 5,
            "slippage_bps": 10
        }
        json.dump(config, temp)
        temp.close()
        yield Path(temp.name)
        Path(temp.name).unlink()

    @patch('main.load_dotenv')
    @patch('main.os.getenv')
    @patch('main.ExtendedClient')
    def test_successful_initialization(self, mock_client_class, mock_getenv, mock_dotenv, config_file):
        """Test successful bot initialization."""
        # Mock environment variables
        mock_getenv.side_effect = lambda k: {
            "EXTENDED_ACCOUNT_ID": "account",
            "EXTENDED_API_KEY": "key",
            "EXTENDED_API_SECRET": "secret"
        }.get(k)

        # Mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        bot = HedgeBot(config_file)

        assert bot.config["capital_pct"] == 95
        assert bot.config["leverage"] == 4
        assert bot.running is True
        assert bot.stoploss_triggered is False
        assert bot.client == mock_client

    @patch('main.load_dotenv')
    @patch('main.os.getenv')
    def test_missing_environment_variables(self, mock_getenv, mock_dotenv, config_file):
        """Test initialization with missing env vars."""
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="Missing env vars"):
            HedgeBot(config_file)

    def test_invalid_config_file(self):
        """Test initialization with missing config file."""
        fake_path = Path("/nonexistent/config.json")

        with pytest.raises(FileNotFoundError):
            HedgeBot(fake_path)


class TestConfigAccess:
    """Test configuration access methods."""

    @pytest.fixture
    def bot(self):
        """Create mock bot."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                with patch('main.ExtendedClient'):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 80,
                        "leverage": 3,
                        "refresh_hours": 6,
                        "stoploss_pct": 4,
                        "slippage_bps": 15
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    yield bot

                    Path(temp.name).unlink()

    def test_get_existing_config(self, bot):
        """Test getting existing config value."""
        assert bot.get_config("capital_pct") == 80
        assert bot.get_config("leverage") == 3

    def test_get_nonexistent_config(self, bot):
        """Test getting non-existent config with default."""
        value = bot.get_config("nonexistent", default=42)
        assert value == 42

    def test_get_nonexistent_config_no_default(self, bot):
        """Test getting non-existent config without default."""
        value = bot.get_config("nonexistent")
        assert value is None


class TestHedgeRatioCalculation:
    """Test hedge ratio calculation and caching."""

    @pytest.fixture
    def bot(self):
        """Create bot with mocks."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                with patch('main.ExtendedClient'):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    yield bot

                    Path(temp.name).unlink()

    @patch('main.calculate_hedge_ratio_auto')
    def test_first_calculation(self, mock_calc, bot):
        """Test first hedge ratio calculation."""
        mock_calc.return_value = (0.75, {"status": "ok"})

        h = bot._get_hedge_ratio()

        assert h == 0.75
        assert bot._cached_h == 0.75
        mock_calc.assert_called_once()

    @patch('main.calculate_hedge_ratio_auto')
    @patch('main.time.time')
    def test_cache_hit(self, mock_time, mock_calc, bot):
        """Test that cached value is used within 1 hour."""
        # First call
        mock_time.return_value = 1000
        mock_calc.return_value = (0.75, {"status": "ok"})
        h1 = bot._get_hedge_ratio()

        # Second call 30 minutes later
        mock_time.return_value = 1000 + 1800  # 30 minutes
        h2 = bot._get_hedge_ratio()

        assert h1 == h2 == 0.75
        mock_calc.assert_called_once()  # Should only call once

    @patch('main.calculate_hedge_ratio_auto')
    @patch('main.time.time')
    def test_cache_expiry(self, mock_time, mock_calc, bot):
        """Test that cache expires after 1 hour."""
        # First call
        mock_time.return_value = 1000
        mock_calc.return_value = (0.75, {"status": "ok"})
        h1 = bot._get_hedge_ratio()

        # Second call 2 hours later
        mock_time.return_value = 1000 + 7200  # 2 hours
        mock_calc.return_value = (0.80, {"status": "ok"})
        h2 = bot._get_hedge_ratio()

        assert h1 == 0.75
        assert h2 == 0.80
        assert mock_calc.call_count == 2

    @patch('main.calculate_hedge_ratio_auto')
    def test_force_refresh(self, mock_calc, bot):
        """Test force refresh bypasses cache."""
        # First call
        mock_calc.return_value = (0.75, {"status": "ok"})
        h1 = bot._get_hedge_ratio()

        # Immediate second call with force
        mock_calc.return_value = (0.80, {"status": "ok"})
        h2 = bot._get_hedge_ratio(force_refresh=True)

        assert h1 == 0.75
        assert h2 == 0.80
        assert mock_calc.call_count == 2

    @patch('main.calculate_hedge_ratio_auto')
    def test_calculation_failure_with_cache(self, mock_calc, bot):
        """Test fallback to cache on calculation failure."""
        # First successful call
        mock_calc.return_value = (0.75, {"status": "ok"})
        h1 = bot._get_hedge_ratio()

        # Second call fails
        mock_calc.side_effect = Exception("Calculation failed")
        h2 = bot._get_hedge_ratio(force_refresh=True)

        assert h1 == h2 == 0.75  # Should use cached

    @patch('main.calculate_hedge_ratio_auto')
    def test_calculation_failure_no_cache(self, mock_calc, bot):
        """Test fallback to 0.85 when no cache exists."""
        mock_calc.side_effect = Exception("Calculation failed")

        h = bot._get_hedge_ratio()

        assert h == 0.85  # Fallback


class TestComputeTargets:
    """Test position sizing calculation."""

    @pytest.fixture
    def bot(self):
        """Create bot with mocks."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                mock_client = Mock()
                mock_client.get_equity.return_value = 1000.0
                mock_client.get_mark_price.side_effect = lambda s: {
                    "BTC": 100000.0,
                    "ETH": 3000.0
                }.get(s)
                mock_client.round_quantity.side_effect = lambda q, s: round(q, 4)
                mock_client.get_max_leverage.return_value = 20

                with patch('main.ExtendedClient', return_value=mock_client):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    bot._cached_h = 0.85  # Set hedge ratio
                    yield bot

                    Path(temp.name).unlink()

    def test_basic_sizing(self, bot):
        """Test basic position sizing calculation."""
        q_btc, q_eth = bot.compute_targets()

        # Equity = 1000
        # Capital = 1000 * 0.95 = 950
        # Gross = 950 * 4 = 3800
        # h = 0.85
        # N_btc = 3800 / (1 + 0.85) = 3800 / 1.85 ≈ 2054
        # N_eth = 0.85 * 2054 ≈ 1746

        assert q_btc > 0
        assert q_eth > 0

        # Check rough proportions
        btc_notional = q_btc * 100000
        eth_notional = q_eth * 3000
        total = btc_notional + eth_notional

        assert 3700 < total < 3900  # Should be close to gross notional

    def test_different_capital_pct(self, bot):
        """Test sizing with different capital percentage."""
        bot.config["capital_pct"] = 50  # Use 50% instead of 95%

        q_btc_50, q_eth_50 = bot.compute_targets()

        bot.config["capital_pct"] = 95
        q_btc_95, q_eth_95 = bot.compute_targets()

        # 50% should give smaller positions
        assert q_btc_50 < q_btc_95
        assert q_eth_50 < q_eth_95

    def test_different_leverage(self, bot):
        """Test sizing with different leverage."""
        bot.config["leverage"] = 2

        q_btc_2x, q_eth_2x = bot.compute_targets()

        bot.config["leverage"] = 10
        q_btc_10x, q_eth_10x = bot.compute_targets()

        # 10x should give larger positions
        assert q_btc_10x > q_btc_2x
        assert q_eth_10x > q_eth_2x

    def test_different_hedge_ratio(self, bot):
        """Test sizing with different hedge ratios."""
        bot._cached_h = 0.5  # BTC-heavy

        q_btc_low, q_eth_low = bot.compute_targets()

        bot._cached_h = 1.5  # ETH-heavy
        q_btc_high, q_eth_high = bot.compute_targets()

        # Lower h = more BTC, less ETH
        assert q_btc_low > q_btc_high
        assert q_eth_low < q_eth_high


class TestStoploss:
    """Test stop-loss checking."""

    @pytest.fixture
    def bot(self):
        """Create bot with mocks."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                mock_client = Mock()
                mock_client.get_lot_size.side_effect = lambda s: 0.001 if s == "BTC" else 0.01

                with patch('main.ExtendedClient', return_value=mock_client):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    yield bot

                    Path(temp.name).unlink()

    def test_no_stoploss_small_loss(self, bot):
        """Test that small loss doesn't trigger stop-loss."""
        bot.client.get_position.side_effect = lambda s: {
            "BTC": {
                "notional": 1000.0,
                "unrealized_pnl": -30.0  # -3% loss
            },
            "ETH": {
                "notional": 1000.0,
                "unrealized_pnl": -20.0  # -2% loss
            }
        }.get(s)

        triggered = bot.check_stoploss()

        assert triggered is False

    def test_stoploss_btc_leg(self, bot):
        """Test stop-loss trigger on BTC leg."""
        bot.client.get_position.side_effect = lambda s: {
            "BTC": {
                "notional": 1000.0,
                "unrealized_pnl": -60.0  # -6% loss (exceeds 5% threshold)
            },
            "ETH": {
                "notional": 1000.0,
                "unrealized_pnl": -20.0
            }
        }.get(s)

        triggered = bot.check_stoploss()

        assert triggered is True

    def test_stoploss_eth_leg(self, bot):
        """Test stop-loss trigger on ETH leg."""
        bot.client.get_position.side_effect = lambda s: {
            "BTC": {
                "notional": 1000.0,
                "unrealized_pnl": -20.0
            },
            "ETH": {
                "notional": 1000.0,
                "unrealized_pnl": -70.0  # -7% loss
            }
        }.get(s)

        triggered = bot.check_stoploss()

        assert triggered is True

    def test_no_stoploss_no_position(self, bot):
        """Test no stop-loss when no positions open."""
        bot.client.get_position.return_value = {
            "notional": 0.0,
            "unrealized_pnl": 0.0
        }

        triggered = bot.check_stoploss()

        assert triggered is False

    def test_stoploss_api_error(self, bot):
        """Test stop-loss check with API error."""
        bot.client.get_position.side_effect = Exception("API error")

        triggered = bot.check_stoploss()

        assert triggered is False  # Should handle gracefully


class TestMarketSimple:
    """Test simple market order execution."""

    @pytest.fixture
    def bot(self):
        """Create bot with mocks."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                mock_client = Mock()
                mock_client.get_mark_price.return_value = 100000.0
                mock_client.round_price.side_effect = lambda p, s: round(p, 1)
                mock_client.place_market_order.return_value = "order_123"

                with patch('main.ExtendedClient', return_value=mock_client):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    yield bot

                    Path(temp.name).unlink()

    def test_buy_order(self, bot):
        """Test buy order execution."""
        bot.market_simple("BTC", "buy", 0.5, reduce_only=False)

        bot.client.place_market_order.assert_called_once()

        call_args = bot.client.place_market_order.call_args
        assert call_args[1]['symbol'] == "BTC"
        assert call_args[1]['side'] == "buy"
        assert call_args[1]['quantity'] == 0.5
        assert call_args[1]['reduce_only'] is False

    def test_sell_order(self, bot):
        """Test sell order execution."""
        bot.market_simple("BTC", "sell", 0.5, reduce_only=True)

        bot.client.place_market_order.assert_called_once()

        call_args = bot.client.place_market_order.call_args
        assert call_args[1]['side'] == "sell"
        assert call_args[1]['reduce_only'] is True

    def test_zero_quantity_skip(self, bot):
        """Test that zero quantity is skipped."""
        bot.market_simple("BTC", "buy", 0.0)

        bot.client.place_market_order.assert_not_called()

    def test_order_error_handling(self, bot):
        """Test error handling during order placement."""
        bot.client.place_market_order.side_effect = Exception("Order failed")

        # Should not crash
        bot.market_simple("BTC", "buy", 0.5)


class TestPairOperations:
    """Test opening and closing pair positions."""

    @pytest.fixture
    def bot(self):
        """Create bot with fully mocked client."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                mock_client = Mock()
                mock_client.get_equity.return_value = 1000.0
                mock_client.get_mark_price.side_effect = lambda s: {
                    "BTC": 100000.0,
                    "ETH": 3000.0
                }.get(s)
                mock_client.round_quantity.side_effect = lambda q, s: round(q, 4)
                mock_client.get_max_leverage.return_value = 20
                mock_client.round_price.side_effect = lambda p, s: round(p, 2)
                mock_client.place_market_order.return_value = "order_123"

                with patch('main.ExtendedClient', return_value=mock_client):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    bot._cached_h = 0.85
                    bot._h_last_update = time.time()
                    yield bot

                    Path(temp.name).unlink()

    @patch('main.time.sleep')
    def test_open_pair(self, mock_sleep, bot):
        """Test opening BTC long and ETH short."""
        bot.open_pair()

        # Should have placed 2 orders (BTC and ETH)
        assert bot.client.place_market_order.call_count == 2
        assert bot.last_deploy_at is not None

        # Check BTC order
        first_call = bot.client.place_market_order.call_args_list[0]
        assert "BTC" in str(first_call)
        assert "buy" in str(first_call)

        # Check ETH order
        second_call = bot.client.place_market_order.call_args_list[1]
        assert "ETH" in str(second_call)
        assert "sell" in str(second_call)

    @patch('main.time.sleep')
    def test_close_pair(self, mock_sleep, bot):
        """Test closing both positions."""
        # Mock positions
        bot.client.get_position.side_effect = lambda s: {
            "BTC": {"qty": 0.5, "entry_price": 100000, "unrealized_pnl": 10, "notional": 50000},
            "ETH": {"qty": -10.0, "entry_price": 3000, "unrealized_pnl": -5, "notional": 30000}
        }.get(s)

        bot.close_pair()

        # Should have placed 2 reduce-only orders
        assert bot.client.place_market_order.call_count == 2

        # Both should be reduce_only
        for call in bot.client.place_market_order.call_args_list:
            assert call[1]['reduce_only'] is True


class TestSignalHandling:
    """Test signal handling for graceful shutdown."""

    @pytest.fixture
    def bot(self):
        """Create bot."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                with patch('main.ExtendedClient'):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    yield bot

                    Path(temp.name).unlink()

    def test_sigint_handler(self, bot):
        """Test SIGINT signal handler."""
        import signal

        assert bot.running is True

        bot._signal_handler(signal.SIGINT, None)

        assert bot.running is False

    def test_sigterm_handler(self, bot):
        """Test SIGTERM signal handler."""
        import signal

        assert bot.running is True

        bot._signal_handler(signal.SIGTERM, None)

        assert bot.running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

class TestBotStartup:
    """Test bot startup logic."""

    @pytest.fixture
    def bot(self):
        """Create bot with mocks."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                mock_client = Mock()
                mock_client.get_lot_size.side_effect = lambda s: 0.001 if s == "BTC" else 0.01

                with patch('main.ExtendedClient', return_value=mock_client):
                    with patch('main.HedgeBot._load_state') as mock_load_state:
                        with patch('main.HedgeBot._save_state') as mock_save_state:
                            temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                            json.dump({
                                "capital_pct": 95,
                                "leverage": 4,
                                "refresh_hours": 8,
                                "stoploss_pct": 5,
                                "slippage_bps": 10
                            }, temp)
                            temp.close()

                            bot = HedgeBot(Path(temp.name))
                            
                            # Mock methods that are not relevant for startup logic
                            bot.check_stoploss = Mock(return_value=False)
                            bot.close_pair = Mock()
                            bot.print_status = Mock()
                            bot.open_pair = Mock()
                            bot.reconcile_and_update_state = Mock()
                            bot.compute_targets = Mock(return_value=(0.1, 3.0))
                            
                            yield bot

                            Path(temp.name).unlink()

    def test_startup_with_existing_positions(self, bot):
        """Test that bot does not open new positions if positions exist."""
        # Mock existing positions
        opened_at = (datetime.now() - timedelta(hours=2)).timestamp()
        position_map = {
            "BTC": {
                "qty": 0.1,
                "entry_price": 50000,
                "unrealized_pnl": 0,
                "notional": 5100,
                "opened_at": opened_at
            },
            "ETH": {
                "qty": -3.0,
                "entry_price": 3000,
                "unrealized_pnl": 0,
                "notional": 9150,
                "opened_at": opened_at
            }
        }
        bot.client.get_position.side_effect = lambda s: position_map.get(s, {"qty": 0, "entry_price": 0, "unrealized_pnl": 0, "notional": 0, "opened_at": None})
        
        # Mock loaded state
        bot.saved_btc_qty = 0.1
        bot.saved_eth_qty = 3.0

        bot.running = False # To prevent infinite loop in run()
        bot.run()

        bot.open_pair.assert_not_called()
        bot.reconcile_and_update_state.assert_not_called()

    def test_startup_with_no_positions(self, bot):
        """Test that bot opens new positions if no positions exist."""
        # Mock no positions
        bot.client.get_position.return_value = {
            "qty": 0,
            "entry_price": 0,
            "unrealized_pnl": 0,
            "notional": 0,
            "opened_at": None
        }

        # Mock no state
        bot.saved_btc_qty = None
        bot.saved_eth_qty = None

        bot.running = False # To prevent infinite loop in run()
        bot.run()

        bot.open_pair.assert_called_once()

    def test_startup_with_aligned_positions_without_timestamp(self, bot):
        """Existing aligned positions without timestamp should assume current time."""
        fixed_now = datetime(2024, 1, 1, 12, 0, 0)
        position_map = {
            "BTC": {
                "qty": 0.1,
                "entry_price": 50000,
                "unrealized_pnl": 0,
                "notional": 5000,
                "opened_at": None
            },
            "ETH": {
                "qty": -3.0,
                "entry_price": 3000,
                "unrealized_pnl": 0,
                "notional": 9000,
                "opened_at": None
            }
        }
        bot.client.get_position.side_effect = lambda s: position_map.get(s)

        # Mock loaded state
        bot.saved_btc_qty = 0.1
        bot.saved_eth_qty = 3.0

        with patch('main.datetime') as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp

            bot.running = False
            bot.run()

        bot.open_pair.assert_not_called()
        bot.reconcile_and_update_state.assert_not_called()

    def test_startup_with_misaligned_positions_opens_pair(self, bot):
        """Mismatched quantities should trigger deployment."""
        position_map = {
            "BTC": {
                "qty": 0.02,  # way below target
                "entry_price": 50000,
                "unrealized_pnl": 0,
                "notional": 1000,
                "opened_at": None
            },
            "ETH": {
                "qty": -1.0,  # magnitude off target
                "entry_price": 3000,
                "unrealized_pnl": 0,
                "notional": 3000,
                "opened_at": None
            }
        }
        bot.client.get_position.side_effect = lambda s: position_map.get(s)

        bot.running = False
        bot.run()

        bot.reconcile_and_update_state.assert_called_once()
        bot.open_pair.assert_not_called()


class TestPositionMatching:
    """Unit tests for position matching helpers."""

    @pytest.fixture
    def bot(self):
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                mock_client = Mock()
                mock_client.get_lot_size.side_effect = lambda s: 0.001 if s == "BTC" else 0.01

                with patch('main.ExtendedClient', return_value=mock_client):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    yield bot

                    Path(temp.name).unlink()

    def test_positions_matching_with_tolerance(self, bot):
        """Allow small drift while considering positions matched."""
        pos_btc = {"qty": 0.101, "entry_price": 50000, "unrealized_pnl": 0, "notional": 5050}
        pos_eth = {"qty": -3.02, "entry_price": 3000, "unrealized_pnl": 0, "notional": 9060}
        assert bot._positions_match_targets(pos_btc, pos_eth, 0.1, 3.0)

    def test_positions_matching_rejects_wrong_direction(self, bot):
        """Reject matches if legs are not long BTC / short ETH."""
        pos_btc = {"qty": -0.1, "entry_price": 50000, "unrealized_pnl": 0, "notional": 5000}
        pos_eth = {"qty": -3.0, "entry_price": 3000, "unrealized_pnl": 0, "notional": 9000}
        assert not bot._positions_match_targets(pos_btc, pos_eth, 0.1, 3.0)

    def test_positions_matching_rejects_large_mismatch(self, bot):
        """Reject matches when quantities drift beyond tolerance."""
        pos_btc = {"qty": 0.15, "entry_price": 50000, "unrealized_pnl": 0, "notional": 7500}
        pos_eth = {"qty": -3.0, "entry_price": 3000, "unrealized_pnl": 0, "notional": 9000}
        assert not bot._positions_match_targets(pos_btc, pos_eth, 0.1, 3.0)

class TestFrozenTargets:
    """Test that target quantities are frozen between refreshes."""

    @pytest.fixture
    def bot(self):
        """Create bot with fully mocked client."""
        with patch('main.load_dotenv'):
            with patch('main.os.getenv') as mock_getenv:
                mock_getenv.side_effect = lambda k: {
                    "EXTENDED_ACCOUNT_ID": "account",
                    "EXTENDED_API_KEY": "key",
                    "EXTENDED_API_SECRET": "secret"
                }.get(k)

                mock_client = Mock()
                mock_client.get_equity.return_value = 1000.0
                mock_client.get_mark_price.side_effect = lambda s: {
                    "BTC": 100000.0,
                    "ETH": 3000.0
                }.get(s)
                mock_client.round_quantity.side_effect = lambda q, s: round(q, 4)
                mock_client.get_max_leverage.return_value = 20
                mock_client.round_price.side_effect = lambda p, s: round(p, 2)
                mock_client.place_market_order.return_value = "order_123"

                with patch('main.ExtendedClient', return_value=mock_client):
                    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                    json.dump({
                        "capital_pct": 95,
                        "leverage": 4,
                        "refresh_hours": 8,
                        "stoploss_pct": 5,
                        "slippage_bps": 10
                    }, temp)
                    temp.close()

                    bot = HedgeBot(Path(temp.name))
                    bot._cached_h = 0.85
                    bot._h_last_update = time.time()
                    yield bot

                    Path(temp.name).unlink()

    @patch('main.time.sleep')
    def test_open_pair_uses_saved_quantities(self, mock_sleep, bot):
        """Test that open_pair uses saved quantities if available."""
        bot.saved_btc_qty = 0.123
        bot.saved_eth_qty = 4.567

        bot.open_pair()

        # Check BTC order
        btc_call = bot.client.place_market_order.call_args_list[0]
        assert btc_call[1]['symbol'] == 'BTC'
        assert btc_call[1]['quantity'] == 0.123

        # Check ETH order
        eth_call = bot.client.place_market_order.call_args_list[1]
        assert eth_call[1]['symbol'] == 'ETH'
        assert eth_call[1]['quantity'] == 4.567

    @patch('main.time.sleep')
    def test_open_pair_uses_computed_targets_when_no_saved_quantities(self, mock_sleep, bot):
        """Test that open_pair computes targets if no saved quantities."""
        bot.saved_btc_qty = None
        bot.saved_eth_qty = None
        bot.compute_targets = Mock(return_value=(0.1, 3.0))

        bot.open_pair()

        bot.compute_targets.assert_called_once()

    @patch('main.time.sleep')
    def test_open_pair_uses_passed_targets(self, mock_sleep, bot):
        """Test that open_pair uses passed targets over saved ones."""
        bot.saved_btc_qty = 0.123
        bot.saved_eth_qty = 4.567

        bot.open_pair(targets=(0.987, 6.543))

        # Check BTC order
        btc_call = bot.client.place_market_order.call_args_list[0]
        assert btc_call[1]['symbol'] == 'BTC'
        assert btc_call[1]['quantity'] == 0.987

        # Check ETH order
        eth_call = bot.client.place_market_order.call_args_list[1]
        assert eth_call[1]['symbol'] == 'ETH'
        assert eth_call[1]['quantity'] == 6.543