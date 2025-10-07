"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path
import pytest

from .stub_solders import ensure_stub

ensure_stub()
from solders.keypair import Keypair

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT = PROJECT_ROOT.parent

# Ensure both the project package and its parent are importable
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Generate a valid test keypair once for all tests
TEST_KEYPAIR = Keypair()
TEST_PRIVATE_KEY = str(TEST_KEYPAIR)


@pytest.fixture(scope="session")
def sample_prices_btc():
    """Sample BTC price data for testing."""
    return [
        100000 + i * 10 + (i % 5) * 50
        for i in range(1000)
    ]


@pytest.fixture(scope="session")
def sample_prices_eth():
    """Sample ETH price data for testing."""
    return [
        3000 + i * 0.5 + (i % 3) * 5
        for i in range(1000)
    ]


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("SOL_WALLET", "test_sol_wallet")
    monkeypatch.setenv("API_PUBLIC", "test_api_public")
    monkeypatch.setenv("API_PRIVATE", TEST_PRIVATE_KEY)


@pytest.fixture(scope="session")
def test_keypair_private():
    """Valid test private key for all tests."""
    return TEST_PRIVATE_KEY


@pytest.fixture
def sample_config():
    """Sample bot configuration."""
    return {
        "capital_pct": 95,
        "leverage": 4,
        "refresh_hours": 8,
        "stoploss_pct": 5,
        "slippage_bps": 10
    }


@pytest.fixture
def sample_market_info():
    """Sample market information."""
    return {
        "BTC-PERP": {
            "tick_size": 0.1,
            "lot_size": 0.001,
            "min_notional": 10.0,
            "max_leverage": 20
        },
        "ETH-PERP": {
            "tick_size": 0.01,
            "lot_size": 0.01,
            "min_notional": 10.0,
            "max_leverage": 20
        }
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, no external deps)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (long-running)"
    )
