"""
Setup verification script for hedge bot.
Run this before starting the bot to verify configuration.

Usage:
    python test_setup.py
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv


def test_config():
    """Test config.json exists and is valid."""
    print("1. Testing config.json...")

    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print("   ❌ config.json not found")
        return False

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        required_keys = ["capital_pct", "leverage", "refresh_hours", "stoploss_pct", "slippage_bps"]
        for key in required_keys:
            if key not in config:
                print(f"   ❌ Missing config key: {key}")
                return False

        print(f"   ✅ config.json valid")
        print(f"      - capital_pct: {config['capital_pct']}%")
        print(f"      - leverage: {config['leverage']}x")
        print(f"      - refresh_hours: {config['refresh_hours']}h")
        print(f"      - stoploss_pct: {config['stoploss_pct']}%")
        print(f"      - slippage_bps: {config['slippage_bps']} bps")

        return True

    except Exception as e:
        print(f"   ❌ Error reading config.json: {e}")
        return False


def test_env():
    """Test .env exists and has required variables."""
    print("\n2. Testing .env...")

    env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        print(f"   ❌ .env not found at {env_path}")
        return False

    load_dotenv(env_path)

    sol_wallet = os.getenv("SOL_WALLET")
    api_public = os.getenv("API_PUBLIC")
    api_private = os.getenv("API_PRIVATE")

    if not sol_wallet:
        print("   ❌ SOL_WALLET not set in .env")
        return False

    if not api_public:
        print("   ❌ API_PUBLIC not set in .env")
        return False

    if not api_private:
        print("   ❌ API_PRIVATE not set in .env")
        return False

    print("   ✅ .env valid")
    print(f"      - SOL_WALLET: {sol_wallet[:8]}...")
    print(f"      - API_PUBLIC: {api_public[:8]}...")
    print(f"      - API_PRIVATE: {'*' * 10}")

    return True


def test_market_data():
    """Test fetching recent klines from Pacifica API."""
    print("\n3. Testing Pacifica market data (klines)...")

    try:
        from ewma_hedge_ratio import load_prices_from_api

        prices_btc, prices_eth, metadata = load_prices_from_api(
            window_hours=1,
            interval="5m"
        )

        sample_count = metadata.get("samples", len(prices_btc))
        start_ts = metadata.get("start_timestamp")
        end_ts = metadata.get("end_timestamp")

        print("   ✅ Kline data fetched successfully")
        print(f"      - Samples: {sample_count}")
        if start_ts and end_ts:
            print(f"      - Window: {start_ts} → {end_ts}")

        return True

    except Exception as exc:
        print(f"   ❌ Failed to fetch klines: {exc}")
        print("      Ensure API credentials are valid and network access is available.")
        return False


def test_dependencies():
    """Test required Python packages are installed."""
    print("\n4. Testing dependencies...")

    required = [
        "requests",
        "websockets",
        "python-dotenv",
        "solders",
        "pandas",
        "numpy"
    ]

    missing = []

    for package in required:
        try:
            # Map package names to import names
            import_name = package
            if package == "python-dotenv":
                import_name = "dotenv"

            __import__(import_name)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"   ❌ Missing packages: {', '.join(missing)}")
        print(f"      Run: pip install {' '.join(missing)}")
        return False

    print("   ✅ All dependencies installed")
    return True


def test_connection():
    """Test connection to Pacifica API."""
    print("\n5. Testing Pacifica connection...")

    try:
        import requests
        from pacifica_sdk.common.constants import REST_URL

        # Add parent to path for SDK imports
        sys.path.insert(0, str(Path(__file__).parent.parent))

        response = requests.get(f"{REST_URL}/markets", timeout=10)
        response.raise_for_status()

        markets = response.json()
        btc_found = any(m.get("symbol") == "BTC-PERP" for m in markets)
        eth_found = any(m.get("symbol") == "ETH-PERP" for m in markets)

        if not btc_found or not eth_found:
            print("   ❌ BTC-PERP or ETH-PERP not found on exchange")
            return False

        print("   ✅ Pacifica API accessible")
        print(f"      - BTC-PERP: available")
        print(f"      - ETH-PERP: available")

        return True

    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HEDGE BOT SETUP VERIFICATION")
    print("=" * 60)

    tests = [
        test_config,
        test_env,
        test_market_data,
        test_dependencies,
        test_connection
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    critical_passed = all(results[:2]) and results[3] and results[4]  # config, env, deps, connection
    warnings = not results[2]  # data_dir

    if critical_passed:
        print("✅ All critical tests passed")
        if warnings:
            print("⚠️  Some warnings (non-critical)")
        print("\nYou can now run: python main.py")
    else:
        print("❌ Some critical tests failed")
        print("\nPlease fix the issues above before running the bot")

    print("=" * 60)

    return 0 if critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
