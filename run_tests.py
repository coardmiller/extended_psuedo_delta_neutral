"""
Test runner script for hedge bot.

Runs all unit tests and generates coverage report.

Usage:
    python run_tests.py                 # Run all tests
    python run_tests.py --fast          # Run only fast unit tests
    python run_tests.py --coverage      # Run with coverage report
    python run_tests.py --verbose       # Verbose output
    python run_tests.py --failed        # Re-run only failed tests
"""
import sys
import subprocess
from pathlib import Path


def run_tests(args=None):
    """Run pytest with specified arguments."""
    if args is None:
        args = sys.argv[1:]

    # Base pytest command
    cmd = ["pytest", "tests/"]

    # Parse custom arguments
    if "--fast" in args:
        cmd.extend(["-m", "not slow"])
        args.remove("--fast")

    if "--coverage" in args:
        cmd.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
        args.remove("--coverage")

    if "--verbose" in args:
        cmd.append("-vv")
        args.remove("--verbose")

    if "--failed" in args:
        cmd.append("--lf")  # Last failed
        args.remove("--failed")

    # Add remaining args
    cmd.extend(args)

    print("=" * 70)
    print("RUNNING HEDGE BOT TESTS")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    print()

    # Run tests
    result = subprocess.run(cmd)

    print()
    print("=" * 70)
    if result.returncode == 0:
        print("[PASS] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("=" * 70)

    if "--coverage" in sys.argv:
        print()
        print("Coverage report generated: htmlcov/index.html")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
