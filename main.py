"""Pseudo delta-neutral farming bot for Extended Exchange.

This script implements a local BTC/ETH hedge strategy designed to farm
volume on https://extended.exchange while attempting to remain close to
delta-neutral exposure.  The bot focuses on being data-driven and
resilient so it can be left running unattended:

* 📈 Goes long BTC and shorts ETH using a minimum-variance hedge ratio
  computed from the last 365 days of daily returns sourced from the
  exchange directly with an open CCXT (Binance/OKX/Kraken) fallback.
* 💾 Persists state (targets, hedge ratio, refresh schedule) in
  ``state/state.json`` so it can recover after a restart.
* 🧠 On start-up it reconciles the saved state with actual exchange
  positions and rebalances if they diverge.
* 🛡️ Enforces a configurable per-leg stop-loss.
* 🔌 Handles Ctrl+C / SIGTERM cleanly and keeps positions open on exit to
  avoid unintended liquidations.

Environment variables expected (e.g. via ``.env``):
    EXTENDED_ACCOUNT_ID, EXTENDED_API_KEY, EXTENDED_API_SECRET
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, init
from dotenv import load_dotenv

from extended_client import ExtendedClient
from market_data import fetch_recent_klines

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
LOG_FILE = Path(__file__).with_name("hedge_bot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("hedge_bot")

# Initialise colour output once
init(autoreset=True)


@dataclass
class BotState:
    """Serializable on-disk state for the bot."""

    hedge_ratio: Optional[float] = None
    btc_qty_target: float = 0.0
    eth_qty_target: float = 0.0
    last_deploy_at: Optional[str] = None
    next_refresh: Optional[str] = None
    stoploss_triggered: bool = False
    reference_account_value: Optional[float] = None

    @staticmethod
    def from_file(path: Path) -> Optional["BotState"]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return BotState(**payload)
        except Exception as exc:
            logger.error("Failed to load state from %s: %s", path, exc)
            return None

    def to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)


class HedgeBot:
    """BTC/ETH pseudo delta-neutral strategy manager."""

    BTC_SYMBOL = "BTC"
    ETH_SYMBOL = "ETH"

    def __init__(self, config_path: Path) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file missing: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = json.load(handle)
        logger.info("Loaded configuration: %s", self.config)

        load_dotenv(Path(__file__).with_name(".env"))
        account_id = os.getenv("EXTENDED_ACCOUNT_ID")
        api_key = os.getenv("EXTENDED_API_KEY")
        api_secret = os.getenv("EXTENDED_API_SECRET")
        if not all([account_id, api_key, api_secret]):
            raise RuntimeError(
                "Missing EXTENDED_ACCOUNT_ID / EXTENDED_API_KEY / EXTENDED_API_SECRET"
            )

        self.client = ExtendedClient(
            account_id=account_id,
            api_key=api_key,
            api_secret=api_secret,
            slippage_bps=int(self.get_config("slippage_bps", 50)),
            allow_fallback=True,
        )

        self.state_path = Path(__file__).parent / "state" / "state.json"
        self.state: BotState = BotState.from_file(self.state_path) or BotState()
        if self.state.hedge_ratio is not None:
            logger.info("Restored hedge ratio %.4f from state", self.state.hedge_ratio)

        self._cached_hedge_ratio: Optional[float] = self.state.hedge_ratio
        self._hedge_ratio_timestamp: float = 0.0

        self.last_deploy_at: Optional[datetime] = (
            datetime.fromisoformat(self.state.last_deploy_at)
            if self.state.last_deploy_at
            else None
        )
        self.next_refresh: Optional[datetime] = (
            datetime.fromisoformat(self.state.next_refresh)
            if self.state.next_refresh
            else None
        )
        self.stoploss_triggered: bool = self.state.stoploss_triggered
        self.reference_account_value: Optional[float] = self.state.reference_account_value

        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    # ------------------------------------------------------------------
    # Signal handling / persistence
    # ------------------------------------------------------------------
    def _signal_handler(self, signum: int, _frame: Any) -> None:
        logger.info("Received signal %s, preparing graceful shutdown.", signum)
        self.running = False

    def _persist_state(self) -> None:
        payload = BotState(
            hedge_ratio=self._cached_hedge_ratio,
            btc_qty_target=self.state.btc_qty_target,
            eth_qty_target=self.state.eth_qty_target,
            last_deploy_at=self.last_deploy_at.isoformat() if self.last_deploy_at else None,
            next_refresh=self.next_refresh.isoformat() if self.next_refresh else None,
            stoploss_triggered=self.stoploss_triggered,
            reference_account_value=self.reference_account_value,
        )
        self.state = payload
        payload.to_file(self.state_path)

    # ------------------------------------------------------------------
    # Hedge ratio logic
    # ------------------------------------------------------------------
    def _fetch_daily_prices(self, symbol: str, days: int) -> pd.Series:
        bars = max(days + 1, 2)
        df, source, metadata = fetch_recent_klines(symbol, "1d", bars=bars)
        if df.empty:
            extra = ", ".join(f"{k}={v}" for k, v in metadata.items())
            raise RuntimeError(f"No {symbol} price data available ({source}; {extra})")
        df = df.tail(days + 1)
        series = df.set_index("timestamp")["price"].astype(float)
        logger.info(
            "Fetched %d daily prices for %s via %s endpoint",
            len(series),
            symbol,
            source,
        )
        return series

    def _minimum_variance_hedge_ratio(self, force_refresh: bool = False) -> float:
        cache_ttl = 3600.0
        now = time.time()
        if (
            not force_refresh
            and self._cached_hedge_ratio is not None
            and now - self._hedge_ratio_timestamp < cache_ttl
        ):
            return float(self._cached_hedge_ratio)

        days = int(self.get_config("hedge_ratio_days", 365))
        fallback = float(self.get_config("fallback_hedge_ratio", 0.85))
        ratio_min = float(self.get_config("hedge_ratio_min", 0.3))
        ratio_max = float(self.get_config("hedge_ratio_max", 2.0))

        try:
            btc_prices = self._fetch_daily_prices(self.BTC_SYMBOL, days)
            eth_prices = self._fetch_daily_prices(self.ETH_SYMBOL, days)
            prices = pd.DataFrame({"BTC": btc_prices, "ETH": eth_prices}).dropna()
            if len(prices) < 60:
                raise ValueError("insufficient overlapping history")
            returns = np.log(prices).diff().dropna()
            if len(returns) < 30:
                raise ValueError("insufficient return samples")
            covariance = returns.cov()
            var_eth = covariance.loc["ETH", "ETH"]
            if var_eth <= 0:
                raise ValueError("ETH variance not positive")
            raw_ratio = covariance.loc["BTC", "ETH"] / var_eth
            clipped_ratio = float(np.clip(raw_ratio, ratio_min, ratio_max))
            self._cached_hedge_ratio = clipped_ratio
            self._hedge_ratio_timestamp = now
            logger.info(
                "Minimum-variance hedge ratio %.4f (raw=%.4f, samples=%d)",
                clipped_ratio,
                raw_ratio,
                len(returns),
            )
            return clipped_ratio
        except Exception as exc:
            logger.warning("Falling back to hedge ratio %.4f: %s", fallback, exc, exc_info=True)
            self._cached_hedge_ratio = fallback
            self._hedge_ratio_timestamp = now
            return fallback

    # ------------------------------------------------------------------
    # Target sizing
    # ------------------------------------------------------------------
    def compute_targets(self) -> Tuple[float, float]:
        equity = float(self.client.get_equity())
        if equity <= 0:
            raise RuntimeError("Equity reported as non-positive")
        logger.info("Account equity: $%.2f", equity)

        capital_pct = float(self.get_config("capital_pct", 95)) / 100.0
        leverage = float(self.get_config("leverage", 4))
        deployable = equity * capital_pct
        gross_notional = deployable * leverage
        logger.info(
            "Deploying %.1f%% of equity with %.1fx leverage (gross $%.2f)",
            capital_pct * 100,
            leverage,
            gross_notional,
        )

        hedge_ratio = self._minimum_variance_hedge_ratio()
        btc_mark = float(self.client.get_mark_price(self.BTC_SYMBOL))
        eth_mark = float(self.client.get_mark_price(self.ETH_SYMBOL))
        if btc_mark <= 0 or eth_mark <= 0:
            raise RuntimeError("Invalid mark price returned from exchange")
        logger.info("Mark prices BTC=$%.2f ETH=$%.2f", btc_mark, eth_mark)

        btc_notional = gross_notional / (1.0 + hedge_ratio)
        eth_notional = btc_notional * hedge_ratio
        btc_qty = btc_notional / btc_mark
        eth_qty = -(eth_notional / eth_mark)  # negative denotes a short

        btc_qty = float(self.client.round_quantity(btc_qty, self.BTC_SYMBOL))
        eth_qty = -float(self.client.round_quantity(abs(eth_qty), self.ETH_SYMBOL))
        logger.info("Target sizes BTC=%.4f ETH=%.4f (hedge ratio %.4f)", btc_qty, eth_qty, hedge_ratio)

        # Sanity-check margin requirements; scale down if needed
        max_lev_btc = float(self.client.get_max_leverage(self.BTC_SYMBOL) or 1)
        max_lev_eth = float(self.client.get_max_leverage(self.ETH_SYMBOL) or 1)
        required_margin = abs(btc_qty * btc_mark) / max_lev_btc + abs(eth_qty * eth_mark) / max_lev_eth
        if required_margin > equity:
            scale = equity / required_margin * 0.95
            logger.info(
                "Required margin $%.2f exceeds equity; scaling positions by %.2f%%",
                required_margin,
                scale * 100,
            )
            btc_qty = float(self.client.round_quantity(btc_qty * scale, self.BTC_SYMBOL))
            eth_qty = -float(
                self.client.round_quantity(abs(eth_qty) * scale, self.ETH_SYMBOL)
            )

        return btc_qty, eth_qty

    # ------------------------------------------------------------------
    # Exchange actions
    # ------------------------------------------------------------------
    def open_pair(self, targets: Optional[Tuple[float, float]] = None) -> None:
        btc_qty, eth_qty = targets or self.compute_targets()
        if btc_qty <= 0 or eth_qty >= 0:
            raise RuntimeError("Invalid target quantities for long/short pair")

        logger.info("Opening BTC long %.4f and ETH short %.4f", btc_qty, abs(eth_qty))
        btc_order = self.client.place_market_order(
            symbol=self.BTC_SYMBOL,
            side="buy",
            quantity=btc_qty,
            reduce_only=False,
        )
        logger.info("BTC market buy submitted (order %s)", btc_order)
        time.sleep(0.5)
        eth_order = self.client.place_market_order(
            symbol=self.ETH_SYMBOL,
            side="sell",
            quantity=abs(eth_qty),
            reduce_only=False,
        )
        logger.info("ETH market sell submitted (order %s)", eth_order)

        self.state.btc_qty_target = btc_qty
        self.state.eth_qty_target = eth_qty
        self.last_deploy_at = datetime.utcnow()
        refresh_hours = float(self.get_config("refresh_hours", 8))
        self.next_refresh = self.last_deploy_at + timedelta(hours=refresh_hours)
        self.stoploss_triggered = False
        if self.reference_account_value is None:
            self.reference_account_value = float(self.client.get_equity())
        self._persist_state()

    def close_pair(self) -> None:
        logger.info("Closing hedge pair positions")
        pos_btc = self.client.get_position(self.BTC_SYMBOL)
        pos_eth = self.client.get_position(self.ETH_SYMBOL)

        if pos_btc.get("qty", 0.0) > 0:
            order = self.client.place_market_order(
                symbol=self.BTC_SYMBOL,
                side="sell",
                quantity=pos_btc["qty"],
                reduce_only=True,
            )
            logger.info("Closed BTC long via order %s", order)
            time.sleep(0.5)

        if pos_eth.get("qty", 0.0) < 0:
            order = self.client.place_market_order(
                symbol=self.ETH_SYMBOL,
                side="buy",
                quantity=abs(pos_eth["qty"]),
                reduce_only=True,
            )
            logger.info("Closed ETH short via order %s", order)

        self.state.btc_qty_target = 0.0
        self.state.eth_qty_target = 0.0
        self._persist_state()

    def reconcile_positions(self) -> None:
        logger.info("Reconciling on-disk state with live positions")
        self.close_pair()
        time.sleep(1.0)
        self.open_pair()

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------
    def _positions_match_targets(
        self, pos_btc: Dict[str, Any], pos_eth: Dict[str, Any], target_btc: float, target_eth: float
    ) -> bool:
        tolerance_pct = float(self.get_config("position_tolerance_pct", 5)) / 100.0
        btc_lot = max(self.client.get_lot_size(self.BTC_SYMBOL), 1e-6)
        eth_lot = max(self.client.get_lot_size(self.ETH_SYMBOL), 1e-6)

        def _matches(actual: float, target: float, lot: float) -> bool:
            if abs(target) < lot:
                return abs(actual) < lot
            allowance = max(abs(target) * tolerance_pct, lot)
            return abs(actual - target) <= allowance

        btc_match = _matches(pos_btc.get("qty", 0.0), target_btc, btc_lot)
        eth_match = _matches(pos_eth.get("qty", 0.0), target_eth, eth_lot)
        logger.info(
            "Position check -> BTC match=%s (actual %.4f target %.4f) | ETH match=%s (actual %.4f target %.4f)",
            btc_match,
            pos_btc.get("qty", 0.0),
            target_btc,
            eth_match,
            pos_eth.get("qty", 0.0),
            target_eth,
        )
        return btc_match and eth_match

    def check_stoploss(self) -> bool:
        threshold = float(self.get_config("stoploss_pct", 10)) / 100.0
        if threshold <= 0:
            return False
        try:
            pos_btc = self.client.get_position(self.BTC_SYMBOL)
            pos_eth = self.client.get_position(self.ETH_SYMBOL)
        except Exception as exc:
            logger.error("Failed to fetch positions for stop-loss check: %s", exc)
            return False

        def _leg_hit(position: Dict[str, Any]) -> bool:
            notional = float(position.get("notional") or 0.0)
            pnl = float(position.get("unrealized_pnl") or 0.0)
            if notional <= 0:
                return False
            pct = pnl / notional
            return pct <= -threshold

        btc_hit = _leg_hit(pos_btc)
        eth_hit = _leg_hit(pos_eth)
        if btc_hit or eth_hit:
            leg = "BTC" if btc_hit else "ETH"
            logger.warning("Stop-loss triggered on %s leg", leg)
            return True
        return False

    def print_status(self) -> None:
        try:
            equity = float(self.client.get_equity())
            pos_btc = self.client.get_position(self.BTC_SYMBOL)
            pos_eth = self.client.get_position(self.ETH_SYMBOL)
        except Exception as exc:
            logger.error("Failed to fetch status: %s", exc)
            return

        total_pnl = float(pos_btc.get("unrealized_pnl", 0.0)) + float(
            pos_eth.get("unrealized_pnl", 0.0)
        )
        pnl_colour = Fore.GREEN if total_pnl >= 0 else Fore.RED
        logger.info(Fore.MAGENTA + "-" * 72)
        logger.info(
            "Equity $%.2f | Combined PnL %s$%.2f",
            equity,
            pnl_colour,
            total_pnl,
        )
        if self._cached_hedge_ratio is not None:
            h = float(self._cached_hedge_ratio)
            btc_pct = 100.0 / (1.0 + h)
            eth_pct = 100.0 * h / (1.0 + h)
            logger.info("Hedge ratio %.4f -> %.1f%% BTC / %.1f%% ETH", h, btc_pct, eth_pct)
        if self.reference_account_value:
            delta = equity - self.reference_account_value
            pct = delta / self.reference_account_value * 100.0
            colour = Fore.GREEN if delta >= 0 else Fore.RED
            logger.info(
                "Long-term PnL %s$%.2f (%.2f%%) vs reference $%.2f",
                colour,
                delta,
                pct,
                self.reference_account_value,
            )
        logger.info(
            "BTC qty %.4f entry $%.2f | ETH qty %.4f entry $%.2f",
            pos_btc.get("qty", 0.0),
            pos_btc.get("entry_price", 0.0),
            pos_eth.get("qty", 0.0),
            pos_eth.get("entry_price", 0.0),
        )
        if self.next_refresh:
            eta = self.next_refresh - datetime.utcnow()
            minutes = max(eta.total_seconds() / 60.0, 0.0)
            logger.info("Next refresh in %.1f minutes", minutes)
        if self.stoploss_triggered:
            logger.info(Fore.RED + "Stop-loss triggered; awaiting next refresh window")
        logger.info(Fore.MAGENTA + "-" * 72)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("Starting hedge bot")
        self._minimum_variance_hedge_ratio(force_refresh=True)
        try:
            self.client.cancel_all_orders()
        except Exception as exc:
            logger.warning("Failed to cancel resting orders: %s", exc)

        pos_btc = self.client.get_position(self.BTC_SYMBOL)
        pos_eth = self.client.get_position(self.ETH_SYMBOL)

        if self.state.btc_qty_target or self.state.eth_qty_target:
            logger.info("State targets detected; validating live positions")
            if not self._positions_match_targets(
                pos_btc, pos_eth, self.state.btc_qty_target, self.state.eth_qty_target
            ):
                logger.warning("Stored state and live positions diverge; reconciling")
                self.reconcile_positions()
        else:
            if pos_btc.get("qty", 0.0) or pos_eth.get("qty", 0.0):
                logger.warning("Live positions found without state; reconciling")
                self.reconcile_positions()
            else:
                logger.info("No open positions; deploying fresh pair")
                self.open_pair()

        status_interval = float(self.get_config("status_interval_seconds", 60))
        loop_sleep = float(self.get_config("loop_sleep_seconds", 15))
        last_status = 0.0

        while self.running:
            now = time.time()
            try:
                if not self.stoploss_triggered and self.check_stoploss():
                    logger.warning("Stop-loss hit -> closing pair")
                    self.close_pair()
                    self.stoploss_triggered = True
                    refresh_hours = float(self.get_config("refresh_hours", 8))
                    self.next_refresh = datetime.utcnow() + timedelta(hours=refresh_hours)
                    self._persist_state()

                if self.next_refresh and datetime.utcnow() >= self.next_refresh:
                    logger.info("Refresh window reached; recycling positions")
                    self.close_pair()
                    time.sleep(1.0)
                    self._minimum_variance_hedge_ratio(force_refresh=True)
                    if not self.stoploss_triggered:
                        self.open_pair()
                    else:
                        logger.info(
                            "Stop-loss active; delaying new deployment until next cycle"
                        )
                        self.next_refresh = datetime.utcnow() + timedelta(
                            hours=float(self.get_config("refresh_hours", 8))
                        )
                        self._persist_state()

                if now - last_status >= status_interval:
                    self.print_status()
                    last_status = now
            except Exception as exc:
                logger.error("Error in main loop: %s", exc, exc_info=True)
            time.sleep(loop_sleep)

        logger.info("Exiting main loop; leaving positions open by design")
        self._persist_state()


def main() -> None:
    config_path = Path(__file__).with_name("config.json")
    bot = HedgeBot(config_path)
    bot.run()


if __name__ == "__main__":
    main()
