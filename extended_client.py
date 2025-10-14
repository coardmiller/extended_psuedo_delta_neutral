"""Extended Exchange client wrapper for hedge bot.

Handles all exchange interactions: positions, orders, market data.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests
from requests import HTTPError, RequestException

from extended_endpoints import get_rest_base_urls, join_url

logger = logging.getLogger(__name__)

@dataclass
class _MarketPrecision:
    """Precision configuration for a market."""

    tick_size: float
    lot_size: float
    min_notional: float
    max_leverage: int
    tick_size_dec: Decimal
    lot_size_dec: Decimal


class ExtendedClient:
    """Simplified Extended exchange client for hedge bot operations."""

    def __init__(
        self,
        account_id: str,
        api_key: str,
        api_secret: str,
        slippage_bps: int = 50,
        allow_fallback: bool = True,
    ) -> None:
        self.account_id = account_id
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.slippage_bps = slippage_bps
        self.allow_fallback = allow_fallback
        self.session = requests.Session()
        self.session.headers.update({"X-EXT-APIKEY": api_key})

        self._rest_base_urls = get_rest_base_urls()
        if not self._rest_base_urls:
            raise RuntimeError("No REST base URLs configured for Extended API")

        self._market_info: Dict[str, _MarketPrecision] = {}
        self._load_market_info()

    # ------------------------------------------------------------------
    # HTTP utilities
    # ------------------------------------------------------------------
    def _sign(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]],
        body: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Create authentication headers as defined by Extended API."""

        timestamp = str(int(time.time() * 1000))
        canonical_query = ""
        if params:
            canonical_query = urlencode(sorted((k, v) for k, v in params.items() if v is not None))
        payload = ""
        if body:
            payload = json.dumps(body, separators=(",", ":"), sort_keys=True)

        message = f"{timestamp}{method.upper()}{path}{canonical_query}{payload}".encode()
        signature = hmac.new(self.api_secret, message, hashlib.sha256).hexdigest()
        return timestamp, signature

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        private: bool = False,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """Perform an HTTP request to the Extended API."""

        headers: Dict[str, str] = {}
        if private:
            timestamp, signature = self._sign(method, path, params, json_body)
            headers.update(
                {
                    "X-EXT-APIKEY": self.api_key,
                    "X-EXT-TIMESTAMP": timestamp,
                    "X-EXT-SIGNATURE": signature,
                    "Content-Type": "application/json",
                }
            )

        last_exc: Optional[Exception] = None
        for base_url in self._rest_base_urls:
            url = join_url(base_url, path)
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("success") is False:
                    raise ValueError(payload.get("error", "Unknown API error"))
                # Promote the successful base to the front for future requests.
                if base_url != self._rest_base_urls[0]:
                    self._rest_base_urls.remove(base_url)
                    self._rest_base_urls.insert(0, base_url)
                return payload
            except HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                if status not in {404, 405}:
                    raise
            except RequestException as exc:  # pragma: no cover - network failure path
                last_exc = exc
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError("Extended API request failed without exception")

    # ------------------------------------------------------------------
    # Market metadata
    # ------------------------------------------------------------------
    def _load_market_info(self) -> None:
        """Load market information for BTC-PERP and ETH-PERP."""

        try:
            result = self._request_with_path_fallbacks(
                "GET",
                "/public/v1/markets",
                path_options=(
                    "/markets",
                    "/public/markets",
                    "/v1/markets",
                    "/api/public/v1/markets",
                    "/exchange/public/v1/markets",
                    "/api/exchange/public/v1/markets",
                ),
            )
            markets = result.get("data") or result.get("result") or []

            for market in markets:
                symbol = str(market.get("symbol") or market.get("name") or "").upper()
                if not symbol:
                    continue
                normalized = symbol.replace("/USDC", "").replace("-PERP", "")
                if normalized not in {"BTC", "ETH"}:
                    continue

                tick_size = self._extract_float(market, ("tick_size", "tickSize", "priceIncrement"), 0.1)
                lot_size = self._extract_float(market, ("lot_size", "lotSize", "sizeIncrement", "quantityIncrement"), 0.001)
                min_notional = self._extract_float(market, ("min_notional", "minNotional", "minOrderValue"), 10.0)
                leverage = int(self._extract_float(market, ("max_leverage", "maxLeverage"), 20))

                tick_dec = Decimal(str(round(tick_size, 12))) if tick_size > 0 else Decimal("0.000001")
                lot_dec = Decimal(str(lot_size)) if lot_size > 0 else Decimal("0.001")

                self._market_info[normalized] = _MarketPrecision(
                    tick_size=float(tick_dec),
                    lot_size=float(lot_dec),
                    min_notional=float(min_notional),
                    max_leverage=leverage,
                    tick_size_dec=tick_dec,
                    lot_size_dec=lot_dec,
                )

            if not self._market_info:
                raise ValueError("Failed to load BTC/ETH market info")
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Failed to load market info from Extended API: %s", exc)
            if not self.allow_fallback:
                raise RuntimeError("Failed to load market info from Extended API") from exc

            fallback_specs = {
                "BTC": _MarketPrecision(
                    tick_size=0.1,
                    lot_size=0.001,
                    min_notional=10.0,
                    max_leverage=20,
                    tick_size_dec=Decimal("0.1"),
                    lot_size_dec=Decimal("0.001"),
                ),
                "ETH": _MarketPrecision(
                    tick_size=0.01,
                    lot_size=0.01,
                    min_notional=10.0,
                    max_leverage=20,
                    tick_size_dec=Decimal("0.01"),
                    lot_size_dec=Decimal("0.01"),
                ),
            }
            self._market_info.update(fallback_specs)

    @staticmethod
    def _extract_float(data: Dict[str, Any], keys: Tuple[str, ...], default: float) -> float:
        for key in keys:
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    continue
        return float(default)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_symbol(symbol: str) -> str:
        symbol = symbol.upper()
        if symbol.endswith("-PERP"):
            return symbol
        return f"{symbol}-PERP"

    def _market_defaults(self) -> _MarketPrecision:
        return _MarketPrecision(
            tick_size=0.1,
            lot_size=0.001,
            min_notional=10.0,
            max_leverage=20,
            tick_size_dec=Decimal("0.1"),
            lot_size_dec=Decimal("0.001"),
        )

    def get_tick_size(self, symbol: str) -> float:
        return self._market_info.get(symbol, self._market_defaults()).tick_size

    def get_lot_size(self, symbol: str) -> float:
        return self._market_info.get(symbol, self._market_defaults()).lot_size

    def get_min_notional(self, symbol: str) -> float:
        return self._market_info.get(symbol, self._market_defaults()).min_notional

    def get_max_leverage(self, symbol: str) -> int:
        return self._market_info.get(symbol, self._market_defaults()).max_leverage

    def round_price(self, price: float, symbol: str, side: str = "bid") -> float:
        market = self._market_info.get(symbol)
        if not market:
            tick = self.get_tick_size(symbol)
            return round(price / tick) * tick

        tick_size_dec = market.tick_size_dec
        price_dec = Decimal(str(price))
        rounding_mode = ROUND_DOWN if side == "bid" else ROUND_UP
        try:
            rounded = price_dec.quantize(tick_size_dec, rounding=rounding_mode)
            return float(rounded)
        except Exception:  # pragma: no cover - fallback path
            tick = float(tick_size_dec)
            return round(price / tick) * tick

    def round_quantity(self, qty: float, symbol: str) -> float:
        market = self._market_info.get(symbol)
        if not market:
            lot = self.get_lot_size(symbol)
            return round(qty / lot) * lot

        lot_size_dec = market.lot_size_dec
        qty_dec = Decimal(str(qty))
        try:
            lots = (qty_dec / lot_size_dec).quantize(Decimal("1"), rounding=ROUND_DOWN)
            rounded = (lots * lot_size_dec).normalize()
            return float(rounded)
        except Exception:  # pragma: no cover - fallback path
            lot = float(lot_size_dec)
            return round(qty / lot) * lot

    # ------------------------------------------------------------------
    # Account data
    # ------------------------------------------------------------------
    def get_equity(self) -> float:
        try:
            payload = self._request(
                "GET",
                "/private/v1/account-summary",
                params={"accountId": self.account_id},
                private=True,
            )
            data = payload.get("data") or payload
            if isinstance(data, dict):
                balance = float(data.get("equity") or data.get("totalValue") or 0.0)
                if balance:
                    return balance
                cash = float(data.get("cashBalance") or data.get("balance") or 0.0)
                unrealized = float(data.get("unrealizedPnl") or data.get("unrealizedPnL") or 0.0)
                return cash + unrealized
            raise ValueError("Unexpected account payload")
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Failed to get equity from Extended API: %s", exc)
            if self.allow_fallback:
                return 10000.0
            raise RuntimeError("Failed to get equity from Extended API") from exc

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_mark_price(self, symbol: str) -> float:
        formatted = self._format_symbol(symbol)
        try:
            payload = self._request_with_path_fallbacks(
                "GET",
                "/public/v1/mark-price",
                params={"symbol": formatted},
                path_options=(
                    "/public/mark-price",
                    "/mark-price",
                    "/public/v1/markPrice",
                    "/markPrice",
                    "/api/public/v1/mark-price",
                ),
            )
            data = payload.get("data") or payload
            if isinstance(data, dict):
                price_fields = ("mark", "markPrice", "price")
                for field in price_fields:
                    if field in data and data[field] is not None:
                        price = float(data[field])
                        if price > 0:
                            return price
            if isinstance(data, list):
                for entry in data:
                    if entry.get("symbol") == formatted:
                        price = float(entry.get("mark") or entry.get("markPrice") or entry.get("price") or 0)
                        if price > 0:
                            return price
            raise ValueError("Mark price unavailable")
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Failed HTTP mark price for %s: %s", symbol, exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Failed to get mark price for {symbol}") from exc
            if symbol == "BTC":
                return 60000.0
            if symbol == "ETH":
                return 3000.0
            return 0.0

    # ------------------------------------------------------------------
    # Position data
    # ------------------------------------------------------------------
    def get_position(self, symbol: str) -> Dict[str, Any]:
        formatted = self._format_symbol(symbol)
        try:
            payload = self._request(
                "GET",
                "/private/v1/positions",
                params={"accountId": self.account_id, "symbol": formatted},
                private=True,
            )
            positions = payload.get("data") or payload.get("result") or []
            if isinstance(positions, dict):
                positions = [positions]

            for pos in positions:
                pos_symbol = str(pos.get("symbol") or "").upper()
                if pos_symbol != formatted:
                    continue

                qty = float(pos.get("size") or pos.get("amount") or 0.0)
                side = str(pos.get("side") or pos.get("positionSide") or "").lower()
                if side in {"sell", "short"}:
                    qty = -abs(qty)
                else:
                    qty = abs(qty)

                entry_price = float(pos.get("entryPrice") or pos.get("entry_price") or pos.get("avgPrice") or 0.0)
                mark_price = float(pos.get("markPrice") or pos.get("mark") or 0.0) or self.get_mark_price(symbol)
                unrealized = float(pos.get("unrealizedPnl") or pos.get("unrealized_pnl") or 0.0)
                if not unrealized and entry_price:
                    unrealized = (mark_price - entry_price) * qty

                opened_at = self._parse_position_timestamp(pos)
                return {
                    "qty": qty,
                    "entry_price": entry_price,
                    "unrealized_pnl": unrealized,
                    "notional": abs(qty * entry_price),
                    "opened_at": opened_at,
                }

            return {
                "qty": 0.0,
                "entry_price": 0.0,
                "unrealized_pnl": 0.0,
                "notional": 0.0,
                "opened_at": None,
            }
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Failed to get position for %s: %s", symbol, exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Failed to get position for {symbol}") from exc
            return {
                "qty": 0.0,
                "entry_price": 0.0,
                "unrealized_pnl": 0.0,
                "notional": 0.0,
                "opened_at": None,
            }

    @staticmethod
    def _parse_position_timestamp(pos: Dict[str, Any]) -> Optional[float]:
        timestamp_fields = (
            "opened_at",
            "open_time",
            "created_at",
            "updated_at",
            "timestamp",
            "createdAt",
            "updatedAt",
        )

        for field in timestamp_fields:
            if field not in pos:
                continue
            raw = pos.get(field)
            if raw is None:
                continue

            if isinstance(raw, (int, float)):
                value = float(raw)
                if value > 1e12:
                    return value / 1000.0
                if value > 1e10:
                    return value / 1e6
                return value

            if isinstance(raw, str):
                cleaned = raw.strip()
                if not cleaned:
                    continue
                try:
                    if cleaned.endswith("Z"):
                        cleaned = cleaned[:-1] + "+00:00"
                    dt = datetime.fromisoformat(cleaned)
                    return dt.timestamp()
                except Exception:
                    try:
                        value = float(cleaned)
                        if value > 1e12:
                            return value / 1000.0
                        if value > 1e10:
                            return value / 1e6
                        return value
                    except Exception:
                        continue
        return None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = True,
    ) -> str:
        formatted = self._format_symbol(symbol)

        market = self._market_info.get(symbol)
        if market:
            rounding_mode = ROUND_DOWN if side.lower() in {"bid", "buy"} else ROUND_UP
            price_dec = Decimal(str(price)).quantize(market.tick_size_dec, rounding=rounding_mode)
            qty_dec = (Decimal(str(quantity)) / market.lot_size_dec).quantize(Decimal("1"), rounding=ROUND_DOWN)
            amount_dec = qty_dec * market.lot_size_dec
            price_str = str(price_dec.normalize())
            amount_str = str(amount_dec.normalize())
        else:
            price_str = str(self.round_price(price, symbol, side="bid" if side.lower() in {"bid", "buy"} else "ask"))
            amount_str = str(self.round_quantity(quantity, symbol))

        client_order_id = str(uuid.uuid4())
        tif = "POST_ONLY" if post_only else "GTC"
        payload = {
            "accountId": self.account_id,
            "symbol": formatted,
            "side": "BUY" if side.lower() in {"bid", "buy"} else "SELL",
            "type": "LIMIT",
            "price": price_str,
            "size": amount_str,
            "timeInForce": tif,
            "reduceOnly": reduce_only,
            "clientOrderId": client_order_id,
        }

        try:
            response = self._request("POST", "/private/v1/orders", json_body=payload, private=True)
            data = response.get("data") or response
            order_id = data.get("orderId") or data.get("order_id")
            if not order_id:
                raise ValueError(f"Unexpected response: {response}")
            return str(order_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to place limit order: {exc}")

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        formatted = self._format_symbol(symbol)
        payload = {"accountId": self.account_id, "symbol": formatted, "orderId": order_id}
        try:
            self._request("POST", "/private/v1/orders/cancel", json_body=payload, private=True)
            return True
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Failed to cancel order %s: %s", order_id, exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Failed to cancel order {order_id}") from exc
            return False

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> str:
        formatted = self._format_symbol(symbol)
        rounded_quantity = self.round_quantity(quantity, symbol)
        client_order_id = str(uuid.uuid4())

        payload = {
            "accountId": self.account_id,
            "symbol": formatted,
            "side": "BUY" if side.lower() == "buy" else "SELL",
            "type": "MARKET",
            "size": str(rounded_quantity),
            "reduceOnly": reduce_only,
            "clientOrderId": client_order_id,
            "slippageBps": self.slippage_bps,
        }

        try:
            response = self._request("POST", "/private/v1/orders", json_body=payload, private=True)
            data = response.get("data") or response
            order_id = data.get("orderId") or data.get("order_id")
            if not order_id:
                raise ValueError(f"Unexpected response: {response}")
            return str(order_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to place market order: {exc}")

    def wait_fills_or_cancel(self, symbol: str, order_id: str, ttl_ms: int = 2500) -> None:
        time.sleep(ttl_ms / 1000.0)
        self.cancel_order(symbol, order_id)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"accountId": self.account_id}
        if symbol:
            payload["symbol"] = self._format_symbol(symbol)
        try:
            self._request("POST", "/private/v1/orders/cancel-all", json_body=payload, private=True)
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Failed to cancel all orders: %s", exc)
            if not self.allow_fallback:
                raise RuntimeError("Failed to cancel all orders") from exc

