"""Shared market data helpers with Extended-first, Pacifica-fallback klines."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests

from extended_endpoints import iter_candidate_urls, join_url

logger = logging.getLogger(__name__)

# Supported kline intervals mapped to seconds (shared across modules)
INTERVAL_SECONDS: Dict[str, int] = {
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

# Extended REST discovery mirrors the exhaustive list already used by the bot
_EXTENDED_KLINE_PATH_OPTIONS: Tuple[str, ...] = (
    "public/v1/klines",
    "public/v1/kline",
    "public/klines",
    "public/kline",
    "v1/public/klines",
    "v1/public/kline",
    "v1/klines",
    "v1/kline",
    "api/public/v1/klines",
    "api/public/v1/kline",
    "api/public/klines",
    "api/public/kline",
    "api/v1/public/klines",
    "api/v1/public/kline",
    "api/v1/klines",
    "api/v1/kline",
    "exchange/public/v1/klines",
    "exchange/public/v1/kline",
    "exchange/public/klines",
    "exchange/public/kline",
    "exchange/v1/public/klines",
    "exchange/v1/public/kline",
    "exchange/v1/klines",
    "exchange/v1/kline",
    "api/exchange/public/v1/klines",
    "api/exchange/public/v1/kline",
    "api/exchange/public/klines",
    "api/exchange/public/kline",
    "api/exchange/v1/klines",
    "api/exchange/v1/kline",
    "exchange/api/public/v1/klines",
    "exchange/api/public/v1/kline",
    "exchange/api/public/klines",
    "exchange/api/public/kline",
    "exchange/api/v1/klines",
    "exchange/api/v1/kline",
    "api/klines",
    "api/kline",
    "exchange/klines",
    "exchange/kline",
    "api/exchange/klines",
    "api/exchange/kline",
    "exchange/api/klines",
    "exchange/api/kline",
    "klines",
    "kline",
)

# Pacifica REST defaults with optional overrides
_PACIFICA_DEFAULT_BASES: Tuple[str, ...] = (
    "https://api.pacifica.fi/api/v1",
    "https://api.pacifica.fi",
)

_MAX_KLINES_PER_REQUEST = 3000

# Open-source spot exchanges powered by CCXT (https://github.com/ccxt/ccxt)
_CCXT_DEFAULT_EXCHANGES: Tuple[str, ...] = (
    "binance",
    "okx",
    "kraken",
)


def _parse_env_urls(*names: str) -> List[str]:
    """Parse comma-separated URL overrides from the environment."""
    urls: List[str] = []
    for name in names:
        raw = os.getenv(name)
        if not raw:
            continue
        for candidate in raw.split(","):
            cleaned = candidate.strip()
            if cleaned:
                urls.append(cleaned.rstrip("/"))
    return urls


def _iter_pacifica_base_urls() -> Iterator[str]:
    """Yield configured Pacifica REST base URLs without duplicates."""
    seen: set[str] = set()
    for url in _parse_env_urls("PACIFICA_REST_BASE_URLS", "PACIFICA_REST_BASE_URL"):
        if url not in seen:
            seen.add(url)
            yield url
    for url in _PACIFICA_DEFAULT_BASES:
        if url not in seen:
            seen.add(url)
            yield url


def _pacifica_headers() -> Dict[str, str]:
    """Return optional Pacifica authentication headers for market data."""
    headers: Dict[str, str] = {}
    api_key = (
        os.getenv("PACIFICA_API_KEY")
        or os.getenv("PACIFICA_API_PUBLIC")
        or os.getenv("PACIFICA_REST_API_KEY")
    )
    if api_key:
        header_name = os.getenv("PACIFICA_REST_API_KEY_HEADER", "X-API-KEY")
        headers[header_name] = api_key
    return headers


def _iter_ccxt_exchanges() -> Iterator[str]:
    """Yield configured CCXT exchange ids without duplicates."""
    seen: set[str] = set()
    raw = os.getenv("CCXT_SPOT_EXCHANGES", "")
    for entry in raw.split(","):
        cleaned = entry.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            yield cleaned
    for exchange in _CCXT_DEFAULT_EXCHANGES:
        if exchange not in seen:
            seen.add(exchange)
            yield exchange


def _normalise_ccxt_symbol(symbol: str) -> str:
    """Map Extended symbols to CCXT spot market pairs (defaulting to USDT)."""
    candidate = symbol.upper().replace("-", "").replace("/", "")
    if candidate.endswith("USDT") and len(candidate) > 4:
        base = candidate[:-4]
    elif candidate in {"BTC", "ETH"}:
        base = candidate
    else:
        base = candidate
    return f"{base}/USDT"


def _extract_timestamp(entry: object) -> Optional[int]:
    """Extract a millisecond timestamp from a kline entry."""
    if isinstance(entry, dict):
        for key in ("t", "timestamp", "time", "openTime", "open_time"):
            value = entry.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(entry, (list, tuple)) and entry:
        try:
            return int(entry[0])
        except (TypeError, ValueError):
            return None
    return None


def _extract_price(entry: object) -> Optional[float]:
    """Extract a numeric close/price value from a kline entry."""
    if isinstance(entry, dict):
        for key in ("c", "close", "price", "p"):
            value = entry.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(entry, (list, tuple)):
        # Common array-based candlestick: [time, open, high, low, close, ...]
        candidates = entry[4:5] or entry[-2:-1]
        for value in candidates:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _klines_to_dataframe(klines: List[object]) -> pd.DataFrame:
    """Convert raw kline payloads to a normalized DataFrame."""
    if not klines:
        return pd.DataFrame()

    timestamps: List[int] = []
    prices: List[float] = []

    for entry in klines:
        ts = _extract_timestamp(entry)
        price = _extract_price(entry)
        if ts is None or price is None:
            continue
        timestamps.append(ts)
        prices.append(price)

    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, unit="ms"), "price": prices})
    df = df.dropna(subset=["timestamp", "price"]).drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _format_errors(errors: List[str]) -> str:
    return "; ".join(errors) if errors else "no detailed errors"


def _fetch_extended_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: Optional[int] = None,
) -> Tuple[List[object], List[str]]:
    """Fetch klines from Extended REST endpoints."""
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(INTERVAL_SECONDS)}")

    interval_ms = INTERVAL_SECONDS[interval] * 1000
    max_per_request = _MAX_KLINES_PER_REQUEST
    klines: List[object] = []
    errors: List[str] = []
    current_start = start_ms
    attempts = 0
    endpoints = list(iter_candidate_urls(_EXTENDED_KLINE_PATH_OPTIONS))

    if not endpoints:
        errors.append("No Extended REST base URLs configured")
        return [], errors

    while current_start < end_ms:
        current_end = min(current_start + max_per_request * interval_ms, end_ms)
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_time": current_start,
            "end_time": current_end,
            "limit": max_per_request,
        }

        last_error: Optional[str] = None
        payload: Optional[object] = None

        for url in endpoints:
            try:
                response = requests.get(url, params=params, timeout=15)
            except requests.RequestException as exc:
                last_error = f"{url}: {exc}"
                continue

            if response.status_code == 404:
                last_error = f"{url}: 404"
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_error = f"{url}: {exc}"
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                last_error = f"{url}: invalid JSON ({exc})"
                continue

            if isinstance(payload, dict) and payload.get("success") is False:
                last_error = f"{url}: {payload.get('error', 'Unknown API error')}"
                payload = None
                continue

            if payload is not None:
                break

        if payload is None:
            if last_error:
                errors.append(last_error)
            break

        if isinstance(payload, dict):
            batch = (
                payload.get("data")
                or payload.get("result")
                or payload.get("klines")
                or payload.get("candles")
                or []
            )
        elif isinstance(payload, list):
            batch = payload
        else:
            batch = []

        if not batch:
            # No more data available at this endpoint range
            break

        klines.extend(batch)

        last_timestamp = _extract_timestamp(batch[-1])
        if last_timestamp is None:
            break

        current_start = last_timestamp + interval_ms
        attempts += 1

        if limit and len(klines) >= limit:
            break

        if current_start >= end_ms:
            break

    return klines, errors


def _fetch_pacifica_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: Optional[int] = None,
) -> Tuple[List[object], List[str]]:
    """Fetch klines from the Pacifica REST API as a fallback."""
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(INTERVAL_SECONDS)}")

    interval_ms = INTERVAL_SECONDS[interval] * 1000
    headers = _pacifica_headers()
    max_per_request = _MAX_KLINES_PER_REQUEST
    klines: List[object] = []
    errors: List[str] = []
    current_start = start_ms

    bases = list(_iter_pacifica_base_urls())
    if not bases:
        errors.append("No Pacifica REST base URLs configured")
        return [], errors

    while current_start < end_ms:
        current_end = min(current_start + max_per_request * interval_ms, end_ms)
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_time": current_start,
            "end_time": current_end,
            "limit": max_per_request,
        }

        last_error: Optional[str] = None
        payload: Optional[object] = None

        for base in bases:
            url = join_url(base, "kline")
            try:
                response = requests.get(url, params=params, headers=headers, timeout=20)
            except requests.RequestException as exc:
                last_error = f"{url}: {exc}"
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_error = f"{url}: {exc}"
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                last_error = f"{url}: invalid JSON ({exc})"
                continue

            if isinstance(payload, dict) and payload.get("success") is False:
                last_error = f"{url}: {payload.get('error', 'Unknown API error')}"
                payload = None
                continue

            if payload is not None:
                break

        if payload is None:
            if last_error:
                errors.append(last_error)
            break

        if isinstance(payload, dict):
            batch = (
                payload.get("data")
                or payload.get("result")
                or payload.get("klines")
                or payload.get("candles")
                or []
            )
        elif isinstance(payload, list):
            batch = payload
        else:
            batch = []

        if not batch:
            break

        klines.extend(batch)

        last_timestamp = _extract_timestamp(batch[-1])
        if last_timestamp is None:
            break

        current_start = last_timestamp + interval_ms

        if limit and len(klines) >= limit:
            break

        if current_start >= end_ms:
            break

    return klines, errors


def _fetch_ccxt_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: Optional[int] = None,
) -> Tuple[List[object], str, List[str]]:
    """Fetch klines from open CCXT exchanges such as Binance or OKX."""
    try:
        import ccxt  # type: ignore
    except ImportError:
        return [], "unavailable", ["ccxt library not installed"]

    interval_seconds = INTERVAL_SECONDS.get(interval)
    if interval_seconds is None:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(INTERVAL_SECONDS)}")

    interval_ms = interval_seconds * 1000

    pair = _normalise_ccxt_symbol(symbol)
    max_per_request = 1000
    ohlcvs: List[object] = []
    errors: List[str] = []

    for exchange_id in _iter_ccxt_exchanges():
        exchange = getattr(ccxt, exchange_id, None)
        if exchange is None:
            errors.append(f"{exchange_id}: not available in ccxt")
            continue

        try:
            client = exchange({"enableRateLimit": True})
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{exchange_id}: failed to initialise ({exc})")
            continue

        try:
            since = start_ms
            ohlcvs.clear()

            while since < end_ms:
                remaining = (limit - len(ohlcvs)) if limit is not None else None
                if remaining is not None and remaining <= 0:
                    break
                batch_limit = min(
                    max_per_request,
                    remaining if remaining is not None else max_per_request,
                )
                batch_limit = max(1, batch_limit)
                try:
                    batch = client.fetch_ohlcv(  # type: ignore[call-arg]
                        pair,
                        timeframe=interval,
                        since=since,
                        limit=batch_limit,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append(f"{exchange_id}: {exc}")
                    ohlcvs.clear()
                    break

                if not batch:
                    break

                ohlcvs.extend(batch)
                last_timestamp = _extract_timestamp(batch[-1])
                if last_timestamp is None:
                    break
                since = max(last_timestamp + interval_ms, since + interval_ms)

                if limit is not None and len(ohlcvs) >= limit:
                    break

                if since >= end_ms:
                    break

            if ohlcvs:
                return list(ohlcvs), exchange_id, errors
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pylint: disable=broad-except
                    pass

    return [], "unavailable", errors


def fetch_historical_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """Fetch historical klines using Extended first, then Pacifica fallback."""
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if start_ms >= end_ms:
        return pd.DataFrame(), "extended", {"extended_errors": "start >= end"}

    ext_data, ext_errors = _fetch_extended_klines(
        symbol, interval, start_ms, end_ms, limit=limit
    )
    if ext_data:
        df = _klines_to_dataframe(ext_data)
        if not df.empty:
            return df, "extended", {"extended_errors": _format_errors(ext_errors)}

    logger.info(
        "Extended kline lookup for %s %s returned no data (%s); trying Pacifica fallback",
        symbol,
        interval,
        _format_errors(ext_errors),
    )

    pac_data, pac_errors = _fetch_pacifica_klines(
        symbol, interval, start_ms, end_ms, limit=limit
    )
    if pac_data:
        df = _klines_to_dataframe(pac_data)
        if not df.empty:
            return df, "pacifica", {
                "extended_errors": _format_errors(ext_errors),
                "pacifica_errors": _format_errors(pac_errors),
            }

    logger.info(
        "Pacifica kline lookup for %s %s returned no data (%s); trying CCXT fallback",
        symbol,
        interval,
        _format_errors(pac_errors),
    )

    ccxt_data, ccxt_exchange, ccxt_errors = _fetch_ccxt_klines(
        symbol, interval, start_ms, end_ms, limit=limit
    )
    if ccxt_data:
        df = _klines_to_dataframe(ccxt_data)
        if not df.empty:
            return df, f"ccxt:{ccxt_exchange}", {
                "extended_errors": _format_errors(ext_errors),
                "pacifica_errors": _format_errors(pac_errors),
                "ccxt_exchange": ccxt_exchange,
                "ccxt_errors": _format_errors(ccxt_errors),
            }

    error_info = {
        "extended_errors": _format_errors(ext_errors),
        "pacifica_errors": _format_errors(pac_errors),
        "ccxt_errors": _format_errors(ccxt_errors),
    }
    return pd.DataFrame(), "unavailable", error_info


def fetch_recent_klines(
    symbol: str,
    interval: str,
    *,
    bars: int,
) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """Convenience helper to fetch a fixed number of bars ending 'now'."""
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(INTERVAL_SECONDS)}")

    interval_ms = INTERVAL_SECONDS[interval] * 1000
    end_time = int(time.time() * 1000)
    start_time = end_time - bars * interval_ms
    return fetch_historical_klines(symbol, interval, start_time, end_time, limit=bars)
