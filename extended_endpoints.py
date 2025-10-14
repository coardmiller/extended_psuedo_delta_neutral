"""Utilities for building Extended API endpoint URLs with graceful fallbacks."""

from __future__ import annotations

import os
from typing import Iterable, Iterator, List, Optional

_DEFAULT_REST_BASES: tuple[str, ...] = (
    "https://api.extended.exchange",
    "https://api.extended.exchange/api",
    "https://api.extended.exchange/api/v1",
)


def _parse_env_urls(value: Optional[str]) -> List[str]:
    """Parse a comma-separated list of URLs from an environment variable."""
    if not value:
        return []
    urls: List[str] = []
    for raw in value.split(","):
        cleaned = raw.strip()
        if not cleaned:
            continue
        urls.append(cleaned.rstrip("/"))
    return urls


def get_rest_base_urls() -> List[str]:
    """Return ordered list of REST base URLs to try."""
    candidates: List[str] = []
    candidates.extend(_parse_env_urls(os.getenv("EXTENDED_REST_BASE_URLS")))
    candidates.extend(_parse_env_urls(os.getenv("EXTENDED_REST_BASE_URL")))
    candidates.extend(_DEFAULT_REST_BASES)

    seen: set[str] = set()
    ordered: List[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def join_url(base: str, path: str) -> str:
    """Join a base URL with a path fragment."""
    if not path:
        return base.rstrip("/")
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def iter_candidate_urls(path_options: Iterable[str]) -> Iterator[str]:
    """Yield fully-qualified URLs for the given path options."""
    seen: set[str] = set()
    for base in get_rest_base_urls():
        for path in path_options:
            url = join_url(base, path)
            if url in seen:
                continue
            seen.add(url)
            yield url
