"""Minimal fallback implementation of ``python-dotenv``'s load_dotenv."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union


def _iter_env_lines(contents: Iterable[str]):
    """Yield key/value pairs from dotenv-style lines."""

    for line in contents:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        yield key.strip(), value.strip().strip('"').strip("'")


def load_dotenv(
    dotenv_path: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    override: bool = False,
) -> bool:
    """Load environment variables from a ``.env`` file.

    Only implements the subset of behaviour the hedge bot relies on.
    Returns ``True`` if any variables were loaded.
    """

    path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
    if not path.exists() or not path.is_file():
        return False

    loaded = False
    with path.open("r", encoding="utf-8") as handle:
        for key, value in _iter_env_lines(handle):
            if key and (override or key not in os.environ):
                os.environ[key] = value
                loaded = True
    return loaded


__all__ = ["load_dotenv"]
