"""Minimal subset of colorama for environments without the package."""

from __future__ import annotations


class _Color:
    def __getattr__(self, name: str) -> str:  # pragma: no cover - trivial
        return ""


Fore = _Color()
Style = _Color()


def init(*_args, **_kwargs) -> None:  # pragma: no cover - trivial
    """Placeholder init function to mirror ``colorama.init``."""
    return None


__all__ = ["Fore", "Style", "init"]
