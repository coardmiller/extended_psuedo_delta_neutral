"""
Utility to provide a light-weight stub for the solders package during tests.

The real solders dependency is heavy and not always available in CI or local
test harnesses. This stub provides just enough surface area for the test suite:
- Keypair() constructor that yields deterministic string representations
- Keypair.from_base58_string for reconstructing instances
- sign_message returning bytes so payload signing still works
- pubkey method returning a predictable placeholder
"""
import hashlib
import os
import sys
import types
from typing import Any


def ensure_stub() -> None:
    """Install a stub solders.keypair module if the real dependency is missing."""
    try:
        import solders.keypair  # type: ignore # pragma: no cover
        return
    except ModuleNotFoundError:
        pass

    solders_module = types.ModuleType("solders")
    keypair_module = types.ModuleType("solders.keypair")

    class Keypair:
        """Minimal stand-in for solders.keypair.Keypair used in tests."""

        def __init__(self, seed: Any | None = None) -> None:
            if seed is None:
                seed_bytes = os.urandom(32)
            elif isinstance(seed, bytes):
                seed_bytes = seed
            else:
                seed_bytes = str(seed).encode("utf-8")

            self._private = hashlib.sha256(seed_bytes).hexdigest()

        @staticmethod
        def from_base58_string(value: str) -> "Keypair":
            instance = Keypair(value)
            instance._private = value
            return instance

        def __str__(self) -> str:
            return self._private

        def sign_message(self, message: bytes) -> bytes:
            return hashlib.sha256(self._private.encode("utf-8") + message).digest()

        def pubkey(self) -> str:
            return f"{self._private}-pub"

    keypair_module.Keypair = Keypair
    solders_module.keypair = keypair_module
    solders_module.__all__ = ("keypair",)
    solders_module.__path__ = []  # type: ignore[attr-defined]

    sys.modules["solders"] = solders_module
    sys.modules["solders.keypair"] = keypair_module
