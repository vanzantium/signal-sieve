"""Public interface for Signal Sieve's declared-language risk layer."""

from .engine import analyze, normalize_language

__all__ = ["analyze", "normalize_language"]
