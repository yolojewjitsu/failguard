"""FailGuard - Detect silent failures in AI agents."""

from .core import (
    failguard,
    Monitor,
    FailGuardError,
    FailureStatus,
    FailureType,
)

__version__ = "0.1.0"
__all__ = [
    "failguard",
    "Monitor",
    "FailGuardError",
    "FailureStatus",
    "FailureType",
    "__version__",
]
