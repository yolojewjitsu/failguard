"""FailGuard - Detect silent failures in AI agents."""

from .core import (
    FailGuardError,
    FailureStatus,
    FailureType,
    Monitor,
    failguard,
)

__version__ = "0.1.0"
__all__ = [
    "FailGuardError",
    "FailureStatus",
    "FailureType",
    "Monitor",
    "__version__",
    "failguard",
]
