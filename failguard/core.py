"""Core failure detection logic for AI agents."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

__all__ = [
    "FailGuardError",
    "FailureStatus",
    "FailureType",
    "Monitor",
    "failguard",
]

T = TypeVar("T")

# Use monotonic time for internal tracking
_get_time = time.monotonic


class FailureType:
    """Types of failures that can be detected."""

    LATENCY_DRIFT = "latency_drift"
    STUCK = "stuck"
    CYCLE = "cycle"


class FailGuardError(Exception):
    """Raised when a failure is detected.

    Attributes:
        failure_type: Type of failure detected.
        message: Description of the failure.
        metrics: Relevant metrics at time of failure.

    """

    __slots__ = ("failure_type", "message", "metrics")

    def __init__(
        self,
        failure_type: str,
        message: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.failure_type = failure_type
        self.message = message
        self.metrics = metrics or {}
        super().__init__(f"{failure_type}: {message}")

    def __repr__(self) -> str:
        return f"FailGuardError({self.failure_type!r}, {self.message!r})"


@dataclass
class FailureStatus:
    """Status report from failure detection."""

    has_failure: bool = False
    failure_types: list[str] = field(default_factory=list)

    # Latency metrics
    latency_ms: float = 0.0
    latency_baseline_ms: float = 0.0
    latency_drift_ratio: float = 1.0
    has_latency_drift: bool = False

    # Stuck metrics
    identical_count: int = 0
    is_stuck: bool = False

    # Cycle metrics
    has_cycle: bool = False
    cycle_pattern: list[str] = field(default_factory=list)
    cycle_length: int = 0


def _hash_output(value: Any) -> str:
    """Create a hash of the output for comparison."""
    # Use surrogatepass to handle any malformed unicode in repr()
    return hashlib.sha256(repr(value).encode("utf-8", errors="surrogatepass")).hexdigest()[:16]


def _detect_cycle(
    sequence: list[str],
    min_length: int = 2,
    max_length: int = 5,
) -> tuple[bool, list[str]]:
    """Detect repeating patterns in a sequence.

    Returns (has_cycle, pattern).
    """
    if len(sequence) < min_length * 2:
        return False, []

    for length in range(min_length, min(max_length + 1, len(sequence) // 2 + 1)):
        pattern = sequence[-length:]
        prev = sequence[-length * 2 : -length]
        if pattern == prev:
            return True, pattern

    return False, []


_WARMUP_SAMPLES = 10  # Number of samples before switching to EMA


class _FailGuardState:
    """Internal state tracking for a guarded function."""

    def __init__(self, window_size: int = 100) -> None:
        self.lock = threading.Lock()
        self.call_history: deque[tuple[float, str, float]] = deque(
            maxlen=window_size,
        )
        self.output_sequence: deque[str] = deque(maxlen=window_size)
        # After warmup, _latency_ema_scaled holds EMA * _WARMUP_SAMPLES
        # so that baseline = _latency_ema_scaled / _WARMUP_SAMPLES
        self._latency_ema_scaled: float = 0.0
        self._latency_count: int = 0

    @property
    def latency_baseline(self) -> float:
        """Rolling average latency in ms."""
        with self.lock:
            if self._latency_count == 0:
                return 0.0
            return self._latency_ema_scaled / self._latency_count

    def record_call(
        self,
        output_hash: str,
        latency_ms: float,
        step_name: str | None = None,
    ) -> None:
        """Record a function call."""
        now = _get_time()
        with self.lock:
            # Update latency baseline (exponential moving average after warmup)
            if self._latency_count < _WARMUP_SAMPLES:
                # Warmup: accumulate sum
                self._latency_ema_scaled += latency_ms
                self._latency_count += 1
            else:
                # After warmup: EMA with alpha=0.1, scaled by _WARMUP_SAMPLES
                self._latency_ema_scaled = self._latency_ema_scaled * 0.9 + latency_ms

            self.call_history.append((now, output_hash, latency_ms))
            self.output_sequence.append(step_name or output_hash)

    def check_latency_drift(
        self, current_ms: float, threshold: float
    ) -> tuple[bool, float]:
        """Check if current latency exceeds threshold * baseline."""
        baseline = self.latency_baseline
        if baseline <= 0:
            return False, 1.0
        ratio = current_ms / baseline
        return ratio > threshold, ratio

    def check_stuck(
        self,
        output_hash: str,
        max_identical: int,
        window_sec: float,
    ) -> tuple[bool, int]:
        """Check if output is stuck (repeating identical values within window).

        Args:
            output_hash: Hash of the current output.
            max_identical: Threshold for stuck detection. Use 0 to disable.
            window_sec: Time window in seconds to consider.

        Returns:
            Tuple of (is_stuck, count_in_window).
        """
        # max_identical=0 disables stuck detection
        if max_identical <= 0:
            return False, 0
        now = _get_time()
        count = 0
        with self.lock:
            for ts, h, _ in self.call_history:
                if now - ts < window_sec and h == output_hash:
                    count += 1
        return count >= max_identical, count

    def check_cycle(
        self, min_length: int = 2, max_length: int = 5
    ) -> tuple[bool, list[str]]:
        """Check for repeating patterns in output sequence."""
        with self.lock:
            return _detect_cycle(list(self.output_sequence), min_length, max_length)

    def reset(self) -> None:
        """Clear all state."""
        with self.lock:
            self.call_history.clear()
            self.output_sequence.clear()
            self._latency_ema_scaled = 0.0
            self._latency_count = 0


def failguard(
    *,
    max_latency_drift: float = 3.0,
    max_identical_outputs: int = 5,
    stuck_window: float = 60.0,
    detect_cycles: bool = True,
    cycle_min_length: int = 2,
    cycle_max_length: int = 5,
    on_failure: Callable[[FailureStatus], Any] | None = None,
    raise_on_failure: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Detect silent failures in AI agent functions.

    Monitors for:
    - Latency drift: Function taking much longer than baseline
    - Stuck states: Same output repeated multiple times
    - Cycles: Repeating patterns like A->B->A->B

    Args:
        max_latency_drift: Alert if latency > N * baseline (default 3.0)
        max_identical_outputs: Alert after N identical outputs (default 5)
        stuck_window: Time window in seconds for stuck detection (default 60)
        detect_cycles: Whether to detect A->B->A patterns (default True)
        cycle_min_length: Minimum cycle length to detect (default 2)
        cycle_max_length: Maximum cycle length to detect (default 5)
        on_failure: Optional callback when failure detected.
                    If provided, receives FailureStatus.
                    If it returns a non-None value, that's used as the return value.
                    If it returns None, the original result is returned.
        raise_on_failure: Whether to raise FailGuardError (default True).
                         Ignored if on_failure handles the failure.

    Example:
        @failguard(max_latency_drift=2.0, max_identical_outputs=3)
        def agent_step(query: str) -> str:
            return llm.complete(query)

    Raises:
        FailGuardError: If a failure is detected and raise_on_failure=True.

    Note:
        This decorator does not support async functions. For async code,
        use the Monitor class with explicit latency_ms measurements.

    """
    state = _FailGuardState()

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = _get_time()
            result = fn(*args, **kwargs)
            latency_ms = (_get_time() - start) * 1000

            output_hash = _hash_output(result)
            state.record_call(output_hash, latency_ms)

            # Build status
            status = FailureStatus(latency_ms=latency_ms)
            status.latency_baseline_ms = state.latency_baseline

            # Check latency drift
            has_drift, ratio = state.check_latency_drift(latency_ms, max_latency_drift)
            status.latency_drift_ratio = ratio
            if has_drift:
                status.has_latency_drift = True
                status.has_failure = True
                status.failure_types.append(FailureType.LATENCY_DRIFT)

            # Check stuck
            is_stuck, count = state.check_stuck(
                output_hash, max_identical_outputs, stuck_window
            )
            status.identical_count = count
            if is_stuck:
                status.is_stuck = True
                status.has_failure = True
                status.failure_types.append(FailureType.STUCK)

            # Check cycles
            if detect_cycles:
                has_cycle, pattern = state.check_cycle(
                    cycle_min_length, cycle_max_length
                )
                status.has_cycle = has_cycle
                status.cycle_pattern = pattern  # Already a new list from slice
                status.cycle_length = len(pattern)
                if has_cycle:
                    status.has_failure = True
                    status.failure_types.append(FailureType.CYCLE)

            # Handle failure
            if status.has_failure:
                if on_failure is not None:
                    handler_result = on_failure(status)
                    if handler_result is not None:
                        return handler_result
                elif raise_on_failure:
                    raise FailGuardError(
                        failure_type=status.failure_types[0],
                        message=_format_failure_message(status),
                        metrics={
                            "latency_ms": status.latency_ms,
                            "latency_baseline_ms": status.latency_baseline_ms,
                            "identical_count": status.identical_count,
                            "cycle_pattern": status.cycle_pattern,  # Already copied
                        },
                    )

            return result

        def reset() -> None:
            """Clear all failure detection state."""
            state.reset()

        def get_status() -> FailureStatus:
            """Get current status without making a call.

            Returns a FailureStatus with only latency_baseline_ms populated.
            Other fields have default values since no check is performed.
            """
            status = FailureStatus()
            status.latency_baseline_ms = state.latency_baseline
            return status

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_status = get_status  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _format_failure_message(status: FailureStatus) -> str:
    """Format a human-readable failure message."""
    parts = []
    if status.has_latency_drift:
        parts.append(
            f"Latency drift: {status.latency_ms:.1f}ms "
            f"(baseline: {status.latency_baseline_ms:.1f}ms, "
            f"ratio: {status.latency_drift_ratio:.1f}x)"
        )
    if status.is_stuck:
        parts.append(f"Stuck: {status.identical_count} identical outputs")
    if status.has_cycle:
        parts.append(f"Cycle detected: {' -> '.join(status.cycle_pattern)}")
    return "; ".join(parts) or "Unknown failure"


class Monitor:
    """Inline monitor for checking outputs without decorators.

    Example:
        monitor = Monitor()
        for step in workflow:
            result = agent.run(step)
            status = monitor.check(result, step_name="process")
            if status.has_failure:
                handle_failure(status)

    """

    def __init__(
        self,
        *,
        max_latency_drift: float = 3.0,
        max_identical_outputs: int = 5,
        stuck_window: float = 60.0,
        detect_cycles: bool = True,
        cycle_min_length: int = 2,
        cycle_max_length: int = 5,
    ) -> None:
        self._state = _FailGuardState()
        self._max_latency_drift = max_latency_drift
        self._max_identical_outputs = max_identical_outputs
        self._stuck_window = stuck_window
        self._detect_cycles = detect_cycles
        self._cycle_min_length = cycle_min_length
        self._cycle_max_length = cycle_max_length
        self._last_check_time: float | None = None

    def check(
        self,
        value: Any,
        *,
        step_name: str | None = None,
        latency_ms: float | None = None,
    ) -> FailureStatus:
        """Check a value for failures.

        Args:
            value: The output to check.
            step_name: Optional name for cycle detection.
            latency_ms: Optional latency override. If not provided,
                        uses time since last check.

        Returns:
            FailureStatus with all detected issues.

        """
        now = _get_time()

        # Calculate latency if not provided
        if latency_ms is None:
            if self._last_check_time is not None:
                latency_ms = (now - self._last_check_time) * 1000
            else:
                latency_ms = 0.0
        self._last_check_time = now

        output_hash = _hash_output(value)
        self._state.record_call(output_hash, latency_ms, step_name)

        # Build status
        status = FailureStatus(latency_ms=latency_ms)
        status.latency_baseline_ms = self._state.latency_baseline

        # Check latency drift
        if latency_ms > 0:
            has_drift, ratio = self._state.check_latency_drift(
                latency_ms, self._max_latency_drift
            )
            status.latency_drift_ratio = ratio
            if has_drift:
                status.has_latency_drift = True
                status.has_failure = True
                status.failure_types.append(FailureType.LATENCY_DRIFT)

        # Check stuck
        is_stuck, count = self._state.check_stuck(
            output_hash, self._max_identical_outputs, self._stuck_window
        )
        status.identical_count = count
        if is_stuck:
            status.is_stuck = True
            status.has_failure = True
            status.failure_types.append(FailureType.STUCK)

        # Check cycles
        if self._detect_cycles:
            has_cycle, pattern = self._state.check_cycle(
                self._cycle_min_length, self._cycle_max_length
            )
            status.has_cycle = has_cycle
            status.cycle_pattern = pattern  # Already a new list from slice
            status.cycle_length = len(pattern)
            if has_cycle:
                status.has_failure = True
                status.failure_types.append(FailureType.CYCLE)

        return status

    def reset(self) -> None:
        """Clear all state."""
        self._state.reset()
        self._last_check_time = None

    @property
    def latency_baseline(self) -> float:
        """Current latency baseline in ms."""
        return self._state.latency_baseline
