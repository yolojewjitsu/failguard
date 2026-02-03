"""Tests for failguard."""

import pytest

from failguard import failguard, Monitor, FailGuardError, FailureStatus, FailureType


class TestFailguardDecorator:
    def test_normal_operation(self):
        """Test that normal calls don't trigger failures."""
        call_count = 0

        @failguard(max_identical_outputs=5)
        def func(x):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # Different outputs each time - no failure
        for _ in range(10):
            func(1)

    def test_stuck_detection(self):
        """Test detection of stuck (identical) outputs."""
        @failguard(max_identical_outputs=3, stuck_window=60)
        def stuck_func():
            return "same_output"

        stuck_func()
        stuck_func()

        with pytest.raises(FailGuardError) as exc:
            stuck_func()

        assert exc.value.failure_type == FailureType.STUCK
        assert "identical" in exc.value.message.lower()

    def test_cycle_detection(self):
        """Test detection of A->B->A->B cycles."""
        outputs = ["A", "B", "A", "B", "A", "B"]
        idx = 0

        @failguard(detect_cycles=True, cycle_min_length=2, max_identical_outputs=100)
        def cycling_func():
            nonlocal idx
            result = outputs[idx % len(outputs)]
            idx += 1
            return result

        # First few calls build up the pattern
        cycling_func()  # A
        cycling_func()  # B
        cycling_func()  # A

        # Fourth call completes the A->B->A->B pattern, cycle detected
        with pytest.raises(FailGuardError) as exc:
            cycling_func()  # B - cycle detected here (A->B repeated twice)

        assert exc.value.failure_type == FailureType.CYCLE

    def test_on_failure_handler(self):
        """Test custom failure handler."""
        handler_called = []

        def handler(status: FailureStatus):
            handler_called.append(status)
            return "fallback"

        @failguard(max_identical_outputs=2, on_failure=handler)
        def func():
            return "same"

        func()
        result = func()

        assert result == "fallback"
        assert len(handler_called) == 1
        assert handler_called[0].is_stuck

    def test_on_failure_returns_none(self):
        """Test on_failure handler that returns None - should return original result."""
        @failguard(max_identical_outputs=2, on_failure=lambda status: None)
        def func():
            return "same"

        func()
        result = func()

        # When on_failure returns None, the original result is returned
        assert result == "same"

    def test_preserves_function_metadata(self):
        """Test that @failguard preserves function metadata."""
        @failguard()
        def documented_func():
            """This is a docstring."""
            return "value"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    def test_raise_on_failure_false(self):
        """Test that raise_on_failure=False suppresses exceptions."""
        @failguard(max_identical_outputs=2, raise_on_failure=False)
        def func():
            return "same"

        # Should not raise
        func()
        result = func()
        assert result == "same"

    def test_reset(self):
        """Test that reset() clears state."""
        @failguard(max_identical_outputs=2)
        def func():
            return "same"

        func()
        func.reset()
        func()  # Should not raise after reset

    def test_get_status(self):
        """Test get_status() method."""
        @failguard()
        def func():
            return "result"

        status = func.get_status()
        assert isinstance(status, FailureStatus)
        assert status.latency_baseline_ms == 0.0

        func()
        status = func.get_status()
        assert status.latency_baseline_ms > 0


class TestMonitor:
    def test_basic_usage(self):
        """Test basic Monitor usage."""
        monitor = Monitor(max_identical_outputs=3)

        status = monitor.check("output1")
        assert not status.has_failure

        status = monitor.check("output2")
        assert not status.has_failure

    def test_stuck_detection(self):
        """Test Monitor stuck detection."""
        monitor = Monitor(max_identical_outputs=3, stuck_window=60)

        monitor.check("same")
        monitor.check("same")
        status = monitor.check("same")

        assert status.is_stuck
        assert status.has_failure
        assert FailureType.STUCK in status.failure_types

    def test_cycle_detection(self):
        """Test Monitor cycle detection."""
        monitor = Monitor(detect_cycles=True, max_identical_outputs=100)

        monitor.check("A", step_name="step_a")
        monitor.check("B", step_name="step_b")
        monitor.check("A", step_name="step_a")
        status = monitor.check("B", step_name="step_b")

        assert status.has_cycle
        assert status.cycle_pattern == ["step_a", "step_b"]

    def test_detect_cycles_disabled(self):
        """Test that detect_cycles=False disables cycle detection."""
        monitor = Monitor(detect_cycles=False, max_identical_outputs=100)

        # Build up a pattern that would trigger cycle detection
        monitor.check("A", step_name="step_a")
        monitor.check("B", step_name="step_b")
        monitor.check("A", step_name="step_a")
        status = monitor.check("B", step_name="step_b")

        # Cycle detection disabled, so no failure
        assert not status.has_cycle
        assert not status.has_failure
        assert status.cycle_pattern == []

    def test_reset(self):
        """Test Monitor reset."""
        monitor = Monitor(max_identical_outputs=2)

        monitor.check("same")
        monitor.reset()
        status = monitor.check("same")

        assert not status.is_stuck

    def test_latency_baseline(self):
        """Test latency baseline tracking."""
        monitor = Monitor()

        # First few calls establish baseline
        for _ in range(5):
            monitor.check("output", latency_ms=100)

        assert monitor.latency_baseline > 0

    def test_custom_latency(self):
        """Test providing custom latency."""
        monitor = Monitor(max_latency_drift=2.0)

        # Establish baseline
        for _ in range(10):
            monitor.check("output", latency_ms=100)

        # Check with high latency
        status = monitor.check("output", latency_ms=500)

        assert status.has_latency_drift
        assert status.latency_drift_ratio > 2.0


class TestFailureStatus:
    def test_default_values(self):
        """Test FailureStatus default values."""
        status = FailureStatus()
        assert not status.has_failure
        assert status.failure_types == []
        assert status.latency_ms == 0.0
        assert not status.has_latency_drift
        assert not status.is_stuck
        assert not status.has_cycle


class TestFailGuardError:
    def test_error_attributes(self):
        """Test FailGuardError attributes."""
        err = FailGuardError(
            failure_type=FailureType.STUCK,
            message="test message",
            metrics={"count": 5},
        )
        assert err.failure_type == FailureType.STUCK
        assert err.message == "test message"
        assert err.metrics == {"count": 5}

    def test_error_str(self):
        """Test FailGuardError string representation."""
        err = FailGuardError(FailureType.CYCLE, "cycle detected")
        assert "cycle" in str(err).lower()

    def test_error_repr(self):
        """Test FailGuardError repr."""
        err = FailGuardError(FailureType.LATENCY_DRIFT, "slow")
        assert "FailGuardError" in repr(err)


class TestCycleDetection:
    def test_no_cycle_short_sequence(self):
        """Test that short sequences don't trigger false cycles."""
        monitor = Monitor(detect_cycles=True)

        status = monitor.check("A", step_name="a")
        assert not status.has_cycle

        status = monitor.check("B", step_name="b")
        assert not status.has_cycle

    def test_longer_cycle(self):
        """Test detection of longer cycles (A->B->C->A->B->C)."""
        monitor = Monitor(detect_cycles=True, cycle_min_length=2, cycle_max_length=5)

        # Need exactly 6 elements to detect A->B->C repeated twice
        steps = ["A", "B", "C", "A", "B", "C"]
        for step in steps[:-1]:
            status = monitor.check(step, step_name=step)
            assert not status.has_cycle  # Not yet complete

        # Sixth element completes the cycle
        status = monitor.check("C", step_name="C")
        assert status.has_cycle
        assert len(status.cycle_pattern) == 3

    def test_no_false_positive_different_values(self):
        """Test that different values don't trigger cycles."""
        monitor = Monitor(detect_cycles=True, max_identical_outputs=100)

        for i in range(20):
            status = monitor.check(f"unique_{i}", step_name=f"step_{i}")
            assert not status.has_cycle


class TestIntegration:
    def test_with_loopguard_pattern(self):
        """Test that failguard complements loopguard patterns."""
        # LoopGuard catches A->A->A (same args)
        # FailGuard catches A->B->A->B (cycles with different outputs)

        @failguard(detect_cycles=True, max_identical_outputs=10)
        def agent_action(query):
            # Simulate cycling between two responses
            if not hasattr(agent_action, "_toggle"):
                agent_action._toggle = False
            agent_action._toggle = not agent_action._toggle
            return "response_A" if agent_action._toggle else "response_B"

        # Build up cycle
        agent_action("query")
        agent_action("query")
        agent_action("query")

        with pytest.raises(FailGuardError) as exc:
            agent_action("query")

        assert exc.value.failure_type == FailureType.CYCLE

    def test_multiple_simultaneous_failures(self):
        """Test that multiple failure types can occur simultaneously."""
        # Use Monitor to check status without raising
        monitor = Monitor(
            max_identical_outputs=2,
            detect_cycles=True,
            cycle_min_length=2,
            stuck_window=60,
        )

        # Build pattern: A, B, A, B (same outputs cycling)
        # This triggers both STUCK and CYCLE detection
        monitor.check("same", step_name="A")
        monitor.check("same", step_name="B")  # 2nd "same" triggers stuck
        monitor.check("same", step_name="A")
        status = monitor.check("same", step_name="B")  # Completes A->B->A->B cycle

        # Both stuck and cycle should be detected
        assert status.has_failure
        assert status.is_stuck
        assert status.has_cycle
        assert FailureType.STUCK in status.failure_types
        assert FailureType.CYCLE in status.failure_types
        assert len(status.failure_types) == 2


class TestLatencyDriftDecorator:
    def test_latency_drift_detection_in_decorator(self):
        """Test that latency drift is detected and raises error."""
        import time

        call_count = [0]

        @failguard(
            max_latency_drift=2.0,
            max_identical_outputs=100,
            detect_cycles=False,  # Disable cycle detection for this test
        )
        def variable_speed_func():
            call_count[0] += 1
            # First 10 calls are fast, then one slow
            if call_count[0] > 10:
                time.sleep(0.01)  # 10ms sleep
            return f"result_{call_count[0]}"

        # Establish fast baseline (< 1ms each)
        for _ in range(10):
            variable_speed_func()

        # This call should trigger drift (much slower than baseline)
        with pytest.raises(FailGuardError) as exc:
            variable_speed_func()

        assert exc.value.failure_type == FailureType.LATENCY_DRIFT

    def test_latency_drift_error_message(self):
        """Test that latency drift error message is formatted correctly."""
        from failguard import FailureStatus
        from failguard.core import _format_failure_message

        status = FailureStatus(
            has_failure=True,
            has_latency_drift=True,
            latency_ms=500.0,
            latency_baseline_ms=100.0,
            latency_drift_ratio=5.0,
        )
        msg = _format_failure_message(status)
        assert "Latency drift" in msg
        assert "500.0ms" in msg
        assert "100.0ms" in msg
