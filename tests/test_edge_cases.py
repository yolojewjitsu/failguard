"""Edge case tests for failguard."""

from failguard import Monitor


class TestEdgeCases:
    def test_max_identical_outputs_zero(self):
        """Test with max_identical_outputs=0 - should disable stuck detection."""
        monitor = Monitor(max_identical_outputs=0, stuck_window=60)
        # max_identical_outputs=0 disables stuck detection entirely
        status = monitor.check("value")
        assert not status.is_stuck
        assert status.identical_count == 0

        # Even multiple identical outputs shouldn't trigger stuck
        status = monitor.check("value")
        assert not status.is_stuck

    def test_stuck_window_zero(self):
        """Test with stuck_window=0 - no calls should be within window."""
        monitor = Monitor(max_identical_outputs=2, stuck_window=0)
        monitor.check("same")
        status = monitor.check("same")
        # With window=0, nothing is "recent" so stuck should not trigger
        assert not status.is_stuck

    def test_latency_drift_zero_threshold(self):
        """Test with max_latency_drift=0."""
        monitor = Monitor(max_latency_drift=0)
        # Establish baseline
        for _ in range(10):
            monitor.check("x", latency_ms=100)
        # Any positive latency should exceed threshold 0
        status = monitor.check("x", latency_ms=100)
        assert status.has_latency_drift

    def test_empty_output(self):
        """Test with empty string output."""
        monitor = Monitor(max_identical_outputs=2)
        monitor.check("")
        status = monitor.check("")
        assert status.is_stuck

    def test_none_output(self):
        """Test with None output."""
        monitor = Monitor(max_identical_outputs=2)
        monitor.check(None)
        status = monitor.check(None)
        assert status.is_stuck

    def test_very_long_output(self):
        """Test with very long output - should still work via hashing."""
        monitor = Monitor(max_identical_outputs=2)
        long_output = "x" * 1000000  # 1MB string
        monitor.check(long_output)
        status = monitor.check(long_output)
        assert status.is_stuck

    def test_cycle_invalid_min_length(self):
        """Test with cycle_min_length=0 - should disable cycle detection gracefully."""
        monitor = Monitor(
            detect_cycles=True,
            cycle_min_length=0,  # Invalid: must be >= 1
            cycle_max_length=5,
            max_identical_outputs=100,
        )
        # Build up what would be a cycle with valid parameters
        for step in ["A", "B", "A", "B"]:
            status = monitor.check(step, step_name=step)
            # Should never detect cycle with invalid min_length
            assert not status.has_cycle

    def test_cycle_min_greater_than_max(self):
        """Test with cycle_min_length > cycle_max_length - should disable cycle detection."""
        monitor = Monitor(
            detect_cycles=True,
            cycle_min_length=5,  # Invalid: greater than max
            cycle_max_length=2,
            max_identical_outputs=100,
        )
        # Build up what would be a cycle with valid parameters
        for step in ["A", "B", "A", "B"]:
            status = monitor.check(step, step_name=step)
            # Should never detect cycle with invalid parameters
            assert not status.has_cycle
