"""Thread safety tests for failguard."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from failguard import failguard, Monitor


class TestThreadSafety:
    def test_monitor_concurrent_access(self):
        """Test Monitor with concurrent access from multiple threads."""
        monitor = Monitor(max_identical_outputs=1000, stuck_window=60)
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    monitor.check(f"output_{thread_id}_{i}", latency_ms=10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"

    def test_decorator_concurrent_calls(self):
        """Test decorated function called from multiple threads."""
        call_count = [0]
        lock = threading.Lock()

        # Disable latency drift to avoid false positives on fast calls
        @failguard(max_identical_outputs=1000, max_latency_drift=1000.0)
        def counted_func(thread_id, call_id):
            with lock:
                call_count[0] += 1
            return f"result_{thread_id}_{call_id}"

        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    counted_func(thread_id, i)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for f in futures:
                f.result()

        assert not errors, f"Errors occurred: {errors}"
        assert call_count[0] == 500  # 10 threads * 50 calls

    def test_reset_during_concurrent_access(self):
        """Test reset() is safe during concurrent access."""
        monitor = Monitor(max_identical_outputs=1000)
        stop_flag = threading.Event()
        errors = []

        def checker():
            while not stop_flag.is_set():
                try:
                    monitor.check("output", latency_ms=1)
                except Exception as e:
                    errors.append(e)

        def resetter():
            for _ in range(10):
                time.sleep(0.01)
                monitor.reset()

        checker_thread = threading.Thread(target=checker)
        resetter_thread = threading.Thread(target=resetter)

        checker_thread.start()
        resetter_thread.start()

        resetter_thread.join()
        stop_flag.set()
        checker_thread.join()

        assert not errors, f"Errors occurred: {errors}"
