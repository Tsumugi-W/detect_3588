"""Real-thread tests for the ROS-independent latest-frame processor."""
import threading
import time
import unittest

from panel_detection.frame_processing import LatestFrameProcessor


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class FrameProcessingTest(unittest.TestCase):
    def make(self, callback, **kwargs):
        processor = LatestFrameProcessor(callback, **kwargs)
        self.addCleanup(processor.close)
        return processor

    def test_empty_queue_never_runs_callback(self):
        received = []
        processor = self.make(received.append)
        time.sleep(0.03)
        self.assertEqual(received, [])
        processor.close()
        self.assertFalse(processor._worker.is_alive())

    def test_slow_processing_keeps_only_newest_pair_and_does_not_block_submit(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        def process(color, depth):
            received.append((color, depth))
            if color == 0:
                entered.set()
                release.wait(1)

        processor = self.make(process)
        self.addCleanup(release.set)
        processor.submit(0, "depth0")
        self.assertTrue(entered.wait(1))
        for index in range(1, 101):
            processor.submit(index, "depth" + str(index))
        self.assertEqual(processor.received, 101)
        self.assertEqual(processor.dropped, 99)
        release.set()
        self.assertTrue(wait_until(lambda: len(received) == 2))
        self.assertEqual(received, [(0, "depth0"), (100, "depth100")])

    def test_rate_limited_without_reprocessing_last_image(self):
        times = []
        processor = self.make(lambda value: times.append(time.monotonic()), max_fps=10)
        deadline = time.monotonic() + 0.34
        while time.monotonic() < deadline:
            processor.submit(object())
            time.sleep(0.005)
        time.sleep(0.12)
        self.assertGreaterEqual(len(times), 2)
        self.assertLessEqual(len(times), 5)
        self.assertTrue(all(b - a >= 0.09 for a, b in zip(times, times[1:])))
        count = len(times)
        time.sleep(0.12)
        self.assertEqual(len(times), count)

    def test_activity_resumes_immediately_from_idle(self):
        received = []
        processor = self.make(received.append, max_fps=10, idle_fps=1)
        processor.set_activity("READY_FOR_COMMANDS")
        processor.submit("idle")
        self.assertTrue(wait_until(lambda: received == ["idle"]))
        processor.submit("active")
        time.sleep(0.03)
        self.assertEqual(received, ["idle"])
        processor.set_activity("COMMAND_PRESS_ID_2_ACQUIRE")
        self.assertTrue(wait_until(lambda: received == ["idle", "active"], 0.2))

    def test_unknown_activity_uses_active_rate(self):
        processor = self.make(lambda value: None, max_fps=5, idle_fps=1)
        for status in ("IDLE", "DONE", "STOPPED", "READY_FOR_COMMANDS", "FAILED: stopped"):
            processor.set_activity(status)
            self.assertEqual(processor._fps, 1)
        for status in ("UNKNOWN", "STARTING", "WAITING_OPERATION_GATE", "COMMAND_DONE_PRESS_5"):
            processor.set_activity(status)
            self.assertEqual(processor._fps, 5)

    def test_old_pending_frame_is_dropped_without_changing_timestamps(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        def process(value):
            received.append(value)
            entered.set()
            release.wait(1)

        processor = self.make(process, max_age_seconds=0.02)
        self.addCleanup(release.set)
        processor.submit("first")
        self.assertTrue(entered.wait(1))
        processor.submit("stale")
        time.sleep(0.05)
        release.set()
        self.assertTrue(wait_until(lambda: processor.dropped == 1))
        self.assertEqual(received, ["first"])

    def test_worker_recovers_after_callback_exception(self):
        errors, received = [], []

        def process(value):
            if value == "bad":
                raise ValueError("invalid frame")
            received.append(value)

        processor = self.make(process, on_error=errors.append)
        processor.submit("bad")
        self.assertTrue(wait_until(lambda: len(errors) == 1))
        processor.submit("good")
        self.assertTrue(wait_until(lambda: received == ["good"]))
        self.assertEqual(processor.errors, 1)

    def test_close_rejects_new_frames(self):
        received = []
        processor = self.make(received.append)
        processor.close()
        processor.submit("after close")
        self.assertEqual(processor.received, 0)
        self.assertEqual(received, [])

    def test_invalid_rates_are_rejected_before_thread_start(self):
        for kwargs in ({"max_fps": -1}, {"max_fps": float("nan")},
                       {"max_fps": 5, "idle_fps": 10}, {"max_fps": 5, "idle_fps": 0},
                       {"max_age_seconds": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LatestFrameProcessor(lambda value: None, **kwargs)


if __name__ == "__main__":
    unittest.main()
