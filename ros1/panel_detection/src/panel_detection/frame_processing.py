"""Bounded, rate-limited image processing without ROS or busy waiting."""
import math
import threading
import time


class LatestFrameProcessor:
    """One worker and one pending frame; newer frames replace pending ones.

    Callbacks must not retain/mutate submitted messages. Rate limits apply to
    processing starts, not ROS publication. Original image stamps are kept.
    """

    def __init__(self, callback, max_fps=0.0, idle_fps=None,
                 max_age_seconds=0.4, on_error=None):
        self.max_fps = float(max_fps)
        self.idle_fps = self.max_fps if idle_fps is None else float(idle_fps)
        self.max_age_seconds = float(max_age_seconds)
        for value in (self.max_fps, self.idle_fps, self.max_age_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("processing rates and maximum age must be finite and nonnegative")
        if self.max_fps and (not self.idle_fps or self.idle_fps > self.max_fps):
            raise ValueError("idle_fps must be positive and no greater than max_fps")
        self.callback = callback
        self.on_error = on_error
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._fps = self.max_fps
        self._last_start = None
        self.received = self.processed = self.dropped = self.errors = 0
        self.last_processing_seconds = 0.0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, *messages):
        with self._condition:
            if self._closed:
                return
            self.received += 1
            if self._pending is not None:
                self.dropped += 1
            self._pending = (messages, time.monotonic())
            self._condition.notify()

    def set_activity(self, status):
        # Unknown states are active (fail open for perception). Only explicit
        # idle/final states lower inference rate; cameras never stop publishing.
        idle = status in ("IDLE", "READY_FOR_COMMANDS", "DONE", "STOPPED") or status.startswith("FAILED")
        fps = self.idle_fps if idle else self.max_fps
        with self._condition:
            if fps != self._fps:
                self._fps = fps
                self._last_start = None
                self._condition.notify()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=2.0)

    def _run(self):
        while True:
            with self._condition:
                while not self._closed:
                    if self._pending is None:
                        self._condition.wait()
                        continue
                    now = time.monotonic()
                    interval = 1.0 / self._fps if self._fps else 0.0
                    delay = 0.0 if self._last_start is None else self._last_start + interval - now
                    if delay > 0:
                        self._condition.wait(timeout=delay)
                        continue
                    messages, received_at = self._pending
                    self._pending = None
                    if self.max_age_seconds and now - received_at > self.max_age_seconds:
                        self.dropped += 1
                        continue
                    self._last_start = now
                    break
                if self._closed:
                    return
            start = time.monotonic()
            try:
                self.callback(*messages)
                self.processed += 1
            except Exception as error:
                self.errors += 1
                if self.on_error is not None:
                    self.on_error(error)
            finally:
                self.last_processing_seconds = time.monotonic() - start
