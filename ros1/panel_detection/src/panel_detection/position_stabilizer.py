"""ROS2-compatible temporal stabilization for panel target positions."""

from collections import defaultdict, deque

import numpy as np


class PositionStabilizer:
    """Filter 3D positions only after the image has remained still.

    While an eye-in-hand camera is moving, raw measurements are returned and
    any old stable value is discarded.  This mirrors the ROS2 detector and
    avoids dragging a target's previous camera-frame position into a new view.
    """

    def __init__(
        self,
        enabled=True,
        still_time=3.0,
        pixel_thresh=5.0,
        window_size=45,
        ema_alpha=0.25,
        depth_std_thresh=0.01,
    ):
        self.enabled = bool(enabled)
        self.still_time = float(still_time)
        self.pixel_thresh = float(pixel_thresh)
        self.window_size = max(3, int(window_size))
        self.ema_alpha = float(ema_alpha)
        self.depth_std_thresh = float(depth_std_thresh)
        self._last_pixels = {}
        self._still_since = {}
        self._histories = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._stable_xyz = {}

    def update(self, measurements, now):
        """Return ``{target_id: filtered_xyz}`` for current measurements."""

        raw = {
            target_id: np.asarray(xyz, dtype=np.float64)
            for target_id, _, xyz in measurements
        }
        if not self.enabled or not measurements:
            return {
                target_id: xyz.tolist()
                for target_id, xyz in raw.items()
            }

        pixels = {
            target_id: np.asarray(pixel_xy, dtype=np.float64)
            for target_id, pixel_xy, _ in measurements
        }
        current_ids = set(raw)
        for target_id in list(self._last_pixels):
            if target_id not in current_ids:
                self._last_pixels.pop(target_id, None)
                self._still_since.pop(target_id, None)

        filtered = {}
        for target_id, xyz in raw.items():
            last_pixel = self._last_pixels.get(target_id)
            is_still = (
                last_pixel is not None
                and np.linalg.norm(pixels[target_id] - last_pixel)
                <= self.pixel_thresh
            )
            if is_still:
                self._still_since.setdefault(target_id, now)
                self._histories[target_id].append(xyz)
            else:
                self._still_since[target_id] = now
                self._histories[target_id].clear()
                self._histories[target_id].append(xyz)
                self._stable_xyz.pop(target_id, None)

            self._last_pixels[target_id] = pixels[target_id]
            history = self._histories[target_id]
            static_ready = (
                now - self._still_since.get(target_id, now)
                >= self.still_time
                and len(history) >= 3
            )
            if static_ready:
                stacked = np.vstack(history)
                depth_stable = (
                    float(np.std(stacked[:, 2]))
                    <= self.depth_std_thresh
                )
                if depth_stable:
                    median_xyz = np.median(stacked, axis=0)
                    previous = self._stable_xyz.get(target_id)
                    stable = (
                        median_xyz
                        if previous is None
                        else (1.0 - self.ema_alpha) * previous
                        + self.ema_alpha * median_xyz
                    )
                    self._stable_xyz[target_id] = stable
                    filtered[target_id] = stable.tolist()
                    continue

            filtered[target_id] = self._stable_xyz.get(
                target_id, xyz
            ).tolist()

        return filtered
