"""
Multi-object tracking: stable IDs for shapes across frames.

Detection is per-frame and memoryless, which is enough to draw outlines but not
enough to *follow* a shape: the moment two shapes overlap, one of them is
partly hidden, its centroid jumps, and a frame or two later it may vanish
entirely behind the other.  A downstream controller (which is the point of the
exercise — an aircraft picking a landing target) needs "shape 3 is still shape
3, and it is roughly here even though I cannot see it right now".

So each track carries a constant-velocity estimate, detections are matched to
predictions by a gated cost, and unmatched tracks coast on their prediction for
a while before being dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .detector import Detection


@dataclass
class Track:
    """One tracked shape."""

    track_id: int
    center: tuple[float, float]
    velocity: tuple[float, float]
    detection: Detection | None  # most recent detection, None while coasting
    label: str
    color: tuple[float, float, float]  # mean BGR, used as a matching cue
    area: float
    hits: int = 1  # total matched frames
    age: int = 1  # frames since creation
    misses: int = 0  # consecutive unmatched frames
    trail: list[tuple[float, float]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        """Seen enough times to be worth reporting (kills one-frame flickers)."""
        return self.hits >= 3

    @property
    def coasting(self) -> bool:
        """Currently predicted rather than observed — i.e. probably occluded."""
        return self.misses > 0

    def predict(self) -> tuple[float, float]:
        return (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])


class ShapeTracker:
    """Greedy nearest-prediction tracker with gating and occlusion coasting.

    Greedy assignment (rather than Hungarian) is deliberate: with fewer than a
    dozen targets that are far apart relative to their per-frame motion, the two
    give the same answer, and greedy adds no dependency and no allocation.

    Parameters
    ----------
    max_misses:
        How many consecutive frames a track may coast before it is dropped.
        At 30 fps, 15 frames is half a second of full occlusion.
    gate_frac:
        Matching gate as a fraction of the frame diagonal.
    smoothing:
        EMA factor on the velocity estimate.
    """

    def __init__(
        self,
        max_misses: int = 15,
        gate_frac: float = 0.08,
        smoothing: float = 0.5,
        trail_len: int = 16,
    ) -> None:
        self.max_misses = max_misses
        self.gate_frac = gate_frac
        self.smoothing = smoothing
        self.trail_len = trail_len
        self.tracks: list[Track] = []
        self._next_id = 1

    # -- main entry point ---------------------------------------------------

    def update(self, detections: list[Detection], frame_shape: tuple[int, ...]) -> list[Track]:
        """Advance all tracks by one frame. Returns the confirmed tracks."""
        diag = math.hypot(frame_shape[0], frame_shape[1])
        gate = self.gate_frac * diag

        # Cost for every (track, detection) pair, then take the cheapest pairs
        # first, skipping any whose track or detection is already spoken for.
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            px, py = tr.predict()
            for di, det in enumerate(detections):
                dist = math.hypot(det.center[0] - px, det.center[1] - py)
                if dist > gate:
                    continue
                pairs.append((self._cost(tr, det, dist, diag), ti, di))
        pairs.sort()

        matched_t: dict[int, int] = {}
        matched_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t[ti] = di
            matched_d.add(di)

        for ti, tr in enumerate(self.tracks):
            if ti in matched_t:
                self._apply(tr, detections[matched_t[ti]])
            else:
                self._coast(tr)

        for di, det in enumerate(detections):
            if di not in matched_d:
                self.tracks.append(self._spawn(det))

        # A track that stops being detected at the edge of the frame has left
        # the field of view, not gone behind something: coasting it would
        # invent a shape that is no longer there.
        h, w = frame_shape[:2]
        self.tracks = [
            t
            for t in self.tracks
            if t.misses <= self.max_misses
            and not (t.coasting and not (0 <= t.center[0] < w and 0 <= t.center[1] < h))
        ]
        return [t for t in self.tracks if t.confirmed]

    # -- internals ----------------------------------------------------------

    def _cost(self, tr: Track, det: Detection, dist: float, diag: float) -> float:
        """Distance, plus cheap appearance terms to break ties during overlap.

        When two shapes are on top of each other, position alone is ambiguous
        and swaps IDs.  Mean colour and area are nearly free to compute and are
        exactly what disambiguates a red rectangle from a yellow circle.
        """
        d_pos = dist / diag

        d_col = (
            sum((a - b) ** 2 for a, b in zip(tr.color, det.mean_color)) ** 0.5 / 441.7
        )  # normalised by the max distance in BGR space

        big = max(tr.area, det.area)
        d_area = abs(tr.area - det.area) / big if big > 0 else 1.0

        # Area is the least reliable of the three during occlusion (a half-
        # hidden shape legitimately shrinks), so it gets the smallest weight.
        return 1.0 * d_pos + 0.5 * d_col + 0.25 * d_area

    def _apply(self, tr: Track, det: Detection) -> None:
        """Fold a matched detection into a track."""
        vx = det.center[0] - tr.center[0]
        vy = det.center[1] - tr.center[1]
        if tr.misses > 0:  # spread the jump over the frames it coasted
            vx /= tr.misses + 1
            vy /= tr.misses + 1
        a = self.smoothing
        tr.velocity = (a * tr.velocity[0] + (1 - a) * vx, a * tr.velocity[1] + (1 - a) * vy)
        tr.center = det.center
        tr.detection = det
        tr.area = 0.7 * tr.area + 0.3 * det.area
        tr.color = tuple(0.8 * c + 0.2 * d for c, d in zip(tr.color, det.mean_color))
        if not det.touches_border:
            tr.label = det.label
        tr.hits += 1
        tr.age += 1
        tr.misses = 0
        self._push_trail(tr)

    def _coast(self, tr: Track) -> None:
        """No detection this frame: run on the motion model (likely occluded)."""
        tr.center = tr.predict()
        tr.detection = None
        tr.misses += 1
        tr.age += 1
        # Bleed off velocity so a long occlusion does not fling the track away.
        tr.velocity = (tr.velocity[0] * 0.9, tr.velocity[1] * 0.9)
        self._push_trail(tr)

    def _spawn(self, det: Detection) -> Track:
        tr = Track(
            track_id=self._next_id,
            center=det.center,
            velocity=(0.0, 0.0),
            detection=det,
            label=det.label,
            color=det.mean_color,
            area=det.area,
        )
        self._next_id += 1
        self._push_trail(tr)
        return tr

    def _push_trail(self, tr: Track) -> None:
        tr.trail.append(tr.center)
        if len(tr.trail) > self.trail_len:
            del tr.trail[0]


def id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR colour per track ID (golden-angle hue)."""
    import cv2

    hue = int((track_id * 137.508) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 235, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
