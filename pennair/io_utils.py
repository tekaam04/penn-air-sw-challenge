"""
Frame-by-frame video I/O.

The brief is explicit that the video must be treated as a stream — the aircraft
does not get to see the whole flight in advance — so this module never holds
more than the current frame, and nothing downstream is allowed to seek.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass
class StreamInfo:
    width: int
    height: int
    fps: float
    frame_count: int  # 0 when unknown (a live camera)


class FrameStream:
    """Iterate a video file (or camera) one frame at a time.

    Used as a context manager:

        with FrameStream("clip.mp4") as stream:
            for i, frame in stream:
                ...
    """

    def __init__(self, source: str | int, max_frames: int | None = None) -> None:
        self.source = source
        self.max_frames = max_frames
        self.cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "FrameStream":
        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        if isinstance(src, str) and not Path(src).exists():
            raise FileNotFoundError(f"no such video: {src}")
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source: {self.source}")
        return self

    def __exit__(self, *exc) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def info(self) -> StreamInfo:
        assert self.cap is not None
        return StreamInfo(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __iter__(self) -> Iterator:
        assert self.cap is not None, "use FrameStream as a context manager"
        i = 0
        while True:
            if self.max_frames is not None and i >= self.max_frames:
                return
            ok, frame = self.cap.read()
            if not ok:
                return
            yield i, frame
            i += 1


class VideoWriter:
    """Thin wrapper that sizes itself from the first frame it is given."""

    def __init__(self, path: str | Path, fps: float, fourcc: str = "avc1") -> None:
        self.path = str(path)
        self.fps = fps
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h)
            )
            if not self._writer.isOpened():
                # H.264 ("avc1") keeps the file small enough to commit and plays
                # in a browser; fall back if this OpenCV build lacks the encoder.
                self._writer = cv2.VideoWriter(
                    self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
                )
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open writer for {self.path}")
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class FpsMeter:
    """Rolling processing-rate meter (excludes disk I/O if you time only work)."""

    def __init__(self, window: int = 30) -> None:
        self.samples: deque[float] = deque(maxlen=window)
        self._t0: float | None = None

    def tick(self) -> None:
        now = time.perf_counter()
        if self._t0 is not None:
            self.samples.append(now - self._t0)
        self._t0 = now

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Record the elapsed time since start() and return it in ms."""
        assert self._t0 is not None
        dt = time.perf_counter() - self._t0
        self.samples.append(dt)
        self._t0 = None
        return dt * 1000.0

    @property
    def fps(self) -> float:
        if not self.samples:
            return 0.0
        mean = sum(self.samples) / len(self.samples)
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def ms(self) -> float:
        if not self.samples:
            return 0.0
        return 1000.0 * sum(self.samples) / len(self.samples)
