"""
Part 4 — from pixels to metric 3D.

A pinhole camera throws away scale, so one known length has to come from
somewhere.  The challenge supplies it: the circle is 10 inches in radius.  Its
image is therefore a direct ruler.

    r_px / f = R / Z        ->        Z = f * R / r_px

and once Z is known for the plane, every other centre back-projects with

    X = (u - cx) * Z / fx        Y = (v - cy) * Z / fy

The "assume a flat surface" in the brief is what makes this enough for *all*
the shapes rather than just the circle: they all sit on one plane, viewed
head-on, so they share the circle's depth.  Each shape's physical size then
falls out of the same scale factor for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .detector import Detection

# Given in the challenge brief.  Note cx = cy = 0: taken literally the
# principal point sits in the top-left corner of the image, which no real
# camera does.  See CameraModel.principal_point for how that is handled.
K_CHALLENGE = np.array(
    [
        [2564.3186869, 0.0, 0.0],
        [0.0, 2569.70273111, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

CIRCLE_RADIUS_IN = 10.0  # inches, given

# The intrinsics belong to a resolution; the videos are 1080p and the static
# image is a half-size render of the same view.
CALIB_SIZE = (1920, 1080)


@dataclass
class Pose3D:
    """A shape's centre in camera coordinates, inches."""

    x: float
    y: float
    z: float
    size_in: float  # equivalent-circle radius of the shape, in inches
    from_circle: bool  # True if the scale came from a circle in THIS frame

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class CameraModel:
    """Pinhole model + the metric scale recovered from the reference circle.

    Parameters
    ----------
    K:
        3x3 intrinsic matrix.
    image_size:
        Size of the frames being processed.  If it differs from the resolution
        the intrinsics were calibrated at, K is rescaled — fx, fy, cx and cy are
        all in pixels, so they all scale with the image.
    assume_centered_principal_point:
        The supplied K has cx = cy = 0.  With that taken literally, the image
        centre back-projects to a point ~19 inches off-axis at 21 ft, and every
        X/Y is biased by half the sensor.  Defaulting to the image centre is the
        physically sensible reading of a simplified brief; pass False to use the
        matrix exactly as given.
    """

    def __init__(
        self,
        K: np.ndarray = K_CHALLENGE,
        image_size: tuple[int, int] = CALIB_SIZE,
        assume_centered_principal_point: bool = True,
        circle_radius_in: float = CIRCLE_RADIUS_IN,
    ) -> None:
        self.K = np.array(K, dtype=float)
        self.circle_radius_in = circle_radius_in
        self.assume_centered = assume_centered_principal_point
        self.set_image_size(image_size)
        self._last_scale: float | None = None  # inches per pixel, on the plane
        self._rejects = 0

    # -- geometry -----------------------------------------------------------

    def set_image_size(self, image_size: tuple[int, int]) -> None:
        """Rescale the intrinsics to the resolution actually being processed."""
        w, h = image_size
        sx = w / float(CALIB_SIZE[0])
        sy = h / float(CALIB_SIZE[1])
        self.image_size = (w, h)
        self.fx = float(self.K[0, 0]) * sx
        self.fy = float(self.K[1, 1]) * sy
        if self.assume_centered:
            self.cx, self.cy = w / 2.0, h / 2.0
        else:
            self.cx = float(self.K[0, 2]) * sx
            self.cy = float(self.K[1, 2]) * sy

    @property
    def f_mean(self) -> float:
        return 0.5 * (self.fx + self.fy)

    def depth_from_circle(self, radius_px: float) -> float:
        """Z = f * R / r.  The one place scale enters the whole pipeline."""
        if radius_px <= 0:
            return float("nan")
        return self.f_mean * self.circle_radius_in / radius_px

    def backproject(self, u: float, v: float, z: float) -> tuple[float, float, float]:
        """Pixel + depth -> 3D point in camera coordinates (inches)."""
        return ((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z)

    # -- applied to a frame's detections -------------------------------------

    def locate(self, detections: list[Detection]) -> list[Detection]:
        """Attach a Pose3D to every detection, in place, and return them.

        The reference circle sets the plane's depth for the whole frame.  If no
        circle is visible this frame the last known scale is reused and the
        result is flagged (`from_circle=False`) rather than dropped — losing
        sight of the ruler should not blind the rest of the frame.
        """
        circle = self._reference_circle(detections)
        radius_px = circle.fit_radius if circle is not None else None

        if radius_px is not None and self._implausible(radius_px):
            # The scene is rigid, so the ruler cannot change size by 25% between
            # frames: something else got mistaken for the circle.  Ride on the
            # previous scale, but give up and re-acquire if this persists (which
            # is what a real altitude change would look like).
            self._rejects += 1
            if self._rejects > 30:
                self._last_scale = None
                self._rejects = 0
            radius_px = None
        elif radius_px is not None:
            self._rejects = 0

        if radius_px is not None:
            z = self.depth_from_circle(radius_px)
            self._last_scale = self.circle_radius_in / radius_px  # inches per px
            fresh = True
        elif self._last_scale is not None:
            z = self.f_mean * self._last_scale
            fresh = False
        else:
            return detections  # nothing to scale with, yet

        for det in detections:
            u, v = det.center
            x, y, zz = self.backproject(u, v, z)
            det.extras["pose"] = Pose3D(
                x=x,
                y=y,
                z=zz,
                size_in=det.equiv_radius * self._last_scale,
                from_circle=fresh,
            )
        return detections

    def _implausible(self, radius_px: float) -> bool:
        """Would accepting this radius imply an impossible jump in scale?"""
        if self._last_scale is None:
            return False
        ratio = (self.circle_radius_in / radius_px) / self._last_scale
        return not (0.75 <= ratio <= 1.33)

    @staticmethod
    def _reference_circle(detections: list[Detection]) -> Detection | None:
        """The fully visible, ellipse-fitted circle, if there is one.

        Three conditions, each earned the hard way on the videos:

        * it must have survived the ellipse fit (`fit_radius`), because an
          area-derived radius is a different, noisier measurement and mixing the
          two makes the depth bimodal;
        * it must be genuinely round -- a hexagon scored 0.903 circularity in one
          frame and, accepted as the ruler, moved the whole plane;
        * it must not touch the border, since a clipped circle under-reports its
          radius and would push the scene away.
        """
        best, best_circ = None, 0.0
        for det in detections:
            if det.touches_border or det.fit_radius is None:
                continue
            if det.label != "circle" or det.circularity < 0.95:
                continue
            if det.circularity > best_circ:
                best, best_circ = det, det.circularity
        return best


def format_pose(pose: Pose3D) -> str:
    """Compact one-line rendering used by the overlay and the ROS2 node."""
    if pose is None or any(math.isnan(v) for v in pose.as_tuple()):
        return "3D: n/a"
    return f"X{pose.x:+.0f} Y{pose.y:+.0f} Z{pose.z:.0f} in"
