"""
Tests that run without the challenge media.

Every fixture here is synthesised, so the suite works on a fresh clone and
pins down the behaviour that actually matters: the detector keys on texture
rather than colour, it finds the right shapes, it survives gradients and
overlaps, and the 3D maths inverts correctly.

    pytest -q
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pennair import CameraModel, ShapeDetector, ShapeTracker
from pennair.depth import K_CHALLENGE

RNG = np.random.default_rng(0xA17)


# ---------------------------------------------------------------------------
# synthetic scenes
# ---------------------------------------------------------------------------


def textured_background(w: int = 960, h: int = 540, base=(60, 120, 60), grain: int = 55):
    """A noisy background: the thing the detector must learn to ignore."""
    bg = np.full((h, w, 3), base, np.uint8)
    noise = RNG.integers(-grain, grain, (h, w, 3), dtype=np.int16)
    speckle = cv2.GaussianBlur(noise.astype(np.float32), (3, 3), 0)
    return np.clip(bg.astype(np.float32) + speckle, 0, 255).astype(np.uint8)


def add_polygon(img, pts, color):
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)
    return img


def add_gradient_circle(img, center, radius, c0, c1):
    """A circle filled with a linear ramp — the Part 3 extra-credit case."""
    x0, y0 = center
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    t = np.clip((xs - (x0 - radius)) / (2.0 * radius), 0, 1)[..., None]
    ramp = (np.array(c0) * (1 - t) + np.array(c1) * t).astype(np.uint8)
    mask = ((xs - x0) ** 2 + (ys - y0) ** 2) <= radius**2
    img[mask] = ramp[mask]
    return img


@pytest.fixture
def scene():
    """Three shapes on grass-green noise: triangle, square, gradient circle."""
    img = textured_background()
    add_polygon(img, [(120, 120), (240, 120), (240, 240), (120, 240)], (200, 40, 40))
    add_polygon(img, [(480, 380), (600, 380), (540, 260)], (40, 40, 210))
    add_gradient_circle(img, (760, 180), 70, (20, 200, 240), (200, 40, 200))
    return img


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_finds_every_shape(scene):
    dets = ShapeDetector().detect(scene)
    assert len(dets) == 3, [d.label for d in dets]


def test_labels_and_centres(scene):
    dets = {d.label: d for d in ShapeDetector().detect(scene)}
    assert set(dets) == {"quad", "triangle", "circle"}

    assert dets["quad"].center == pytest.approx((180, 180), abs=2.0)
    assert dets["circle"].center == pytest.approx((760, 180), abs=2.0)
    # Centroid of a triangle is the mean of its vertices.
    assert dets["triangle"].center == pytest.approx((540, 340), abs=3.0)


def test_gradient_fill_is_not_split(scene):
    """The whole point of the detrended texture cue: a ramp stays one shape."""
    circles = [d for d in ShapeDetector().detect(scene) if d.label == "circle"]
    assert len(circles) == 1
    expected = math.pi * 70**2
    assert circles[0].area == pytest.approx(expected, rel=0.08)


@pytest.mark.parametrize(
    "base,grain",
    [
        ((60, 120, 60), 55),  # grass green
        ((40, 40, 40), 70),  # dark gravel, like the hard video
        ((200, 200, 205), 45),  # bright concrete
        ((120, 60, 160), 50),  # something no one would tune for
    ],
)
def test_background_agnostic(base, grain):
    """Same shapes, four unrelated backgrounds, no parameter changes."""
    img = textured_background(base=base, grain=grain)
    add_polygon(img, [(120, 120), (240, 120), (240, 240), (120, 240)], (10, 10, 10))
    add_polygon(img, [(480, 380), (600, 380), (540, 260)], (250, 250, 250))
    dets = ShapeDetector().detect(img)
    assert len(dets) == 2, f"{base}: {[d.label for d in dets]}"


def test_overlapping_shapes_stay_separate():
    """Two shapes that touch must not merge into one blob."""
    img = textured_background()
    add_polygon(img, [(300, 200), (460, 200), (460, 360), (300, 360)], (30, 30, 220))
    add_polygon(img, [(420, 320), (580, 320), (580, 480), (420, 480)], (220, 200, 30))
    dets = ShapeDetector().detect(img)
    assert len(dets) == 2, [(d.label, d.area) for d in dets]


def test_empty_background_finds_nothing():
    assert ShapeDetector().detect(textured_background()) == []


def test_handles_degenerate_input():
    det = ShapeDetector()
    assert det.detect(None) == []
    assert det.detect(np.zeros((0, 0, 3), np.uint8)) == []


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------


def _moving_scene(step: int):
    img = textured_background()
    x = 200 + 6 * step
    add_polygon(img, [(x, 150), (x + 110, 150), (x + 110, 260), (x, 260)], (40, 40, 220))
    add_gradient_circle(img, (700, 380), 60, (20, 200, 240), (60, 240, 120))
    return img


def test_track_ids_are_stable_across_frames():
    det, trk = ShapeDetector(), ShapeTracker()
    ids_per_frame = []
    for step in range(12):
        frame = _moving_scene(step)
        tracks = trk.update(det.detect(frame), frame.shape)
        if tracks:
            ids_per_frame.append({t.track_id for t in tracks})

    assert ids_per_frame, "tracker never confirmed a track"
    assert all(s == ids_per_frame[-1] for s in ids_per_frame[-6:])
    assert len(ids_per_frame[-1]) == 2


def test_track_coasts_through_a_dropout():
    """A shape that vanishes for a few frames keeps its ID and is flagged."""
    det, trk = ShapeDetector(), ShapeTracker()
    for step in range(6):
        frame = _moving_scene(step)
        trk.update(det.detect(frame), frame.shape)
    before = {t.track_id for t in trk.tracks if t.confirmed}

    blank = textured_background()
    tracks = trk.update(det.detect(blank), blank.shape)
    assert {t.track_id for t in tracks} == before
    assert all(t.coasting for t in tracks)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def test_depth_inverts_the_projection():
    """Project a known circle at a known depth; the model must recover it."""
    cam = CameraModel(image_size=(1920, 1080))
    true_z = 254.0
    radius_px = cam.f_mean * cam.circle_radius_in / true_z
    assert cam.depth_from_circle(radius_px) == pytest.approx(true_z)


def test_backprojection_round_trip():
    cam = CameraModel(image_size=(1920, 1080))
    u, v, z = 1200.0, 300.0, 254.0
    x, y, zz = cam.backproject(u, v, z)
    assert zz == z
    assert x * cam.fx / z + cam.cx == pytest.approx(u)
    assert y * cam.fy / z + cam.cy == pytest.approx(v)


def test_intrinsics_scale_with_resolution():
    """Half-resolution frames need half the focal length, or depth doubles."""
    full = CameraModel(image_size=(1920, 1080))
    half = CameraModel(image_size=(960, 540))
    assert half.fx == pytest.approx(full.fx / 2)
    assert half.cx == pytest.approx(full.cx / 2)
    assert full.depth_from_circle(100.0) == pytest.approx(half.depth_from_circle(50.0))


def test_literal_principal_point_is_available():
    cam = CameraModel(image_size=(1920, 1080), assume_centered_principal_point=False)
    assert (cam.cx, cam.cy) == (K_CHALLENGE[0, 2], K_CHALLENGE[1, 2])


def test_scene_gets_metric_positions():
    """Everything on the plane shares the circle's depth, and X/Y follow signs."""
    img = textured_background()
    add_gradient_circle(img, (760, 180), 70, (20, 200, 240), (20, 200, 240))
    add_polygon(img, [(120, 300), (260, 300), (260, 440), (120, 440)], (200, 40, 40))

    cam = CameraModel(image_size=(img.shape[1], img.shape[0]))
    dets = cam.locate(ShapeDetector().detect(img))
    poses = [d.extras["pose"] for d in dets]

    assert len(poses) == 2
    assert poses[0].z == pytest.approx(poses[1].z)  # one flat plane
    assert poses[0].z > 0
    # The square is left of and below centre; the circle right of and above it.
    square = next(p for d, p in zip(dets, poses) if d.label != "circle")
    circle = next(p for d, p in zip(dets, poses) if d.label == "circle")
    assert square.x < 0 < circle.x
    assert circle.y < 0 < square.y
