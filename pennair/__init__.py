"""PennAiR 2026 Software Challenge — background-agnostic shape detection."""

from .depth import CIRCLE_RADIUS_IN, K_CHALLENGE, CameraModel
from .detector import Detection, ShapeDetector
from .overlay import draw_detections, draw_hud
from .tracker import ShapeTracker, Track

__all__ = [
    "Detection",
    "ShapeDetector",
    "Track",
    "ShapeTracker",
    "CameraModel",
    "K_CHALLENGE",
    "CIRCLE_RADIUS_IN",
    "draw_detections",
    "draw_hud",
]
