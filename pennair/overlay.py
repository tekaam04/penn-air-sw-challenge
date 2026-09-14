"""Drawing: outlines, centres, labels, 3D readouts and a small HUD."""

from __future__ import annotations

import cv2
import numpy as np

from .depth import Pose3D
from .detector import Detection
from .tracker import Track, id_color

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scaled(frame: np.ndarray) -> tuple[float, int]:
    """Font scale and line thickness that look the same at 540p and 1080p."""
    s = max(0.62, frame.shape[1] / 1400.0)
    return s * 0.62, max(1, int(round(s * 1.8)))


def _text(
    img: np.ndarray,
    s: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thick: int,
) -> None:
    """Text with a dark halo, so it survives both grass and dark gravel."""
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    items: list[Detection] | list[Track],
    show_3d: bool = False,
    show_trails: bool = True,
) -> np.ndarray:
    """Annotate a copy of `frame` with outlines, centres and labels.

    Accepts either raw detections (Part 1/2 style) or tracks (Part 6 style); a
    track additionally gets a stable colour, an ID, a motion trail, and stays
    drawn — dashed — while it is occluded.
    """
    out = frame.copy()
    scale, thick = _scaled(frame)

    for item in items:
        is_track = isinstance(item, Track)
        det = item.detection if is_track else item
        color = id_color(item.track_id) if is_track else (0, 255, 255)
        coasting = is_track and item.coasting

        if is_track and show_trails and len(item.trail) > 1:
            pts = np.array(item.trail, np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, color, max(1, thick - 1), cv2.LINE_AA)

        if det is not None and not coasting:
            cv2.drawContours(out, [det.contour], -1, color, thick + 1, cv2.LINE_AA)
        elif is_track:
            # Occluded: show where the tracker believes the shape is.
            cx, cy = int(item.center[0]), int(item.center[1])
            r = int(max(12, (item.area / 3.14159) ** 0.5))
            _dashed_circle(out, (cx, cy), r, color, thick)

        cx, cy = (int(item.center[0]), int(item.center[1]))
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, int(18 * scale / 0.62), thick)
        cv2.circle(out, (cx, cy), max(2, thick), (255, 255, 255), -1, cv2.LINE_AA)

        lines = []
        if is_track:
            lines.append(f"#{item.track_id} {item.label}" + (" occluded" if coasting else ""))
        else:
            lines.append(item.label)
        lines.append(f"({cx}, {cy}) px")

        pose: Pose3D | None = det.extras.get("pose") if det is not None else None
        if show_3d and pose is not None:
            lines.append(f"X {pose.x:+.1f}  Y {pose.y:+.1f} in")
            lines.append(f"Z {pose.z:.1f} in" + ("" if pose.from_circle else " held"))

        anchor = _label_anchor(det, item, out.shape, lines, scale, thick)
        _label_block(out, lines, anchor, color, scale, thick)

    return out


def _label_anchor(
    det: Detection | None,
    item,
    shape: tuple[int, ...],
    lines: list[str],
    scale: float,
    thick: int,
) -> tuple[int, int]:
    """Put the caption beside the shape, not on top of it, and keep it on screen."""
    pad = int(8 * scale / 0.62)
    lh = int(21 * scale / 0.62)
    bw = max(cv2.getTextSize(s, FONT, scale, thick)[0][0] for s in lines) + 2 * pad
    bh = lh * len(lines) + pad

    if det is not None:
        x, y, w, _ = cv2.boundingRect(det.contour)
    else:  # coasting: no contour, so box the predicted position instead
        r = int(max(12, (item.area / 3.14159) ** 0.5))
        x, y, w = int(item.center[0]) - r, int(item.center[1]) - r, 2 * r

    H, W = shape[:2]
    ax = x + w + pad  # first choice: to the right of the shape
    if ax + bw > W - 4:
        ax = x - bw - pad  # otherwise to its left
    ax = max(4, min(ax, W - bw - 4))
    ay = max(4, min(y, H - bh - 4))
    return ax, ay


def _label_block(
    img: np.ndarray,
    lines: list[str],
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thick: int,
) -> None:
    """Caption on a darkened plate, with a colour bar tying it to its shape."""
    pad = int(8 * scale / 0.62)
    lh = int(21 * scale / 0.62)
    bw = max(cv2.getTextSize(s, FONT, scale, thick)[0][0] for s in lines) + 2 * pad
    bh = lh * len(lines) + pad
    x, y = anchor

    roi = img[y : y + bh, x : x + bw]
    if roi.size:
        roi[:] = (0.30 * roi).astype(np.uint8)
        cv2.rectangle(img, (x, y), (x + max(2, pad // 3), y + bh), color, -1)

    ty = y + lh - int(5 * scale / 0.62)
    for i, line in enumerate(lines):
        _text(
            img,
            line,
            (x + pad + max(2, pad // 3), ty),
            color if i == 0 else (235, 235, 235),
            scale,
            max(1, thick - 1),
        )
        ty += lh


def _dashed_circle(
    img: np.ndarray, center: tuple[int, int], r: int, color: tuple[int, int, int], thick: int
) -> None:
    for a in range(0, 360, 30):
        cv2.ellipse(img, center, (r, r), 0, a, a + 16, color, thick, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (14, 14),
) -> np.ndarray:
    """Top-left status panel: fps, counts, mode."""
    scale, thick = _scaled(frame)
    pad = int(10 * scale / 0.62)
    h = int(24 * scale / 0.62)
    w = max(cv2.getTextSize(s, FONT, scale, thick)[0][0] for s in lines) + 2 * pad

    x, y = origin
    panel = frame[y : y + h * len(lines) + pad, x : x + w]
    if panel.size:
        panel[:] = (0.30 * panel).astype(np.uint8)

    ty = y + h - int(6 * scale / 0.62)
    for line in lines:
        _text(frame, line, (x + pad, ty), (255, 255, 255), scale, thick - 1 or 1)
        ty += h
    return frame
