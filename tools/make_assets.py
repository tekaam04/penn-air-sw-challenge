#!/usr/bin/env python3
"""
Build the figures and GIF previews embedded in the README.

    python tools/make_assets.py

Not part of the detector — this only exists so the documentation images can be
regenerated from the videos rather than being mystery binaries in the repo.
Needs Pillow (`pip install pillow`) on top of the runtime requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pennair import CameraModel, ShapeDetector, ShapeTracker, draw_detections  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def gif_from_video(src: Path, dst: Path, start: int, count: int, width: int, fps: int) -> None:
    """Sample `count` frames starting at `start` into an animated GIF."""
    from PIL import Image

    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))

    frames = []
    i = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = round(frame.shape[0] * width / frame.shape[1])
            small = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames read from {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # A textured background is the worst case for GIF: every frame is full of
    # unique colours.  One shared 64-colour palette keeps the file committable.
    palette = frames[0].quantize(colors=64, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for f in frames]
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"wrote {dst}  ({len(frames)} frames, {dst.stat().st_size/1e6:.1f} MB)")


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Caption strip under a panel."""
    bar = np.full((34, img.shape[1], 3), 24, np.uint8)
    cv2.putText(
        bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([img, bar])


def pipeline_figure(frame: np.ndarray, dst: Path, panel_w: int = 470) -> None:
    """Four-panel figure: input, texture energy, mask, result."""
    det = ShapeDetector()
    detections = det.detect(frame, collect_debug=True)
    dbg = det.debug

    def prep(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h = round(img.shape[0] * panel_w / img.shape[1])
        return cv2.resize(img, (panel_w, h), interpolation=cv2.INTER_AREA)

    result = draw_detections(frame, detections)
    panels = [
        label(prep(frame), "1. input frame"),
        label(prep(255 - dbg["sigma"]), "2. texture energy (inverted: dark = rough)"),
        label(prep(dbg["mask"]), "3. smooth-region mask (robust threshold)"),
        label(prep(result), "4. watershed + model fit -> outlines and centres"),
    ]
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    fig = np.vstack([top, bottom])
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), fig)
    print(f"wrote {dst}")


def occlusion_figure(
    src: Path,
    dst: Path,
    frames: list[int],
    crop: tuple[int, int, int, int] | None = None,
    width: int = 620,
) -> None:
    """Strip of frames showing a shape tracked while another passes over it."""
    detector, tracker = ShapeDetector(), ShapeTracker()
    camera = CameraModel(image_size=(1920, 1080))
    cap = cv2.VideoCapture(str(src))
    wanted = set(frames)
    panels = []
    i = 0
    while i <= max(frames):
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector.detect(frame)
        camera.locate(dets)
        tracks = tracker.update(dets, frame.shape)
        if i in wanted:
            out = draw_detections(frame, tracks, show_3d=False, show_trails=False)
            if crop:
                x0, y0, x1, y1 = crop
                out = out[y0:y1, x0:x1]
            h = round(out.shape[0] * width / out.shape[1])
            panel = cv2.resize(out, (width, h), interpolation=cv2.INTER_AREA)
            panels.append(label(panel, f"frame {i}"))
        i += 1
    cap.release()
    cv2.imwrite(str(dst), np.hstack(panels))
    print(f"wrote {dst}")


def main() -> int:
    DOCS.mkdir(exist_ok=True)

    static = cv2.imread(str(ROOT / "data" / "static.png"))
    if static is not None:
        pipeline_figure(static, DOCS / "pipeline_static.png")

    cap = cv2.VideoCapture(str(ROOT / "data" / "dynamic_hard.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1800)
    ok, hard = cap.read()
    cap.release()
    if ok:
        pipeline_figure(hard, DOCS / "pipeline_hard.png")

    for name, start in [("dynamic_annotated.mp4", 520), ("hard_annotated.mp4", 520)]:
        src = ROOT / "outputs" / name
        if src.exists():
            gif_from_video(src, DOCS / (src.stem + ".gif"), start=start, count=40, width=440, fps=8)

    static_out = ROOT / "outputs" / "static_annotated.png"
    if static_out.exists():
        print(f"static result already at {static_out}")

    dyn = ROOT / "data" / "dynamic.mp4"
    if dyn.exists():
        occlusion_figure(
            dyn, DOCS / "occlusion.png", frames=[596, 604, 612], crop=(150, 300, 1150, 1010)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
