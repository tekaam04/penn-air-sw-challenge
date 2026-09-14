#!/usr/bin/env python3
"""
Measure the things the report claims.

    python tools/evaluate.py                  # everything
    python tools/evaluate.py --skip-video     # static accuracy + timing only

Three groups of numbers:

* **Accuracy** on the static image.  Ground truth comes from a *colour* flood —
  a completely different cue from the texture one the detector uses, so it is an
  independent reference rather than a restatement of the algorithm.  The seeds
  are five hand-picked points, one inside each shape.
* **Speed** at several working resolutions, on real 1080p frames.
* **Stability** over both videos: shapes found per frame, how often the circle
  is usable as a depth reference, and the spread of the recovered depth (the
  scene is rigid, so any spread is measurement noise).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pennair import CameraModel, ShapeDetector, ShapeTracker  # noqa: E402
from pennair.io_utils import FrameStream  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# One point inside each shape of data/static.png, read off the image by hand.
STATIC_SEEDS = [(112, 76), (279, 320), (691, 344), (553, 104), (839, 148)]


def ground_truth_mask(img: np.ndarray, seed: tuple[int, int], tol: float = 22.0) -> np.ndarray:
    """Colour-similarity region grown from a seed point, holes filled."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    sx, sy = seed
    dist = np.linalg.norm(lab - lab[sy, sx], axis=2)
    m = (dist < tol).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels = cv2.connectedComponents(m)
    m = (labels == labels[sy, sx]).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    flood = m.copy()
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    return cv2.bitwise_or(m, cv2.bitwise_not(flood))


def static_accuracy() -> list[str]:
    img = cv2.imread(str(ROOT / "data" / "static.png"))
    if img is None:
        return ["_data/static.png not found — skipped._"]

    dets = ShapeDetector().detect(img)
    rows = [
        "| shape | IoU vs colour GT | centre error (px) | vertices |",
        "| --- | --- | --- | --- |",
    ]
    ious, errs = [], []
    for seed in STATIC_SEEDS:
        gt = ground_truth_mask(img, seed) > 0
        mom = cv2.moments(gt.astype(np.uint8))
        gcx, gcy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]

        best_iou, best = 0.0, None
        for det in dets:
            pm = np.zeros(gt.shape, np.uint8)
            cv2.drawContours(pm, [det.contour], -1, 255, cv2.FILLED)
            inter = np.count_nonzero((pm > 0) & gt)
            union = np.count_nonzero((pm > 0) | gt)
            if union and inter / union > best_iou:
                best_iou, best = inter / union, det
        if best is None:
            rows.append("| (missed) | 0 | — | — |")
            ious.append(0.0)
            continue
        err = float(np.hypot(best.center[0] - gcx, best.center[1] - gcy))
        ious.append(best_iou)
        errs.append(err)
        rows.append(f"| {best.label} | {best_iou:.4f} | {err:.2f} | {best.vertices or '—'} |")

    rows.append(
        f"| **mean** | **{np.mean(ious):.4f}** | **{np.mean(errs):.2f}** | {len(dets)} found / "
        f"{len(STATIC_SEEDS)} expected |"
    )
    return rows


def timing(frames: int = 60) -> list[str]:
    src = ROOT / "data" / "dynamic_hard.mp4"
    if not src.exists():
        return ["_data/dynamic_hard.mp4 not found — skipped._"]

    buffered = []
    with FrameStream(str(src), max_frames=frames) as stream:
        for _, f in stream:
            buffered.append(f)

    rows = ["| working width | ms / frame | fps | shapes / frame |", "| --- | --- | --- | --- |"]
    for width in (640, 960, 1280, 1920):
        det = ShapeDetector(work_width=width)
        det.detect(buffered[0])  # warm up (first call pays for lazy allocations)
        counts, t0 = [], time.perf_counter()
        for f in buffered:
            counts.append(len(det.detect(f)))
        dt = (time.perf_counter() - t0) / len(buffered)
        rows.append(
            f"| {width} px | {dt*1000:.1f} | {1/dt:.0f} | {np.mean(counts):.2f} |"
            + (" *(default)*" if width == 960 else "")
        )
    return rows


def video_stability(path: Path, expected: int = 5) -> list[str]:
    if not path.exists():
        return [f"_{path.name} not found — skipped._"]

    det, trk = ShapeDetector(), ShapeTracker()
    cam = CameraModel(image_size=(1920, 1080))
    counts, depths, radii, fresh, ids = [], [], [], 0, set()
    t0 = time.perf_counter()

    with FrameStream(str(path)) as stream:
        for _, frame in stream:
            dets = det.detect(frame)
            cam.locate(dets)
            tracks = trk.update(dets, frame.shape)
            counts.append(len(dets))
            ids.update(t.track_id for t in tracks)

            ref = cam._reference_circle(dets)
            if ref is not None:
                fresh += 1
                r = ref.fit_radius or ref.equiv_radius
                radii.append(r)
                depths.append(cam.depth_from_circle(r))

    dt = (time.perf_counter() - t0) / max(1, len(counts))
    counts_a = np.array(counts)
    out = [
        f"**{path.name}** — {len(counts)} frames, {dt*1000:.1f} ms/frame ({1/dt:.0f} fps)",
        "",
        f"* shapes per frame: mean {counts_a.mean():.2f}, median {np.median(counts_a):.0f}, "
        f"max {counts_a.max()}",
        f"* frames with the full set visible ({expected}): "
        f"{100*np.mean(counts_a >= expected):.1f}%  "
        f"(the rest have shapes genuinely off-frame)",
        f"* frames with 0 detections: {int(np.sum(counts_a == 0))}",
        f"* distinct track IDs: {len(ids)} (shapes leave and re-enter the frame)",
    ]
    if depths:
        d = np.array(depths)
        r = np.array(radii)
        out += [
            f"* circle usable as depth reference: {fresh}/{len(counts)} frames "
            f"({100*fresh/len(counts):.1f}%)",
            f"* fitted circle radius: {r.mean():.2f} ± {r.std():.2f} px",
            f"* recovered depth: {d.mean():.1f} ± {d.std():.2f} in "
            f"({d.mean()/12:.2f} ft), spread {100*d.std()/d.mean():.2f}% of range",
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("-o", "--output", default=str(ROOT / "docs" / "metrics.md"))
    args = ap.parse_args()

    md = ["# Measured results", "", "Regenerate with `python tools/evaluate.py`.", ""]
    md += ["## Part 1 — static image accuracy", ""] + static_accuracy() + [""]
    md += ["## Speed vs working resolution", "", "60 frames of 1080p video, single thread.", ""]
    md += timing() + [""]

    if not args.skip_video:
        md += ["## Parts 2/3 — stability over the full videos", ""]
        for name in ("dynamic.mp4", "dynamic_hard.mp4"):
            md += video_stability(ROOT / "data" / name) + [""]

    text = "\n".join(md)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
