#!/usr/bin/env python3
"""
PennAiR 2026 Software Challenge — command line entry point.

    python main.py image data/static.png -o outputs/static_annotated.png
    python main.py video data/dynamic.mp4 -o outputs/dynamic_annotated.mp4
    python main.py video data/dynamic_hard.mp4 -o outputs/hard_annotated.mp4 --3d
    python main.py video 0 --show            # live camera

Parts 1-4 and 6 all run through here; Part 5 (ROS2) wraps the same package in
ros2_ws/src/pennair_vision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from pennair import CameraModel, ShapeDetector, ShapeTracker, draw_detections, draw_hud
from pennair.io_utils import FpsMeter, FrameStream, VideoWriter


def build_detector(args: argparse.Namespace) -> ShapeDetector:
    return ShapeDetector(
        work_width=args.work_width,
        sigma_window=args.texture_window,
        outlier_k=args.outlier_k,
        min_area_frac=args.min_area_frac,
    )


def detections_to_json(items, frame_index: int) -> dict:
    """Serialisable record of one frame — the same payload the ROS2 node sends."""
    out = []
    for it in items:
        det = getattr(it, "detection", it)
        rec = {
            "center": [round(it.center[0], 2), round(it.center[1], 2)],
            "label": getattr(it, "label", None) or (det.label if det else None),
        }
        if hasattr(it, "track_id"):
            rec["id"] = it.track_id
            rec["occluded"] = it.coasting
        if det is not None:
            rec["area_px"] = round(det.area, 1)
            rec["vertices"] = det.vertices
            rec["contour"] = det.contour.reshape(-1, 2).tolist()
            pose = det.extras.get("pose")
            if pose is not None:
                rec["xyz_in"] = [round(pose.x, 2), round(pose.y, 2), round(pose.z, 2)]
                rec["size_in"] = round(pose.size_in, 2)
                rec["depth_from_circle"] = pose.from_circle
        out.append(rec)
    return {"frame": frame_index, "shapes": out}


# ---------------------------------------------------------------------------
# Part 1: static image
# ---------------------------------------------------------------------------


def run_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(args.source)
    if frame is None:
        print(f"error: could not read image {args.source}", file=sys.stderr)
        return 1

    detector = build_detector(args)
    meter = FpsMeter()
    meter.start()
    detections = detector.detect(frame, collect_debug=args.debug)
    ms = meter.stop()

    camera = None
    if args.three_d:
        camera = CameraModel(
            image_size=(frame.shape[1], frame.shape[0]),
            assume_centered_principal_point=not args.literal_principal_point,
        )
        camera.locate(detections)

    print(f"{len(detections)} shapes in {ms:.1f} ms")
    for det in detections:
        line = (
            f"  {det.label:9s} center=({det.center[0]:7.1f},{det.center[1]:7.1f}) "
            f"area={det.area:9.0f}px  verts={det.vertices}  circ={det.circularity:.3f}"
        )
        pose = det.extras.get("pose")
        if pose is not None:
            line += f"  XYZ=({pose.x:+.1f},{pose.y:+.1f},{pose.z:.1f}) in"
        print(line)

    annotated = draw_detections(frame, detections, show_3d=args.three_d)
    draw_hud(
        annotated,
        [f"{len(detections)} shapes", f"{ms:.1f} ms/frame"],
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, annotated)
        print(f"wrote {args.output}")
    if args.debug:
        stem = Path(args.output or "outputs/debug").with_suffix("")
        for name, img in detector.debug.items():
            p = f"{stem}_{name}.png"
            cv2.imwrite(p, img)
            print(f"wrote {p}")
    if args.json:
        Path(args.json).write_text(json.dumps(detections_to_json(detections, 0), indent=2))
        print(f"wrote {args.json}")
    if args.show:
        cv2.imshow("PennAiR — static", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# Parts 2/3/4/6: video, one frame at a time
# ---------------------------------------------------------------------------


def run_video(args: argparse.Namespace) -> int:
    detector = build_detector(args)
    tracker = None if args.no_track else ShapeTracker()
    camera = None
    writer = None
    meter = FpsMeter(window=60)
    records = []

    with FrameStream(args.source, max_frames=args.max_frames) as stream:
        info = stream.info
        print(
            f"source {args.source}: {info.width}x{info.height} @ {info.fps:.2f} fps, "
            f"{info.frame_count or 'unknown'} frames"
        )
        if args.three_d:
            camera = CameraModel(
                image_size=(info.width, info.height),
                assume_centered_principal_point=not args.literal_principal_point,
            )
        if args.output:
            writer = VideoWriter(args.output, info.fps)

        shape_counts = []
        for i, frame in stream:
            meter.start()
            detections = detector.detect(frame)
            if camera is not None:
                camera.locate(detections)
            items = tracker.update(detections, frame.shape) if tracker else detections
            meter.stop()

            shape_counts.append(len(detections))
            if args.json:
                records.append(detections_to_json(items, i))

            annotated = draw_detections(frame, items, show_3d=args.three_d)
            hud = [
                f"frame {i}" + (f"/{info.frame_count}" if info.frame_count else ""),
                f"{len(items)} shapes",
                f"{meter.ms:.1f} ms  ({meter.fps:.0f} fps)",
            ]
            if tracker is not None:
                occluded = sum(1 for t in items if t.coasting)
                if occluded:
                    hud.append(f"{occluded} occluded (tracked)")
            draw_hud(annotated, hud)

            if args.out_width and annotated.shape[1] != args.out_width:
                oh = round(annotated.shape[0] * args.out_width / annotated.shape[1])
                annotated = cv2.resize(
                    annotated, (args.out_width, oh), interpolation=cv2.INTER_AREA
                )
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("PennAiR — video", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.progress and i % 100 == 0:
                print(f"  frame {i}: {len(items)} shapes, {meter.ms:.1f} ms")

    if writer is not None:
        writer.release()
        print(f"wrote {args.output}")
    if args.show:
        cv2.destroyAllWindows()
    if args.json:
        Path(args.json).write_text(json.dumps(records))
        print(f"wrote {args.json}")

    n = len(shape_counts)
    if n:
        print(
            f"processed {n} frames, mean {sum(shape_counts)/n:.2f} shapes/frame, "
            f"{meter.ms:.1f} ms/frame ({meter.fps:.0f} fps)"
        )
    return 0


# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect solid shapes on any textured background.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", choices=["image", "video"])
    p.add_argument("source", help="image path, video path, or camera index")
    p.add_argument("-o", "--output", help="where to write the annotated result")
    p.add_argument("--json", help="also dump detections as JSON")
    p.add_argument("--show", action="store_true", help="display a window")
    p.add_argument("--debug", action="store_true", help="save intermediate stages (image mode)")
    p.add_argument("--progress", action="store_true", help="print progress every 100 frames")

    p.add_argument("--3d", dest="three_d", action="store_true", help="Part 4: metric X/Y/Z")
    p.add_argument(
        "--literal-principal-point",
        action="store_true",
        help="use cx=cy=0 exactly as the brief's K states, instead of the image centre",
    )
    p.add_argument("--no-track", action="store_true", help="per-frame detections only, no IDs")
    p.add_argument("--max-frames", type=int, help="stop after N frames")
    p.add_argument("--out-width", type=int, help="downscale the annotated output to this width")

    p.add_argument("--work-width", type=int, default=960, help="processing resolution")
    p.add_argument("--texture-window", type=int, default=9, help="texture window, px")
    p.add_argument("--outlier-k", type=float, default=3.0, help="smoothness threshold, in MADs")
    p.add_argument("--min-area-frac", type=float, default=0.0012, help="smallest shape accepted")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_image(args) if args.mode == "image" else run_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
