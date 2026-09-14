#!/usr/bin/env python3
"""
Node 2 of Part 5: run the detector on every incoming image and publish results.

Subscribes
    /camera/image_raw           sensor_msgs/Image

Publishes
    /pennair/shapes             pennair_msgs/ShapeArray   (centres, outlines, 3D)
    /pennair/image_annotated    sensor_msgs/Image         (the overlay, for rqt)
    /pennair/markers            visualization_msgs/MarkerArray (for RViz)

The detection code itself is *not* duplicated here — this node is a thin ROS
wrapper around the same `pennair` package the CLI uses, so there is one
implementation to get right.
"""

from __future__ import annotations

import os
import sys

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Point32, Polygon
from pennair_msgs.msg import Shape, ShapeArray
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

# The algorithm lives in the repository root as a plain Python package.  Best
# practice is `pip install -e .` from the repo root so it is simply importable;
# PENNAIR_REPO is a fallback for a workspace built without that step.
try:
    from pennair import CameraModel, ShapeDetector, ShapeTracker, draw_detections, draw_hud
except ImportError:  # pragma: no cover - environment-dependent
    repo = os.environ.get("PENNAIR_REPO")
    if not repo:
        raise
    sys.path.insert(0, repo)
    from pennair import CameraModel, ShapeDetector, ShapeTracker, draw_detections, draw_hud


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("shape_detector")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("work_width", 960)
        self.declare_parameter("texture_window", 9)
        self.declare_parameter("outlier_k", 3.0)
        self.declare_parameter("track", True)
        self.declare_parameter("estimate_3d", True)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("literal_principal_point", False)

        self.detector = ShapeDetector(
            work_width=int(self.get_parameter("work_width").value),
            sigma_window=int(self.get_parameter("texture_window").value),
            outlier_k=float(self.get_parameter("outlier_k").value),
        )
        self.tracker = ShapeTracker() if bool(self.get_parameter("track").value) else None
        self.estimate_3d = bool(self.get_parameter("estimate_3d").value)
        self.camera: CameraModel | None = None  # sized on the first frame
        self.bridge = CvBridge()
        self.frames = 0

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.sub = self.create_subscription(
            Image, str(self.get_parameter("image_topic").value), self.on_image, sensor_qos
        )
        self.pub_shapes = self.create_publisher(ShapeArray, "/pennair/shapes", 10)
        self.pub_image = self.create_publisher(Image, "/pennair/image_annotated", sensor_qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/pennair/markers", 10)
        self.get_logger().info("shape detector ready")

    # -- callback -----------------------------------------------------------

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        detections = self.detector.detect(frame)
        if self.estimate_3d:
            if self.camera is None:
                self.camera = CameraModel(
                    image_size=(frame.shape[1], frame.shape[0]),
                    assume_centered_principal_point=not bool(
                        self.get_parameter("literal_principal_point").value
                    ),
                )
            self.camera.locate(detections)

        items = self.tracker.update(detections, frame.shape) if self.tracker else detections
        self.frames += 1

        self.pub_shapes.publish(self._to_msg(items, msg))
        if bool(self.get_parameter("publish_annotated").value):
            annotated = draw_detections(frame, items, show_3d=self.estimate_3d)
            draw_hud(annotated, [f"frame {self.frames}", f"{len(items)} shapes"])
            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.pub_image.publish(out)
        if bool(self.get_parameter("publish_markers").value):
            self.pub_markers.publish(self._to_markers(items, msg))

    # -- conversions --------------------------------------------------------

    @staticmethod
    def _detection_of(item):
        """Items are Tracks when tracking is on, Detections when it is off."""
        return getattr(item, "detection", item)

    def _to_msg(self, items, src: Image) -> ShapeArray:
        out = ShapeArray()
        out.header = src.header

        for item in items:
            det = self._detection_of(item)
            s = Shape()
            s.id = int(getattr(item, "track_id", 0))
            s.label = str(getattr(item, "label", "") or (det.label if det else ""))
            s.occluded = bool(getattr(item, "coasting", False))
            s.center_px = Point(x=float(item.center[0]), y=float(item.center[1]), z=0.0)

            if det is not None:
                poly = Polygon()
                poly.points = [
                    Point32(x=float(p[0]), y=float(p[1]), z=0.0)
                    for p in det.contour.reshape(-1, 2)
                ]
                s.outline = poly
                s.area_px = float(det.area)
                s.vertices = int(det.vertices)

                pose = det.extras.get("pose")
                if pose is not None:
                    s.has_3d = True
                    s.position = Point(x=float(pose.x), y=float(pose.y), z=float(pose.z))
                    s.depth_from_circle = bool(pose.from_circle)
                    s.size_in = float(pose.size_in)
            out.shapes.append(s)
        return out

    def _to_markers(self, items, src: Image) -> MarkerArray:
        """Outlines as RViz line strips, in the image plane (pixels, z = 0)."""
        arr = MarkerArray()
        clear = Marker()
        clear.header = src.header
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, item in enumerate(items):
            det = self._detection_of(item)
            if det is None:
                continue
            m = Marker()
            m.header = src.header
            m.ns = "outlines"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 2.0
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            pts = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in det.contour.reshape(-1, 2)]
            m.points = pts + pts[:1]  # close the loop
            arr.markers.append(m)
        return arr


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
