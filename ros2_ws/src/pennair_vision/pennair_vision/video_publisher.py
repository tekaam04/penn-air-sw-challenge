#!/usr/bin/env python3
"""
Node 1 of Part 5: stream a video file out as `sensor_msgs/Image`.

This stands in for the aircraft's camera driver.  It reads one frame at a time
and publishes on a timer at the file's own frame rate, so the detector node
downstream sees exactly what it would see in flight — a stream it cannot seek
in and cannot get ahead of.

    ros2 run pennair_vision video_publisher --ros-args \
        -p video_path:=/abs/path/dynamic_hard.mp4 -p loop:=true
"""

from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self) -> None:
        super().__init__("video_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("loop", True)
        self.declare_parameter("fps", 0.0)  # 0 = use the file's own rate

        path = self.get_parameter("video_path").value
        if not path:
            raise RuntimeError("set the video_path parameter")

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {path}")

        fps = float(self.get_parameter("fps").value) or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.loop = bool(self.get_parameter("loop").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.bridge = CvBridge()
        self.frames = 0

        topic = str(self.get_parameter("topic").value)
        self.pub = self.create_publisher(Image, topic, 10)
        self.timer = self.create_timer(1.0 / fps, self.tick)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"publishing {path} ({w}x{h} @ {fps:.2f} fps) on {topic}")

    def tick(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            self.get_logger().info(f"end of stream after {self.frames} frames")
            self.timer.cancel()
            rclpy.shutdown()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)
        self.frames += 1

    def destroy_node(self) -> bool:
        self.cap.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VideoPublisher()
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
