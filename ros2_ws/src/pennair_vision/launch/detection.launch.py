"""
Bring up the whole Part 5 system: video stream -> detector -> topics.

    ros2 launch pennair_vision detection.launch.py \
        video_path:=$HOME/penn-air-sw-challenge/data/dynamic_hard.mp4

Add `rviz:=true` for the RViz marker view, or watch the overlay with
`ros2 run rqt_image_view rqt_image_view /pennair/image_annotated`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    from launch_ros.actions import Node

    video_path = LaunchConfiguration("video_path")
    loop = LaunchConfiguration("loop")
    track = LaunchConfiguration("track")
    estimate_3d = LaunchConfiguration("estimate_3d")
    work_width = LaunchConfiguration("work_width")
    rviz = LaunchConfiguration("rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("video_path", description="absolute path to the input video"),
            DeclareLaunchArgument("loop", default_value="true"),
            DeclareLaunchArgument("track", default_value="true"),
            DeclareLaunchArgument("estimate_3d", default_value="true"),
            DeclareLaunchArgument("work_width", default_value="960"),
            DeclareLaunchArgument("rviz", default_value="false"),
            Node(
                package="pennair_vision",
                executable="video_publisher",
                name="video_publisher",
                output="screen",
                parameters=[{"video_path": video_path, "loop": loop}],
            ),
            Node(
                package="pennair_vision",
                executable="detector_node",
                name="shape_detector",
                output="screen",
                parameters=[
                    {
                        "track": track,
                        "estimate_3d": estimate_3d,
                        "work_width": work_width,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                condition=IfCondition(rviz),
                output="screen",
            ),
        ]
    )
