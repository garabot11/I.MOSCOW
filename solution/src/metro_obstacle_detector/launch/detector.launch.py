"""Launch the obstacle detector (and optionally RViz2).

    ros2 launch metro_obstacle_detector detector.launch.py                      # input_topic:=auto
    ros2 launch metro_obstacle_detector detector.launch.py input_topic:=/lidar_points
    ros2 launch metro_obstacle_detector detector.launch.py input_topic:=/sensing/lidar/hesai128/pointcloud rviz:=true fixed_frame:=lidar_livox

With ``rviz:=true`` the RViz config is copied with the cloud display switched to
``input_topic`` and, if ``fixed_frame`` is given, the fixed frame set to it (it
must equal the ``header.frame_id`` of the clouds: there is no TF in the data;
``bag_tools info --frame-id`` prints it for a bag).
"""
import os
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("metro_obstacle_detector")
    default_cfg = os.path.join(share, "config", "detector.yaml")
    default_rviz = os.path.join(share, "config", "view.rviz")
    args = [
        DeclareLaunchArgument("input_topic", default_value="auto", description="PointCloud2 topic to subscribe, or 'auto' (default) = the only PointCloud2 topic in the graph"),
        DeclareLaunchArgument("config_file", default_value=default_cfg, description="detector YAML parameters"),
        DeclareLaunchArgument("qos_reliability", default_value="reliable", description="reliable (rosbag2 playback, recorded QoS) | best_effort (live sensors publishing best-effort)"),
        DeclareLaunchArgument("qos_depth", default_value="1", description="subscription history depth (1 = freshest frame)"),
        DeclareLaunchArgument("log_path", default_value="", description="optional JSONL log of every processed frame"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="false", description="also start RViz2"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
        DeclareLaunchArgument("fixed_frame", default_value="", description="RViz fixed frame = header.frame_id of the clouds (default: the one in rviz_config, hesai_lidar)"),
        DeclareLaunchArgument("rviz_geometry", default_value="1600x1000+40+40", description="X11 window geometry passed to RViz2 (-geometry)"),
    ]
    detector = Node(
        package="metro_obstacle_detector",
        executable="detector_node",
        name="metro_obstacle_detector",
        output="screen",
        parameters=[
            {
                "input_topic": LaunchConfiguration("input_topic"),
                "config_file": LaunchConfiguration("config_file"),
                "qos_reliability": LaunchConfiguration("qos_reliability"),
                "qos_depth": LaunchConfiguration("qos_depth"),
                "log_path": LaunchConfiguration("log_path"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )
    rviz = OpaqueFunction(function=_rviz, condition=IfCondition(LaunchConfiguration("rviz")))
    return LaunchDescription(args + [detector, rviz])


def _rviz(context):
    """RViz2 with the cloud display on ``input_topic`` and the requested fixed frame."""
    fixed = LaunchConfiguration("fixed_frame").perform(context).strip()
    topic = LaunchConfiguration("input_topic").perform(context)
    with open(LaunchConfiguration("rviz_config").perform(context), encoding="utf-8") as fh:
        text = fh.read()
    if topic.lower() != "auto":  # with auto the RViz cloud display keeps the topic of the config
        text = re.sub(r"(?m)^([ \t]*Value: )/lidar_points[ \t]*$", lambda m: m.group(1) + topic, text)
    if fixed:
        text = re.sub(r"(?m)^(\s*Fixed Frame: ).*$", lambda m: m.group(1) + fixed, text)
    with tempfile.NamedTemporaryFile("w", prefix="metro_view_", suffix=".rviz", delete=False) as fh:
        fh.write(text)
    # the fixed frame goes into the config only: rviz2 -f would also reset the camera of the config
    arguments = ["-d", fh.name, "-geometry", LaunchConfiguration("rviz_geometry").perform(context)]
    return [Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=arguments,
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )]
