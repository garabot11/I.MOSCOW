import os
from glob import glob

from setuptools import setup

package_name = "metro_obstacle_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml") + glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Metro obstacle team",
    maintainer_email="gagik.rabota.1@gmail.com",
    description="Geometric obstacle detection ahead of a metro train from 3D LiDAR point clouds.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector_node = metro_obstacle_detector.node:main",
            "offline = metro_obstacle_detector.offline:main",
            "status_monitor = metro_obstacle_detector.status_monitor:main",
            "bag_tools = metro_obstacle_detector.bag_tools:main",
        ],
    },
)
