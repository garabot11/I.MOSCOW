"""metro_obstacle_detector: LiDAR obstacle detection for metro tunnels.

The package is split into a ROS-independent core (``cloud_io``, ``config``,
``track_model``, ``detector``, ``temporal``) and thin adapters
(``node`` for ROS 2 streaming, ``offline`` for sequential bag evaluation).
"""

__version__ = "0.1.0"
