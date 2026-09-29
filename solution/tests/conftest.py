import os
import sys

# make the package importable without a colcon install (host or container)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "metro_obstacle_detector"))
