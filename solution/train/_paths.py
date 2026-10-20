"""Make ``metro_obstacle_detector`` (solution/src/...) and the sibling ``d3_*`` modules importable."""
import os
import sys

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.dirname(TRAIN_DIR)
PKG_DIR = os.path.join(SOLUTION_DIR, "src", "metro_obstacle_detector")
for _p in (PKG_DIR, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SYNTHETIC_ROOT = "/home/gag/i.mos/synthetic"
WORK_DIR = "/home/gag/i.mos/work/ml_d3"            # cache + runs, outside git (override with --work)
MODELS_DIR = os.path.join(SOLUTION_DIR, "models")
