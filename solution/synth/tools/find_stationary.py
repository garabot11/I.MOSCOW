"""Find stationary segments of the real bags (cached in solution/synth/results/stationary_<bag>.json)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth.insert import stationary_segments  # noqa: E402
from synth.realbag import RealBag  # noqa: E402

BAGS = ["doubleT_platform", "squareT_platform_squareT_switch", "roundT_doubleT", "roundT_pressureGate_roundT",
        "roundT_squareT_pressureGate_squareT"]

if __name__ == "__main__":
    for name in (sys.argv[1:] or BAGS):
        b = RealBag(name)
        print(name, len(b), "stationary segments:", stationary_segments(b), flush=True)
