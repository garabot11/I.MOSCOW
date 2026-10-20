"""Add ``grade_profile`` to scene.json of already generated parametric scenes (same content the generator now writes;
computed from the stored tunnel spec, no regeneration).

    python solution/synth/tools/patch_grade_profile.py /home/gag/i.mos/synthetic/val_hesai_param [...]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src", "metro_obstacle_detector")))
from synth.parametric import grade_profile  # noqa: E402
from synth.tunnel import TunnelSpec  # noqa: E402


def main(argv):
    n = 0
    for set_dir in argv:
        for path in sorted(glob.glob(os.path.join(set_dir, "*", "scene.json"))):
            sj = json.load(open(path))
            if sj.get("source") != "parametric":
                continue
            spec = dict(sj["tunnel"]["spec"])
            if spec.get("platform") is not None:
                spec["platform"] = tuple(spec["platform"])
            mount = sj.get("mount", {})
            import math
            sj["grade_profile"] = grade_profile(TunnelSpec(**spec), sj["sensor_start_s_m"], mount.get("height_above_rail_m", 1.3),
                                                math.radians(mount.get("pitch_deg", 0.0)))
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(sj, fh, indent=1)
            os.replace(tmp, path)
            n += 1
    print(f"patched {n} scene.json files")


if __name__ == "__main__":
    main(sys.argv[1:])
