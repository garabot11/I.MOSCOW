"""Print the labels of a fake-style scene (frames 0 and last) and check them against the bag:
n_hits == labelled message points, labelled points have the object's range, message layout."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth.realbag import xyz_of  # noqa: E402


def read_mcap(bag_dir):
    from mcap.reader import make_reader
    from metro_obstacle_detector.cloud_io import parse_pointcloud2_cdr, pointcloud2_to_struct

    f = glob.glob(os.path.join(bag_dir, "*.mcap"))[0]
    with open(f, "rb") as fh:
        for _, _, msg in make_reader(fh).iter_messages():
            m = parse_pointcloud2_cdr(msg.data)
            yield m, np.array(pointcloud2_to_struct(m))


def main(scene):
    labels = [json.loads(l) for l in open(os.path.join(scene, "labels.jsonl"))]
    sl = np.load(os.path.join(scene, "slot_labels.npz"))
    info = json.load(open(os.path.join(scene, "scene.json")))
    print(os.path.basename(scene), "style", info["style"], "speed", info["speed_mps"], "s_total", info["ego_s_total_m"],
          "points/frame", info["points_per_frame"])
    errs = 0
    for j, (m, c) in enumerate(read_mcap(os.path.join(scene, "bag"))):
        lab = labels[j]
        sel = sl["frame"] == j
        xyz = xyz_of(c).astype(np.float64)
        for o in lab["objects"]:
            idx = sl["slot"][sel & (sl["obj_id"] == o["id"])]
            if len(idx) != o["n_hits"]:
                errs += 1
            if len(idx) and abs(np.linalg.norm(xyz[idx], axis=1).min() - o["distance_m"]) > 0.01:
                errs += 1
        if j in (0, len(labels) - 1):
            print(f"  frame {j}: point_step {m.point_step} n {len(c)} fields {[f.name for f in m.fields]}")
            for o in lab["objects"]:
                print(f"    id {o['id']} {o['class']:16s} dist {o['distance_m']:7.2f} hits {o['n_hits']:5d} del {o.get('n_deleted')} "
                      f"lat {o['lateral_m']:5.2f} bottom {o['bottom_above_rail_m']:5.2f} top {o['top_above_rail_m']:5.2f} "
                      f"obstacle {o['is_obstacle']}")
    print("  consistency errors:", errs)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
