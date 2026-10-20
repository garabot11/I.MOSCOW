"""Consistency checks of a generated set (no ROS needed).

    python solution/synth/tools/verify_set.py /home/gag/i.mos/synthetic/val_hesai_insert [--json out.json]

Per scene: files present; one label line per bag message with matching stamps; per frame and object
n_hits == number of slot_labels with that obj_id; labelled slots are returns (non-zero points); for Hesai
layouts labelled slots are fired slots; is_obstacle implies in_clearance; dust never is_obstacle.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src", "metro_obstacle_detector")))
from metro_obstacle_detector.cloud_io import parse_pointcloud2_cdr, pointcloud2_to_struct  # noqa: E402


def read_mcap(path):
    from mcap.reader import make_reader

    files = glob.glob(os.path.join(path, "*.mcap"))
    with open(files[0], "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages():
            yield ch.topic, msg.log_time, msg.data


def check_scene(sdir):
    errs = []
    scene = json.load(open(os.path.join(sdir, "scene.json")))
    labels = [json.loads(l) for l in open(os.path.join(sdir, "labels.jsonl"))]
    sl = np.load(os.path.join(sdir, "slot_labels.npz"))
    fired = None
    style = scene.get("style")
    if scene.get("source") == "insert_real" and "renderer" in scene and style is None:
        errs.append("fake_style scene without 'style' in scene.json")
    if scene["sensor"] == "hesai128" and style != "fake_a":
        from metro_obstacle_detector.beam_model import hesai128

        # organised layouts; fake_style in_place / physical keep the 2400-column base layout
        fired = hesai128(scene["layout"].get("n_cols", 2400)).fired_mask()
    n_msgs, stamps, n_obj_frames, hits_total = 0, [], 0, 0
    classes = {}
    for k, (topic, t, raw) in enumerate(read_mcap(os.path.join(sdir, "bag"))):
        msg = parse_pointcloud2_cdr(raw)
        pts = pointcloud2_to_struct(msg)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if topic != scene["topic"]:
            errs.append(f"frame {k}: topic {topic} != {scene['topic']}")
        if msg.header.frame_id != scene["frame_id"]:
            errs.append(f"frame {k}: frame_id {msg.header.frame_id}")
        if k >= len(labels):
            errs.append(f"more messages than label lines ({k + 1} > {len(labels)})")
            break
        lab = labels[k]
        if lab["stamp_ns"] != stamp:
            errs.append(f"frame {k}: label stamp {lab['stamp_ns']} != header {stamp}")
        m = sl["frame"] == k
        slots, ids = sl["slot"][m], sl["obj_id"][m]
        xyz = np.stack([pts["x"][slots], pts["y"][slots], pts["z"][slots]], 1) if len(slots) else np.zeros((0, 3))
        if len(slots) and (np.abs(xyz).sum(1) == 0).any():
            errs.append(f"frame {k}: labelled slot without a return")
        if len(slots) and int(slots.max()) >= len(pts):
            errs.append(f"frame {k}: slot label beyond the message ({int(slots.max())} >= {len(pts)})")
            slots = slots[slots < len(pts)]
        if fired is not None and len(pts) != len(fired):
            errs.append(f"frame {k}: {len(pts)} points, organised layout expects {len(fired)}")
            fired = None
        if fired is not None and len(slots) and not fired[slots].all():
            errs.append(f"frame {k}: labelled slot not fired")
        if style is not None:
            if "n_points" in lab and lab["n_points"] != len(pts):
                errs.append(f"frame {k}: label n_points {lab['n_points']} != message {len(pts)}")
            if style == "fake_a":
                if msg.point_step != 16 or [f.name for f in msg.fields] != ["x", "y", "z", "intensity"]:
                    errs.append(f"frame {k}: fake_a layout (point_step {msg.point_step})")
                n_app = int(sum(o["n_hits"] for o in lab["objects"]))
                tail = np.arange(len(pts) - n_app, len(pts))
                if len(slots) != n_app or (len(slots) and not np.array_equal(np.sort(slots), tail)):
                    errs.append(f"frame {k}: fake_a labelled points are not exactly the appended tail")
            if style in ("fake_a", "in_place") and len(slots) and not np.all(pts["intensity"][slots] == 1.0):
                errs.append(f"frame {k}: {style} object point with intensity != 1.0")
        for o in lab["objects"]:
            n = int((ids == o["id"]).sum())
            classes[o["class"]] = classes.get(o["class"], 0) + 1
            n_obj_frames += 1
            hits_total += n
            if n != o["n_hits"]:
                errs.append(f"frame {k} obj {o['id']}: n_hits {o['n_hits']} != slot labels {n}")
            if o["visible"] != (o["n_hits"] > 0):
                errs.append(f"frame {k} obj {o['id']}: visible flag")
            if o["is_obstacle"] and not o["in_clearance"]:
                errs.append(f"frame {k} obj {o['id']}: obstacle outside clearance")
            if o["class"] == "dust" and o["is_obstacle"]:
                errs.append(f"frame {k} obj {o['id']}: dust is an obstacle")
        known = {o["id"] for o in lab["objects"]}
        if len(ids) and not set(np.unique(ids).tolist()) <= known:
            errs.append(f"frame {k}: slot labels with unknown ids")
        n_msgs += 1
    if n_msgs != len(labels):
        errs.append(f"messages {n_msgs} != label lines {len(labels)}")
    return {"scene": os.path.basename(sdir), "style": style, "messages": n_msgs, "object_frames": n_obj_frames,
            "hits_total": hits_total, "classes": classes, "errors": errs[:20], "n_errors": len(errs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rows = []
    for sdir in sorted(glob.glob(os.path.join(args.set_dir, "*", "scene.json"))):
        r = check_scene(os.path.dirname(sdir))
        rows.append(r)
        print(json.dumps(r), flush=True)
    bad = sum(r["n_errors"] for r in rows)
    print(f"{len(rows)} scenes, {sum(r['messages'] for r in rows)} frames, errors: {bad}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
