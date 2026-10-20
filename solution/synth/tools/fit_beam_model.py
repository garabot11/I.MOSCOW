"""Fit the beam models of both message layouts from the real bags and report their accuracy.

    python solution/synth/tools/fit_beam_model.py [--out solution/synth/data] [--print-const]

Writes ``hesai128_2400.npz`` (307 200-point /lidar_points layout, from roundT_doubleT) and
``hesai128_7200.npz`` (921 600-point layout, from doubleT_obstacle), checks every model against
frames of all six bags and optionally prints the constants block for ``beam_model.hesai128``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth.realbag import RealBag  # noqa: E402
from metro_obstacle_detector.beam_model import BeamModel  # noqa: E402

FIT = {2400: ("roundT_doubleT", [0, 1, 2, 3, 60, 120, 180, 240]),
       7200: ("doubleT_obstacle", [0, 1, 2, 3, 100, 101, 199, 200])}
CHECK = ["roundT_doubleT", "doubleT_obstacle", "roundT_squareT_pressureGate_squareT", "doubleT_platform",
         "roundT_pressureGate_roundT", "squareT_platform_squareT_switch"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--print-const", action="store_true")
    ap.add_argument("--check-frames", type=int, default=6)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    models = {}
    for ncol, (bag, idx) in FIT.items():
        b = RealBag(bag)
        frames = [b.frame(i) for i in idx]
        m = BeamModel.estimate([f.cloud for f in frames], stamps_ns=[f.stamp_ns for f in frames])
        m.meta.update({"fitted_on": bag, "frames": idx})
        m.save(os.path.join(args.out, f"hesai128_{ncol}.npz"))
        models[ncol] = m
        print(f"layout {ncol}: dual={m.dual_return} step={m.meta['firing_step_deg']:.7f} window={m.az_window_deg} "
              f"fit max residual {m.meta['fit_max_residual_deg']:.2e} deg", flush=True)
    same_el = np.abs(models[2400].elevation_deg - models[7200].elevation_deg).max()
    same_off = np.abs(models[2400].ring_az_offset_deg - models[7200].ring_az_offset_deg).max()
    print(f"2400 vs 7200: max |d elevation| {same_el:.2e} deg, max |d ring offset| {same_off:.2e} deg; "
          f"parity equal {np.array_equal(models[2400].ring_fire_parity, models[7200].ring_fire_parity)}")
    report = {}
    for bag in CHECK:
        b = RealBag(bag)
        idx = np.unique(np.linspace(0, len(b) - 1, args.check_frames).astype(int))
        worst, n_valid, unfired_valid = 0.0, 0, 0
        for i in idx:
            f = b.frame(int(i))
            m = models[len(f.cloud) // 128]
            err = m.angular_error_deg(f.cloud)
            worst = max(worst, float(err.max()))
            n_valid += len(err)
            xyz = np.stack([f.cloud["x"], f.cloud["y"], f.cloud["z"]], 1)
            v = ~(xyz == 0).all(1)
            unfired_valid += int((v & ~m.fired_mask()).sum())
        report[bag] = {"frames": idx.tolist(), "valid_returns": n_valid, "max_angle_err_deg": worst,
                       "valid_in_unfired_slots": unfired_valid}
        print(bag, json.dumps(report[bag]), flush=True)
    if args.print_const:
        m2, m7 = models[2400], models[7200]
        t2 = m2.col_time_s[0::2]
        per = float((t2[-1] - t2[0]) / (len(t2) - 1))
        const = {
            "step_deg": round(m2.meta["firing_step_deg"], 9),
            "firing_period_s": per,
            "elevation_deg": [round(float(v), 7) for v in m2.elevation_deg],
            "ring_az_offset_deg": [round(float(v), 7) for v in m2.ring_az_offset_deg],
            "ring_fire_parity": [int(v) for v in m2.ring_fire_parity],
            "layouts": {
                2400: {"az0_deg": round(float(m2.col_az_deg[0]), 7), "t0_s": float(m2.col_time_s[0]),
                       "az_window_deg": [round(v, 5) for v in m2.az_window_deg]},
                7200: {"az0_deg": round(float(m7.col_az_deg[0]), 7), "t0_s": float(m7.col_time_s[0]),
                       "az_window_deg": None},
            },
        }
        print("_HESAI_CONST = " + repr(const))
    return 0


if __name__ == "__main__":
    sys.exit(main())
