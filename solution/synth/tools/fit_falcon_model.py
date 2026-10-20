"""Fit the Falcon Prime slot table from the real recording and report its accuracy on held-out frames.

    python solution/synth/tools/fit_falcon_model.py [--bag /home/gag/i.mos/dataset/falcon/falcon_test_second_dark]

Messages are fetched one by one by id (the bag is 20 GB).  Writes ``solution/synth/data/falcon_prime_seyond.npz``
and ``solution/synth/results/falcon_beam_model.json``.
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
from metro_obstacle_detector.beam_model import FalconBeamModel  # noqa: E402

DEFAULT_BAG = "/home/gag/i.mos/dataset/falcon/falcon_test_second_dark"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=DEFAULT_BAG)
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "falcon_prime_seyond.npz"))
    ap.add_argument("--report", default=os.path.join(os.path.dirname(HERE), "results", "falcon_beam_model.json"))
    args = ap.parse_args(argv)
    bag = RealBag(args.bag)
    n = len(bag)
    fit_idx = np.linspace(5, n - 6, args.n_fit).astype(int)
    test_idx = sorted(set((fit_idx[:-1] + np.diff(fit_idx) // 2)[::8].tolist()))  # between fit frames
    frames = [bag.frame(int(i)) for i in fit_idx]
    m = FalconBeamModel.estimate([f.cloud for f in frames], [f.stamp_ns for f in frames])
    m.meta.update({"bag": os.path.basename(args.bag), "fit_frames": fit_idx.tolist()})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    m.save(args.out)
    m2 = FalconBeamModel.load(args.out)
    assert np.allclose(m2.az_deg, m.az_deg, atol=1e-9) and np.allclose(m2.el_deg, m.el_deg, atol=1e-9)
    rows = []
    for i in test_idx:
        f = bag.frame(int(i))
        c = f.cloud
        first = c["is_2nd_return"] == 0
        j = m.slot_of_cloud(c[first])
        raw = m.angular_error_deg(c, corrected=False)
        cor = m.angular_error_deg(c, corrected=True)
        nr = m.no_return_mask(c)
        rows.append({"frame": int(i), "points": int(len(c)), "first_returns": int(first.sum()),
                     "second_returns": int((~first).sum()), "not_in_table": int((j < 0).sum()),
                     "no_return_slots": int(nr.sum()), "no_return_frac": float(nr.mean()),
                     "raw_err_p50": float(np.percentile(raw, 50)), "raw_err_p95": float(np.percentile(raw, 95)),
                     "raw_err_p99": float(np.percentile(raw, 99)),
                     "cor_err_p50": float(np.percentile(cor, 50)), "cor_err_p95": float(np.percentile(cor, 95)),
                     "cor_err_p99": float(np.percentile(cor, 99)), "cor_err_max": float(cor.max())})
        print(json.dumps(rows[-1]), flush=True)
    keys = ["not_in_table", "no_return_frac", "raw_err_p50", "raw_err_p95", "raw_err_p99", "cor_err_p50",
            "cor_err_p95", "cor_err_p99", "cor_err_max"]
    summary = {k: float(np.median([r[k] for r in rows])) for k in keys}
    rep = {"n_slots": m.n_slots, "meta": m.meta, "held_out_frames": test_idx, "held_out_median": summary, "rows": rows,
           "axes": {"forward": m.forward_axis, "up": m.up_axis},
           "seen_frac_hist": np.histogram(m.seen_frac, bins=[0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0001])[0].tolist()}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("slots", m.n_slots, "meta", json.dumps(m.meta)[:400])
    print("held-out median", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
