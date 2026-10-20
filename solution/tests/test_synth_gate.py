"""Gate: a synthetic 0.9 x 0.9 x 0.85 m box at the real object's position in doubleT_obstacle gives the same number
of returns as the real object (+-20 %).  Needs the dataset (skipped otherwise); ~30 s."""
import glob
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "synth"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "synth", "tools"))


def test_gate_box_vs_real_object(tmp_path):
    root = os.environ.get("IMOS_DATASET", "/home/gag/i.mos/dataset/for_hackathon")
    if not glob.glob(os.path.join(root, "doubleT_obstacle", "*.db3")):
        pytest.skip("doubleT_obstacle not available")
    import gate_hesai

    out = tmp_path / "gate.json"
    assert gate_hesai.main(["--out", str(out)]) == 0
    rep = json.load(open(out))
    real = rep["real"]["slots_mean"]
    spec = rep["synthetic"][0]
    assert 120 < real < 180                      # independent real count (per-slot differencing)
    assert abs(spec["ratio_to_real_slots"] - 1.0) <= 0.2, spec
    assert rep["verdict"]["pass_pm20"]
