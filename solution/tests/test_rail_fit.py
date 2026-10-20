"""fit_rail_level: an object standing on a sparse bed must not lever the near-range fit (floor_step_up_max_m)."""
import dataclasses

import numpy as np

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.track_model import fit_rail_level

Z0, GRADE = -2.0, -0.042          # Falcon-like: sensor 2.5 m above the bed, pitched up ~2.4 deg


def _bed(rng, start, counts, z0=Z0, grade=GRADE, lateral=0.6, noise=0.08):
    """Bed returns per 2 m bin: ``counts[k]`` returns in the bin starting at ``start + 2k`` (0 = no returns)."""
    d, l, h = [], [], []
    for k, n in enumerate(counts):
        if n <= 0:
            continue
        lo = start + 2.0 * k
        dd = rng.uniform(lo, lo + 2.0, n)
        d.append(dd)
        l.append(rng.uniform(-lateral, lateral, n))
        h.append(z0 + grade * dd + rng.uniform(-noise, noise, n))
    return np.concatenate(d), np.concatenate(l), np.concatenate(h)


def _object(rng, d0, n, h_above_bed=1.2, z0=Z0, grade=GRADE):
    dd = rng.uniform(d0, d0 + 2.0, n)
    return dd, rng.uniform(-0.3, 0.3, n), z0 + grade * dd + h_above_bed + rng.uniform(-0.3, 0.3, n)


def _scene(rng, with_object):
    # dense bed 10-25 m (bin counts fall with range), a bed bin at 26 m just too sparse to be listed (< floor_min_points),
    # then the sparse dark bed to 100 m (>= floor_min_points_far per bin); the object stands at 28-30 m
    counts = [120, 190, 180, 140, 105, 75, 60, 45, 35, 30, 28, 20, 17, 14, 12, 10, 9, 9, 9, 9] + [9] * 25
    d, l, h = _bed(rng, 10.0, counts)
    if with_object:
        for d0 in (28.0, 30.0):
            od, ol, oh = _object(rng, d0, 220)
            d, l, h = np.concatenate([d, od]), np.concatenate([l, ol]), np.concatenate([h, oh])
    return d, l, h


ON = dataclasses.replace(DetectorConfig(), floor_step_up_max_m=0.5)   # the rule is off by default (v1); enabled per deployment


def test_the_rule_is_off_by_default():
    assert DetectorConfig().floor_step_up_max_m == 0.0


def test_an_object_on_the_bed_does_not_tilt_the_near_fit():
    cfg = ON
    rng = np.random.default_rng(3)
    d, l, h = _scene(rng, with_object=True)
    rail = fit_rail_level(d, l, h, cfg, 0.0)
    assert rail.ok
    assert abs(rail.grade - GRADE) < 0.01 and abs(rail.z0 - Z0) < 0.15
    assert 20.0 <= rail.max_d_observed < 28.0      # the torso bins at 28-30 m are not part of the observed bed
    for dd in (20.0, 28.0, 60.0):
        assert abs(float(rail(np.array(dd))) - (Z0 + GRADE * dd)) < 0.2


def test_an_object_close_ahead_keeps_the_fit():
    # the torso bin at 14 m is bridged (the run goes on over the bed behind it) and rejected by the stage-1 gates:
    # the fit must not fail for want of near bins (v1 kept the person at 4-18 m this way)
    cfg = ON
    rng = np.random.default_rng(5)
    d, l, h = _scene(rng, with_object=False)
    od, ol, oh = _object(rng, 14.0, 1200)         # a torso at 14 m gives ~1000 returns in the Falcon ROI
    d, l, h = np.concatenate([d, od]), np.concatenate([l, ol]), np.concatenate([h, oh])
    rail = fit_rail_level(d, l, h, cfg, 0.0)
    assert rail.ok
    assert abs(rail.grade - GRADE) < 0.01 and abs(rail.z0 - Z0) < 0.15
    assert rail.max_d_observed >= 22.0            # the run continued over the bed behind the object
    v1 = fit_rail_level(d, l, h, dataclasses.replace(cfg, floor_step_up_max_m=0.0), 0.0)
    assert (v1.z0, v1.grade, v1.n_bins, v1.max_d_observed) == (rail.z0, rail.grade, rail.n_bins, rail.max_d_observed)


def test_v1_behaviour_is_reproduced_with_the_rule_off():
    # the object's dense bins at 28-30 m are reached across the unlisted sparse bin at 26 m and lever the fit (v1)
    cfg = dataclasses.replace(DetectorConfig(), floor_step_up_max_m=0.0)
    rng = np.random.default_rng(3)
    d, l, h = _scene(rng, with_object=True)
    rail = fit_rail_level(d, l, h, cfg, 0.0)
    assert rail.ok
    assert rail.grade > GRADE + 0.03 or rail.max_d_observed < 40.0


def test_a_plain_bed_is_unchanged_by_the_rule():
    rng = np.random.default_rng(7)
    d, l, h = _scene(rng, with_object=False)
    a = fit_rail_level(d, l, h, ON, 0.0)
    b = fit_rail_level(d, l, h, DetectorConfig(), 0.0)
    assert a.ok and b.ok
    assert (a.z0, a.grade, a.curv, a.n_bins, a.max_d_observed) == (b.z0, b.grade, b.curv, b.n_bins, b.max_d_observed)


def test_a_gap_and_a_step_down_do_not_end_the_run():
    # an empty bin inside the dense run (shadow) and a bin whose median sits lower (trough) keep the v1 behaviour:
    # the rule looks at steps up only
    cfg = ON
    rng = np.random.default_rng(11)
    counts = [120, 190, 0, 140, 105, 75, 60, 45, 44, 42, 41, 20, 17, 14, 12, 10, 9, 9, 9, 9] + [9] * 25
    d, l, h = _bed(rng, 10.0, counts)
    low = (d >= 18.0) & (d < 20.0)
    h = h.copy()
    h[low] -= 0.4                                   # a trough-dominated bin, 0.4 m lower
    a = fit_rail_level(d, l, h, cfg, 0.0)
    b = fit_rail_level(d, l, h, DetectorConfig(), 0.0)
    assert a.ok and b.ok
    assert (a.z0, a.grade, a.n_bins, a.max_d_observed) == (b.z0, b.grade, b.n_bins, b.max_d_observed)
    assert abs(a.grade - GRADE) < 0.01
