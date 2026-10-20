"""median_model: one kind of rail level only; the far part stays consistent."""
import dataclasses

import numpy as np

from metro_obstacle_detector.track_model import RailLevel, RailLevelExtended, TrackCurve, median_model


def _rail(z0, grade, max_d):
    return RailLevel(z0=z0, grade=grade, curv=0.0, n_bins=10, ok=True, residual_rms=0.03, max_d_observed=max_d)


def _ext(z0, grade, far_max, d_switch=60.0, height=4.3):
    base = _rail(z0, grade, far_max)
    return RailLevelExtended(**dataclasses.asdict(base), far_coef=(3.0, -0.005, 0.0), height=height,
                             d_switch=d_switch, far_max_d=far_max, n_far_bins=5, bed_max_d=d_switch)


def _curve(b=1e-5):
    return TrackCurve(l0=0.0, a=0.0, b=b, ok=True, n_used=10, max_d_supported=80.0, max_d_absolute=80.0)


def test_a_plain_last_fit_is_not_mixed_with_extended_ones():
    models = [(_ext(-1.3, 0.001, 150.0), _curve()) for _ in range(4)] + [(_rail(-1.31, 0.0012, 60.0), _curve())]
    rail, curve = median_model(models, majority_kind=False)             # v1: the last fit's kind
    assert type(rail) is RailLevel and rail.max_d_observed <= 60.0      # no reach borrowed from the extended fits


def test_majority_kind_ignores_a_single_failed_extension():
    # four extended fits and a plain last fit (this frame's ceiling extension failed): the extended median, built
    # from the extended fits only (the plain fit's parameters are not mixed in)
    models = [(_ext(-1.3, 0.001, 150.0), _curve()) for _ in range(4)] + [(_rail(-2.5, 0.05, 60.0), _curve())]
    rail, _ = median_model(models)
    assert isinstance(rail, RailLevelExtended) and rail.far_max_d == 150.0
    assert abs(rail.z0 + 1.3) < 1e-9 and abs(rail.grade - 0.001) < 1e-12


def test_majority_kind_plain_and_ties():
    plain = [(_rail(-1.31 - 0.01 * k, 0.0012, 60.0 + k), _curve()) for k in range(3)]
    rail, _ = median_model([(_ext(-1.3, 0.001, 150.0), _curve())] * 2 + plain)
    assert type(rail) is RailLevel and rail.max_d_observed == 61.0       # 3 plain vs 2 extended: plain median
    rail, _ = median_model(plain[:2] + [(_ext(-1.3, 0.001, 150.0), _curve())] * 2)
    assert isinstance(rail, RailLevelExtended)                          # a tie: the last fit's kind


def test_an_extended_last_fit_keeps_the_extension_as_in_v1():
    # three plain fits and an extended last fit (the ceiling extension succeeded in this frame, e.g. behind dust):
    # the last fit's kind, as in v1 -- only a plain last fit is overruled by an extended majority
    plain = [(_rail(-1.31 - 0.01 * k, 0.0012, 13.0 + k), _curve()) for k in range(3)]
    last = (_ext(-1.3, 0.001, 35.5), _curve())
    rail, _ = median_model(plain + [last])
    rail_v1, _ = median_model(plain + [last], majority_kind=False)
    assert isinstance(rail, RailLevelExtended) and rail.far_max_d == 35.5
    assert rail == rail_v1


def test_extended_rails_keep_a_consistent_far_part():
    reaches = [150.0, 96.0, 150.0, 156.0, 96.0]
    models = [(_ext(-1.3 + 0.01 * k, 0.001, rc, d_switch=60.0 + k), _curve()) for k, rc in enumerate(reaches)]
    rail, _ = median_model(models)
    assert isinstance(rail, RailLevelExtended) and rail.far_max_d == 150.0
    assert rail.d_switch in (60.0, 62.0)                                 # the whole far part comes from a median-reach rail, not the last
    assert abs(rail.z0 - np.median([-1.3 + 0.01 * k for k in range(5)])) < 1e-9


def test_curves_follow_the_rail_kind():
    # four extended fits with support 150 m and a plain last fit with support 60 m: the plain result must not
    # carry the extended curves' support
    ext = [(_ext(-1.3, 0.001, 150.0), TrackCurve(l0=0.0, a=0.0, b=1e-5, ok=True, n_used=10, max_d_supported=150.0, max_d_absolute=150.0)) for _ in range(4)]
    plain = (_rail(-1.31, 0.0012, 60.0), TrackCurve(l0=0.0, a=0.0, b=1e-5, ok=True, n_used=10, max_d_supported=60.0, max_d_absolute=60.0))
    rail, curve = median_model(ext + [plain], majority_kind=False)
    assert type(rail) is RailLevel and curve.max_d_absolute == 60.0
    rail, curve = median_model(ext + [plain])                           # majority: extended rails with their own curves
    assert isinstance(rail, RailLevelExtended) and curve.max_d_absolute == 150.0


def test_the_extended_majority_does_not_overrule_a_plain_last_fit_in_a_bend():
    # four extended fits and a plain last one, all in a bend (|b| 3e-4 > straight_keep_abs_b): the v1 kind (plain)
    bend = [(_ext(-1.3, 0.001, 150.0), _curve(b=3e-4)) for _ in range(4)] + [(_rail(-1.31, 0.0012, 60.0), _curve(b=3e-4))]
    rail, _ = median_model(bend, straight_keep_abs_b=1.5e-4)
    assert type(rail) is RailLevel and rail.max_d_observed <= 60.0
    rail, _ = median_model(bend)                                        # no gate given: the majority rules
    assert isinstance(rail, RailLevelExtended)
    # the same window in a straight section: the extended majority overrules the single failed extension
    straight = [(_ext(-1.3, 0.001, 150.0), _curve(b=1e-5)) for _ in range(4)] + [(_rail(-1.31, 0.0012, 60.0), _curve(b=1e-5))]
    rail, _ = median_model(straight, straight_keep_abs_b=1.5e-4)
    assert isinstance(rail, RailLevelExtended) and rail.far_max_d == 150.0
    # the gate is on the window's median |b|: two bent fits among five do not block it, three do
    mixed2 = [(_ext(-1.3, 0.001, 150.0), _curve(b=b)) for b in (1e-5, 3e-4, 1e-5, 3e-4)] + [(_rail(-1.31, 0.0012, 60.0), _curve(b=1e-5))]
    assert isinstance(median_model(mixed2, straight_keep_abs_b=1.5e-4)[0], RailLevelExtended)
    mixed3 = [(_ext(-1.3, 0.001, 150.0), _curve(b=b)) for b in (3e-4, 3e-4, 1e-5, 3e-4)] + [(_rail(-1.31, 0.0012, 60.0), _curve(b=1e-5))]
    assert type(median_model(mixed3, straight_keep_abs_b=1.5e-4)[0]) is RailLevel
