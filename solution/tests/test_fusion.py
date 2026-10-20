"""Fusion of detector candidates over time (evidence accumulation, agreement rule, hysteresis)."""
import numpy as np
import pytest

from metro_obstacle_detector.candidates import DetCandidate
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import Candidate, FrameResult
from metro_obstacle_detector.fusion import FusionFilter, FusionParams
from metro_obstacle_detector.track_model import RailLevel

R = DetectorConfig().axis_matrix()
RAIL = RailLevel(-1.3, 0.0, 0.0, 10, True, 0.02, 60.0)


def d1(d, l=0.0, n=20, conf=0.8):
    h = -1.3 + 0.8
    center = R.T @ np.array([d, l, h])
    return Candidate(distance_m=float(np.hypot(d, l)), forward_m=d, lateral_m=l, height_m=h, center_sensor=center,
                     extent=np.array([0.5, 0.5, 1.6]), n_points=n, lateral_offset_from_track=l, height_above_rail=0.8,
                     confidence=conf)


def d2(d, l=0.0, n=12, score=0.8, cls="obstacle"):
    return DetCandidate("d2", d, l, 0.8, (0.4, 0.5, 1.5), float(np.hypot(d, l)), n, score, cls=cls, lateral_offset=l)


def d3(d, l=0.0, n=100, score=1.0, cls="obstacle", extent=(3.0, 1.0, 1.5)):
    return DetCandidate("d3", d, l, 0.8, extent, float(np.hypot(d, l)), n, score, cls=cls, lateral_offset=l)


def frame(cands_d1=()):
    res = FrameResult(status="clear")
    res.candidates = list(cands_d1)
    return res


def run(seq, params=None):
    """seq: list of (d1 list, d2 list) per frame -> list of decisions."""
    f = FusionFilter(DetectorConfig(), params)
    out = []
    for c1, c2 in seq:
        res = frame(c1)
        dec = f.update(res, c2, RAIL, R)
        out.append((dec.status, dec.distance_m, res))
    return out, f


def test_two_detectors_agreeing_confirm_within_two_frames_and_keep_the_track_through_misses():
    seq = [([d1(50.0)], [d2(50.0)])] * 2 + [([], [])] * 2 + [([d1(49.0)], [d2(49.0)])]
    out, f = run(seq)
    assert out[0][0] == "clear" and out[1][0] == "detected"          # one frame is never enough; 2 x 3.16 >= 3.0 with two frames
    assert out[2][0] == "detected" and out[3][0] == "detected"       # hysteresis: two misses keep it (6.0 -> 4.4)
    assert out[4][0] == "detected" and abs(out[4][1] - 49.0) < 0.01
    assert all(c.confirmed for c in out[4][2].candidates)


def test_a_coasting_confirmed_track_appears_as_a_synthetic_candidate_in_the_result():
    out, f = run([([d1(50.0)], [d2(50.0)])] * 2 + [([], [])])
    status, dist, res = out[2]
    assert status == "detected" and len(res.candidates) == 1 and res.candidates[0].confirmed
    assert abs(res.candidates[0].distance_m - 50.0) < 0.01 and res.candidates[0].n_points == 20


def test_a_single_detector_near_range_does_not_confirm_on_weak_evidence():
    out, _ = run([([], [d2(60.0, n=1, score=0.05)])] * 12)
    assert all(s == "clear" for s, _, _ in out)


def test_a_single_detector_within_120_m_never_confirms_alone_however_persistent():
    out, _ = run([([], [d2(70.0, n=30, score=0.9)])] * 12)
    assert all(s == "clear" for s, _, _ in out)                     # "two of three" is strict within 120 m


def test_a_weak_second_detector_does_not_count_as_agreement():
    out, _ = run([([d1(80.0, n=4, conf=0.4)], [d2(80.0, n=1, score=0.05)])] * 8)
    assert all(s == "clear" for s, _, _ in out)


def test_a_dust_verdict_is_not_sticky():
    seq = [([], [d2(50.0, n=20, score=0.8, cls="dust")])] * 2 + [([d1(50.0)], [d2(50.0)])] * 3
    out, f = run(seq)
    assert out[1][0] == "clear" and out[-1][0] == "detected"


def test_far_single_detector_does_not_confirm_by_default():
    out, _ = run([([], [d2(180.0, n=4, score=0.7)])] * 8)
    assert all(s == "clear" for s, _, _ in out)                     # two detectors must agree at every range


def test_far_persistence_of_one_detector_counts_only_when_enabled():
    out, _ = run([([], [d2(180.0, n=4, score=0.7)])] * 8, params=FusionParams(far_single_detector=True))
    assert out[-1][0] == "detected"
    first = next(i for i, (s, _, _) in enumerate(out) if s == "detected")
    assert 3 <= first <= 6


def test_dust_class_is_vetoed_and_a_track_without_hits_dies():
    out, f = run([([], [d2(40.0, n=30, score=0.9, cls="dust")])] * 10)
    assert all(s == "clear" for s, _, _ in out)
    out, f = run([([d1(50.0)], [d2(50.0)])] * 2 + [([], [])] * 12)
    assert out[1][0] == "detected" and out[-1][0] == "clear" and not f.tracks


def test_unknown_frames_never_become_clear_and_reset_drops_tracks():
    f = FusionFilter(DetectorConfig())
    for _ in range(2):
        f.update(frame([d1(50.0)]), [d2(50.0)], RAIL, R)
    res = FrameResult(status="unknown", reason="rail level not observable")
    dec = f.update(res, [], None, R, allow_clear=False)
    assert dec.status == "detected"                                  # the confirmed track still coasts
    res = FrameResult(status="unknown", reason="rail level not observable")
    f.reset()
    dec = f.update(res, [], None, R, allow_clear=False)
    assert dec.status == "unknown" and not f.tracks


def test_association_keeps_one_track_for_an_approaching_object():
    seq = [([d1(100.0 - 2.0 * i)], [d2(100.0 - 2.0 * i)]) for i in range(10)]
    out, f = run(seq)
    assert len(f.tracks) == 1 and out[-1][0] == "detected"
    assert abs(out[-1][1] - 82.0) < 0.01


def test_far_two_detectors_confirm_after_four_frames():
    out, _ = run([([d1(180.0, n=6)], [d2(180.0, n=6, score=0.8)])] * 6)
    assert [s for s, _, _ in out] == ["clear", "clear", "clear", "detected", "detected", "detected"]


def test_a_confirmed_track_survives_frames_with_weak_hits():
    # confirmed by two detectors, then seen only with low scores for a while (the Falcon person walking
    # at close range): the hysteresis keeps it, a run of complete misses ends it
    seq = [([d1(30.0)], [d2(30.0)])] * 3 + [([d1(29.0, n=4, conf=0.2)], [d2(29.0, n=1, score=0.05)])] * 6 + [([], [])] * 9
    out, _ = run(seq)
    assert all(s == "detected" for s, _, _ in out[2:9])
    assert out[-1][0] == "clear"


def test_d3_hits_add_evidence_but_never_start_or_move_a_track():
    # confirmed by D1 + D2 at 50 m; afterwards D2 follows the object while a large D3 blob sits 1 m nearer:
    # the blob must not take the D2 hits away from the track (which would coast and die)
    seq = [([d1(50.0)], [d2(50.0)])] * 2 + [([], [d2(50.0 - 0.7 * k), d3(49.3 - 0.7 * k)]) for k in range(1, 12)]
    out, f = run(seq)
    assert all(s == "detected" for s, _, _ in out[1:])
    assert len([tr for tr in f.tracks if tr.confirmed]) == 1
    assert abs(out[-1][1] - (50.0 - 0.7 * 11)) < 0.3            # the reported range follows D2, not the blob
    out2, f2 = run([([], [d3(80.0)])] * 10)
    assert all(s == "clear" for s, _, _ in out2) and f2.tracks == []   # D3 alone neither votes nor tracks


def test_d3_hits_alone_do_not_keep_a_confirmed_track_alive():
    # confirmed by D1 + D2, then only a weak D3 blob near the (now empty) place: the track must end like a miss
    seq = [([d1(50.0)], [d2(50.0)])] * 3 + [([], [d3(50.0, n=3, score=0.3)])] * 12
    out, _ = run(seq)
    assert out[2][0] == "detected" and out[-1][0] == "clear"


def test_a_confirmed_track_beyond_the_verified_range_is_not_reported():
    f = FusionFilter(DetectorConfig())
    out = []
    for k in range(4):
        res = frame([d1(80.0)])
        out.append(f.update(res, [d2(80.0)], RAIL, R, max_d=100.0 if k < 3 else 60.0).status)
    assert out[2] == "detected" and out[3] == "clear"          # the corridor was cut to 60 m: nothing beyond is claimed
