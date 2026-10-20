"""wall_sweep_limit: a wall crossing the corridor caps it; faces, posts and parallel sides do not."""
import numpy as np

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.track_model import wall_sweep_limit

CFG = DetectorConfig()


def _wall(d0, d1, offset, side=1.0, step=0.25, heights=(0.6, 1.2, 1.8, 2.4, 2.9)):
    """Vertical wall surface from d0 to d1 at lateral offset(d) (callable or number), one side."""
    ds = np.arange(d0, d1, step)
    off = np.array([offset(x) if callable(offset) else offset for x in ds])
    d = np.repeat(ds, len(heights)); dl = side * np.repeat(off, len(heights)); dh = np.tile(np.array(heights), len(ds))
    return d, dl, dh


def _cat(*parts):
    return tuple(np.concatenate(x).astype(np.float32) for x in zip(*parts))


def _straight():
    return _cat(_wall(20, 200, 2.0, +1.0), _wall(20, 200, 2.0, -1.0))


def test_straight_walls_do_not_cap():
    d, dl, dh = _straight()
    lim, info = wall_sweep_limit(d, dl, dh, CFG)
    assert lim == float("inf") and info == {}


def test_oblique_wall_entering_the_corridor_caps_at_the_first_bin_inside():
    left = _wall(20, 160, lambda x: 2.0 if x < 100 else max(0.1, 2.0 - 0.06 * (x - 100)), +1.0)
    d, dl, dh = _cat(left, _wall(20, 200, 2.0, -1.0))
    lim, info = wall_sweep_limit(d, dl, dh, CFG)
    assert 100.0 <= lim <= 110.0, (lim, info)
    assert info["side"] == "left" and len(info["bins"]) == 3


def test_a_train_standing_ahead_is_not_a_cap():
    # parallel sides at 1.5 m from 100.5 m on and a face across the corridor at 100.5 m
    face_l = np.repeat(np.linspace(-1.4, 1.4, 29), 5); face_h = np.tile(np.array([0.5, 1.2, 1.9, 2.6, 3.2]), 29)
    face = (np.full(face_l.shape, 100.5), face_l, face_h)
    d, dl, dh = _cat(_straight(), _wall(100.5, 120, 1.5, +1.0), _wall(100.5, 120, 1.5, -1.0), face)
    lim, info = wall_sweep_limit(d, dl, dh, CFG)
    assert lim == float("inf"), (lim, info)


def test_a_wide_face_after_a_narrowing_wall_is_not_a_cap():
    # walls 2.2 m to 95 m then 2.0 m; a 2.8 m wide face at 101 m (the reviewer's counter-example)
    face_l = np.repeat(np.linspace(-1.4, 1.4, 29), 5); face_h = np.tile(np.array([0.5, 1.2, 1.9, 2.6, 3.2]), 29)
    face = (np.full(face_l.shape, 101.0), face_l, face_h)
    walls = _cat(_wall(20, 200, lambda x: 2.2 if x < 95 else 2.0, +1.0), _wall(20, 200, lambda x: 2.2 if x < 95 else 2.0, -1.0))
    d, dl, dh = _cat(walls, face)
    lim, info = wall_sweep_limit(d, dl, dh, CFG)
    assert lim == float("inf"), (lim, info)


def test_a_single_post_in_the_corridor_is_not_a_cap():
    post = (np.full(10, 90.0), np.full(10, 0.9), np.linspace(0.4, 2.6, 10))
    d, dl, dh = _cat(_straight(), post)
    lim, info = wall_sweep_limit(d, dl, dh, CFG)
    assert lim == float("inf"), (lim, info)
