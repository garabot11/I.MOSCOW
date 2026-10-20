"""D2 labelling (WP-S S4): ``BeamDetector.group_labels`` (bounding-box window) is identical to the v1 labelling.

``_ref_labels`` is the labelling block of ``BeamDetector.process`` of v1 (code 57c8cdb) copied verbatim (whole
image).  ``group_labels`` must return the same label image, the same number of plain groups and the same
``find_objects`` slices, on random masks of several densities, masks touching the image edges, single
pixels, an empty mask and face-bridging cases (fragments sampled every other pixel)."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.det_beam import BeamDetector, BeamDetectorParams, _EIGHT

H, W = 128, 600


def _ref_labels(p, inbox, ri, dl, dh):
    """v1 (57c8cdb) det_beam.process lines 'plain, n_plain = ndimage.label(...)' .. 'objs = find_objects(labels)'."""
    plain, n_plain = ndimage.label(inbox, structure=_EIGHT)
    if n_plain == 0:
        return None, [], 0
    labels = plain
    small = np.zeros(1, dtype=bool)
    if p.merge_gap_px > 0:
        sizes = np.bincount(plain.ravel(), minlength=n_plain + 1)
        small = sizes <= p.bridge_fragment_px
        small[0] = False
    if p.merge_gap_px > 0 and small.any():
        grouped = ndimage.binary_dilation(inbox, structure=_EIGHT, iterations=p.merge_gap_px)
        lab2, n2 = ndimage.label(grouped, structure=_EIGHT)
        lab2 = np.where(inbox, lab2, 0)
        flat2, flat1 = lab2.ravel(), plain.ravel()
        nz = np.flatnonzero(flat2)
        c2v, c1v = flat2[nz], flat1[nz]
        n_big = np.bincount(c2v, weights=(~small[c1v]).astype(np.float64), minlength=n2 + 1)
        tot = np.bincount(c2v, minlength=n2 + 1)
        key = np.unique(c2v.astype(np.int64) * (n_plain + 1) + c1v)
        n_frag = np.bincount(key // (n_plain + 1), minlength=n2 + 1)
        cand_comp = np.flatnonzero((n_big == 0) & (tot >= p.face_min_px) & (n_frag >= 2))
        if len(cand_comp):
            dlh_flat = ri.dlh[0].reshape(-1, 3)
            dl_flat, dh_flat = dl.ravel(), dh.ravel()
            order2 = np.argsort(c2v, kind="stable")
            c2s = c2v[order2]
            for c2 in cand_comp:
                lo, hi = np.searchsorted(c2s, c2), np.searchsorted(c2s, c2, side="right")
                pix = nz[order2[lo:hi]]
                pts = dlh_flat[pix]
                ext2 = pts.max(axis=0) - pts.min(axis=0)
                dl_c2 = float(np.median(dl_flat[pix]))
                if (ext2[1] >= p.face_min_width_m and ext2[0] <= p.face_max_depth_m and ext2[2] >= p.face_min_height_m
                        and float(dh_flat[pix].min()) <= p.face_max_bottom_m and abs(dl_c2) <= p.face_max_center_offset_m):
                    labels = labels.copy() if labels is plain else labels
                    labels.flat[pix] = n_plain + int(c2)
    objs = ndimage.find_objects(labels)
    return labels, objs, n_plain


def _scene(rng, h=H, w=W, d0=20.0, jitter=0.02):
    """A (fake) range image: a planar face across the corridor at d0, l along the columns, h along the rows
    (5 cm per pixel), so that sparse fragments pass the face test of the bridging."""
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    d = d0 + jitter * rng.standard_normal((h, w)).astype(np.float32)
    l = (cols - w / 2) * 0.05
    hh = (h - rows) * 0.05 - 2.0
    dlh = np.stack([d, l, hh], axis=-1)[None].repeat(2, axis=0)
    ri = SimpleNamespace(dlh=dlh, valid=np.ones((2, h, w), bool), organized=False)
    dl = l.astype(np.float32)
    dh = (hh + 1.6).astype(np.float32)          # rail 1.6 m below the sensor
    return ri, dl, dh


def _check(det, inbox, ri, dl, dh):
    ref_lab, ref_objs, ref_n = _ref_labels(det.p, inbox, ri, dl, dh)
    lab, objs, n_plain, n_big = det.group_labels(inbox, ri, dl, dh)
    assert n_plain == ref_n
    assert n_big == n_plain
    if ref_n == 0:
        assert objs == [] and not lab.any()
        return False
    assert lab.dtype == ref_lab.dtype
    assert np.array_equal(lab, ref_lab)
    assert len(objs) == len(ref_objs)
    for a, b in zip(objs, ref_objs):
        assert a == b
    return bool(lab.max() > ref_n)     # True = a bridged face label exists


@pytest.fixture()
def det():
    return BeamDetector(DetectorConfig(), BeamDetectorParams())


def test_empty_and_single_pixels(det):
    ri, dl, dh = _scene(np.random.default_rng(0))
    inbox = np.zeros((H, W), bool)
    _check(det, inbox, ri, dl, dh)
    for r, c in [(0, 0), (H - 1, W - 1), (0, W - 1), (H - 1, 0), (64, 300)]:
        inbox = np.zeros((H, W), bool)
        inbox[r, c] = True
        _check(det, inbox, ri, dl, dh)


@pytest.mark.parametrize("density", [0.002, 0.01, 0.05, 0.15, 0.3, 0.6])
def test_random_masks(det, density):
    rng = np.random.default_rng(int(density * 1000))
    for k in range(6):
        ri, dl, dh = _scene(rng)
        inbox = rng.random((H, W)) < density
        if k % 2:                                  # a corridor-like band only
            inbox[:40] = False
            inbox[:, :200] = False
            inbox[:, 420:] = False
        _check(det, inbox, ri, dl, dh)


def test_edges(det):
    rng = np.random.default_rng(7)
    ri, dl, dh = _scene(rng)
    for edge in ["top", "bottom", "left", "right", "all"]:
        inbox = np.zeros((H, W), bool)
        sub = rng.random((H, W)) < 0.1
        if edge in ("top", "all"):
            inbox[:3] = sub[:3]
        if edge in ("bottom", "all"):
            inbox[-3:] = sub[-3:]
        if edge in ("left", "all"):
            inbox[:, :3] = sub[:, :3]
        if edge in ("right", "all"):
            inbox[:, -3:] = sub[:, -3:]
        _check(det, inbox, ri, dl, dh)


def _face(inbox, r0, r1, c0, c1, step=2):
    inbox[r0:r1:step, c0:c1:step] = True


def test_face_bridging_cases(det):
    rng = np.random.default_rng(3)
    ri, dl, dh = _scene(rng, jitter=0.01)
    bridged = 0
    # a face sampled every other pixel, centred (bridged), one at the image edges, one with a big fragment
    # inside (never bridged), two faces side by side, a too-narrow one
    cases = []
    m = np.zeros((H, W), bool); _face(m, 70, 110, 270, 330); cases.append(m)
    m = np.zeros((H, W), bool); _face(m, 70, 110, 270, 330); m[90:94, 290:296] = True; cases.append(m)
    m = np.zeros((H, W), bool); _face(m, 88, H, 260, 340); _face(m, 0, 30, 0, 40); cases.append(m)
    m = np.zeros((H, W), bool); _face(m, 70, 110, 250, 300); _face(m, 70, 110, 304, 350); cases.append(m)
    m = np.zeros((H, W), bool); _face(m, 70, 110, 295, 305); cases.append(m)
    m = np.zeros((H, W), bool); _face(m, 60, 110, 260, 340, step=3); cases.append(m)
    for m in cases:
        bridged += _check(det, m, ri, dl, dh)
    assert bridged >= 3          # the bridging path is really exercised


def test_face_bridging_random_fragments(det):
    rng = np.random.default_rng(11)
    bridged = 0
    for k in range(20):
        ri, dl, dh = _scene(rng)
        inbox = np.zeros((H, W), bool)
        inbox[60:120:2, 250:350:2] = rng.random((30, 50)) < 0.8
        noise = rng.random((H, W)) < 0.01
        inbox |= noise
        bridged += _check(det, inbox, ri, dl, dh)
    assert bridged >= 5


def test_merge_gap_variants():
    rng = np.random.default_rng(5)
    for gap in (0, 1, 2):
        det = BeamDetector(DetectorConfig(), BeamDetectorParams(merge_gap_px=gap))
        for k in range(4):
            ri, dl, dh = _scene(rng)
            inbox = np.zeros((H, W), bool)
            inbox[60:120:3, 250:350:3] = rng.random((20, 34)) < 0.9
            inbox[0:2, :] = rng.random((2, W)) < 0.05
            _check(det, inbox, ri, dl, dh)
