"""Detector D2: sparse obstacles in the range image (per-beam reasoning).

The cluster detector (D1) needs a handful of voxels per object, which a person-sized target
only gives up to ~100-150 m.  D2 works on the range image instead: every return inside the
clearance corridor is a candidate pixel, candidate pixels are grouped by image adjacency
(8-neighbourhood), and a group is judged by what surrounds it *in the image*:

* **background** neighbours - no return at all (the beam passed the object and hit nothing in
  range; only meaningful for organised sensors, where an empty pixel is a fired beam) or a
  return much farther away (the object stands in front of the tunnel);
* **attached** neighbours - returns at a similar range that lie outside the corridor: the group
  is the part of a continuous surface (wall, duct, floor patch) that pokes into the corridor;
* **grounded** columns - the nearest return below the group is the track bed at the same
  distance (an object standing on the track; positive evidence at ranges where the bed is seen).

A group of one or two beams is enough to become a candidate; the fusion stage weighs the
candidate by its score and confirms it over time.  No 3-D neighbourhood searches: everything
is array indexing on a (H, W) image, a few milliseconds per frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import ndimage

from .candidates import DetCandidate
from .config import DetectorConfig
from .range_image import RangeImage
from .track_model import curvature_shrink

_EIGHT = np.ones((3, 3), bool)


def _bridge_empty_bins(plain: np.ndarray, n_plain: int, inbox: np.ndarray, ri: RangeImage, p, win):
    """Empty-bin bridging of small fragments in binned images (WP-R rule ``d2_bridge_empty_bins``).

    ``plain``/``inbox`` are crops of the window ``win`` = (row slice, column slice) of the range image;
    the matching crops of the image are ``ri.valid[0][win]`` etc., ``ri.organized`` tells binned images
    apart.  Returns (plain, n_plain, n_big_ids): labels 1..n_big_ids are plain groups, labels above it are
    joined fragments (``n_big_ids == n_plain`` = nothing joined = v1 behaviour).

    In a binned image of an unorganised cloud an empty bin is "no sample", not "no return": a sensor sampled
    coarser than the bins, or an object inserted as a coarse grid of points (the organisers' plate: 1-2 px
    pieces every other bin), leaves empty bins between the pixels of one surface.  Small plain groups
    (<= ``bridge_fragment_px``) separated by at most 2 * ``bridge_empty_px`` empty bins are joined into one
    group; larger groups keep their plain labels (joining them too merged a person at 23 m on the Falcon bag
    with the structure beside him).  Label numbers of the plain groups are unchanged, so this runs after any
    plain labelling.

    Stream gate: organised images are never bridged (there an empty pixel is a fired beam that returned
    nothing), and only a stream with the vertical field of view of the Hesai class
    (``bridge_min_vspan_deg``, 128-ring OT128 = 39.5 deg) is bridged: on the Falcon Prime (26 deg) the
    rule gains nothing and loses frames (its bag from frame 0: close walk 684/710 instead of 710, the
    person at 24-28 m joined to the structure beside him where the corridor model is unstable)."""
    if n_plain == 0 or not p.bridge_empty_bins or ri.organized:
        return plain, n_plain, n_plain
    el = getattr(ri, "el_deg", None)                     # unknown vertical span (a stand-in image): not bridged
    if el is None or len(el) < 2 or float(el.max() - el.min()) < p.bridge_min_vspan_deg:
        return plain, n_plain, n_plain
    frag = np.bincount(plain.ravel(), minlength=n_plain + 1) <= p.bridge_fragment_px
    frag[0] = False
    frag_px = frag[plain]
    if not frag_px.any():
        return plain, n_plain, n_plain
    empty = ~ri.valid[0][win]
    grown = frag_px | (ndimage.binary_dilation(frag_px, structure=_EIGHT, iterations=p.bridge_empty_px) & empty)
    lab_s, n_s = ndimage.label(grown, structure=_EIGHT)
    rng = getattr(ri, "rng", None)                       # a stand-in image without ranges: not split
    if p.bridge_split_range and n_s and rng is not None:
        lab_s, n_s = _split_joined_by_range(lab_s, plain, frag_px, rng[0][win], p)
    out = np.where(frag_px, lab_s + n_plain, plain).astype(plain.dtype)
    return out, n_plain + n_s, n_plain


def _split_joined_by_range(lab_s: np.ndarray, plain: np.ndarray, frag_px: np.ndarray, rng: np.ndarray, p):
    """Split joined fragment groups where the fragments' ranges jump (``bridge_split_range``).

    One surface sampled coarser than the bins has fragments at continuous ranges.  Through the gap between the
    head and the shoulder of a person at 20 m the tunnel returns 204 m behind him are in-box fragments too, one
    empty bin away: joined to the person they made a group 185 m deep, dropped as structure, and the person had
    no D2 vote (val_sweep, val_fake_style scene 002).  The fragments of each joined group are sorted by their
    mean range and a new group starts where two consecutive ones differ by more than the attachment margin
    (``range_margin_m + range_margin_per_m * r``).  Returns (labels at the fragment pixels, group count)."""
    f_pix = plain[frag_px]
    n_f = int(plain.max()) + 1
    cnt = np.bincount(f_pix, minlength=n_f)
    r_f = np.bincount(f_pix, weights=rng[frag_px].astype(np.float64), minlength=n_f) / np.maximum(cnt, 1)
    comp_f = np.zeros(n_f, np.int64)
    comp_f[f_pix] = lab_s[frag_px]
    fs = np.flatnonzero(cnt)
    fs = fs[np.lexsort((r_f[fs], comp_f[fs]))]
    cs, rs = comp_f[fs], r_f[fs]
    start = np.ones(len(fs), dtype=bool)
    start[1:] = (cs[1:] != cs[:-1]) | (rs[1:] - rs[:-1] > p.range_margin_m + p.range_margin_per_m * rs[:-1])
    sub = np.cumsum(start)
    new_f = np.zeros(n_f, np.int64)
    new_f[fs] = sub
    out = np.zeros_like(lab_s)
    out[frag_px] = new_f[f_pix]
    return out, int(sub[-1]) if len(sub) else 0


@dataclass
class BeamDetectorParams:
    range_margin_m: float = 0.5        # neighbours within +-(range_margin + range_margin_per_m * r) are "attached"
    range_margin_per_m: float = 0.01
    background_gap_m: float = 3.0      # a neighbour this much farther (or 5 %) is background
    attached_max_frac: float = 0.35    # reject a group with more attached boundary pixels than this
    max_extent_d: float = 6.0          # longer groups are infrastructure
    far_max_extent_d: float = 2.5      # beyond far_extent_from_d a person/box is never this long along the track
    far_extent_from_d: float = 60.0
    object_size_m: float = 0.5         # reference object size for the expected pixel count
    min_score: float = 0.0             # groups below this score are dropped (0 = keep all, let the fusion weigh them)
    ring_px: int = 1                   # boundary ring thickness in pixels
    grow_px: int = 24                  # window (pixels) in which an edge group is grown through its range band
    straddle_max_extent_d: float = 2.5 # a grown region shorter than this along the track ...
    straddle_max_ratio: float = 6.0    # ... and at most this many times the group is a compact object straddling the edge
    bottom_margin_beyond_per_m: float = 0.002   # D2's own box bottom growth beyond the observed bed (D1: 0.008):
                                                # a floor patch lifted into the box is rejected as "attached" instead
    see_through_min_frac: float = 0.5  # fraction of a group's beams with a second echo well behind the first -> semi-transparent (dust)
    see_through_gap_m: float = 1.0     # "well behind" = at least this (or 3 % of the range) farther
    see_through_min_px: int = 3
    rough_min_px: int = 6              # volumetric scatter: median |r - 3x3 median r| within the group, normalised by
    rough_ratio: float = 2.5           # (rough_base + rough_per_m * r), above this ratio -> dust (single-return dust)
    rough_base_m: float = 0.03
    rough_per_m: float = 0.003
    sliver_max_width_m: float = 0.35   # thin (lateral) and long (along the track) group hugging the lateral edge of the
    sliver_min_length_m: float = 1.5   # corridor: platform coping, duct, rail of the next track -> class "structure"
    sliver_edge_m: float = 0.5
    extends_above_min_frac: float = 0.5  # fraction of the group's columns continuing above the box top -> structure
    embedded_min_known: int = 6        # a group with no background neighbour among this many known ones is part of a surface
    pole_max_width_m: float = 0.2      # thin vertical post (lateral extent) ...
    pole_edge_m: float = 0.45          # ... within this of the lateral corridor edge -> structure
    hanging_bottom_m: float = 1.5      # a group whose lowest pixel is this high above the rail does not stand on the track
    hanging_beyond_d: float = 50.0     # ... (lamps, cables, signs): beyond this range its score is scaled by hanging_factor
    hanging_factor: float = 0.25
    face_min_width_m: float = 1.5      # a group this wide across the corridor, thin along the track and centred is a face (obstacle)
    face_max_depth_m: float = 1.0
    face_max_center_offset_m: float = 0.9
    face_min_px: int = 12
    flat_sliver_max_h_m: float = 0.12  # a group this thin vertically and >= flat_sliver_min_w_m wide is a cable, duct or floor patch
    flat_sliver_min_w_m: float = 0.4
    flat_sliver_min_h_above_rail_m: float = 1.5
    long_edge_min_length_m: float = 3.0  # a group this long along the track, >= 3x longer than wide, at the corridor edge is structure
    long_edge_m: float = 0.6
    merge_gap_px: int = 1              # dilation steps bridging gaps of up to 2*merge_gap_px pixels: a synthetic face sampled every other pixel fell apart into 4-px groups
    bridge_fragment_px: int = 8        # only fragments up to this size are bridged, and only when their union is a face across the corridor
    face_min_height_m: float = 1.5
    face_max_bottom_m: float = 1.5
    bridge_empty_bins: Optional[bool] = None  # binned images: join small in-box fragments across empty bins (None = cfg.d2_bridge_empty_bins)
    bridge_min_vspan_deg: Optional[float] = None  # ... only streams with at least this vertical span (None = cfg.d2_bridge_min_vspan_deg)
    bridge_empty_px: int = 1           # ... each fragment grows this many empty bins (gaps up to 2 * this are joined)
    bridge_split_range: bool = True    # ... and a joined group is split where its fragments' ranges jump (background seen through a gap)
    ring_look_px: int = 3              # ... and the neighbourhood of a joined group is the first returns reached through at most this many empty bins
    flat_edge_max_h_m: float = 0.1     # ... and a joined group flatter than this with attached neighbours is not rescued as "straddling"


class BeamDetector:
    def __init__(self, cfg: DetectorConfig, params: Optional[BeamDetectorParams] = None):
        self.cfg = cfg
        self.p = params or BeamDetectorParams()
        if self.p.bridge_empty_bins is None:
            self.p.bridge_empty_bins = bool(getattr(cfg, "d2_bridge_empty_bins", False))
        if self.p.bridge_min_vspan_deg is None:
            self.p.bridge_min_vspan_deg = float(getattr(cfg, "d2_bridge_min_vspan_deg", 33.0))

    # ------------------------------------------------------------------
    def corridor_mask(self, ri: RangeImage, rail, curve):
        """Pixels inside the clearance corridor (same geometry as D1) and the box bottom/top maps."""
        cfg = self.cfg
        d = ri.dlh[0][..., 0]
        l = ri.dlh[0][..., 1]
        h = ri.dlh[0][..., 2]
        valid = ri.valid[0] & (d >= cfg.d_min) & (d <= cfg.d_max)
        lc = curve(d)
        hr = rail(d)
        dl = l - lc
        dh = h - hr
        halfwidth = np.where(dh < cfg.low_zone_top_above_rail, cfg.gauge_halfwidth_low, cfg.gauge_halfwidth_body)
        halfwidth = halfwidth - (cfg.lateral_shrink_per_m * d + curvature_shrink(cfg, curve, d))
        halfwidth = np.where(halfwidth < cfg.lateral_min_halfwidth, -1.0, halfwidth)
        bed_max = getattr(rail, "bed_max_d", rail.max_d_observed)
        beyond_bed = np.maximum(0.0, d - bed_max)
        beyond = np.maximum(0.0, d - rail.max_d_observed)
        rms_term = cfg.margin_rms_factor * (rail.residual_rms if math.isfinite(rail.residual_rms) else 0.0)
        bottom = (cfg.box_bottom_above_rail + cfg.bottom_margin_per_m * d + self.p.bottom_margin_beyond_per_m * (beyond_bed - beyond)
                  + cfg.bottom_margin_beyond_per_m * beyond + rms_term)
        top = cfg.box_top_above_rail - cfg.top_margin_per_m * d - cfg.top_margin_beyond_per_m * beyond - rms_term
        inbox = valid & (np.abs(dl) <= halfwidth) & (dh >= bottom) & (dh <= top)
        return inbox, valid, dl, dh, bottom, halfwidth, top

    # ------------------------------------------------------------------
    def group_labels(self, inbox: np.ndarray, ri: RangeImage, dl: np.ndarray, dh: np.ndarray):
        """Groups of in-corridor pixels: (labels (H, W) int32, find_objects slices, n_plain, n_big_ids).

        Plain 8-connected in-box pixels (the behaviour of every evaluation before the organisers' edited
        bag).  Bridging across one-pixel gaps is applied ONLY to a cluster of small fragments whose union
        looks like a face across the corridor (wide, thin along the track, tall, standing low, centred):
        the inserted 2.9 x 4 m wall is sampled every other pixel and fell apart into 1-4 px pieces.
        Bridging everything merged far persons into wall groups (val_hesai_insert_tunnel 50-100 m recall
        0.32 -> 0.05) and, with range splits, turned wall pieces into obstacles.

        Speed (WP-S S4): everything runs on the bounding box of the in-box pixels plus ``margin`` pixels
        (the dilation of the face bridging never reaches farther than ``merge_gap_px`` from an in-box
        pixel; ``binary_dilation`` uses border_value 0, so the clipped image edges behave the same).
        ``ndimage.label`` numbers components in raster order of their first pixel and the window keeps
        the raster order, so label numbers, pixel order and the find_objects slices (shifted back to
        image coordinates) are identical to labelling the whole image (tests/test_det_beam_labels.py).
        ``n_big_ids``: labels 1..n_big_ids are plain groups; = n_plain unless a bridging rule (hook
        ``_bridge_empty_bins``) appends joined-fragment labels after them.
        """
        p = self.p
        H, W = inbox.shape
        rows = np.flatnonzero(inbox.any(axis=1))
        if len(rows) == 0:
            return np.zeros((H, W), np.int32), [], 0, 0
        cols = np.flatnonzero(inbox.any(axis=0))
        # a bridging rule that looks farther than merge_gap_px from an in-box pixel must raise the margin
        margin = max(int(p.merge_gap_px), 0) + 1
        win = (slice(max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, H)),
               slice(max(int(cols[0]) - margin, 0), min(int(cols[-1]) + 1 + margin, W)))
        r0, c0 = win[0].start, win[1].start
        box = inbox[win]
        plain, n_plain = ndimage.label(box, structure=_EIGHT)
        # ---- hook (WP-R): empty-bin bridging of small plain fragments in binned images.  All arrays are
        # crops of the window ``win`` (ri.valid[0][win], ri.dlh[0][win], ...); it returns the (possibly
        # renumbered) crop labels, their count and n_big_ids (labels > n_big_ids = joined fragments).
        plain, n_plain, n_big_ids = _bridge_empty_bins(plain, n_plain, box, ri, p, win)
        if n_plain == 0:
            return np.zeros((H, W), np.int32), [], 0, 0
        labels = plain
        small = np.zeros(1, dtype=bool)
        if p.merge_gap_px > 0:
            sizes = np.bincount(plain.ravel(), minlength=n_plain + 1)
            small = sizes <= p.bridge_fragment_px
            small[0] = False
        if p.merge_gap_px > 0 and small.any():
            grouped = ndimage.binary_dilation(box, structure=_EIGHT, iterations=p.merge_gap_px)
            lab2, n2 = ndimage.label(grouped, structure=_EIGHT)
            lab2 = np.where(box, lab2, 0)
            flat2, flat1 = lab2.ravel(), plain.ravel()
            nz = np.flatnonzero(flat2)
            c2v, c1v = flat2[nz], flat1[nz]
            # bridged components made only of small fragments, with enough pixels in total (integer keys:
            # np.unique(axis=0) took 40 ms on a crowded image)
            n_big = np.bincount(c2v, weights=(~small[c1v]).astype(np.float64), minlength=n2 + 1)
            tot = np.bincount(c2v, minlength=n2 + 1)
            key = np.unique(c2v.astype(np.int64) * (n_plain + 1) + c1v)
            n_frag = np.bincount(key // (n_plain + 1), minlength=n2 + 1)
            cand_comp = np.flatnonzero((n_big == 0) & (tot >= p.face_min_px) & (n_frag >= 2))
            if len(cand_comp):
                dlh_flat = ri.dlh[0][win].reshape(-1, 3)
                dl_flat, dh_flat = dl[win].ravel(), dh[win].ravel()
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
        objs = [None if sl is None else (slice(sl[0].start + r0, sl[0].stop + r0), slice(sl[1].start + c0, sl[1].stop + c0))
                for sl in ndimage.find_objects(labels)]
        full = np.zeros((H, W), labels.dtype)
        full[win] = labels
        return full, objs, n_plain, n_big_ids

    # ------------------------------------------------------------------
    def process(self, ri: RangeImage, rail, curve, max_d_report: float = 1e9) -> List[DetCandidate]:
        if rail is None or curve is None or not rail.ok:
            return []
        p = self.p
        inbox, valid, dl, dh, bottom, halfwidth, top_img = self.corridor_mask(ri, rail, curve)
        if not inbox.any():
            return []
        r = ri.rng[0]
        r_far_img = ri.rng[1]
        d = ri.dlh[0][..., 0]
        H, W = ri.shape
        labels, objs, n_plain, n_big_ids = self.group_labels(inbox, ri, dl, dh)
        if n_plain == 0:
            return []
        # pixel spacing (metres per pixel at range r) for the expected support of a reference object
        el = ri.el_deg.astype(np.float64)
        row_step = np.abs(np.gradient(el)) if H > 1 else np.full(H, 0.125)
        col_step = float(abs(ri.az_deg[1] - ri.az_deg[0])) if W > 1 else 0.1
        out: List[DetCandidate] = []
        empty_img = ~ri.valid[0] if n_big_ids < n_plain else None   # joined fragments exist (empty-bin bridging): no return at all
        for lab, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            look = n_big_ids < lab <= n_plain      # a group of small fragments joined across empty bins
            pad = max(p.ring_px, p.ring_look_px) if look else p.ring_px
            rows_sl = slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, H))
            cols_sl = slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, W))
            comp = labels[rows_sl, cols_sl] == lab
            n = int(comp.sum())
            rr = r[rows_sl, cols_sl][comp]
            dd = d[rows_sl, cols_sl][comp]
            r_med = float(np.median(rr))
            d_med = float(np.median(dd))
            if d_med > max_d_report:
                continue
            # ---- boundary ring and its classification
            if look:
                # the neighbours of a joined group are the first returns reached through empty bins
                empty_w = empty_img[rows_sl, cols_sl]
                reach = comp.copy()
                ring = np.zeros_like(comp)
                for _ in range(max(p.ring_look_px, p.ring_px)):
                    front = ndimage.binary_dilation(reach, structure=_EIGHT) & ~reach
                    ring |= front & ~empty_w
                    step = front & empty_w
                    if not step.any():
                        break
                    reach |= step
            else:
                ring = ndimage.binary_dilation(comp, structure=_EIGHT, iterations=p.ring_px) & ~comp
            v_ring = valid[rows_sl, cols_sl] & ring
            r_ring = r[rows_sl, cols_sl]
            margin = p.range_margin_m + p.range_margin_per_m * r_med
            similar = v_ring & (np.abs(r_ring - r_med) <= margin)
            attached = similar & ~inbox[rows_sl, cols_sl]
            farther = v_ring & (r_ring > r_med + max(p.background_gap_m, 0.05 * r_med))
            empty = ring & ~valid[rows_sl, cols_sl]
            n_ring = int(ring.sum())
            n_attached = int(attached.sum())
            n_bg = int(farther.sum()) + (int(empty.sum()) if ri.organized else 0)
            n_known = n_attached + n_bg + int((similar & inbox[rows_sl, cols_sl]).sum())
            attached_frac = n_attached / max(n_known, 1)
            straddling = False
            # a joined group that is flat (one scan line) and continued by attached material at the same range is the
            # top edge of something on the bed that barely enters the box (a 0.2 m step across the track at 69 m in the
            # organisers' platform frames), not a compact object straddling the corridor edge
            flat_edge = False
            if look and n_attached > 0:
                dh_g = dh[rows_sl, cols_sl][comp]
                flat_edge = float(dh_g.max() - dh_g.min()) < p.flat_edge_max_h_m
            if n_attached > 0 and attached_frac > p.attached_max_frac * 0.5 and not flat_edge:
                # is the attached material a compact object straddling the corridor edge, or extended
                # structure (wall, platform edge, duct)?  Grow the group through pixels of the same
                # range band in a window around it and look at the extent of what it is connected to.
                w_r = slice(max(sl[0].start - p.grow_px, 0), min(sl[0].stop + p.grow_px, H))
                w_c = slice(max(sl[1].start - p.grow_px, 0), min(sl[1].stop + p.grow_px, W))
                band = valid[w_r, w_c] & (np.abs(r[w_r, w_c] - r_med) <= margin)
                lab_w, _ = ndimage.label(band, structure=_EIGHT)
                own = np.unique(lab_w[rows_sl.start - w_r.start:rows_sl.stop - w_r.start, cols_sl.start - w_c.start:cols_sl.stop - w_c.start][comp])
                own = own[own > 0]
                if len(own):
                    region = np.isin(lab_w, own)
                    reg_dlh = ri.dlh[0][w_r, w_c][region]
                    reg_ext = reg_dlh.max(axis=0) - reg_dlh.min(axis=0)
                    n_region = int(region.sum())
                    if reg_ext[0] <= p.straddle_max_extent_d and n_region <= p.straddle_max_ratio * n:
                        straddling = True
                        attached_frac = 0.0
            # ---- grounded: nearest return below the group within 4 rows is the bed at the same distance
            comp_rows, comp_cols = np.nonzero(comp)
            grounded = 0.0
            cols_u = np.unique(comp_cols)
            if len(cols_u):
                hits = 0
                for c in cols_u:
                    r0 = comp_rows[comp_cols == c].max() + rows_sl.start
                    c0 = c + cols_sl.start
                    for rb in range(r0 + 1, min(r0 + 5, H)):
                        if valid[rb, c0]:
                            if abs(d[rb, c0] - d_med) <= 2.0 and dh[rb, c0] < bottom[rb, c0]:
                                hits += 1
                            break
                grounded = hits / len(cols_u)
            # ---- extends above: returns at the same range right above the group (within 4 rows) that lie
            # above the box top -> column / wall / gate / hanging cable reaching the ceiling (objects end in air)
            extends_above = 0.0
            if len(cols_u):
                hits = 0
                for c in cols_u:
                    r0 = comp_rows[comp_cols == c].min() + rows_sl.start
                    c0 = c + cols_sl.start
                    for ra in range(r0 - 1, max(r0 - 5, -1), -1):
                        if valid[ra, c0]:
                            if abs(r[ra, c0] - r_med) <= margin and dh[ra, c0] > top_img[ra, c0]:
                                hits += 1
                            break
                extends_above = hits / len(cols_u)
            # ---- geometry of the group in the analysis frame
            dlh = ri.dlh[0][rows_sl, cols_sl][comp]
            ext = dlh.max(axis=0) - dlh.min(axis=0)
            if ext[0] > p.max_extent_d or (d_med >= p.far_extent_from_d and ext[0] > p.far_max_extent_d):
                continue
            rows_abs = comp_rows + rows_sl.start
            px_h = r_med * math.radians(float(np.mean(row_step[rows_abs])))
            px_w = r_med * math.radians(col_step)
            n_exp = max(1.0, (p.object_size_m / max(px_w, 1e-3)) * (p.object_size_m / max(px_h, 1e-3)))
            support = min(1.0, n / n_exp)
            score = support * (1.0 - attached_frac)
            if attached_frac > p.attached_max_frac or score < p.min_score:
                continue
            dist = float(np.percentile(rr, 10)) if n >= 10 else float(rr.min())
            cen = dlh.mean(axis=0)
            dh_comp = dh[rows_sl, cols_sl][comp]
            dh_c = float(np.mean(dh_comp))
            dl_c = float(np.mean(dl[rows_sl, cols_sl][comp]))
            cls = "obstacle"
            # dual-return cue: a solid object returns one echo per beam (or first == last); a beam that
            # also returns from the tunnel behind the group passed through a semi-transparent medium
            r_far = r_far_img[rows_sl, cols_sl][comp]
            see_through = float(np.mean(r_far > rr + np.maximum(p.see_through_gap_m, 0.03 * rr)))
            # volumetric scatter: a surface is smooth across neighbouring beams, a dust puff is not
            roughness = 0.0
            if n >= p.rough_min_px:
                rw = r[rows_sl, cols_sl]
                vw = valid[rows_sl, cols_sl]
                med = ndimage.median_filter(np.where(vw, rw, np.nan), size=3, mode="nearest")
                med = np.where(np.isfinite(med), med, rw)
                dev = np.abs(rw - med)[comp]
                roughness = float(np.median(dev) / (p.rough_base_m + p.rough_per_m * r_med))
            hw_here = float(np.median(halfwidth[rows_sl, cols_sl][comp]))
            # a face across the corridor (wide, thin along the track, centred) is an obstacle even when it reaches
            # above the clearance or has no free space behind it (a train ahead, a closed gate, a panel)
            face_across = (ext[1] >= p.face_min_width_m and ext[0] <= p.face_max_depth_m and n >= p.face_min_px
                           and ext[2] >= p.face_min_height_m and float(dh_comp.min()) <= p.face_max_bottom_m
                           and abs(dl_c) <= p.face_max_center_offset_m)
            if (n >= p.see_through_min_px and see_through >= p.see_through_min_frac) or roughness >= p.rough_ratio:
                cls = "dust"
            elif face_across:
                cls = "obstacle"
            elif (ext[2] < p.flat_sliver_max_h_m and max(ext[0], ext[1]) >= p.flat_sliver_min_w_m
                  and dh_c >= p.flat_sliver_min_h_above_rail_m and int(rows_abs.max() - rows_abs.min()) >= 1):
                score *= 0.1                   # a few centimetres thick, wide and hanging high: a cable or a duct
                cls = "structure"              # (a group of one row is thin by construction: a far person's shoulders are not a duct)
            elif ext[0] >= p.long_edge_min_length_m and ext[1] <= 1.0 and ext[0] >= 3.0 * max(ext[1], 0.05) and abs(dl_c) >= hw_here - p.long_edge_m:
                score *= 0.1                   # long along the track, narrow, at the corridor edge: wall, pipe, platform
                cls = "structure"              # (new_data: a 6 m x 0.7 m x 2.6 m wall chunk at the edge scored 0.83)
            elif n_bg == 0 and n_known >= p.embedded_min_known:
                score *= 0.1                   # no free space around it at all: a patch of a larger in-corridor surface
                cls = "structure"
            elif extends_above >= p.extends_above_min_frac:
                score *= 0.1                   # continues above the clearance: structure reaching the ceiling
                cls = "structure"
            elif ext[1] <= p.sliver_max_width_m and ext[0] >= p.sliver_min_length_m and abs(dl_c) >= hw_here - p.sliver_edge_m:
                score *= 0.1                   # thin and long along the track at the corridor edge: structure
                cls = "structure"
            elif ext[1] <= p.pole_max_width_m and ext[2] >= 2.0 * max(ext[1], 0.05) and abs(dl_c) >= hw_here - p.pole_edge_m and n >= 3:
                score *= 0.1                   # a thin vertical post at the corridor edge (signs, platform posts)
                cls = "structure"
            elif d_med >= p.hanging_beyond_d and float(dh_comp.min()) >= p.hanging_bottom_m:
                score *= p.hanging_factor      # floating high above the track: hanging infrastructure, most likely
                cls = "hanging"
            slots = ri.slot_of_pixel[0][rows_sl, cols_sl][comp]
            out.append(DetCandidate(
                detector="d2", d=float(cen[0]), l=float(cen[1]), h_above_rail=dh_c,
                extent_dlh=(float(ext[0]), float(ext[1]), float(ext[2])), distance_m=dist, n_support=n,
                score=float(score), cls=cls, lateral_offset=dl_c, slots=slots[slots >= 0],
                info={"attached_frac": attached_frac, "n_bg": n_bg, "n_ring": n_ring, "grounded": grounded,
                      "n_exp": n_exp, "dh_min": float(dh_comp.min()), "see_through": see_through, "roughness": roughness, "straddling": straddling,
                      "extends_above": extends_above, "rows": (int(rows_abs.min()), int(rows_abs.max())),
                      "cols": (int(comp_cols.min() + cols_sl.start), int(comp_cols.max() + cols_sl.start))},
            ))
        return out
