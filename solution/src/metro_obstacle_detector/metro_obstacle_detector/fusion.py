"""Fusion of the detectors over time: evidence accumulation per track (sequential test).

Every detector contributes candidates (``DetCandidate`` or D1's ``Candidate``) with a score
0..1.  Candidates are associated to tracks in (d, l); each track accumulates a log-likelihood
ratio (LLR): a hit of detector *k* adds ``llr_hit[k] * score``, two or more detectors hitting
in the same frame add ``agree_bonus``, a frame without any hit adds ``llr_miss``.  The sum is
clipped to ``[llr_delete, llr_max]`` so that a long-confirmed track still disappears a few
frames after the object is gone.  A track is confirmed when the LLR reaches the threshold
*and* the agreement rule holds:

* at every range at least two distinct detectors must have voted for the track in the last
  ``agree_window`` frames ("minimum two of three"; D3 adds evidence but does not vote);
* beyond ``two_of_three_max_d`` the LLR threshold is higher and the track must have been seen in
  ``far_min_frames`` frames; the old exception "persistence of one detector counts as the second
  vote" is behind ``far_single_detector`` (off).

Once confirmed, a track stays confirmed while its LLR is above ``llr_keep`` (hysteresis), so a
detection does not blink when single frames are missed.  Confirmed tracks are written back
into the frame result as confirmed candidates (a track hit only by D2/D3, or coasting without a
hit this frame, becomes a synthetic candidate), so the node, the markers and the offline tool
need no change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .candidates import DetCandidate
from .config import DetectorConfig
from .detector import Candidate, FrameResult
from .temporal import TemporalDecision


@dataclass
class FusionParams:
    llr_hit: Dict[str, float] = field(default_factory=lambda: {"d1": 2.0, "d2": 1.2, "d3": 1.0})
    vote_min_score_det: Dict[str, float] = field(default_factory=lambda: {"d3": 1.01})  # per-detector vote threshold; D3 (model 1) adds evidence but does not vote until validated on unseen tunnels
    far_min_frames: int = 4            # beyond two_of_three_max_d a track must be seen in at least this many frames (3-frame blips of far wall returns confirmed)
    far_single_detector: bool = False  # beyond two_of_three_max_d one detector alone may confirm after persist_frames_far frames: OFF -
                                       # two detectors must agree at every range (a wall swept into the corridor at 133 m
                                       # was confirmed by D1 alone while D2 called it structure)
    score_floor: Dict[str, float] = field(default_factory=lambda: {"d1": 0.3, "d2": 0.0, "d3": 0.0})  # a hit counts at least this fraction of llr_hit (D1 candidates are vetted by its rules; a 1-pixel D2 group is not)
    agree_bonus: float = 0.6
    agree_min_score: float = 0.3       # the agreement bonus needs two detectors with at least this score
    llr_miss: float = -0.8
    llr_confirm: float = 3.0
    llr_confirm_far: float = 4.0       # beyond two_of_three_max_d: ~4 consistent single-detector frames
    llr_keep: float = 1.2              # confirmed tracks stay confirmed above this
    llr_delete: float = -2.0
    llr_max: float = 6.0
    two_of_three_max_d: float = 120.0
    agree_window: int = 5
    persist_frames_far: int = 4
    strong_enabled: bool = False       # one detector alone confirming within two_of_three_max_d by persistence: OFF -
    strong_llr: float = 5.0            # on the real clear bags static structure (niches, brackets) persisted with
    strong_frames: int = 5             # high D2 scores for seconds and was confirmed; "two of three" stays strict
    strong_min_score: float = 0.5
    vote_min_score: float = 0.3        # a detector votes for the agreement rule only with at least this score
    max_misses: int = 8
    dust_veto_llr: float = -1.5        # a "dust" classification of the hit subtracts this
    d3_dust_min_score: float = 0.5     # a D3 dust component with at least this score makes the track dust even if D1 sees a surface


@dataclass
class _Hit:
    det: str
    d: float
    l: float
    distance_m: float
    h_above_rail: float
    h: float                           # analysis-frame height of the centre (for the sensor-frame position)
    extent: Tuple[float, float, float]
    score: float
    n_support: int
    cls: str
    ref: object                        # the originating candidate object
    lateral_offset: float = 0.0        # l - l_c(d)


@dataclass
class FusedTrack:
    track_id: int
    d: float
    l: float
    distance_m: float
    h_above_rail: float
    h: float
    extent: Tuple[float, float, float]
    n_support: int
    llr: float = 0.0
    recent: List[Tuple[int, Set[str], Set[str]]] = field(default_factory=list)   # (frame, voting detectors, all hitting detectors)
    first_frame: int = 0
    last_hit_frame: int = 0
    misses: int = 0
    confirmed: bool = False
    cls: str = "obstacle"
    lateral_offset: float = 0.0
    n_frames_seen: int = 0        # frames in which a point-based detector (D1/D2) placed the track

    def votes(self, frame: int, window: int) -> Tuple[int, int]:
        """(distinct detectors, frames with a hit) within the last ``window`` frames."""
        dets: Set[str] = set()
        n_frames = 0
        for f, ks, hs in self.recent:
            if f > frame - window and (hs & {"d1", "d2"}):
                # frames with D1/D2 hits below the voting score count as "seen": the Falcon person walking
                # 100 -> 4 m is hit weakly in ~190 of 710 frames and the hysteresis must carry him; D3-only
                # frames do not count (weak D3 blobs kept a stale track alive for 40 frames on synthetic
                # scenes); confirmation itself still needs two voting detectors
                dets |= ks
                n_frames += 1
        return len(dets), n_frames


class FusionFilter:
    def __init__(self, cfg: DetectorConfig, params: Optional[FusionParams] = None):
        self.cfg = cfg
        self.p = params or FusionParams()
        self.tracks: List[FusedTrack] = []
        self._next_id = 1
        self._frame = 0

    def reset(self) -> None:
        self.tracks.clear()

    # ------------------------------------------------------------------
    def _far_no_vote(self, res: FrameResult, forward_m: float, n_points: int) -> bool:
        """far_vote_d1_min_d: on a moving train a far D1 cluster with only a few voxels adds evidence but never votes
        (sparse far wall returns at 156-165 m on the 20-minute ride) - by any path, the compact pair vote included."""
        cfg = self.cfg
        far_d = float(getattr(cfg, "far_vote_d1_min_d", 0.0) or 0.0)
        standing = float(getattr(res, "static_frac", 0.0) or 0.0) >= getattr(cfg, "far_weak_d1_static_frac", 0.85)
        return far_d > 0.0 and not standing and forward_m >= far_d and n_points < cfg.far_vote_d1_min_points

    def _hits_from_d1(self, res: FrameResult, det_cands: Sequence[DetCandidate] = ()) -> List[_Hit]:
        cfg = self.cfg
        static_frac = float(getattr(res, "static_frac", 0.0) or 0.0)
        standing = static_frac >= getattr(cfg, "far_weak_d1_static_frac", 0.85)
        plate_bg = float(getattr(cfg, "plate_d2_min_bg_frac", 0.5))
        d2_free = None
        out = []
        for c in res.candidates:
            score = float(c.confidence)
            if self._far_no_vote(res, c.forward_m, c.n_points):
                # moving train, far and only a few voxels: evidence (the score floor keeps its LLR gain) but no vote
                score = 0.0
            if getattr(c, "plate", False):
                # accepted only by the plate exception: a plate on the bed casts a shadow (background behind its D2
                # group), a step in the bed is continuous with the surface behind it -> evidence only, no vote
                if d2_free is None:
                    d2_free = [k for k in det_cands if k.detector == "d2" and k.cls == "obstacle" and k.score >= self.p.vote_min_score
                               and int((getattr(k, "info", None) or {}).get("n_bg", 0))
                               >= plate_bg * max(int((getattr(k, "info", None) or {}).get("n_ring", 0)), 1)]
                if not any(abs(k.d - c.forward_m) <= cfg.assoc_d_farther_m and abs(k.l - c.lateral_m) <= cfg.assoc_lateral_m
                           for k in d2_free):
                    score = 0.0
            out.append(_Hit("d1", float(c.forward_m), float(c.lateral_m), float(c.distance_m), float(c.height_above_rail),
                            float(c.height_m), tuple(float(v) for v in c.extent), score, int(c.n_points),
                            "obstacle", c, float(c.lateral_offset_from_track)))
        if getattr(cfg, "far_weak_d1_vote", False) and standing:
            # weak far clusters (rejected only for their support) vote when D2 sees an obstacle at the same place
            d2v = [k for k in det_cands if k.detector == "d2" and k.cls == "obstacle" and k.score >= self.p.vote_min_score]
            for c in res.rejected:
                if not getattr(c, "weak", False):
                    continue
                if not any(abs(k.d - c.forward_m) <= cfg.assoc_d_farther_m and abs(k.l - c.lateral_m) <= cfg.assoc_lateral_m
                           for k in d2v):
                    continue
                out.append(_Hit("d1", float(c.forward_m), float(c.lateral_m), float(c.distance_m), float(c.height_above_rail),
                                float(c.height_m), tuple(float(v) for v in c.extent), self.p.vote_min_score, int(c.n_points),
                                "obstacle", c, float(c.lateral_offset_from_track)))
        return out

    def _compact_pairs(self, res: FrameResult, hits: List[_Hit]) -> List[_Hit]:
        """compact_pair_vote: a small compact D1 cluster and an isolated D2 obstacle group at the same place in the
        same frame both vote, whatever their size-normalised scores.  A 0.3 m cube scores 0.15-0.25 in both detectors
        (support relative to a 0.5 m object) and was never confirmed although both saw it for 30 frames.  The D2 group
        must stand in free space (no attached neighbours, background around almost all of its ring) and both must be
        small; D1 clusters rejected only for their support or as a short edge sliver (``Candidate.compact``) take part
        as extra D1 hits.  Raises the scores of the paired hits in place and returns the extra D1 hits."""
        cfg, p = self.cfg, self.p
        vote = p.vote_min_score
        d2, d2_free = [], set()          # free-space D2 groups; d2_free: background around almost all of the ring
        min_bg = min(cfg.compact_min_bg_frac, getattr(cfg, "straddle_min_bg_frac", 1.0)) if getattr(cfg, "straddle_compact_exempt", False) \
            else cfg.compact_min_bg_frac
        for h in hits:
            if h.det != "d2" or h.cls != "obstacle" or h.n_support < cfg.compact_min_d2_px or max(h.extent) > cfg.compact_max_extent_m:
                continue
            info = getattr(h.ref, "info", None) or {}
            n_ring = int(info.get("n_ring", 0))
            if n_ring > 0 and float(info.get("attached_frac", 1.0)) <= cfg.compact_max_attached:
                n_bg = int(info.get("n_bg", 0))
                if n_bg >= cfg.compact_min_bg_frac * n_ring:
                    d2_free.add(id(h))
                if n_bg >= min_bg * n_ring:
                    d2.append(h)
        if not d2:
            return []
        d1 = [h for h in hits if h.det == "d1" and not getattr(h.ref, "weak", False)
              and h.n_support >= cfg.compact_min_d1_points and max(h.extent) <= cfg.compact_max_extent_m
              and not self._far_no_vote(res, h.d, h.n_support)]
        emitted = {id(h.ref) for h in hits if h.det == "d1"}   # a weak cluster already voting is not added twice
        extra = []
        for c in res.rejected:
            if getattr(c, "compact", False) and id(c) not in emitted and not self._far_no_vote(res, c.forward_m, c.n_points):
                extra.append(_Hit("d1", float(c.forward_m), float(c.lateral_m), float(c.distance_m), float(c.height_above_rail),
                                  float(c.height_m), tuple(float(v) for v in c.extent), 0.0, int(c.n_points),
                                  "obstacle", c, float(c.lateral_offset_from_track)))
        all_d1 = list(res.candidates) + list(res.rejected)

        def isolated(c) -> bool:
            """A compact object stands alone: no bigger D1 cluster within compact_isolation_{d,l}_m of it, and no
            other fragment of at least compact_min_d1_points outside the pair window.  A dust cloud breaks into
            many small clusters a metre apart (val_dust: a 15-voxel fragment and a 5-pixel speck inside a cloud
            of 100-200-voxel clusters confirmed as an obstacle); the organisers' 0.3 m cubes have nothing within
            3 m along / 2.5 m across."""
            for o in all_d1:
                if o is c:
                    continue
                dd, dl = abs(float(o.forward_m) - float(c.forward_m)), abs(float(o.lateral_m) - float(c.lateral_m))
                if dd > cfg.compact_isolation_d_m or dl > cfg.compact_isolation_l_m:
                    continue
                if float(np.max(o.extent)) > cfg.compact_max_extent_m:
                    return False
                if int(o.n_points) >= cfg.compact_min_d1_points and (dd > cfg.compact_pair_dd_m or dl > cfg.compact_pair_dl_m):
                    return False
            return True

        out = []
        for is_extra, group in ((False, d1), (True, extra)):
            for h1 in group:
                # a cluster compact only through the straddle exemption may pair with a D2 group bordered by the rest of the object
                straddle = getattr(h1.ref, "straddle", False)
                mate = [h2 for h2 in d2 if abs(h2.d - h1.d) <= cfg.compact_pair_dd_m and abs(h2.l - h1.l) <= cfg.compact_pair_dl_m
                        and (straddle or id(h2) in d2_free)]
                if not mate or not isolated(h1.ref):
                    continue
                h1.score = max(h1.score, vote)
                for h2 in mate:
                    h2.score = max(h2.score, vote)
                if is_extra:
                    out.append(h1)
        return out

    def _hits_from_det(self, cands: Sequence[DetCandidate], rail) -> List[_Hit]:
        out = []
        for c in cands:
            h = float(c.h_above_rail + (rail(c.d) if rail is not None else 0.0))
            out.append(_Hit(c.detector, float(c.d), float(c.l), float(c.distance_m), float(c.h_above_rail), h,
                            tuple(float(v) for v in c.extent_dlh), float(c.score), int(c.n_support), c.cls, c,
                            float(c.lateral_offset)))
        return out

    def _associate(self, hits: List[_Hit], frame: int) -> Dict[int, List[_Hit]]:
        cfg = self.cfg
        assigned: Dict[int, List[_Hit]] = {tr.track_id: [] for tr in self.tracks}
        free = list(hits)
        for tr in sorted(self.tracks, key=lambda t: t.distance_m):
            used_dets: Set[str] = set()
            for det in ("d1", "d3", "d2"):
                best, best_cost = None, None
                for hname in free:
                    if hname.det != det or det in used_dets:
                        continue
                    dd = hname.d - tr.d
                    if dd < -cfg.assoc_d_closer_m or dd > cfg.assoc_d_farther_m:
                        continue
                    dl = abs(hname.l - tr.l)
                    if dl > cfg.assoc_lateral_m:
                        continue
                    cost = abs(dd) + dl
                    if best_cost is None or cost < best_cost:
                        best, best_cost = hname, cost
                if best is not None:
                    free.remove(best)
                    used_dets.add(det)
                    assigned[tr.track_id].append(best)
        # new tracks from the remaining D1/D2 hits; a D2 hit joins a track born from a D1 hit this frame.
        # D3 hits never start a track: a D3 blob born as a track one frame earlier stole the D2 hits of
        # the real (confirmed) track, which then coasted and died while the blob's track could not confirm
        new_tracks: List[FusedTrack] = []
        for det in ("d1", "d2"):
            for hname in [x for x in free if x.det == det]:
                joined = False
                for tr in new_tracks:
                    if abs(hname.d - tr.d) <= cfg.assoc_d_farther_m + 1.0 and abs(hname.l - tr.l) <= cfg.assoc_lateral_m:
                        assigned[tr.track_id].append(hname)
                        joined = True
                        break
                if joined:
                    continue
                tr = FusedTrack(self._next_id, hname.d, hname.l, hname.distance_m, hname.h_above_rail, hname.h,
                                hname.extent, hname.n_support, first_frame=frame, last_hit_frame=frame, cls=hname.cls,
                                lateral_offset=hname.lateral_offset)
                self._next_id += 1
                new_tracks.append(tr)
                assigned[tr.track_id] = [hname]
        self.tracks.extend(new_tracks)
        return assigned

    # ------------------------------------------------------------------
    def update(self, res: FrameResult, det_cands: Sequence[DetCandidate], rail, R: np.ndarray,
               reset: bool = False, allow_clear: bool = True, max_d: Optional[float] = None) -> TemporalDecision:
        """Fuse this frame's candidates; returns the decision and marks/creates confirmed candidates in ``res``.
        ``max_d``: tracks farther along the track than this (the verified corridor range, which a wall sweeping
        into the corridor can cut below a track confirmed a few frames earlier) are not reported."""
        p = self.p
        self._frame += 1
        frame = self._frame
        hits = self._hits_from_d1(res, det_cands) + self._hits_from_det(det_cands, rail)
        if getattr(self.cfg, "compact_pair_vote", False):
            hits += self._compact_pairs(res, hits)
        assigned = self._associate(hits, frame)
        for tr in self.tracks:
            hs = assigned.get(tr.track_id, [])
            if hs:
                dets = {h.det for h in hs}
                gain = 0.0
                for h in hs:
                    s = max(p.score_floor.get(h.det, 0.0), min(1.0, h.score))
                    g = p.llr_hit.get(h.det, 1.0) * s
                    if h.cls == "dust":
                        g += p.dust_veto_llr
                    elif h.cls in ("hanging", "structure"):
                        g *= 0.5
                    gain += g
                if len({h.det for h in hs if h.score >= p.agree_min_score}) >= 2:
                    gain += p.agree_bonus
                tr.llr = min(p.llr_max, tr.llr + gain)
                tr.recent.append((frame, {h.det for h in hs if h.score >= p.vote_min_score_det.get(h.det, p.vote_min_score) and h.cls != "dust"},
                                  {h.det for h in hs}))
                tr.recent = [x for x in tr.recent if x[0] > frame - max(p.agree_window, 8)]
                tr.misses = 0
                tr.last_hit_frame = frame
                # geometry from the point-based detectors (D1, then D2); D3 components are coarse range-image
                # blobs (on an untextured floor they span 10-20 m), so a frame with D3 hits only adds evidence
                # and leaves the track where D1/D2 last put it
                geo = [h for h in hs if h.det != "d3"] or ([] if tr.n_frames_seen > 0 else hs)
                if geo:
                    best = sorted(geo, key=lambda h: ("d1", "d2", "d3").index(h.det) if h.det in ("d1", "d2", "d3") else 9)[0]
                    tr.d, tr.l, tr.distance_m = best.d, best.l, best.distance_m
                    tr.h_above_rail, tr.h, tr.extent, tr.n_support = best.h_above_rail, best.h, best.extent, best.n_support
                    tr.lateral_offset = best.lateral_offset
                    tr.n_frames_seen += 1
                # the class follows the latest evidence (a dust verdict is not sticky: an object entering the
                # corridor edge-on can look see-through for a frame)
                d3_dust = any(h.det == "d3" and h.cls == "dust" and h.score >= p.d3_dust_min_score for h in hs)
                if d3_dust or (any(h.cls == "dust" for h in hs) and not any(h.det == "d1" or h.cls == "obstacle" for h in hs)):
                    tr.cls = "dust"          # the trained detector has the say on dust; D1 cannot tell dust from a surface
                elif any(h.cls == "obstacle" for h in hs):
                    tr.cls = "obstacle"
            else:
                tr.misses += 1
                tr.llr = max(p.llr_delete, tr.llr + p.llr_miss)
            far = tr.d >= p.two_of_three_max_d
            n_dets, n_frames = tr.votes(frame, p.agree_window)
            # seen in at least two frames, and (near) by two detectors or (far) persistently by one
            rule = n_frames >= (p.far_min_frames if far else 2) and (
                n_dets >= 2 or (far and p.far_single_detector and n_frames >= p.persist_frames_far))
            thr = p.llr_confirm_far if far else p.llr_confirm
            # a single well-supported detector seen frame after frame is the second vote everywhere
            _, n_frames6 = tr.votes(frame, p.agree_window + 1)
            strong = (p.strong_enabled and hs and max(h.score for h in hs) >= p.strong_min_score
                      and n_frames6 >= p.strong_frames and tr.llr >= p.strong_llr)
            if not tr.confirmed:
                tr.confirmed = ((tr.llr >= thr and rule) or strong) and tr.cls != "dust"
            else:
                # hysteresis while D1 or D2 still hits the track in the window (weak hits included: the
                # close-range Falcon person is hit only weakly in a quarter of his frames; D3-only frames do not count);
                # the LLR decay (-0.8 per missed frame) ends it within a few frames after the object left
                tr.confirmed = tr.llr >= p.llr_keep and n_frames >= 1 and tr.cls != "dust"
        self.tracks = [tr for tr in self.tracks if tr.llr > p.llr_delete and tr.misses <= p.max_misses]
        # ---- write back: confirmed candidates in the frame result
        for c in res.candidates:
            c.confirmed = False
        confirmed: List[Candidate] = []
        for tr in self.tracks:
            hs = assigned.get(tr.track_id, [])
            d1 = [h for h in hs if h.det == "d1" and not getattr(h.ref, "weak", False)
                  and not getattr(h.ref, "compact", False)]   # a weak / compact D1 cluster stays in res.rejected
            for h in hs:
                if h.det == "d1":
                    h.ref.track_id = tr.track_id
            if not tr.confirmed or (max_d is not None and tr.d > max_d):
                continue
            if d1:
                cand = d1[0].ref
                cand.confirmed = True
            else:
                center = R.T @ np.array([tr.d, tr.l, tr.h], dtype=np.float64)
                cand = Candidate(distance_m=tr.distance_m, forward_m=tr.d, lateral_m=tr.l, height_m=tr.h,
                                 center_sensor=center, extent=np.array(tr.extent, dtype=np.float64),
                                 n_points=tr.n_support, lateral_offset_from_track=tr.lateral_offset, height_above_rail=tr.h_above_rail,
                                 confidence=min(1.0, max(0.0, tr.llr / p.llr_max)), track_id=tr.track_id, confirmed=True)
                res.candidates.append(cand)
            confirmed.append(cand)
        res.fusion = {"tracks": [{"id": tr.track_id, "d": round(tr.d, 1), "l": round(tr.l, 2), "llr": round(tr.llr, 2),
                                  "confirmed": tr.confirmed, "misses": tr.misses, "cls": tr.cls,
                                  "dets": sorted(set().union(*[ks for _, ks, _ in tr.recent])) if tr.recent else []}
                                 for tr in self.tracks]}
        if confirmed:
            nearest = min(confirmed, key=lambda c: c.distance_m)
            return TemporalDecision("detected", nearest.distance_m, nearest.forward_m, confirmed, reset)
        if not allow_clear:
            return TemporalDecision("unknown", None, None, [], reset, res.reason)
        return TemporalDecision("clear", None, None, [], reset)
