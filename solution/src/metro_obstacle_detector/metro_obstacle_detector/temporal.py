"""Temporal confirmation of single-frame candidates.

Candidates are associated to short-lived tracks by proximity in (d, l).
A track is confirmed once it collected ``confirm_hits`` hits within the
last ``confirm_window`` frames.  The state is reset when time jumps
backwards (bag loop / restart) or when the gap between frames exceeds
``max_gap_s``; a reset never carries an old detection into new data.  The
check runs before the frame is processed (``check_time``), so the geometric
state of the detector is reset for the same frame (``pipeline.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import DetectorConfig
from .detector import Candidate, FrameResult


@dataclass
class Track:
    track_id: int
    forward_m: float
    lateral_m: float
    distance_m: float
    hits: List[int] = field(default_factory=list)  # frame counters with a hit
    misses: int = 0
    last_frame: int = 0

    def recent_hits(self, frame: int, window: int) -> int:
        return sum(1 for f in self.hits if f > frame - window)


@dataclass
class TemporalDecision:
    status: str                    # 'clear' | 'detected' | 'unknown'
    distance_m: Optional[float]    # nearest confirmed obstacle
    forward_m: Optional[float]
    confirmed: List[Candidate]
    reset: bool = False
    reason: str = ""


class TemporalFilter:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self._next_id = 1
        self._frame = 0
        self._last_stamp_ns: Optional[int] = None

    def reset(self) -> None:
        self.tracks.clear()
        self._last_stamp_ns = None

    def _confirmed(self, tr: Track, cand: Candidate, frame: int) -> bool:
        cfg = self.cfg
        if cand.forward_m >= cfg.far_range_m:
            return tr.recent_hits(frame, cfg.confirm_window_far) >= cfg.confirm_hits_far
        return tr.recent_hits(frame, cfg.confirm_window) >= cfg.confirm_hits

    def check_time(self, stamp_ns: Optional[int]) -> bool:
        """Register the stamp of a new frame; drop the tracks on a backward jump or a gap > ``max_gap_s``.

        Returns True when the state was reset.  A frame without a stamp is not a time reference
        (the last valid stamp is kept).  Call it *before* the geometry of the frame is processed
        (see ``pipeline.DetectionPipeline``) so that no state crosses the discontinuity.
        """
        if stamp_ns is None:
            return False
        reset = False
        if self._last_stamp_ns is not None:
            gap = (stamp_ns - self._last_stamp_ns) / 1e9
            if gap < -1e-3 or gap > self.cfg.max_gap_s:
                self.reset()
                reset = True
        self._last_stamp_ns = stamp_ns
        return reset

    def update(self, result: FrameResult, stamp_ns: Optional[int] = None, reset: bool = False) -> TemporalDecision:
        """Temporal decision for one frame.  ``stamp_ns`` is checked here unless the caller already
        did it with ``check_time`` (then pass ``stamp_ns=None`` and its result as ``reset``)."""
        cfg = self.cfg
        if stamp_ns is not None:
            reset = self.check_time(stamp_ns) or reset
        self._frame += 1
        frame = self._frame
        if result.status == "unknown":
            # keep tracks but count a miss; never turn "unknown" into "clear"
            for tr in self.tracks:
                tr.misses += 1
            self.tracks = [tr for tr in self.tracks if tr.misses <= cfg.track_max_misses]
            return TemporalDecision("unknown", None, None, [], reset, result.reason)
        # ---- association (greedy, nearest first)
        unmatched = list(result.candidates)
        for tr in sorted(self.tracks, key=lambda t: t.distance_m):
            best, best_cost = None, None
            for c in unmatched:
                dd = c.forward_m - tr.forward_m  # negative: object came closer
                if dd < -cfg.assoc_d_closer_m or dd > cfg.assoc_d_farther_m:
                    continue
                if abs(c.lateral_m - tr.lateral_m) > cfg.assoc_lateral_m:
                    continue
                cost = abs(dd) + abs(c.lateral_m - tr.lateral_m)
                if best_cost is None or cost < best_cost:
                    best, best_cost = c, cost
            if best is not None:
                unmatched.remove(best)
                tr.forward_m, tr.lateral_m, tr.distance_m = best.forward_m, best.lateral_m, best.distance_m
                tr.hits.append(frame)
                tr.hits = tr.hits[-max(cfg.confirm_window, cfg.confirm_window_far):]
                tr.misses, tr.last_frame = 0, frame
                best.track_id = tr.track_id
                best.confirmed = self._confirmed(tr, best, frame)
            else:
                tr.misses += 1
        for c in unmatched:
            tr = Track(self._next_id, c.forward_m, c.lateral_m, c.distance_m, [frame], 0, frame)
            self._next_id += 1
            self.tracks.append(tr)
            c.track_id = tr.track_id
            c.confirmed = self._confirmed(tr, c, frame)
        self.tracks = [tr for tr in self.tracks if tr.misses <= cfg.track_max_misses]
        confirmed = [c for c in result.candidates if c.confirmed]
        if confirmed:
            nearest = min(confirmed, key=lambda c: c.distance_m)
            return TemporalDecision("detected", nearest.distance_m, nearest.forward_m, confirmed, reset)
        return TemporalDecision("clear", None, None, [], reset)
