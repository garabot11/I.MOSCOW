"""Per-frame state handling shared by the ROS node and the offline evaluator.

Order for every frame: (1) check the header stamp for a backward jump / gap,
(2) on a discontinuity reset *both* the temporal tracks and the geometric state
of the detector (previous track curve), (3) process the geometry, (4) temporal
confirmation.  Doing the check first guarantees that the first frame after a
jump never uses the model of the previous sequence, and using this one class in
both adapters guarantees identical behaviour online and offline.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import DetectorConfig
from .detector import FrameResult, ObstacleDetector
from .temporal import TemporalDecision, TemporalFilter


def header_stamp_ns(msg) -> Optional[int]:
    """Stamp of a message header in ns (None if it has no header)."""
    try:
        st = msg.header.stamp
        return int(st.sec) * 1_000_000_000 + int(st.nanosec)
    except AttributeError:
        return None


class DetectionPipeline:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.detector = ObstacleDetector(cfg)
        self.temporal = TemporalFilter(cfg)

    def reset(self) -> None:
        """Forget all state (stale input, restart)."""
        self.detector.reset()
        self.temporal.reset()

    def _begin(self, stamp_ns: Optional[int]) -> bool:
        reset = self.temporal.check_time(stamp_ns)
        if reset:
            self.detector.reset()
        return reset

    def step(self, xyz: np.ndarray, stamp_ns: Optional[int]) -> Tuple[FrameResult, TemporalDecision]:
        reset = self._begin(stamp_ns)
        res = self.detector.process(xyz)
        return res, self.temporal.update(res, reset=reset)

    def step_invalid(self, reason: str, stamp_ns: Optional[int]) -> Tuple[FrameResult, TemporalDecision]:
        """A frame that could not be decoded: UNKNOWN, but its stamp still counts for the gap check."""
        reset = self._begin(stamp_ns)
        res = FrameResult(status="unknown", reason=reason)
        return res, self.temporal.update(res, reset=reset)
