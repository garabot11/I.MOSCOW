"""Detector-independent candidate record consumed by the fusion stage (interface E)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class DetCandidate:
    detector: str                 # "d1" | "d2" | "d3"
    d: float                      # centre along the track (analysis frame)
    l: float                      # centre lateral
    h_above_rail: float           # centre height above the rail level
    extent_dlh: Tuple[float, float, float]
    distance_m: float             # Euclidean range to the nearest reliable surface
    n_support: int                # points / pixels
    score: float                  # detector-specific confidence 0..1
    cls: str = "obstacle"         # "obstacle" | "dust" | "structure"
    lateral_offset: float = 0.0   # l - l_c(d)
    slots: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector, "d": round(float(self.d), 2), "l": round(float(self.l), 2),
            "h_above_rail": None if self.h_above_rail is None else round(float(self.h_above_rail), 2),
            "extent_dlh": [round(float(v), 2) for v in self.extent_dlh],
            "distance_m": round(float(self.distance_m), 2), "n_support": int(self.n_support),
            "score": round(float(self.score), 3), "cls": self.cls,
            "lateral_offset": None if self.lateral_offset is None else round(float(self.lateral_offset), 2),
            "info": {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in self.info.items()},
        }
