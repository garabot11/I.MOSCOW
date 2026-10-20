"""One input cloud with everything the detectors need, built once per message.

``FrameBuilder.build(msg)`` parses the PointCloud2 buffer through the structured dtype of
``cloud_io``, looks up (or computes, for a new stream) the ``SensorProfile``, applies the
forward crop of organised clouds before any other work and returns a ``Frame`` with the
cropped points in message order, their intensity and ring, and the original slot index of
every point (needed by the range image and to map results back to the message).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .cloud_io import CloudFormatError, pointcloud2_to_struct
from .config import DetectorConfig
from .sensor_profile import SensorProfile, build_profile, detect_layout, signature_of


@dataclass
class Frame:
    xyz: np.ndarray                    # (N, 3) float32, sensor frame, message order (after the crop)
    intensity: Optional[np.ndarray]    # (N,) float32 or None
    ring: Optional[np.ndarray]         # (N,) uint16 or None
    slot: Optional[np.ndarray]         # (N,) int64 message slot of every point; None = identity
    stamp_ns: Optional[int]
    frame_id: str
    n_msg: int                         # points in the message (before the crop)
    profile: SensorProfile
    struct: np.ndarray = field(repr=False, default=None)  # full structured view of the message buffer

    @property
    def n(self) -> int:
        return len(self.xyz)


class FrameBuilder:
    """Parses clouds into ``Frame``s; keeps one ``SensorProfile`` per stream signature."""

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.profiles: Dict[tuple, SensorProfile] = {}
        self.last_new_profile: Optional[SensorProfile] = None

    def reset(self) -> None:
        self.profiles.clear()

    def build(self, msg) -> Frame:
        struct = pointcloud2_to_struct(msg)
        names = struct.dtype.names or ()
        for req in ("x", "y", "z"):
            if req not in names:
                raise CloudFormatError(f"required field '{req}' missing; fields={list(names)}")
        sig = signature_of(msg, names)
        loose = (sig[0], sig[1], sig[3])       # without the point count: an unorganised stream varies in size
        prof = self.profiles.get(sig)
        if prof is None and int(msg.width) * int(msg.height) % 128 != 0:
            prof = self.profiles.get(loose)      # a count that is a multiple of 128 may be an organised layout: detect it
        if prof is None and loose in self.profiles and not detect_layout(struct)[0]:
            # an unorganised frame of a known stream whose count happens to be a multiple of 128 (Falcon: 32 of
            # 4443 frames) keeps the stream's profile: axes re-detected from that one frame flipped the up axis
            # with a person close ahead (a run started at frame 4160 of the Falcon bag)
            prof = self.profiles[sig] = self.profiles[loose]
        self.last_new_profile = None
        if prof is None:
            prof = build_profile(msg, struct, self.cfg)
            self.profiles[sig] = prof
            if not prof.organized:
                # a cloud whose points were filtered or edited (the organisers' clouds with an inserted object
                # have 307187 points, a different count every frame) must not rebuild the profile per frame
                self.profiles[loose] = prof
            self.last_new_profile = prof
        sub = struct
        slot = None
        if prof.keep_slice is not None:
            sub = struct[prof.keep_slice]
            slot = np.arange(prof.keep_slice.start, prof.keep_slice.stop, dtype=np.int64)
        elif prof.keep_cols is not None:
            nr = prof.n_rings
            slot = (prof.keep_cols[:, None] * nr + np.arange(nr)[None, :]).ravel().astype(np.int64)
            sub = struct[slot]
        xyz = np.empty((len(sub), 3), dtype=np.float32)
        xyz[:, 0] = sub["x"]
        xyz[:, 1] = sub["y"]
        xyz[:, 2] = sub["z"]
        inten = np.asarray(sub["intensity"], dtype=np.float32) if "intensity" in names else None
        ring = np.asarray(sub["ring"]) if "ring" in names else None
        st = msg.header.stamp
        stamp_ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
        return Frame(xyz=xyz, intensity=inten, ring=ring, slot=slot, stamp_ns=stamp_ns,
                     frame_id=str(msg.header.frame_id), n_msg=len(struct), profile=prof, struct=struct)
