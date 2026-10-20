"""Conversion of sensor_msgs/PointCloud2 into NumPy arrays.

The clouds in this project have mixed field types (float32 xyz/intensity,
uint16 ring, float64 timestamp) and a 26-byte point step, so the buffer is
read through an explicit structured dtype built from ``msg.fields`` with the
declared offsets and ``point_step`` as itemsize. No automatic alignment is
applied, and ``row_step``/``height`` are honoured for organised clouds.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import numpy as np

# sensor_msgs/msg/PointField datatype codes -> numpy type characters
_PF_TO_NP = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


class CloudFormatError(ValueError):
    """Raised when a PointCloud2 message cannot be interpreted safely."""


@dataclass
class CloudInfo:
    """Metadata of a converted cloud that is useful for logs and diagnostics."""

    frame_id: str
    stamp_ns: int
    width: int
    height: int
    point_step: int
    row_step: int
    field_names: Sequence[str]
    n_points: int


def build_dtype(fields, point_step: int, is_bigendian: bool) -> np.dtype:
    """Build a structured dtype mirroring the PointCloud2 memory layout."""
    names, formats, offsets = [], [], []
    endian = ">" if is_bigendian else "<"
    for f in fields:
        if f.datatype not in _PF_TO_NP:
            raise CloudFormatError(f"unsupported PointField datatype {f.datatype} for '{f.name}'")
        base = _PF_TO_NP[f.datatype]
        count = int(getattr(f, "count", 1) or 1)
        fmt = f"{endian}{base}" if count == 1 else f"{count}{endian}{base}"
        size = np.dtype(fmt).itemsize
        if f.offset + size > point_step:
            raise CloudFormatError(
                f"field '{f.name}' (offset {f.offset}, size {size}) exceeds point_step {point_step}"
            )
        names.append(f.name)
        formats.append(fmt)
        offsets.append(int(f.offset))
    if len(set(names)) != len(names):
        raise CloudFormatError(f"duplicate field names: {names}")
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(point_step)})


class _CdrReader:
    """Minimal XCDR1 reader (plain CDR, little or big endian) with bounds checks."""

    def __init__(self, raw) -> None:
        self.buf = memoryview(raw)
        if len(self.buf) < 4:
            raise CloudFormatError("serialized message shorter than the CDR header")
        kind = bytes(self.buf[0:2])
        if kind == b"\x00\x01":
            self.e = "<"
        elif kind == b"\x00\x00":
            self.e = ">"
        else:
            raise CloudFormatError(f"unsupported CDR encapsulation {kind.hex()}")
        self.o = 4  # alignment is relative to the end of the 4-byte encapsulation header

    def _align(self, n: int) -> None:
        self.o += (-(self.o - 4)) % n

    def _need(self, n: int) -> None:
        if self.o + n > len(self.buf):
            raise CloudFormatError(f"serialized PointCloud2 truncated at byte {self.o} (+{n} of {len(self.buf)})")

    def u32(self) -> int:
        self._align(4)
        self._need(4)
        v = struct.unpack_from(self.e + "I", self.buf, self.o)[0]
        self.o += 4
        return v

    def i32(self) -> int:
        self._align(4)
        self._need(4)
        v = struct.unpack_from(self.e + "i", self.buf, self.o)[0]
        self.o += 4
        return v

    def u8(self) -> int:
        self._need(1)
        v = self.buf[self.o]
        self.o += 1
        return v

    def string(self) -> str:
        n = self.u32()  # length including the terminating NUL
        self._need(n)
        s = bytes(self.buf[self.o:self.o + max(n - 1, 0)]).decode("utf-8", "replace")
        self.o += n
        return s

    def octets(self) -> memoryview:
        n = self.u32()
        self._need(n)
        v = self.buf[self.o:self.o + n]
        self.o += n
        return v


def parse_pointcloud2_cdr(raw):
    """Read a CDR-serialised sensor_msgs/PointCloud2 without rclpy deserialisation.

    Returns an object with the attributes of the message used here (``header.stamp.sec/nanosec``,
    ``header.frame_id``, ``height``, ``width``, ``fields``, ``is_bigendian``, ``point_step``,
    ``row_step``, ``data``, ``is_dense``); ``data`` is a zero-copy view of ``raw``.  This skips
    the conversion of the point buffer into a Python array (≈ 20 ms for a 24 MB cloud), which
    otherwise runs in the node's only thread before every frame.  The decoded values are
    identical to ``rclpy.serialization.deserialize_message`` (tested on every recorded message).
    """
    r = _CdrReader(raw)
    sec = r.i32()
    nanosec = r.u32()
    frame_id = r.string()
    height = r.u32()
    width = r.u32()
    n_fields = r.u32()
    if n_fields > 1024:
        raise CloudFormatError(f"implausible number of fields: {n_fields}")
    fields = []
    for _ in range(n_fields):
        name = r.string()
        offset = r.u32()
        datatype = r.u8()
        count = r.u32()
        fields.append(SimpleNamespace(name=name, offset=offset, datatype=datatype, count=count))
    is_bigendian = bool(r.u8())
    point_step = r.u32()
    row_step = r.u32()
    data = r.octets()
    is_dense = bool(r.u8())
    header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)
    return SimpleNamespace(header=header, height=height, width=width, fields=fields, is_bigendian=is_bigendian,
                           point_step=point_step, row_step=row_step, data=data, is_dense=is_dense)


def pointcloud2_to_struct(msg) -> np.ndarray:
    """Return a structured array view (no copy where possible) of all points."""
    n = int(msg.width) * int(msg.height)
    if msg.point_step <= 0:
        raise CloudFormatError("point_step must be positive")
    dt = build_dtype(msg.fields, msg.point_step, msg.is_bigendian)
    data = msg.data
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = memoryview(data)
    nbytes = len(data)
    if msg.height > 1 and msg.row_step != msg.width * msg.point_step:
        # organised cloud with row padding: gather row by row
        if nbytes < msg.row_step * msg.height:
            raise CloudFormatError(f"data has {nbytes} bytes, expected >= row_step*height={msg.row_step * msg.height}")
        rows = [
            np.frombuffer(data, dtype=dt, count=msg.width, offset=r * msg.row_step)
            for r in range(msg.height)
        ]
        return np.concatenate(rows)
    if nbytes < n * msg.point_step:
        raise CloudFormatError(f"data has {nbytes} bytes, expected >= {n * msg.point_step} for {n} points")
    return np.frombuffer(data, dtype=dt, count=n)


def cloud_to_xyz(msg, intensity: bool = True):
    """Convert a PointCloud2 into ``(xyz float32 (N,3), intensity float32 (N,) or None, CloudInfo)``.

    Rows are returned in message order; no filtering is done here so callers can
    account for invalid (non-finite / all-zero) points explicitly.
    """
    struct = pointcloud2_to_struct(msg)
    names = struct.dtype.names or ()
    for req in ("x", "y", "z"):
        if req not in names:
            raise CloudFormatError(f"required field '{req}' missing; fields={list(names)}")
    xyz = np.empty((len(struct), 3), dtype=np.float32)
    xyz[:, 0] = struct["x"]
    xyz[:, 1] = struct["y"]
    xyz[:, 2] = struct["z"]
    inten: Optional[np.ndarray] = None
    if intensity and "intensity" in names:
        inten = np.asarray(struct["intensity"], dtype=np.float32)
    stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    info = CloudInfo(
        frame_id=str(msg.header.frame_id),
        stamp_ns=stamp_ns,
        width=int(msg.width),
        height=int(msg.height),
        point_step=int(msg.point_step),
        row_step=int(msg.row_step),
        field_names=tuple(names),
        n_points=int(len(struct)),
    )
    return xyz, inten, info


def validity_mask(xyz: np.ndarray) -> Dict[str, np.ndarray]:
    """Return masks for finite points and for the exact (0,0,0) placeholder returns."""
    finite = np.isfinite(xyz).all(axis=1)
    zero = (xyz == 0).all(axis=1)
    return {"finite": finite, "zero": zero, "valid": finite & ~zero}
