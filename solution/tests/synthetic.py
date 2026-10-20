"""Synthetic tunnel clouds with known geometry for unit tests (metres, sensor frame: forward = -y, up = +z)."""
import numpy as np


def make_pointcloud2(xyz, extra_fields=True, bigendian=False, height=1, row_pad=0, drop=None):
    """Build a sensor_msgs/PointCloud2 with the real 26-byte mixed layout (x,y,z,intensity float32, ring uint16, timestamp float64)."""
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    n = len(xyz)
    endian = ">" if bigendian else "<"
    dt = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                   "formats": [endian + "f4", endian + "f4", endian + "f4", endian + "f4", endian + "u2", endian + "f8"],
                   "offsets": [0, 4, 8, 12, 16, 18], "itemsize": 26})
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["intensity"] = 10.0
    arr["ring"] = np.arange(n) % 128
    arr["timestamp"] = 946687297.2
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = "hesai_lidar"
    msg.header.stamp.sec, msg.header.stamp.nanosec = 946687297, 199933052
    fields = [("x", 0, PointField.FLOAT32), ("y", 4, PointField.FLOAT32), ("z", 8, PointField.FLOAT32),
              ("intensity", 12, PointField.FLOAT32), ("ring", 16, PointField.UINT16), ("timestamp", 18, PointField.FLOAT64)]
    if not extra_fields:
        fields = fields[:3]
    if drop:
        fields = [f for f in fields if f[0] not in drop]
    for name, off, dtype in fields:
        pf = PointField()
        pf.name, pf.offset, pf.datatype, pf.count = name, off, dtype, 1
        msg.fields.append(pf)
    msg.is_bigendian = bigendian
    msg.point_step = 26
    if height > 1:
        width = n // height
        msg.height, msg.width = height, width
        msg.row_step = width * 26 + row_pad
        rows = []
        raw = arr.tobytes()
        for r in range(height):
            rows.append(raw[r * width * 26:(r + 1) * width * 26] + b"\0" * row_pad)
        msg.data = b"".join(rows)
    else:
        msg.height, msg.width = 1, n
        msg.row_step = n * 26
        msg.data = arr.tobytes()
    msg.is_dense = False
    return msg


def tunnel_cloud(length=200.0, radius=2.7, sensor_height=1.3, obstacle=None, yaw=0.0, curvature=0.0,
                 rng=None, n_ring=240, n_az=600, zero_fraction=0.4, elev_deg=(-20.0, 10.0), az_deg=(-45.0, 45.0)):
    """Ray-cast a round tunnel (radius ``radius``) whose track centre follows l_c(d) = yaw*d + curvature*d^2.

    The sensor sits ``sensor_height`` above rail level on the track centre.  The track bed carries two
    rail heads (+-0.76 m, 0.15 m high) and a central trough (0.3 m deep, 1.0 m wide) like the real data.
    ``obstacle`` is (d, l, w, h): a box of width w and height h standing on the track bed at forward
    distance d and lateral offset l from the track centre.  Returns an (N,3) float32 array in the sensor
    frame (x lateral, -y forward, z up) including exact-zero placeholder returns like the real sensor.
    Vectorised ray marching (0.1 m steps).
    """
    rng = rng or np.random.default_rng(0)
    # ring layout like a 128-line automotive sensor: fine (0.125 deg) around the horizon, coarser outside
    elev = np.deg2rad(np.concatenate([np.linspace(elev_deg[0], -6.0, 36, endpoint=False), np.linspace(-6.0, 3.0, 72, endpoint=False),
                                      np.linspace(3.0, elev_deg[1], 20)]))
    az = np.deg2rad(np.linspace(az_deg[0], az_deg[1], n_az))
    E, A = np.meshgrid(elev, az)
    dirs = np.column_stack([np.cos(E.ravel()) * np.sin(A.ravel()), np.cos(E.ravel()) * np.cos(A.ravel()), np.sin(E.ravel())])
    t = np.arange(1.0, length, 0.1)
    cz = radius - sensor_height
    out = []
    for start in range(0, len(dirs), 1500):
        D = dirs[start:start + 1500]
        P = t[None, :, None] * D[:, None, :]           # (rays, steps, 3): columns l, d, h
        l, d, h = P[..., 0], P[..., 1], P[..., 2]
        lc = yaw * d + curvature * d * d
        rel = l - lc
        bed = -sensor_height + np.where(np.abs(rel) < 0.35, -0.3, 0.0) + np.where(np.abs(np.abs(rel) - 0.76) < 0.04, 0.08, 0.0)
        hit = ((rel ** 2 + (h - cz) ** 2 >= radius ** 2) | (h <= bed)) & (d > 0.5)
        if obstacle is not None:
            od, ol, ow, oh = obstacle
            hit |= (np.abs(d - od) < 0.15) & (np.abs(rel - ol) < ow / 2) & (h > -sensor_height) & (h < -sensor_height + oh)
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        pts = P[np.arange(len(D)), first]
        out.append(pts[any_hit & (pts[:, 1] <= length)])
    pts = np.vstack(out).astype(np.float32)
    pts += rng.normal(0, 0.01, pts.shape).astype(np.float32)
    n_zero = int(len(pts) * zero_fraction / (1 - zero_fraction))
    pts = np.vstack([pts, np.zeros((n_zero, 3), dtype=np.float32)])
    rng.shuffle(pts)
    pts[:, 1] = -pts[:, 1]  # sensor frame: forward is -y
    return pts
