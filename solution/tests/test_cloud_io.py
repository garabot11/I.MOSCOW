import numpy as np
import pytest

from metro_obstacle_detector.cloud_io import CloudFormatError, cloud_to_xyz, validity_mask
from synthetic import make_pointcloud2


def _xyz(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(0, 10, (n, 3)).astype(np.float32)
    xyz[::7] = 0.0  # placeholder returns
    return xyz


def test_mixed_layout_roundtrip():
    xyz = _xyz()
    out, inten, info = cloud_to_xyz(make_pointcloud2(xyz))
    assert out.shape == (1000, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out, xyz)
    assert inten is not None and np.all(inten == 10.0)
    assert info.point_step == 26 and info.frame_id == "hesai_lidar"
    assert info.stamp_ns == 946687297 * 1_000_000_000 + 199933052


def test_bigendian_layout():
    xyz = _xyz(200, 1)
    out, _, _ = cloud_to_xyz(make_pointcloud2(xyz, bigendian=True))
    np.testing.assert_array_equal(out, xyz)


def test_organised_cloud_with_row_padding():
    xyz = _xyz(400, 2)
    out, _, info = cloud_to_xyz(make_pointcloud2(xyz, height=4, row_pad=8))
    assert info.height == 4 and info.row_step == 100 * 26 + 8
    np.testing.assert_array_equal(out, xyz)


def test_xyz_only_fields():
    xyz = _xyz(50, 3)
    out, inten, _ = cloud_to_xyz(make_pointcloud2(xyz, extra_fields=False))
    np.testing.assert_array_equal(out, xyz)
    assert inten is None


def test_missing_required_field_raises():
    with pytest.raises(CloudFormatError):
        cloud_to_xyz(make_pointcloud2(_xyz(10), drop=("z",)))


def test_short_buffer_raises():
    msg = make_pointcloud2(_xyz(10))
    msg.data = bytes(msg.data[:-30])
    with pytest.raises(CloudFormatError):
        cloud_to_xyz(msg)


def test_validity_mask_separates_zero_and_nonfinite():
    xyz = np.array([[1, 2, 3], [0, 0, 0], [np.nan, 1, 1], [0, 0, 1]], dtype=np.float32)
    m = validity_mask(xyz)
    assert m["valid"].tolist() == [True, False, False, True]
    assert m["zero"].tolist() == [False, True, False, False]
