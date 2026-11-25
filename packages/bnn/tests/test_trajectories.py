import shutil

import numpy as np
from bnn.store import trajectories


def test_numeric_zarr_bundle_preserves_shapes_dtypes_and_stable_identity(tmp_path):
    arrays = {
        "positions": np.arange(90, dtype=np.float64).reshape(3, 10, 3),
        "observables": np.arange(20, dtype=np.float32).reshape(10, 2),
        "scalar": np.array(1.5),
        "target_ids": np.array(["train-a", "heldout-b"]),
        "empty": np.zeros((0, 5), dtype=np.int16),
    }
    first = trajectories.save(tmp_path / "a.zarr", **arrays)
    second = trajectories.save(tmp_path / "b.zarr", **arrays)
    restored = trajectories.load(first)
    for name, value in arrays.items():
        np.testing.assert_array_equal(restored[name], value)
        assert restored[name].dtype == value.dtype
    assert trajectories.checksum(first) == trajectories.checksum(second)
    shutil.move(first, tmp_path / "moved.zarr")
    assert trajectories.checksum(tmp_path / "moved.zarr") == trajectories.checksum(
        second
    )
    assert list(second.rglob("zarr.json"))
