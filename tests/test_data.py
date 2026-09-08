"""Unit tests for src/data.py, using small synthetic 3D arrays -- never
real patient volumes, per the project's AI-assistant data rule.
"""

import numpy as np
import pytest

import data


def _oblique_affine(spacing=2.0, tilt_deg=15.0, offset=(-10.0, -10.0, -10.0)):
    """A synthetic oblique affine (rotation around z), matching the shape
    of the real confound EDA found (`README.md`: 40% of volumes oblique,
    up to 40.3 deg) -- `as_closest_canonical()` would not correct this.
    """
    theta = np.radians(tilt_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    affine = np.eye(4)
    affine[:3, :3] = rot * spacing
    affine[:3, 3] = offset
    return affine


def test_resample_to_spacing_changes_voxel_count_for_oblique_affine():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[8:12, 8:12, 8:12] = 500.0
    affine = _oblique_affine(spacing=2.0, tilt_deg=15.0)

    resampled, new_affine = data.resample_to_spacing(volume, affine, (1.0, 1.0, 1.0))

    # 2mm voxels resampled to 1mm voxels: roughly double the extent per axis.
    assert resampled.shape[0] > volume.shape[0]
    assert new_affine.shape == (4, 4)


def test_resample_to_spacing_preserves_signal_for_identity_affine():
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[4:6, 4:6, 4:6] = 1000.0
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0

    resampled, _ = data.resample_to_spacing(volume, affine, (2.0, 2.0, 2.0))

    assert resampled.shape == volume.shape
    assert resampled.max() == pytest.approx(1000.0, rel=0.05)
