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


def test_crop_or_pad_extracts_target_shape_centered_at_offset():
    volume = np.zeros((40, 40, 40), dtype=np.float32)
    volume[18:22, 18:22, 18:22] = 777.0  # centered at voxel (19.5,19.5,19.5)

    cropped = data.crop_or_pad(volume, spacing=(1.0, 1.0, 1.0),
                                center_mm=(0.0, 0.0, 0.0), target_shape=(10, 10, 10))

    assert cropped.shape == (10, 10, 10)
    assert cropped.max() == pytest.approx(777.0)


def test_crop_or_pad_respects_center_mm_offset():
    volume = np.zeros((40, 40, 40), dtype=np.float32)
    volume[28:32, 18:22, 18:22] = 777.0  # offset +10 voxels on axis 0

    cropped = data.crop_or_pad(volume, spacing=(1.0, 1.0, 1.0),
                                center_mm=(10.0, 0.0, 0.0), target_shape=(10, 10, 10))

    assert cropped.max() == pytest.approx(777.0)
    # Signal should now land near the center of the *cropped* volume.
    assert cropped[4:6, 4:6, 4:6].max() == pytest.approx(777.0)


def test_crop_or_pad_zero_pads_when_target_exceeds_volume():
    """Tight-FOV case: `config.py` documents volumes where TARGET_SHAPE's
    z-extent fits MIN_FOV_MM by only 3mm -- crop_or_pad must pad, not
    crash or silently clip, when the source volume is smaller than the
    target shape on an axis.
    """
    volume = np.full((56, 30, 20), 10.0, dtype=np.float32)

    padded = data.crop_or_pad(volume, spacing=(1.0, 1.0, 1.0),
                               center_mm=(0.0, 0.0, 0.0), target_shape=(56, 30, 44))

    assert padded.shape == (56, 30, 44)
    assert (padded[:, :, 0] == 0.0).all()  # padded edge
    assert (padded[:, :, 22] == 10.0).all()  # original data preserved near center
