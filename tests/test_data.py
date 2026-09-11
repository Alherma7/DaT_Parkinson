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


def test_normalize_intensity_zero_means_the_foreground():
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[:5] = 100.0   # foreground half
    volume[5:] = 0.0     # background half

    normalized = data.normalize_intensity(volume, background_percentile=30,
                                           background_max_fraction=0.05)

    # Foreground (>threshold) voxels should be ~zero-mean.
    assert normalized[:5].mean() == pytest.approx(0.0, abs=1e-4)


def test_normalize_intensity_clips_negative_values_first():
    """A handful of volumes are stored as int16, not uint16, and can carry
    small negative artifacts (EDA section 5, `features.py`'s same
    np.clip guard)."""
    volume = np.array([[[-5.0, 10.0], [20.0, 30.0]]], dtype=np.float32)

    normalized = data.normalize_intensity(volume)

    assert np.isfinite(normalized).all()


def test_normalize_intensity_handles_all_zero_volume():
    volume = np.zeros((5, 5, 5), dtype=np.float32)

    normalized = data.normalize_intensity(volume)

    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, volume)


def test_load_volume_returns_target_shape_with_channel_dim(tmp_path, monkeypatch):
    import nibabel as nib

    volume = np.zeros((60, 60, 40), dtype=np.float32)
    volume[25:35, 25:35, 15:25] = 500.0
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "synthetic_uid.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    out = data.load_volume("synthetic_uid")

    assert out.shape == (1, *data.config.TARGET_SHAPE)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_load_volume_warns_on_degenerate_all_zero_volume(tmp_path, monkeypatch):
    import nibabel as nib

    volume = np.zeros((60, 60, 40), dtype=np.float32)  # no signal anywhere
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "degenerate_uid.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    with pytest.warns(UserWarning, match="degenerate/near-empty volume"):
        out = data.load_volume("degenerate_uid")

    assert out.shape == (1, *data.config.TARGET_SHAPE)


# --- denoise_volume (rung-4 experiment 5) ------------------------------

def _add_gaussian_noise(volume, sigma, seed):
    rng = np.random.RandomState(seed)
    return volume + rng.normal(scale=sigma, size=volume.shape).astype(np.float32)


def test_denoise_volume_preserves_shape():
    clean = np.zeros((12, 12, 12), dtype=np.float32)
    clean[4:8, 4:8, 4:8] = 10.0
    noisy = _add_gaussian_noise(clean, sigma=1.5, seed=0)

    denoised = data.denoise_volume(noisy)

    assert denoised.shape == noisy.shape


def test_denoise_volume_reduces_noise_relative_to_the_clean_signal():
    clean = np.zeros((14, 14, 14), dtype=np.float32)
    clean[5:9, 5:9, 5:9] = 10.0
    noisy = _add_gaussian_noise(clean, sigma=2.0, seed=1)

    denoised = data.denoise_volume(noisy)

    noisy_mse = float(np.mean((noisy - clean) ** 2))
    denoised_mse = float(np.mean((denoised - clean) ** 2))
    assert denoised_mse < noisy_mse


def test_load_volume_applies_denoising_only_when_flag_is_enabled(tmp_path, monkeypatch):
    import nibabel as nib

    volume = np.zeros((60, 60, 40), dtype=np.float32)
    rng = np.random.RandomState(2)
    volume += rng.normal(scale=5.0, size=volume.shape).astype(np.float32)
    volume[25:35, 25:35, 15:25] += 500.0
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "noisy_uid.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    calls = []
    real_denoise = data.denoise_volume

    def spying_denoise(volume, *args, **kwargs):
        calls.append(1)
        return real_denoise(volume, *args, **kwargs)

    monkeypatch.setattr(data, "denoise_volume", spying_denoise)

    monkeypatch.setattr(data.config, "USE_NLM_DENOISING", False)
    data.load_volume("noisy_uid")
    assert calls == []

    monkeypatch.setattr(data.config, "USE_NLM_DENOISING", True)
    data.load_volume("noisy_uid")
    assert calls == [1]


def test_load_slab_returns_2d_shape_with_channel_dim(tmp_path, monkeypatch):
    """Uniform noise everywhere (not a small localized block, unlike
    load_volume's equivalent test) so this test's outcome doesn't depend
    on the exact SLAB_TARGET_SPACING/CROP_CENTER_MM arithmetic landing a
    specific signal region inside the much thinner (12mm-thick) slab crop
    -- shape/dtype/finiteness is what's under test here, not signal
    placement (that's covered by data.py's crop_or_pad tests directly).
    """
    import nibabel as nib

    rng = np.random.RandomState(0)
    volume = rng.uniform(400.0, 600.0, size=(60, 60, 40)).astype(np.float32)
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "synthetic_uid.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    out = data.load_slab("synthetic_uid")

    assert out.shape == (1, data.config.SLAB_TARGET_SHAPE[0], data.config.SLAB_TARGET_SHAPE[1])
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_load_slab_warns_on_degenerate_all_zero_volume(tmp_path, monkeypatch):
    import nibabel as nib

    volume = np.zeros((60, 60, 40), dtype=np.float32)  # no signal anywhere
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "degenerate_uid.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    with pytest.warns(UserWarning, match="degenerate/near-empty volume"):
        out = data.load_slab("degenerate_uid")

    assert out.shape == (1, data.config.SLAB_TARGET_SHAPE[0], data.config.SLAB_TARGET_SHAPE[1])


def test_load_slab_degenerate_warning_never_names_the_uid(tmp_path, monkeypatch):
    """Same disqualification-risk guard as load_volume's -- this function
    would run inside submission_src/main.py if roadmap item 6 is adopted.
    """
    import nibabel as nib

    volume = np.zeros((60, 60, 40), dtype=np.float32)
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    secret_uid = "some_real_test_patient_uid"
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / f"{secret_uid}.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    with pytest.warns(UserWarning) as recorded:
        data.load_slab(secret_uid)

    assert all(secret_uid not in str(w.message) for w in recorded)


def test_load_volume_degenerate_warning_never_names_the_uid(tmp_path, monkeypatch):
    """Regression guard: submission_src/main.py runs this exact function
    against real test volumes, and the competition platform scans
    submission logs for per-sample test-set information and disqualifies
    on it (confirmed 2026-09-09: a real full-submission run got two log
    lines auto-filtered as forbidden content because this warning used
    to embed the uid). The warning must stay uid-free in every context.
    """
    import nibabel as nib

    volume = np.zeros((60, 60, 40), dtype=np.float32)
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    secret_uid = "some_real_test_patient_uid"
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / f"{secret_uid}.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    with pytest.warns(UserWarning) as recorded:
        data.load_volume(secret_uid)

    assert all(secret_uid not in str(w.message) for w in recorded)
