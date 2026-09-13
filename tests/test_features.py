"""Unit tests for src/features.py, using small synthetic 3D arrays --
never real patient volumes, per the project's AI-assistant data rule.
"""

import numpy as np
import pytest

import features


def _blank_volume(shape=(30, 30, 20)):
    return np.zeros(shape, dtype=float)


# --- striatum_mask ---------------------------------------------------------

def test_striatum_mask_finds_two_symmetric_bright_blobs():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)  # mm -> voxel volume 8 mm^3 = 0.008 mL
    # Left blob (low x index = anatomical Left in RAS) and right blob,
    # both well inside the central region, same size (3x4x4 = 48 voxels each).
    volume[5:8, 13:17, 8:12] = 1000.0
    volume[22:25, 13:17, 8:12] = 1000.0
    voxel_ml = np.prod(spacing) / 1000.0
    target_ml = 96 * voxel_ml  # exactly both blobs, nothing else

    mask = features.striatum_mask(volume, spacing, target_ml=target_ml)

    assert mask is not None
    assert mask.sum() == 96
    assert mask[5:8, 13:17, 8:12].all()
    assert mask[22:25, 13:17, 8:12].all()


def test_striatum_mask_excludes_edge_artifact_via_central_margin():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)
    # Two real (central) blobs, plus a brighter, larger artifact right at
    # the edge of the volume -- exactly the failure mode found in EDA
    # section 6a (a peripheral bright region winning over the striatum).
    volume[5:8, 13:17, 8:12] = 1000.0
    volume[22:25, 13:17, 8:12] = 1000.0
    volume[0:2, 0:4, 0:4] = 5000.0  # edge artifact, brighter and bigger
    voxel_ml = np.prod(spacing) / 1000.0
    target_ml = 96 * voxel_ml

    mask = features.striatum_mask(volume, spacing, target_ml=target_ml,
                                   central_margin=0.15)

    assert mask is not None
    assert not mask[0:2, 0:4, 0:4].any()


def test_striatum_mask_returns_none_for_empty_volume():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)

    mask = features.striatum_mask(volume, spacing, target_ml=0.5)

    assert mask is None


# --- striatum_center_mm ------------------------------------------------------

def test_striatum_center_mm_symmetric_blobs_land_at_geometric_center():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)
    volume[5:8, 13:17, 8:12] = 1000.0
    volume[22:25, 13:17, 8:12] = 1000.0
    voxel_ml = np.prod(spacing) / 1000.0
    target_ml = 96 * voxel_ml

    center_mm = features.striatum_center_mm(volume, spacing, target_ml=target_ml)

    assert center_mm is not None
    np.testing.assert_allclose(center_mm, (0.0, 0.0, 0.0), atol=1e-8)


def test_striatum_center_mm_matches_manual_centroid_for_an_offset_blob():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)
    volume[20:24, 13:17, 8:12] = 1000.0  # centroid at voxel (21.5, 14.5, 9.5)

    center_mm = features.striatum_center_mm(volume, spacing, target_ml=0.5)

    # geometric center is voxel (14.5, 14.5, 9.5) -- offset is (7, 0, 0) voxels
    assert center_mm is not None
    np.testing.assert_allclose(center_mm, (14.0, 0.0, 0.0), atol=1e-8)


def test_striatum_center_mm_returns_none_for_empty_volume():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)

    center_mm = features.striatum_center_mm(volume, spacing, target_ml=0.5)

    assert center_mm is None


def test_striatum_center_mm_is_the_inverse_of_crop_or_pad_center_mm():
    """The property notebooks/25's per-subject-centering experiment (and
    data.load_volume's "auto" mode) depends on: feeding striatum_center_mm's
    output back into crop_or_pad's center_mm recenters the crop on the same
    striatum, since the two use algebraically inverse conventions (6th Opus
    review, project memory)."""
    import data

    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)
    volume[20:24, 13:17, 8:12] = 1000.0

    center_mm = features.striatum_center_mm(volume, spacing, target_ml=0.5)
    cropped = data.crop_or_pad(volume, spacing, center_mm, target_shape=(10, 10, 10))

    assert cropped.sum() == pytest.approx(volume.sum())  # nothing clipped
    idx = np.argwhere(cropped > 0)
    crop_center = (np.asarray(cropped.shape) - 1) / 2.0
    centroid = idx.mean(axis=0)
    assert np.max(np.abs(centroid - crop_center)) <= 1.0  # within 1 voxel (rounding)


# --- signed_asymmetry --------------------------------------------------------

def test_signed_asymmetry_positive_when_left_brighter():
    volume = _blank_volume()
    mask = np.zeros_like(volume, dtype=bool)
    # Left (low x index) brighter than right.
    volume[5:8, 13:17, 8:12] = 100.0
    volume[22:25, 13:17, 8:12] = 50.0
    mask[5:8, 13:17, 8:12] = True
    mask[22:25, 13:17, 8:12] = True
    spacing = (2.0, 2.0, 2.0)

    signed = features.signed_asymmetry(volume, mask, spacing)

    assert signed > 0


def test_signed_asymmetry_near_zero_when_symmetric():
    volume = _blank_volume()
    mask = np.zeros_like(volume, dtype=bool)
    volume[5:8, 13:17, 8:12] = 100.0
    volume[22:25, 13:17, 8:12] = 100.0
    mask[5:8, 13:17, 8:12] = True
    mask[22:25, 13:17, 8:12] = True
    spacing = (2.0, 2.0, 2.0)

    signed = features.signed_asymmetry(volume, mask, spacing)

    assert signed == pytest.approx(0.0, abs=1e-9)


def test_signed_asymmetry_matches_hand_computed_value():
    volume = _blank_volume(shape=(10, 5, 5))
    mask = np.zeros_like(volume, dtype=bool)
    volume[2, 2, 2] = 30.0  # left of midline (x=4.5 for shape 10)
    volume[7, 2, 2] = 10.0  # right of midline
    mask[2, 2, 2] = True
    mask[7, 2, 2] = True
    spacing = (1.0, 1.0, 1.0)

    signed = features.signed_asymmetry(volume, mask, spacing)

    expected = (30.0 - 10.0) / (30.0 + 10.0)
    assert signed == pytest.approx(expected)


# --- striatal_ratio ----------------------------------------------------------

def test_striatal_ratio_matches_hand_computed_value():
    volume = _blank_volume(shape=(10, 10, 10))
    mask = np.zeros_like(volume, dtype=bool)
    volume[1:3, 1:3, 1:3] = 40.0  # masked region, mean 40
    mask[1:3, 1:3, 1:3] = True
    volume[5:7, 5:7, 5:7] = 10.0  # other positive-intensity voxels, mean 10

    ratio = features.striatal_ratio(volume, mask)

    # mean(masked) / mean(all positive voxels) = 40 / ((8*40 + 8*10) / 16)
    expected = 40.0 / ((8 * 40.0 + 8 * 10.0) / 16)
    assert ratio == pytest.approx(expected)


# --- fit_combat / apply_combat ----------------------------------------------

def _batch_shifted_features(rng, n_per_batch=10, shift=5.0):
    """Two batches, same within-batch spread, feature 0 offset by `shift`
    between batches, feature 1 batch-invariant (nothing to harmonize)."""
    a = rng.normal(loc=0.0, scale=0.2, size=(n_per_batch, 2))
    b = rng.normal(loc=0.0, scale=0.2, size=(n_per_batch, 2))
    a[:, 0] += 0.0
    b[:, 0] += shift
    X = np.vstack([a, b])
    batch = np.array(["A"] * n_per_batch + ["B"] * n_per_batch)
    return X, batch


def test_apply_combat_removes_known_additive_batch_shift():
    rng = np.random.RandomState(0)
    X, batch = _batch_shifted_features(rng, n_per_batch=15, shift=5.0)

    params = features.fit_combat(X, batch, min_batch_size=3)
    adjusted = features.apply_combat(X, batch, params)

    mean_a = adjusted[batch == "A", 0].mean()
    mean_b = adjusted[batch == "B", 0].mean()
    # Started ~5.0 apart; harmonization should collapse almost all of it.
    assert abs(mean_a - mean_b) < 0.5


def test_apply_combat_is_noop_with_a_single_batch():
    rng = np.random.RandomState(0)
    X = rng.normal(loc=3.0, scale=1.0, size=(20, 2))
    batch = np.array(["only"] * 20)

    params = features.fit_combat(X, batch, min_batch_size=3)
    adjusted = features.apply_combat(X, batch, params)

    assert params.no_op is True
    np.testing.assert_array_equal(adjusted, X)


def test_apply_combat_handles_singleton_rare_batch_without_crashing():
    rng = np.random.RandomState(0)
    X, batch = _batch_shifted_features(rng, n_per_batch=15, shift=5.0)
    # A third batch with a single member -- too small to estimate its own
    # variance, must collapse instead of crashing.
    X = np.vstack([X, [[100.0, 100.0]]])
    batch = np.concatenate([batch, ["C"]])

    params = features.fit_combat(X, batch, min_batch_size=3)
    adjusted = features.apply_combat(X, batch, params)

    assert np.isfinite(adjusted).all()


def test_apply_combat_maps_unseen_batch_to_a_finite_result():
    rng = np.random.RandomState(0)
    X, batch = _batch_shifted_features(rng, n_per_batch=15, shift=5.0)
    params = features.fit_combat(X, batch, min_batch_size=3)

    new_X = np.array([[1.0, 1.0], [2.0, 2.0]])
    new_batch = np.array(["never_seen", "never_seen"])
    adjusted = features.apply_combat(new_X, new_batch, params)

    assert np.isfinite(adjusted).all()


# --- inplane_family ---------------------------------------------------------

def test_inplane_family_buckets_known_spacings():
    assert features.inplane_family(2.46) == "2.46"
    assert features.inplane_family(3.895) == "3.895"
    assert features.inplane_family(1.47) == "~1.47"


def test_inplane_family_falls_back_for_unbucketed_spacing():
    assert features.inplane_family(3.123) == "other(3.123)"


def test_inplane_family_boundary_lower_inclusive_upper_exclusive():
    assert features.inplane_family(2.45) == "2.46"
    assert features.inplane_family(2.47) == "other(2.470)"


def test_inplane_family_boundary_between_adjacent_ranges():
    assert features.inplane_family(1.50) == "~1.5-1.8"


# --- extract_baseline_features -----------------------------------------------

def test_extract_baseline_features_returns_all_three_keys():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)
    volume[5:8, 13:17, 8:12] = 1000.0
    volume[22:25, 13:17, 8:12] = 500.0  # asymmetric on purpose

    result = features.extract_baseline_features(volume, spacing, target_ml=0.5)

    assert result is not None
    assert set(result.keys()) == {"abs_asym", "striatal_ratio", "inplane_family"}
    assert result["abs_asym"] > 0  # asymmetric blobs -> nonzero
    assert result["inplane_family"] == "2.00"


def test_extract_baseline_features_returns_none_for_empty_volume():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)

    result = features.extract_baseline_features(volume, spacing, target_ml=0.5)

    assert result is None
