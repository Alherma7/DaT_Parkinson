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
