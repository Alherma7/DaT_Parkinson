"""Unit tests for src/augment.py, using small synthetic 4D arrays --
never real patient volumes, per the project's AI-assistant data rule.
Array shape throughout is (channel, L-R, A-P, S-I), matching
data.load_volume's (1, *config.TARGET_SHAPE) output.
"""

import numpy as np
import pytest

import augment


class _FixedRNG:
    """Stub with the two numpy.random.Generator methods augment.py
    calls, each pinned to a fixed return value -- avoids flaky
    seeded-value assertions in tests that need an exact, known draw.
    """

    def __init__(self, random_value=0.0, uniform_value=0.0):
        self._random_value = random_value
        self._uniform_value = uniform_value

    def random(self):
        return self._random_value

    def uniform(self, low, high):
        return self._uniform_value


def _asymmetric_volume():
    volume = np.zeros((1, 10, 6, 6), dtype=np.float32)
    volume[0, 2, 2, 2] = 5.0  # left-heavy (low L-R index)
    volume[0, 7, 3, 3] = 1.0  # right-side, different intensity
    return volume


# --- random_flip ----------------------------------------------------------

def test_random_flip_flips_the_l_r_axis_when_forced():
    volume = _asymmetric_volume()
    rng = _FixedRNG(random_value=0.0)  # random() < p=0.5 -> flip happens

    flipped = augment.random_flip(volume, rng, axis=1, p=0.5)

    np.testing.assert_array_equal(flipped, np.flip(volume, axis=1))


def test_random_flip_leaves_volume_unchanged_when_probability_not_met():
    volume = _asymmetric_volume()
    rng = _FixedRNG(random_value=0.99)  # random() >= p=0.5 -> no flip

    result = augment.random_flip(volume, rng, axis=1, p=0.5)

    np.testing.assert_array_equal(result, volume)


# --- random_rotation --------------------------------------------------------

def test_random_rotation_preserves_shape():
    volume = _asymmetric_volume()
    rng = np.random.default_rng(0)

    rotated = augment.random_rotation(volume, rng, max_degrees=10.0)

    assert rotated.shape == volume.shape


def test_random_rotation_is_identity_at_zero_degrees():
    volume = _asymmetric_volume()
    rng = _FixedRNG(uniform_value=0.0)  # uniform(-max, max) -> 0 degrees

    rotated = augment.random_rotation(volume, rng, max_degrees=10.0)

    np.testing.assert_allclose(rotated, volume, atol=1e-5)


def test_random_rotation_never_mixes_the_l_r_axis():
    """The L-R axis (1) must stay the rotation axis, not part of the
    rotated plane -- rotating axes=(2, 3) only (A-P, S-I) keeps a
    pathology's left-right laterality meaningful under augmentation."""
    volume = np.zeros((1, 4, 8, 8), dtype=np.float32)
    volume[0, 1, :, :] = 3.0  # a full "slice" at L-R index 1
    rng = np.random.default_rng(0)

    rotated = augment.random_rotation(volume, rng, max_degrees=10.0)

    # Content stays within L-R index 1 (nothing bled into other L-R slices)
    assert rotated[0, 0, :, :].sum() == pytest.approx(0.0, abs=1e-5)
    assert rotated[0, 2, :, :].sum() == pytest.approx(0.0, abs=1e-5)
    assert rotated[0, 3, :, :].sum() == pytest.approx(0.0, abs=1e-5)


# --- random_brightness_jitter ------------------------------------------------

def test_random_brightness_jitter_scales_by_the_drawn_factor():
    volume = _asymmetric_volume()
    rng = _FixedRNG(uniform_value=1.2)

    jittered = augment.random_brightness_jitter(volume, rng, max_jitter=0.3)

    np.testing.assert_allclose(jittered, volume * 1.2)


def test_random_brightness_jitter_factor_of_one_is_identity():
    volume = _asymmetric_volume()
    rng = _FixedRNG(uniform_value=1.0)

    jittered = augment.random_brightness_jitter(volume, rng, max_jitter=0.3)

    np.testing.assert_array_equal(jittered, volume)


# --- augment_volume (composition) -------------------------------------------

def test_augment_volume_returns_float32_same_shape():
    volume = _asymmetric_volume()
    rng = np.random.default_rng(1)

    result = augment.augment_volume(volume, rng)

    assert result.shape == volume.shape
    assert result.dtype == np.float32


def test_augment_volume_is_reproducible_with_the_same_seed():
    volume = _asymmetric_volume()

    result_1 = augment.augment_volume(volume, np.random.default_rng(42))
    result_2 = augment.augment_volume(volume, np.random.default_rng(42))

    np.testing.assert_array_equal(result_1, result_2)


def test_augment_volume_differs_with_a_different_seed():
    volume = _asymmetric_volume()

    result_1 = augment.augment_volume(volume, np.random.default_rng(1))
    result_2 = augment.augment_volume(volume, np.random.default_rng(2))

    assert not np.array_equal(result_1, result_2)
