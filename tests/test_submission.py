"""Unit tests for src/submission.py -- the submission.zip inference
entrypoint's testable logic. No real data; uids here are arbitrary
strings, not real patient identifiers.
"""

import math
import pytest
import numpy as np

import submission


# --- to_logit / from_logit ----------------------------------------------------

def test_to_logit_from_logit_round_trip():
    for p in [0.01, 0.3, 0.5, 0.7, 0.99]:
        assert submission.from_logit(submission.to_logit(p)) == pytest.approx(p, abs=1e-6)


def test_to_logit_clips_zero_and_one_instead_of_raising():
    z0 = submission.to_logit(0.0)
    z1 = submission.to_logit(1.0)

    assert np.isfinite(z0)
    assert np.isfinite(z1)
    assert z0 < z1


# --- pool_logit_mean ------------------------------------------------------------

def test_pool_logit_mean_of_identical_arrays_returns_that_array():
    p = np.array([0.2, 0.6, 0.9])

    pooled = submission.pool_logit_mean([p, p, p])

    np.testing.assert_allclose(pooled, p, atol=1e-6)


def test_pool_logit_mean_differs_from_probability_space_mean():
    a = np.array([0.9])
    b = np.array([0.3])

    pooled = submission.pool_logit_mean([a, b])
    prob_space_mean = np.mean([a, b], axis=0)

    assert not np.allclose(pooled, prob_space_mean)


# --- combine_predictions -------------------------------------------------------

def _expected_main_blend(cnn_p, baseline_p, a, b, c):
    """Independent (non-to_logit/from_logit) reference computation, so
    the test doesn't just re-assert the implementation against itself."""
    cnn_logit = math.log(cnn_p / (1 - cnn_p))
    baseline_logit = math.log(baseline_p / (1 - baseline_p))
    z = a * cnn_logit + b * baseline_logit + c
    return 1.0 / (1.0 + math.exp(-z))


def _expected_fallback(cnn_p, a1, c1):
    cnn_logit = math.log(cnn_p / (1 - cnn_p))
    z = a1 * cnn_logit + c1
    return 1.0 / (1.0 + math.exp(-z))


def test_combine_predictions_blends_when_both_probabilities_present():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8, "b": 0.2}
    baseline_probs = {"a": 0.6, "b": 0.4}
    a, b, c, a1, c1 = 0.8558, 0.5296, -0.0905, 0.9784, 0.0508

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, a, b, c, a1, c1)

    assert result[0] == pytest.approx(_expected_main_blend(0.8, 0.6, a, b, c), abs=1e-9)
    assert result[1] == pytest.approx(_expected_main_blend(0.2, 0.4, a, b, c), abs=1e-9)


def test_combine_predictions_falls_back_to_cnn_only_calibration_when_baseline_missing():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8, "b": 0.2}
    baseline_probs = {"a": 0.6}  # "b" has no classical features (degenerate mask)
    a, b, c, a1, c1 = 0.8558, 0.5296, -0.0905, 0.9784, 0.0508

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, a, b, c, a1, c1)

    assert result[1] == pytest.approx(_expected_fallback(0.2, a1, c1), abs=1e-9)
    # fallback path must NOT equal the raw uncalibrated CNN probability
    assert result[1] != pytest.approx(0.2)


def test_combine_predictions_preserves_uids_order_regardless_of_dict_order():
    uids = ["z", "a", "m"]
    cnn_probs = {"a": 0.1, "m": 0.5, "z": 0.9}  # inserted in a different order than uids
    baseline_probs = {"a": 0.1, "m": 0.5, "z": 0.9}
    a, b, c, a1, c1 = 0.8558, 0.5296, -0.0905, 0.9784, 0.0508

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, a, b, c, a1, c1)

    expected = [_expected_main_blend(0.9, 0.9, a, b, c),
                _expected_main_blend(0.1, 0.1, a, b, c),
                _expected_main_blend(0.5, 0.5, a, b, c)]
    assert result == pytest.approx(expected, abs=1e-9)


def test_combine_predictions_raises_if_a_uid_is_missing_from_cnn_probs():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8}  # "b" missing -- every test uid must have a CNN prediction
    baseline_probs = {}
    a, b, c, a1, c1 = 0.8558, 0.5296, -0.0905, 0.9784, 0.0508

    with pytest.raises(KeyError):
        submission.combine_predictions(uids, cnn_probs, baseline_probs, a, b, c, a1, c1)
