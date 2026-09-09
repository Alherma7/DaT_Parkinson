"""Unit tests for src/evaluate.py, incl. the metric formula itself."""

import math

import numpy as np
import pytest

import evaluate


# --- log_loss_score -----------------------------------------------------

def test_log_loss_score_matches_hand_computed_value():
    y_true = [0, 1]
    y_pred = [0.1, 0.9]
    expected = -(math.log(1 - 0.1) + math.log(0.9)) / 2
    assert evaluate.log_loss_score(y_true, y_pred) == pytest.approx(expected)


def test_log_loss_score_penalizes_confident_wrong_predictions_more():
    low_confidence_wrong = evaluate.log_loss_score([1], [0.5])
    high_confidence_wrong = evaluate.log_loss_score([1], [0.01])
    assert high_confidence_wrong > low_confidence_wrong


def test_log_loss_score_rewards_confident_correct_predictions():
    low_confidence_right = evaluate.log_loss_score([1], [0.5])
    high_confidence_right = evaluate.log_loss_score([1], [0.99])
    assert high_confidence_right < low_confidence_right


# --- auroc_score ---------------------------------------------------------

def test_auroc_score_perfect_separation_is_one():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    assert evaluate.auroc_score(y_true, y_pred) == pytest.approx(1.0)


def test_auroc_score_all_tied_predictions_is_half():
    y_true = [0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.5]
    assert evaluate.auroc_score(y_true, y_pred) == pytest.approx(0.5)


# --- expected_calibration_error ------------------------------------------

def test_ece_is_zero_when_confidence_matches_accuracy():
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0.5] * 10  # bin confidence 0.5, bin accuracy 5/10 = 0.5
    assert evaluate.expected_calibration_error(y_true, y_pred, n_bins=10) == pytest.approx(0.0)


def test_ece_detects_miscalibration():
    y_true = [0] * 10  # all negative
    y_pred = [0.9] * 10  # confidently (and wrongly) predicts positive
    # bin confidence 0.9, bin accuracy 0.0 -> ece = 1.0 * |0.0 - 0.9|
    assert evaluate.expected_calibration_error(y_true, y_pred, n_bins=10) == pytest.approx(0.9)


# --- combined_score --------------------------------------------------------

def test_combined_score_returns_log_loss_auroc_and_ece():
    y_true = [0, 1, 0, 1]
    y_pred = [0.2, 0.8, 0.3, 0.7]
    result = evaluate.combined_score(y_true, y_pred)
    assert set(result.keys()) == {"log_loss", "auroc", "ece"}


# --- make_folds ------------------------------------------------------------

def test_make_folds_partitions_every_row_exactly_once():
    rng = np.random.RandomState(0)
    n = 200
    target = rng.binomial(1, 0.55, size=n)
    family = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1])

    folds = evaluate.make_folds(target, family, n_splits=5, random_state=42)

    assert len(folds) == 5
    all_test_idx = np.concatenate([test_idx for _, test_idx in folds])
    assert len(all_test_idx) == n
    assert sorted(all_test_idx.tolist()) == list(range(n))


def test_make_folds_preserves_target_ratio_within_tolerance():
    rng = np.random.RandomState(0)
    n = 500
    target = rng.binomial(1, 0.55, size=n)
    family = rng.choice(["A", "B", "C", "D"], size=n)

    folds = evaluate.make_folds(target, family, n_splits=5, random_state=42)
    overall_rate = np.mean(target)

    for _, test_idx in folds:
        fold_rate = np.mean(np.asarray(target)[test_idx])
        assert abs(fold_rate - overall_rate) < 0.1


def test_make_folds_handles_singleton_rare_families_without_crashing():
    rng = np.random.RandomState(0)
    n = 100
    target = rng.binomial(1, 0.5, size=n)
    family = np.array(["common"] * 97 + ["rare1", "rare2", "rare3"])

    folds = evaluate.make_folds(target, family, n_splits=5, random_state=42)

    assert len(folds) == 5


# --- family_oversample_weights ------------------------------------------

def test_family_oversample_weights_boosts_only_named_families():
    family = np.array(["2.46", "2.30", "3.895", "2.30"])

    weights = evaluate.family_oversample_weights(family, boosted_families=["2.46", "3.895"], boost_factor=2.0)

    np.testing.assert_array_equal(weights, [2.0, 1.0, 2.0, 1.0])


def test_family_oversample_weights_defaults_boost_factor_to_one_when_unboosted():
    family = ["a", "b", "c"]

    weights = evaluate.family_oversample_weights(family, boosted_families=[], boost_factor=5.0)

    np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0])


def test_family_oversample_weights_returns_float_array_matching_input_length():
    family = ["a"] * 7

    weights = evaluate.family_oversample_weights(family, boosted_families=["a"], boost_factor=3.0)

    assert weights.dtype == np.float64
    assert len(weights) == 7


# --- paired_bootstrap_ci -----------------------------------------------

def test_paired_bootstrap_ci_is_zero_width_when_predictions_are_identical():
    rng = np.random.RandomState(0)
    y_true = rng.binomial(1, 0.5, size=50)
    probs = rng.uniform(0.1, 0.9, size=50)

    ci_low, ci_high = evaluate.paired_bootstrap_ci(y_true, probs, probs, seed=42)

    assert ci_low == pytest.approx(0.0)
    assert ci_high == pytest.approx(0.0)


def test_paired_bootstrap_ci_favors_the_clearly_better_predictions():
    y_true = np.array([0, 1] * 25)
    probs_a = np.array([0.05, 0.95] * 25)  # confidently correct
    probs_b = np.array([0.5, 0.5] * 25)  # uninformative

    ci_low, ci_high = evaluate.paired_bootstrap_ci(y_true, probs_a, probs_b, seed=42)

    # delta = log_loss(a) - log_loss(b); a is better so delta is negative throughout
    assert ci_high < 0


def test_paired_bootstrap_ci_is_reproducible_with_same_seed():
    rng = np.random.RandomState(1)
    y_true = rng.binomial(1, 0.5, size=50)
    probs_a = rng.uniform(0.1, 0.9, size=50)
    probs_b = rng.uniform(0.1, 0.9, size=50)

    result_1 = evaluate.paired_bootstrap_ci(y_true, probs_a, probs_b, seed=7)
    result_2 = evaluate.paired_bootstrap_ci(y_true, probs_a, probs_b, seed=7)

    assert result_1 == result_2


def test_paired_bootstrap_ci_low_does_not_exceed_high():
    rng = np.random.RandomState(2)
    y_true = rng.binomial(1, 0.5, size=50)
    probs_a = rng.uniform(0.1, 0.9, size=50)
    probs_b = rng.uniform(0.1, 0.9, size=50)

    ci_low, ci_high = evaluate.paired_bootstrap_ci(y_true, probs_a, probs_b, seed=42)

    assert ci_low <= ci_high
