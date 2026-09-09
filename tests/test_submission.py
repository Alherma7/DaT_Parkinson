"""Unit tests for src/submission.py -- the submission.zip inference
entrypoint's testable logic. No real data; uids here are arbitrary
strings, not real patient identifiers.
"""

import pytest

import submission


def test_combine_predictions_blends_when_both_probabilities_present():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8, "b": 0.2}
    baseline_probs = {"a": 0.6, "b": 0.4}

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=0.70)

    assert result[0] == pytest.approx(0.70 * 0.8 + 0.30 * 0.6)
    assert result[1] == pytest.approx(0.70 * 0.2 + 0.30 * 0.4)


def test_combine_predictions_falls_back_to_cnn_alone_when_baseline_missing():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8, "b": 0.2}
    baseline_probs = {"a": 0.6}  # "b" has no classical features (degenerate mask)

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=0.70)

    assert result[1] == pytest.approx(0.2)  # CNN alone, not blended with anything


def test_combine_predictions_preserves_uids_order_regardless_of_dict_order():
    uids = ["z", "a", "m"]
    cnn_probs = {"a": 0.1, "m": 0.5, "z": 0.9}  # inserted in a different order than uids
    baseline_probs = {"a": 0.1, "m": 0.5, "z": 0.9}

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=1.0)

    assert result == [0.9, 0.1, 0.5]


def test_combine_predictions_raises_if_a_uid_is_missing_from_cnn_probs():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8}  # "b" missing -- every test uid must have a CNN prediction
    baseline_probs = {}

    with pytest.raises(KeyError):
        submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=0.70)
