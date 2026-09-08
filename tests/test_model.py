"""Unit tests for src/model.py."""

import numpy as np

import features
import model


def test_build_classical_baseline_fits_and_predicts_valid_probabilities():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 2))
    y = rng.binomial(1, 0.5, size=40)

    pipeline = model.build_classical_baseline()
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]

    assert proba.shape == (40,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_build_classical_baseline_returns_a_new_instance_each_call():
    first = model.build_classical_baseline()
    second = model.build_classical_baseline()

    assert first is not second


# --- build_combat_baseline ---------------------------------------------------

def _batch_shifted_Xy(rng, n_per_batch=20, shift=4.0):
    """Two batches with a strong additive shift on feature 0 that is also
    correlated with the label, so an unharmonized model partly latches
    onto batch identity instead of the real signal -- ComBat should still
    fit cleanly and produce valid probabilities either way."""
    a = rng.normal(loc=0.0, scale=0.5, size=(n_per_batch, 2))
    b = rng.normal(loc=0.0, scale=0.5, size=(n_per_batch, 2))
    b[:, 0] += shift
    X = np.vstack([a, b])
    batch = np.array(["A"] * n_per_batch + ["B"] * n_per_batch)
    y = rng.binomial(1, 0.5, size=2 * n_per_batch)
    return X, y, batch


def test_build_combat_baseline_fits_and_predicts_valid_probabilities():
    rng = np.random.RandomState(0)
    X, y, batch = _batch_shifted_Xy(rng)

    pipeline = model.build_combat_baseline()
    pipeline.fit(X, y, batch)
    proba = pipeline.predict_proba(X, batch)[:, 1]

    assert proba.shape == (40,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_build_combat_baseline_fits_combat_on_controls_only():
    """Regression guard for the leakage-avoidance rule (RESOURCES.md /
    notebooks/04_combat_harmonization.ipynb): harmonization parameters must
    come from the training fold's label=0 rows, not the full training set.
    """
    rng = np.random.RandomState(1)
    X, y, batch = _batch_shifted_Xy(rng)

    pipeline = model.build_combat_baseline()
    pipeline.fit(X, y, batch)

    controls = y == 0
    expected_params = features.fit_combat(X[controls], batch[controls])
    np.testing.assert_array_equal(
        pipeline._combat_params.grand_mean, expected_params.grand_mean)


def test_build_combat_baseline_predict_requires_batch_labels_for_new_rows():
    rng = np.random.RandomState(2)
    X, y, batch = _batch_shifted_Xy(rng)
    pipeline = model.build_combat_baseline()
    pipeline.fit(X, y, batch)

    new_X = np.array([[1.0, 1.0], [5.0, 0.5]])
    new_batch = np.array(["A", "B"])
    proba = pipeline.predict_proba(new_X, new_batch)[:, 1]

    assert proba.shape == (2,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_build_combat_baseline_returns_a_new_instance_each_call():
    first = model.build_combat_baseline()
    second = model.build_combat_baseline()

    assert first is not second
