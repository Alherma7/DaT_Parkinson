"""Unit tests for src/model.py."""

import numpy as np

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
