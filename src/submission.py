"""Pure, unit-testable logic shared by submission_src/main.py. Kept here
(not inlined in main.py) so it's covered by pytest -- main.py itself
can't be run or verified against real data by Claude
(docs/superpowers/specs/2026-09-09-submission-packaging-design.md).
"""
import numpy as np

_EPS = 1e-6


def to_logit(p):
    """Log-odds of `p`, clipped to [EPS, 1-EPS] first so a 0/1 input
    doesn't blow up. Works on a python float or a numpy array.
    """
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def from_logit(z):
    """Inverse of to_logit (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-z))


def pool_logit_mean(prob_arrays):
    """Average multiple checkpoints' per-row probability arrays in logit
    space (sigmoid of the mean of their logits), not probability space.
    Logit-space pooling was the winning pre-registered comparison in
    notebooks/22_calibration_refit_rowwise_cv.ipynb (2026-09-10, delta
    -0.0020 vs probability-space pooling) -- see
    docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md.

    `prob_arrays` is a non-empty sequence of same-shape numpy arrays (one
    per checkpoint). Returns one pooled array, same shape.
    """
    return from_logit(np.mean([to_logit(p) for p in prob_arrays], axis=0))


def combine_predictions(uids, cnn_probs, baseline_probs, a, b, c, a1, c1):
    """Blend per-uid CNN and classical-baseline probabilities into the
    submission's final prediction, in `uids`' exact order.

    `cnn_probs` is the CNN ensemble's already-pooled (see
    `pool_logit_mean`) per-uid probability. `cnn_probs` and
    `baseline_probs` are dicts keyed by uid. Every uid in `uids` must
    have a CNN prediction (raises KeyError otherwise -- the CNN ensemble
    runs on every test volume unconditionally). A uid missing from
    `baseline_probs` (features.extract_baseline_features returned None
    for it -- a degenerate striatum mask) uses the CNN-only fallback
    calibration (a1, c1) instead of the main blend, rather than falling
    back to the raw uncalibrated CNN probability.

    Both paths are a logistic-regression blend in logit space, fit in
    notebooks/22_calibration_refit_rowwise_cv.ipynb (2026-09-10):
        main:     p = sigmoid(a * logit(cnn_p) + b * logit(baseline_p) + c)
        fallback: p = sigmoid(a1 * logit(cnn_p) + c1)

    Returns a list of floats, same length and order as `uids`.
    """
    predictions = []
    for uid in uids:
        cnn_logit = to_logit(cnn_probs[uid])
        if uid in baseline_probs:
            z = a * cnn_logit + b * to_logit(baseline_probs[uid]) + c
        else:
            z = a1 * cnn_logit + c1
        predictions.append(float(from_logit(z)))
    return predictions
