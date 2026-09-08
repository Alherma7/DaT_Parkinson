"""Model pipeline builders."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def build_classical_baseline():
    """StandardScaler + LogisticRegression classical baseline.

    Features: `features.striatum_mask` -> `features.signed_asymmetry`
    (as `abs_asym`) + `features.striatal_ratio`. Validated in
    `notebooks/03_baseline_classical.ipynb` against the real
    `src/features.py` code on all 1362 volumes: the combined model beats
    `abs_asym` alone by -0.0136 log loss, ~20-45x the run's own
    fold-reshuffle sd -- see `README.md` for the full comparison.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)),
    ])
