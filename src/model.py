"""Model pipeline builders."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import features


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


class ComBatHarmonizedPipeline:
    """`build_classical_baseline`'s pipeline, preceded by ComBat batch
    harmonization (`features.fit_combat`/`apply_combat`).

    Not a plain sklearn `Pipeline` step: ComBat needs `y` at fit time (to
    restrict fitting to training-set controls, the leakage-avoidance rule
    from the Frontiers 2022 PPMI precedent in `RESOURCES.md`) and needs
    the batch label again at both fit and predict time, which sklearn's
    2-arg `fit(X, y)`/`predict_proba(X)` convention has no slot for.

    Gate passed in `notebooks/04_combat_harmonization.ipynb` (2026-09-08,
    5 seeds, `evaluate.make_folds`/`log_loss_score`): `combined + ComBat`
    0.5290 vs. `combined` (no ComBat) 0.5753, delta -0.0462 -- ~40-80x
    that run's own per-variant sd (0.0006-0.0011). This is the classical
    baseline to actually use; `build_classical_baseline` stays as the
    (weaker) no-ComBat reference point.
    """

    def __init__(self):
        self._pipeline = build_classical_baseline()
        self._combat_params = None

    def fit(self, X, y, batch):
        X, y, batch = np.asarray(X), np.asarray(y), np.asarray(batch, dtype=object)
        controls = y == 0
        self._combat_params = features.fit_combat(X[controls], batch[controls])
        X_adj = features.apply_combat(X, batch, self._combat_params)
        self._pipeline.fit(X_adj, y)
        return self

    def predict_proba(self, X, batch):
        X_adj = features.apply_combat(np.asarray(X), np.asarray(batch, dtype=object),
                                       self._combat_params)
        return self._pipeline.predict_proba(X_adj)


def build_combat_baseline():
    """`ComBatHarmonizedPipeline` factory -- see its docstring."""
    return ComBatHarmonizedPipeline()
