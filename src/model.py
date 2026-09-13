"""Model pipeline builders."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

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


class DatCNN(nn.Module):
    """Small 3D CNN, trained from scratch (no pretrained backbone --
    RESOURCES.md has three independent findings against ImageNet transfer
    on this modality, and DINOv3/PPMI eligibility is unresolved/ineligible;
    see docs/superpowers/specs/2026-09-08-cnn-data-plumbing-design.md).
    He init / BatchNorm / dropout / Adam, no LR schedule (Geron Ch.11
    default DNN config; matches config.LR_SCHEDULE = None). Input is
    `config.TARGET_SHAPE` = (56, 30, 44), 1 channel. Returns a raw logit
    (not a sigmoid output) -- use with `BCEWithLogitsLoss`.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.BatchNorm3d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


def build_model():
    """`DatCNN` factory -- fresh instance each call, mirroring
    `build_classical_baseline()`'s existing convention."""
    return DatCNN()


class DatSlab2DCNN(nn.Module):
    """2D CNN over a thick axial "slab" through the striatum (roadmap
    item 6, README.md -- architecture diversity for the ensemble, never
    run until now). Input is `data.load_slab`'s output,
    `config.SLAB_TARGET_SHAPE[:2]` = (68, 35), 1 channel.

    Channel schedule (1->64->64->96->128) and the 3x3-conv/batchnorm/
    2x2-maxpool blocks follow Wenzel et al. 2019 (RESOURCES.md) directly.
    Two deliberate deviations from their exact design, both to fit this
    project's existing pipeline rather than for performance reasons:
    (1) `AdaptiveAvgPool2d` before the dense head instead of a hand-tuned
    flatten size -- mirrors `DatCNN`'s own `AdaptiveAvgPool3d(1)` choice,
    robust to the exact slab crop shape; (2) a single raw-logit output
    (`BCEWithLogitsLoss`), not their 2-unit softmax + categorical
    cross-entropy -- mechanically equivalent for binary classification,
    and keeps this variant's OOF array directly poolable alongside the
    other 6 (all single-logit) variants with no special-casing.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(1024, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


def build_slab_model():
    """`DatSlab2DCNN` factory -- fresh instance each call, mirroring
    `build_model()`'s existing convention."""
    return DatSlab2DCNN()


PRODUCTION_VARIANT_PREFIXES = ["coveragefix_persubject"]
# Replaces the old 6-variant/150-checkpoint fixed-crop-center composition
# (rung3 + 5 rung-4 variants) entirely -- notebooks/26_striatum_coverage_retrain.ipynb
# (2026-09-13) found a single variant trained on data.load_volume(uid,
# center_mm="auto") (per-subject striatum centering, 100% coverage vs.
# the old fixed crop's ~71%) beats the whole old ensemble decisively
# (paired_repeat_gate mean=-0.1025, CI=[-0.1449,-0.0601] vs. rung3 alone;
# re-adding the old variants to the blend HURTS it, 6th Opus review) --
# see project memory's striatum-crop-coverage investigation.

PRODUCTION_VARIANT_BUILD_FNS = {prefix: build_model for prefix in PRODUCTION_VARIANT_PREFIXES}
PRODUCTION_VARIANT_BUILD_FNS["slab2d"] = build_slab_model
# Per-prefix architecture factory, consulted by submission_src/main.py
# instead of calling build_model() unconditionally, so a checkpoint's
# architecture is derived from its own filename prefix rather than
# assumed uniform. Added now (additive, no behavior change -- "slab2d" is
# not yet in PRODUCTION_VARIANT_PREFIXES/production_checkpoint_filenames())
# so it's ready if notebooks/24_slab2d_architecture_diversity.ipynb's gate
# clears; harmless if it never does.


def variant_checkpoint_filenames(prefix, seeds=range(config.SEED, config.SEED + 5),
                                  n_folds=config.N_FOLDS):
    """Filenames of one variant's nested-CV checkpoints, e.g.
    "rung4_augment_seed42_fold0.pt" -- the naming convention every
    training notebook's own torch.save calls already use.
    """
    return [f"{prefix}_seed{seed}_fold{fold}.pt" for seed in seeds for fold in range(n_folds)]


def rung3_checkpoint_filenames(seeds=range(config.SEED, config.SEED + 5), n_folds=config.N_FOLDS):
    """Filenames of the rung-3 nested-CV checkpoints (notebooks/07_cnn_rung3.ipynb).
    Thin wrapper around variant_checkpoint_filenames -- kept for the
    existing callers/tests that name it directly.
    """
    return variant_checkpoint_filenames("rung3", seeds, n_folds)


def production_checkpoint_filenames():
    """All 25 checkpoints (1 variant x 5 seeds x 5 folds) the shipped
    ensemble averages -- notebooks/26_striatum_coverage_retrain.ipynb's
    per-subject-centered CNN, per PRODUCTION_VARIANT_PREFIXES above.
    scripts/build_submission_assets.py and submission_src/main.py both
    use this instead of re-deriving the variant list.
    """
    return [name for prefix in PRODUCTION_VARIANT_PREFIXES
            for name in variant_checkpoint_filenames(prefix)]


def predict(model, x):
    """Forward pass -> sigmoid -> numpy probabilities -- the deep-learning
    analogue of `predict_proba` on the classical pipelines. Puts the model
    in eval mode (disables dropout/BatchNorm training behavior), moves the
    input onto the model's own device, and restores the model's prior
    training/eval state afterward so mid-training validation calls don't
    silently leave it in eval mode.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = x.to(next(model.parameters()).device)
            return torch.sigmoid(model(x)).cpu().numpy()
    finally:
        model.train(was_training)
