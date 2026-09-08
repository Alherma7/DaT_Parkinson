# 3D-CNN data plumbing: `data.py`, `dataset.py`, first architecture, smoke test

Status: approved for implementation planning
Date: 2026-09-08

## Purpose

Start the project's main track (3D CNN on resampled DaT-SPECT volumes,
`README.md`'s "Main track" next step) by building and validating just the
plumbing: deterministic geometry/intensity preprocessing (`data.py`), a
`Dataset` wrapper (`dataset.py`), a first CNN architecture (`model.py`),
and a rung-0/rung-1 smoke test per `deep-learning-imaging.md`'s cheap-proxy
ladder. This spec stops once the pipeline runs end-to-end and can overfit
a tiny labeled subset — it does not train a submittable model.

## Explicitly out of scope (deferred to a follow-up spec)

- Full `train.py` (optimizer/scheduler, AMP, checkpointing, early stopping,
  per-fold logging)
- `augment.py` and any augmentation wiring (including validating the flip
  augmentation EDA cleared as safe)
- Rung 2 (single fold, full data) and rung 3 (full CV — the actual gate
  decision against the classical `build_combat_baseline()`)
- Hyperparameter tuning, calibration

None of this spec changes anything already validated: the classical
baseline (`features.py`, `model.py::build_classical_baseline`/
`build_combat_baseline`) does not call `data.py` and is untouched. The
`config.py` constants this spec consumes (`TARGET_SPACING`, `CROP_SIZE_MM`,
`CROP_CENTER_MM`, `TARGET_SHAPE`, `BACKGROUND_PERCENTILE`,
`BACKGROUND_MAX_FRACTION`) were already pinned by the EDA before this spec
and are not changed here.

## Decisions made during brainstorming

- **Train from scratch, no pretrained backbone.** `RESOURCES.md` already
  has three independent findings against ImageNet transfer learning on
  this modality (Wenzel et al. 2019, Chegodaev et al., Kim et al.), DINoV3
  is confirmed ineligible, and PPMI-pretrained eligibility is still
  unresolved. Training from scratch sidesteps the eligibility risk and
  matches the cited prior art.
- **Small 3D CNN, not 2D-slice + attention.** The prior-art scan found
  both architecture families; a plain 3D CNN is simpler to implement and
  test, and the small resampled volume (56x30x44) does not hit the memory
  pressure that would motivate the 2D+attention alternative.
- **Fixed physical crop, not the classical baseline's adaptive mask.**
  `features.striatum_mask` (used by the classical baseline) finds the
  striatum per-volume; the CNN instead gets the same physical box every
  time (`config.CROP_SIZE_MM` centered at `config.CROP_CENTER_MM`), per
  `deep-learning-imaging.md`'s "resample to fixed spacing, then crop/pad
  to fixed shape" pattern. These are two independent pipelines.

## `src/data.py`

One composed function, three separately-tested steps:

```python
def load_volume(uid: str) -> np.ndarray:
    """Returns a (1, *config.TARGET_SHAPE) float32 array. The single
    function both training and inference call -- never reimplemented
    at the inference entry point."""
```

1. **`resample_to_spacing(volume, affine, target_spacing) -> (array, new_affine)`**
   — wraps `nibabel.processing.resample_to_output`, which resamples
   through the full affine (not just axis permutation), addressing the
   EDA finding that 40% of volumes are oblique (up to 40.3 deg) and that
   `as_closest_canonical()` does not correct this. Output is RAS-aligned,
   so no separate reorientation step is needed.
2. **`crop_or_pad(volume, affine, center_mm, size_mm, target_shape) -> array`**
   — crops (or zero-pads, for the handful of tight-FOV volumes on z) to
   `config.TARGET_SHAPE` voxels, centered at the resampled volume's
   geometric center offset by `config.CROP_CENTER_MM`.
3. **`normalize_intensity(volume) -> array`** — background-aware
   per-volume z-score on the *cropped* volume: background threshold =
   `min(percentile(vol, config.BACKGROUND_PERCENTILE), config.BACKGROUND_MAX_FRACTION * vol.max())`;
   z-score uses the foreground (above-threshold) voxels' mean/std,
   applied to the whole cropped volume. `np.clip(volume, 0, None)` first,
   consistent with `features.py`'s existing handling of the 16 `int16`
   volumes.

Each of the three takes an array (+ affine/spacing), not a file path, so
each is unit-testable on tiny synthetic 3D arrays -- including a synthetic
oblique affine for `resample_to_spacing`, and a synthetic tight-FOV shape
for `crop_or_pad`'s padding branch.

## `src/dataset.py`

```python
class DatParkinsonDataset(torch.utils.data.Dataset):
    """index -> data.py::load_volume -> tensor (+ label if provided)."""
    def __init__(self, uids, labels=None): ...
    def __getitem__(self, i):
        # labels is None (inference): returns (tensor, uid)
        # labels given (train/eval): returns (tensor, label)
```

No augmentation wiring in this spec -- `augment.py` does not exist yet.
Glue only, per the skill's `dataset.py` rule: test its output *contract*
(shape, dtype, label type), not its content.

## `src/model.py` additions

Small 3D CNN, He init / BatchNorm / dropout / Adam (Geron Ch.11 default
DNN config), no LR schedule (matches `config.LR_SCHEDULE = None`):

```
Input: (1, 56, 30, 44)                              # TARGET_SHAPE
Block1: Conv3d(1->16, k=3, pad=1) -> BN3d -> ReLU -> MaxPool3d(2)
Block2: Conv3d(16->32, k=3, pad=1) -> BN3d -> ReLU -> MaxPool3d(2)
Block3: Conv3d(32->64, k=3, pad=1) -> BN3d -> ReLU -> MaxPool3d(2)
Block4: Conv3d(64->128, k=3, pad=1) -> BN3d -> ReLU -> AdaptiveAvgPool3d(1)
Flatten -> Dropout(0.3-0.5) -> Linear(128, 1)        # raw logit
```

- `DatCNN(nn.Module)` -- the architecture above, weights initialized with
  `kaiming_normal_` (He init) on `Conv3d`/`Linear` layers.
- `build_model() -> DatCNN` -- fresh-instance factory, mirroring
  `build_classical_baseline()`'s existing convention.
- `predict(model, x) -> np.ndarray` -- forward pass -> sigmoid -> numpy
  probabilities (the deep-learning analogue of `predict_proba`).
- Loss: `BCEWithLogitsLoss` in the training code (not part of `model.py`
  itself) -- the model returns a raw logit, not a sigmoid output, for
  numerical stability.

This lives in the existing `src/model.py` alongside the classical
pipelines (`build_classical_baseline`, `ComBatHarmonizedPipeline`), per
`deep-learning-imaging.md`'s skeleton, which lists `model.py` as holding
both `nn.Module` definitions and `build_model()`/`predict()`.

## Smoke test: `notebooks/05_cnn_smoke_test.ipynb`

Rung 0 (mandatory before any longer run) and, if that passes, rung 1:

- **Rung 0** (seconds): 16-32 labeled volumes (small stratified subset of
  `train_labels.csv`), `DatParkinsonDataset` + a `DataLoader`, a bare
  training loop (no checkpointing/early stopping -- that's the deferred
  spec) run for a handful of epochs. Confirms: right shapes/dtypes end to
  end, and the model can overfit this tiny batch to near-zero loss (Geron
  Ch.10's shape/label/init sanity check).
- **Rung 1** (minutes, only if rung 0 passes): a 10-20% stratified subset
  (via `evaluate.make_folds`), few epochs, one fold. Confirms the loss
  curve is sane and the run isn't catastrophically worse than the
  classical baseline. Logged explicitly as a rung-1 number, per
  `deep-learning-imaging.md`'s "log the rung with the number" rule -- not
  a gate decision.

Both cells touch real `.nii.gz` volumes and row-level labels, so per the
project's AI-assistant data rule (`README.md`), this notebook is marked
**[RUN ME]** -- the user runs it and shares back the printed shapes and
loss curve, not Claude.

## Testing plan

- `tests/test_data.py`: `resample_to_spacing` (incl. a synthetic oblique
  affine), `crop_or_pad` (incl. the tight-FOV padding branch),
  `normalize_intensity` (background-threshold formula on a known array) --
  all on small synthetic arrays, never real patient data.
- `tests/test_dataset.py`: batch contract (shape `(1, 56, 30, 44)`, dtype
  `float32`, label type) via a monkeypatched `load_volume`, never real
  data.
- `tests/test_model.py` additions: forward-pass output shape and
  finiteness on a random tensor; `build_model()` returns a fresh instance
  each call (mirrors the existing classical-baseline test).

## Definition of done

- `pytest tests/` passes (new tests plus the existing 28).
- `notebooks/05_cnn_smoke_test.ipynb` written, `[RUN ME]`-marked, not run
  by Claude.
- `README.md` Progress/Next-steps updated once the user runs the notebook
  and reports the rung-0/1 results back (per the graduation rule, this
  spec's code stays notebook-referenced, not wired into any production
  path, until the follow-up spec's rung 3 clears the gate).
