# Calibrated 6-Variant Ensemble Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire notebook 22's locked recipe (6-variant CNN ensemble, logit-space checkpoint pooling, logistic-regression blend with a CNN-only fallback) into the real submission code path, replacing the still-shipped original rung-3-only/probability-weighted recipe, before the project's one remaining DrivenData submission.

**Architecture:** Generalize `model.py`'s checkpoint-filename generator from rung3-only to all 6 production variants; rewrite `submission.py`'s pure blend logic (new `to_logit`/`from_logit`/`pool_logit_mean` helpers + a new `combine_predictions` signature); point `build_submission_assets.py` at the 150-checkpoint list; update `submission_src/main.py` to pool in logit space and call the new blend with the fitted constants. Everything testable (model.py, submission.py) gets TDD'd; `main.py` itself stays untested per this project's established convention (real data + Docker only).

**Tech Stack:** Python, pytest, numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md`

## Global Constraints

- Only 1 real DrivenData submission remains (2026-09-16 deadline) — every change must be covered by a test where testable; `main.py` changes must be reviewed extra carefully since they can't be.
- 6-variant composition only — do NOT add the denoise variant (explicitly rejected, see spec).
- `combine_predictions`'s signature change is a deliberate breaking change — update every caller, don't shim the old signature.
- Blend constants (exact values, do not round further): `a=0.8558, b=0.5296, c=-0.0905, a1=0.9784, c1=0.0508`.
- `PRODUCTION_VARIANT_PREFIXES = ["rung3", "rung4_familybias", "rung4_lrsched", "rung4_augment", "rung4_classweight", "rung4_fixedepoch"]` (6 items, order matters for reproducible checkpoint-loading order, not for correctness).

---

### Task 1: `model.py` — generalize checkpoint-filename generation

**Files:**
- Modify: `src/model.py:114-121` (replace `rung3_checkpoint_filenames`)
- Test: `tests/test_model.py` (extend, after the existing `rung3_checkpoint_filenames` tests at line 185)

**Interfaces:**
- Consumes: `config.SEED`, `config.N_FOLDS` (existing)
- Produces: `model.variant_checkpoint_filenames(prefix, seeds=..., n_folds=...) -> list[str]`, `model.PRODUCTION_VARIANT_PREFIXES -> list[str]` (6 items), `model.production_checkpoint_filenames() -> list[str]` (150 items) — `scripts/build_submission_assets.py` (Task 4) and `submission_src/main.py` (Task 5) both call `production_checkpoint_filenames()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py`:

```python
def test_variant_checkpoint_filenames_matches_pattern_for_a_non_rung3_prefix():
    names = model_module.variant_checkpoint_filenames("rung4_augment")

    assert len(names) == 25
    assert names[0] == "rung4_augment_seed42_fold0.pt"
    assert names[-1] == "rung4_augment_seed46_fold4.pt"


def test_production_checkpoint_filenames_returns_150_names_across_6_variants():
    names = model_module.production_checkpoint_filenames()

    assert len(names) == 150
    assert len(set(names)) == 150  # no duplicates
    assert len(model_module.PRODUCTION_VARIANT_PREFIXES) == 6
    for name in names:
        assert any(name.startswith(prefix + "_seed") for prefix in model_module.PRODUCTION_VARIANT_PREFIXES)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -k "variant_checkpoint_filenames or production_checkpoint_filenames" -v`
Expected: FAIL with `AttributeError: module 'model' has no attribute 'variant_checkpoint_filenames'` (and similarly for the other two names).

- [ ] **Step 3: Implement**

Replace `src/model.py:114-121` (the current `rung3_checkpoint_filenames` function) with:

```python
PRODUCTION_VARIANT_PREFIXES = [
    "rung3", "rung4_familybias", "rung4_lrsched", "rung4_augment",
    "rung4_classweight", "rung4_fixedepoch",
]  # notebooks/16_multivariant_ensemble.ipynb + 18_fixed_epoch_full_data.ipynb
   # composition, confirmed (denoise excluded) in
   # notebooks/22_calibration_refit_rowwise_cv.ipynb (2026-09-10) --
   # see docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md


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
    """All 150 checkpoints (6 variants x 5 seeds x 5 folds) the shipped
    ensemble averages -- the composition notebooks 16/18/22 adopted.
    scripts/build_submission_assets.py and submission_src/main.py both
    use this instead of re-deriving the variant list.
    """
    return [name for prefix in PRODUCTION_VARIANT_PREFIXES
            for name in variant_checkpoint_filenames(prefix)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS, all tests including the pre-existing `test_rung3_checkpoint_filenames_matches_the_25_files_on_disk` and `test_rung3_checkpoint_filenames_respects_custom_seeds_and_folds` (regression check that the refactor didn't change `rung3_checkpoint_filenames`'s behavior).

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: generalize checkpoint-filename generation to all 6 production variants"
```

---

### Task 2: `submission.py` — logit helpers and logit-space pooling

**Files:**
- Modify: `src/submission.py:1-28` (add helpers above `combine_predictions`, which Task 3 rewrites)
- Test: `tests/test_submission.py` (new tests, before the existing `combine_predictions` tests which Task 3 will replace)

**Interfaces:**
- Consumes: `numpy`
- Produces: `submission.to_logit(p) -> float|ndarray`, `submission.from_logit(z) -> float|ndarray`, `submission.pool_logit_mean(prob_arrays) -> ndarray` — Task 3's `combine_predictions` uses `to_logit`/`from_logit`; `submission_src/main.py` (Task 5) uses `pool_logit_mean`.

- [ ] **Step 1: Write the failing tests**

Add to the top of `tests/test_submission.py` (after the module docstring/imports, before the existing `combine_predictions` tests):

```python
import numpy as np


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_submission.py -k "to_logit or from_logit or pool_logit_mean" -v`
Expected: FAIL with `AttributeError: module 'submission' has no attribute 'to_logit'` (and similarly for the other two names).

- [ ] **Step 3: Implement**

Replace `src/submission.py:1-6` (the module docstring) and insert the new helpers before `combine_predictions`:

```python
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
```

(Leave the existing `combine_predictions` function below this in place for now — Task 3 replaces it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_submission.py -k "to_logit or from_logit or pool_logit_mean" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/submission.py tests/test_submission.py
git commit -m "feat: add logit-space pooling helpers to submission.py"
```

---

### Task 3: `submission.py` — new `combine_predictions` signature

**Files:**
- Modify: `src/submission.py` (replace `combine_predictions`, the function Task 2 left in place)
- Test: `tests/test_submission.py` (replace the 4 existing `combine_predictions` tests)

**Interfaces:**
- Consumes: `submission.to_logit`, `submission.from_logit` (Task 2)
- Produces: `submission.combine_predictions(uids, cnn_probs, baseline_probs, a, b, c, a1, c1) -> list[float]` — `submission_src/main.py` (Task 5) calls this with the fitted constants from `docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md`.

- [ ] **Step 1: Write the failing tests**

Replace the 4 existing tests in `tests/test_submission.py` (`test_combine_predictions_blends_when_both_probabilities_present` through `test_combine_predictions_raises_if_a_uid_is_missing_from_cnn_probs`) with:

```python
import math


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_submission.py -v`
Expected: FAIL — `TypeError: combine_predictions() got an unexpected keyword argument` / wrong positional arity (old signature is `(uids, cnn_probs, baseline_probs, cnn_weight)`).

- [ ] **Step 3: Implement**

Replace `combine_predictions` in `src/submission.py` (currently the function below the helpers Task 2 added) with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_submission.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run the full suite to check nothing else broke**

Run: `pytest tests/ -v`
Expected: PASS (no other module calls `submission.combine_predictions` except `submission_src/main.py`, which Task 5 updates, and it's not covered by pytest).

- [ ] **Step 6: Commit**

```bash
git add src/submission.py tests/test_submission.py
git commit -m "feat: replace combine_predictions with the logit-space logistic blend + CNN-only fallback calibration"
```

---

### Task 4: `scripts/build_submission_assets.py` — package all 150 checkpoints

**Files:**
- Modify: `scripts/build_submission_assets.py:49-63` (`copy_checkpoints`)

**Interfaces:**
- Consumes: `model.production_checkpoint_filenames()` (Task 1)
- Produces: `submission_src/model_assets/checkpoints/` populated with 150 files — `submission_src/main.py` (Task 5) reads from here.

Not unit-tested (real-file-copy script, same convention as the rest of this file) — verify manually per the steps below instead of a pytest step.

- [ ] **Step 1: Update `copy_checkpoints`**

Replace `scripts/build_submission_assets.py:49-63`:

```python
def copy_checkpoints():
    dest = MODEL_ASSETS / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    names = model.production_checkpoint_filenames()
    copied = 0
    for name in names:
        src_path = config.CHECKPOINT_DIR / name
        if not src_path.exists():
            raise FileNotFoundError(
                f"expected checkpoint not found: {src_path} -- did every training "
                "notebook (07/09/10/11/12/18) finish its full 5x5 run for its variant?"
            )
        shutil.copy2(src_path, dest / name)
        copied += 1
    print(f"copied {copied}/{len(names)} production checkpoints (6 variants) -> {dest}")
```

- [ ] **Step 2: Sanity-check the filename list without touching real files**

Run: `python -c "import sys; sys.path.insert(0, 'src'); import model; names = model.production_checkpoint_filenames(); print(len(names), names[0], names[-1])"`
Expected: `150 rung3_seed42_fold0.pt rung4_fixedepoch_seed46_fold4.pt`

- [ ] **Step 3: Commit**

```bash
git add scripts/build_submission_assets.py
git commit -m "feat: package all 150 production checkpoints, not just rung3's 25"
```

(Actually running `scripts/build_submission_assets.py` end-to-end against real checkpoints/labels is the user's `[RUN ME]` step — see Task 6.)

---

### Task 5: `submission_src/main.py` — logit pooling, new blend constants, widened clip

**Files:**
- Modify: `submission_src/main.py` (all of it except the imports and `run_classical_baseline`, which is unchanged)

**Interfaces:**
- Consumes: `model.production_checkpoint_filenames()` (Task 1), `submission.pool_logit_mean` (Task 2), `submission.combine_predictions` (Task 3)
- Produces: `submission.csv` — no downstream consumer, this is the entrypoint.

Not unit-tested (real data + Docker only, same convention as the first packaging pass) — this task is a direct, careful hand-edit; review it against the spec line by line before moving on.

- [ ] **Step 1: Replace `CNN_WEIGHT` with the blend constants**

In `submission_src/main.py`, replace line 34 (`CNN_WEIGHT = 0.70  # README.md, 2026-09-09 leave-one-repeat-out result`) with:

```python
# notebooks/22_calibration_refit_rowwise_cv.ipynb (2026-09-10), 6-variant
# composition, logit-space pooling + LogisticRegression(fit_intercept=True)
# blend -- docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md
BLEND_A = 0.8558      # CNN logit coefficient
BLEND_B = 0.5296      # baseline logit coefficient
BLEND_C = -0.0905     # intercept
FALLBACK_A1 = 0.9784  # CNN-only fallback (degenerate baseline mask)
FALLBACK_C1 = 0.0508  # CNN-only fallback intercept
PROB_CLIP = (0.005, 0.995)  # widened from (1e-6, 1-1e-6) -- tail-risk hedge,
                             # second Opus review finding 7g: one confidently-
                             # wrong row at 1e-6 costs ~0.023 log loss
INFERENCE_BATCH_SIZE = 32   # bumped from config.BATCH_SIZE=8 (a training
                             # default) -- inference-only, model.predict()
                             # already runs in eval mode (deterministic
                             # BatchNorm), pure speed win, finding 7a
```

- [ ] **Step 2: Update `run_cnn_ensemble`'s docstring and checkpoint list**

Replace the docstring and the `expected =` line (currently referencing "25 rung-3 checkpoints"):

```python
def run_cnn_ensemble(uids):
    """Average sigmoid probability across all 150 production checkpoints
    (6 variants x 5 seeds x 5 folds -- notebooks 16/18/22's adopted
    composition), pooled in LOGIT space (submission.pool_logit_mean),
    not probability space -- notebooks/22_calibration_refit_rowwise_cv.ipynb's
    winning pre-registered comparison.

    Each test volume's preprocessing (resample/crop/normalize --
    data.load_volume, the expensive part) runs at most ONCE per uid and
    is cached in memory (a plain dict, not the disk-backed
    cache.CachedVolumeStore -- inference is a single session, nothing to
    persist across runs) so all 150 checkpoints' passes reuse it instead
    of repeating it 150x per volume. Returns {uid: probability}.
    """
    checkpoint_dir = MODEL_ASSETS / "checkpoints"
    expected = model.production_checkpoint_filenames()
    checkpoint_paths = [checkpoint_dir / name for name in expected]
    missing = [p.name for p in checkpoint_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(expected)} production checkpoints missing from {checkpoint_dir} "
            "-- did scripts/build_submission_assets.py run, and did model_assets/ get zipped?"
        )
    print(f"CNN ensemble: {len(checkpoint_paths)} checkpoints, {len(uids)} test volumes, device={DEVICE}")
```

- [ ] **Step 3: Replace the checkpoint loop's accumulation and logging**

Replace the loop body and return (currently accumulating a running probability sum):

```python
    volume_cache = {}

    def cached_load_volume(uid):
        if uid not in volume_cache:
            volume_cache[uid] = data.load_volume(uid)
        return volume_cache[uid]

    ds = dataset.DatParkinsonDataset(uids, load_fn=cached_load_volume)
    all_probs = []
    for i, checkpoint_path in enumerate(checkpoint_paths):
        start = time.time()
        net = model.build_model().to(DEVICE)
        net.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        loader = torch.utils.data.DataLoader(ds, batch_size=INFERENCE_BATCH_SIZE, num_workers=0)
        probs = []
        for x, _ in loader:
            probs.append(model.predict(net, x))
        all_probs.append(np.concatenate(probs))
        if (i + 1) % 10 == 0 or i + 1 == len(checkpoint_paths):
            print(f"  checkpoint {i + 1}/{len(checkpoint_paths)} done in {time.time() - start:.1f}s")

    pooled = submission.pool_logit_mean(all_probs)
    return dict(zip(uids, pooled.tolist()))
```

- [ ] **Step 4: Update `main()`'s blend call and clip**

Replace:

```python
    predictions = submission.combine_predictions(uids, cnn_probs, baseline_probs, CNN_WEIGHT)
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
```

with:

```python
    predictions = submission.combine_predictions(
        uids, cnn_probs, baseline_probs,
        BLEND_A, BLEND_B, BLEND_C, FALLBACK_A1, FALLBACK_C1,
    )
    predictions = np.clip(predictions, *PROB_CLIP)
```

- [ ] **Step 5: Re-read the whole file and diff against the spec**

Open `submission_src/main.py` in full and check line-by-line against
"## `submission_src/main.py`" in
`docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md`.
Confirm: `run_classical_baseline` is untouched; the `NIFTI_DIR` assertion
in `main()` is untouched; no leftover reference to `CNN_WEIGHT` or the
old `1e-6` clip bound remains anywhere in the file.

- [ ] **Step 6: Commit**

```bash
git add submission_src/main.py
git commit -m "feat: wire the 6-variant logit-pooled ensemble and logistic blend into main.py"
```

---

### Task 6: Full verification and handoff to the user's real-data steps

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: PASS, every test (including the pre-existing suite untouched by this plan).

- [ ] **Step 2: Confirm no stale references remain**

Run: `grep -rn "CNN_WEIGHT\|rung3_checkpoint_filenames()" submission_src/main.py scripts/build_submission_assets.py`
Expected: no matches (both files now use the new names — `rung3_checkpoint_filenames` itself still exists in `model.py` for backward compatibility, but nothing in the submission path should call it anymore).

- [ ] **Step 3: Report to the user what still needs real data / Docker (not run by Claude)**

Tell the user, in this order:
1. Run `python scripts/build_submission_assets.py` (re-fits the ComBat
   baseline on 100% of training data, copies all 150 checkpoints, copies
   the updated `src/model.py`/`src/submission.py` into `submission_src/`).
2. Confirm `submission_src/model_assets/checkpoints/` has 150 files and
   `combat_baseline.pkl` exists.
3. Run the runtime repo's `just pack-submission` then `just
   test-submission` against the smoke-test data (per
   `docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md`'s
   "Definition of done") and report back: exit code, wall time, whether
   `submission.csv` matches `submission_format.csv`'s shape/columns, and
   any warnings.
4. Only after a clean smoke test: decide whether to spend the one
   remaining real submission now.
