# Calibrated 6-variant ensemble + logit-space blend: `submission.py`, `main.py`, `model.py`

**Decision this feeds**: `submission_src/main.py` as shipped today still runs
the *original* rung-3-only (25-checkpoint), probability-weighted
(`CNN_WEIGHT=0.70`) recipe from the first submission-packaging pass
(`docs/superpowers/specs/2026-09-09-submission-packaging-design.md`). None
of notebooks 14-22's findings (multi-variant ensemble, recalibration,
logit-space pooling) have been wired in yet — `project_dat_parkinson_strategic_roadmap.md`
(memory) tracked this as "Status: NOT YET IMPLEMENTED" through the whole
analysis phase. **Only one real DrivenData submission remains** before the
2026-09-16 deadline (user-confirmed 2026-09-10) — this is the last chance
to ship the validated improvement, so correctness matters more than speed
of delivery, and every change below is scoped to what the locked recipe
actually requires (see "Explicitly out of scope").

**Final recipe** (`notebooks/22_calibration_refit_rowwise_cv.ipynb`,
2026-09-10, fully re-run and internally consistent — see
`project_dat_parkinson_strategic_roadmap.md` memory for the complete
derivation): 6-variant CNN ensemble — `rung3`, `rung4_familybias`,
`rung4_lrsched`, `rung4_augment`, `rung4_classweight`, `rung4_fixedepoch`
(150 checkpoints, 25 each; **denoise explicitly rejected** by the user —
mechanical delta -0.0016 was only ~6% of that comparison's own row-wise-CV
sd, the same "wins on sign but is noise" pattern that got flip-TTA dropped
in notebook 21, and adding it means a second preprocessing path in
`main.py` for one submission with zero margin), pooled per-row in
**logit space** (not probability space — this is a real behavior change),
blended with the classical ComBat baseline via
`LogisticRegression(fit_intercept=True)` on `[logit(cnn_p), logit(baseline_p)]`:

```
main blend:     p = sigmoid(a * logit(cnn_p) + b * logit(baseline_p) + c)
                a=0.8558  b=0.5296  c=-0.0905

CNN-only fallback (degenerate baseline mask):
                p = sigmoid(a1 * logit(cnn_p) + c1)
                a1=0.9784  c1=0.0508
```

No base-rate shrinkage (`eps=0`, confirmed — log loss degrades
monotonically away from it). Honest row-wise CV mean=0.3617, vs. the
current shipped recipe's never-correctly-measured baseline.

## Architecture

Four files change; nothing else. Same flat-copy convention as the first
packaging pass (`scripts/build_submission_assets.py::copy_modules`) — no
new files, no import rewrites.

```
src/model.py          # generalize checkpoint-filename generation (6 variants, not just rung3)
src/submission.py     # new combine_predictions signature + logit-space pooling helper
scripts/build_submission_assets.py  # copy_checkpoints uses the new 150-name list
submission_src/main.py              # 150-checkpoint logit-pooled ensemble, new blend constants, widened clip, throttled logging
```

## `src/model.py`: generalize checkpoint-filename generation

Today `rung3_checkpoint_filenames()` hardcodes the `"rung3"` prefix.
Extract the pattern into a prefix-parameterized helper, keep the old name
as a thin wrapper (existing tests / callers untouched), and add the
6-variant production list:

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

**Testing (`tests/test_model.py`, extend):**
1. `test_rung3_checkpoint_filenames_matches_the_25_files_on_disk` and
   `test_rung3_checkpoint_filenames_respects_custom_seeds_and_folds`
   (existing) must keep passing unchanged — regression guard that the
   refactor is behavior-preserving.
2. `variant_checkpoint_filenames("rung4_augment")` returns 25 names,
   `"rung4_augment_seed42_fold0.pt"` first, `"rung4_augment_seed46_fold4.pt"`
   last.
3. `production_checkpoint_filenames()` returns exactly 150 names, with no
   duplicates, and every name matches one of the 6
   `PRODUCTION_VARIANT_PREFIXES` (regex or `str.startswith` check per
   name).

## `src/submission.py`: logit-space pooling + new blend signature

Mirrors `notebooks/22_calibration_refit_rowwise_cv.ipynb`'s own
`to_logit`/`logit_mean` helpers almost verbatim, so a reviewer can diff
this file directly against the notebook that validated it. Introduces a
numpy dependency to this module (already a project dependency, used
throughout `src/`) — `submission.py` was pure-Python only because nothing
before this needed array pooling.

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

This is a **breaking signature change** (`cnn_weight` param removed,
5 new required params added) — every existing caller/test of
`combine_predictions` must be updated, not shimmed, since the old
probability-weighted formula is exactly the bug this whole exercise
fixes (finding 2 of the second Opus review, `project_dat_parkinson_strategic_roadmap.md`).

**Testing (`tests/test_submission.py`, rewrite to match the new
signature):**
1. `to_logit`/`from_logit` round-trip: `from_logit(to_logit(p)) == p`
   (`pytest.approx`) for a few values in `(0, 1)`, including near the
   `_EPS` boundary.
2. `to_logit(0.0)` and `to_logit(1.0)` don't raise or return `inf`/`nan`
   (the clip fires).
3. `pool_logit_mean` on two arrays that are the same value returns that
   value unchanged (pooling a constant is a no-op).
4. `pool_logit_mean` on `[0.9, 0.1]` (two single-element arrays) is
   **not** `0.5` (probability-space mean would be `0.5`; logit-space mean
   of `logit(0.9)` and `logit(0.1)` is `0` exactly, since they're
   symmetric around 0.5 in logit space too here -- use an asymmetric pair
   instead, e.g. `[0.9, 0.3]`, and assert the result differs from
   `np.mean([0.9, 0.3])` to prove logit-space pooling is actually being
   used, not silently falling back to a probability-space mean).
5. `combine_predictions` main-blend path: hand-computed expected value
   for one uid with known `cnn_p`, `baseline_p`, `a, b, c` (compute
   `sigmoid(a*logit(cnn_p) + b*logit(baseline_p) + c)` in the test with
   `math.exp`/`math.log`, independent of `to_logit`/`from_logit`, so the
   test doesn't just re-assert the implementation against itself).
6. `combine_predictions` fallback path: a uid missing from
   `baseline_probs` uses `a1, c1` (hand-computed expected value, same
   independent-computation approach).
7. `combine_predictions` preserves `uids` order regardless of dict
   insertion order (keep the existing test, adapted to the new call
   signature).
8. `combine_predictions` raises `KeyError` if a uid is missing from
   `cnn_probs` (keep the existing test, adapted).

## `scripts/build_submission_assets.py`: copy all 150 checkpoints

One-line change in `copy_checkpoints`:

```python
names = model.production_checkpoint_filenames()  # was: model.rung3_checkpoint_filenames()
```

Update the `FileNotFoundError` message (currently says "did
notebooks/07_cnn_rung3.ipynb finish its full 5x5 run?") to name all 6
source notebooks generically, and the final print's `"rung-3 checkpoints"`
wording, so a failure message still points at the right place. No test
needed — this script's only logic (`fit_and_pickle_baseline`,
`copy_checkpoints`, `copy_modules`) is exercised end-to-end when the user
runs it, same as the first packaging pass; the filename list itself is
tested via `production_checkpoint_filenames()` above.

## `submission_src/main.py`

`main.py` is written directly in `submission_src/` (not copied from
`src/`) and is **not unit-tested** — same rule as the first packaging
pass, it can't be run or verified against real data by Claude. Changes:

1. Replace `CNN_WEIGHT = 0.70` with the 5 blend constants:
   ```python
   # notebooks/22_calibration_refit_rowwise_cv.ipynb (2026-09-10), 6-variant
   # composition, logit-space pooling + LogisticRegression(fit_intercept=True)
   # blend -- docs/superpowers/specs/2026-09-10-calibrated-ensemble-blend-design.md
   BLEND_A = 0.8558      # CNN logit coefficient
   BLEND_B = 0.5296      # baseline logit coefficient
   BLEND_C = -0.0905     # intercept
   FALLBACK_A1 = 0.9784  # CNN-only fallback (degenerate baseline mask)
   FALLBACK_C1 = 0.0508  # CNN-only fallback intercept
   PROB_CLIP = (0.005, 0.995)  # widened from (1e-6, 1-1e-6) -- tail-risk
                                # hedge, second Opus review finding 7g:
                                # one confidently-wrong row at 1e-6 costs
                                # ~0.023 log loss
   ```
2. `run_cnn_ensemble`: `expected = model.production_checkpoint_filenames()`
   (150 names, was 25). Collect each checkpoint's probability array into
   a list instead of accumulating a running probability sum, then pool
   once at the end with `submission.pool_logit_mean(all_probs)`:
   ```python
   all_probs = []
   for i, checkpoint_path in enumerate(checkpoint_paths):
       ...
       all_probs.append(np.concatenate(probs))
       if (i + 1) % 10 == 0 or i + 1 == len(checkpoint_paths):
           print(f"  checkpoint {i + 1}/{len(checkpoint_paths)} done in {time.time() - start:.1f}s")
   pooled = submission.pool_logit_mean(all_probs)
   return dict(zip(uids, pooled.tolist()))
   ```
   (throttled to every 10th line + the last one -- 150 unthrottled lines
   risks the platform's ~300-line log cap, second Opus review finding
   7h). Also raise the inference `DataLoader`'s `batch_size` from the
   training default (`config.BATCH_SIZE=8`) to a submission-local
   `INFERENCE_BATCH_SIZE = 32` constant -- inference-only, `model.predict`
   already puts the net in eval mode (deterministic BatchNorm, no
   per-batch-size dependence), so this is a pure speed win with no
   correctness risk (second Opus review finding 7a).
3. `main()`: call `submission.combine_predictions(uids, cnn_probs,
   baseline_probs, BLEND_A, BLEND_B, BLEND_C, FALLBACK_A1, FALLBACK_C1)`,
   then `np.clip(predictions, *PROB_CLIP)`.

## Explicitly out of scope (deliberate, not an oversight)

- **Denoise as a 7th variant**: rejected by the user (see "Final recipe"
  above) — mechanical delta inside the noise floor, added complexity not
  justified with one submission left.
- **sklearn-pickle → plain `.npz` ComBat baseline reconstruction**
  (second Opus review finding 7c): a real code-quality concern (an
  `InconsistentVersionWarning` already fired once between local-fit and
  container-runtime sklearn versions), but the *actual* local Docker
  smoke test that hit this (`README.md`, 2026-09-09) completed with exit
  code 0 despite the warning — empirically survivable, not a crash. With
  only one submission left, rewriting a currently-working path (and
  introducing a new place to get a sigmoid/scaling formula subtly wrong)
  is higher-risk than leaving it alone. Deferred, not fixed.
- **Shared single volume load for CNN + classical baseline** (finding
  7b): `run_cnn_ensemble` uses `data.load_volume` (resampled/cropped/
  normalized, CNN-ready); `run_classical_baseline` needs the *original*
  raw volume/spacing for `features.extract_baseline_features` — sharing
  the initial `nib.load` would touch both paths for a speed optimization
  when GPU/wall time is explicitly not the binding constraint (the
  strategic roadmap memory: "GPU time is NOT the constraint"). Deferred.
- **Batch size / preprocessing for a 7th (denoise) variant**: moot, see
  above.

## Definition of done

- `pytest tests/` passes (extended `test_model.py`, rewritten
  `test_submission.py`, everything else untouched), synthetic data only.
- `src/model.py` and `src/submission.py` are the only `src/*.py` changes;
  `scripts/build_submission_assets.py` and `submission_src/main.py` are
  updated directly (the latter is not copied from `src/`).
- User runs `scripts/build_submission_assets.py`, then the runtime
  repo's `just pack-submission` + `just test-submission` against the
  smoke-test data, and reports back: exit code, wall time (watch the
  6-checkpoint-groups x 25 vs. the old 25-checkpoint run's ~64s — expect
  roughly 6x unless the batch-size bump offsets it), whether
  `submission.csv` matches `submission_format.csv`'s shape/columns, any
  warnings (the ComBat pickle one is expected and known-survivable, per
  "out of scope" above — anything else is new and must be investigated
  before spending the one remaining real submission).
- `README.md` Next steps updated once the smoke test result comes back,
  same as the first packaging pass.
