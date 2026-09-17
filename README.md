# DaT Parkinson's Challenge

Classifies a dopamine-transporter (DaT) SPECT brain scan as normal or
abnormal, from the scan alone.

Built for [DrivenData's DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)
(French Society of Nuclear Medicine, 2026), closed 2026-09-16.

**Project site with the full write-up, charts and the "what worked / what
didn't" analysis:** https://alherma7.github.io/DaT_Parkinson/

| | |
|---|---|
| **Final leaderboard** | **rank #160**, log loss **0.3436** (AUROC 0.9226), 3 submissions used |
| Best submission score (own measurement, at submit time) | log loss 0.2975, AUROC 0.9433 |
| Submissions | 0.4648 → 0.4185 → 0.2975 (own measurement); final confirmed leaderboard score is 0.3436 |

The gap between the score shown at submission time (0.2975) and the final
confirmed leaderboard result (0.3436) is real and reported here honestly —
see [the project site](https://alherma7.github.io/DaT_Parkinson/#results) for the discussion.

## The final pipeline

1. **Resample** each NIfTI to physical spacing (not a fixed voxel shape),
   affine-aware — 40% of volumes have oblique affines despite reporting RAS
   axis codes, so `as_closest_canonical()` alone is not sufficient.
2. **Locate the striatum per subject** and crop around each subject's own
   measured centroid (`center_mm="auto"`) — replacing an earlier fixed,
   population-median crop center that missed the striatum entirely for
   ~30% of subjects.
3. **Classical baseline**: handcrafted asymmetry/ratio features +
   ComBat-harmonized (Fortin et al. 2018) logistic regression, correcting
   for the scanner-family confound found in EDA.
4. **3D CNN** trained on the per-subject-centered crops.
5. **Blend**: CNN + classical baseline, logit-space temperature-calibrated
   weighted average, weights and temperatures selected via leave-one-fold-out
   validation (never on the data used to score the final recipe).

## What worked and what didn't

The single largest lever in the whole project (-0.1025 log loss, more than
10x any other individual change) was fixing a crop-coverage bug: the
production crop was centered on a fixed, population-level point that missed
the striatum for about 30% of subjects. Nearly everything else tried at the
architecture/training-loop level (family oversampling, cosine LR schedule,
augmentation, class weighting, denoising, flip TTA, an alternate 2D-slab
architecture) was measured and rejected — each with its own statistical
gate, delta and confidence interval.

Full tables with every accepted and rejected experiment, their deltas, and
why, are on [the project site](https://alherma7.github.io/DaT_Parkinson/).

## Repository layout

```
src/              production pipeline: config, data loading/resampling, features,
                  dataset, augmentation, model, training, evaluation, submission
submission_src/   what ships inside the competition zip (main.py + copies of src/)
scripts/          scripts/build_submission_assets.py — packages submission_src/
notebooks/        01-27, the full experimental record (EDA through the final
                  coverage-fix retrain), one notebook per gated experiment
tests/            pytest unit tests (TDD throughout, synthetic data only)
docs/
  index.html      project site: results, pipeline, what worked/didn't, lessons
  PROGRESS.md     full dated development log (every decision, gate, bug)
RESOURCES.md      every paper/technique/external asset considered, with license notes
```

## Reproduce

1. Register for the competition and download the training data (this
   repository contains **no scan data**). Create the conda environment from
   `environment.yml` (Python 3.12, PyTorch + CUDA).
2. Run `pytest` — all pipeline code is under test with synthetic data.
3. Run `notebooks/01` through `27` in order to reproduce the experimental
   record, or train directly via `src/train.py` using the settings recorded
   in `src/config.py` (crop size/center, target spacing, production variant
   list).
4. `scripts/build_submission_assets.py` packages `submission_src/` into a
   competition submission zip.

## Data and rules

This repository contains **code, configuration and aggregate metrics only**.
No scans, per-patient labels, per-patient predictions or model weights are
included, per the competition rules. `.pdf` reference papers used during
prior-art review are kept locally only, not committed.

## License

[MIT](LICENSE).
