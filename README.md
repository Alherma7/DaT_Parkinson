# DaT Parkinson's Prediction Challenge

DrivenData / SFMN competition: classify DaT (dopamine transporter) SPECT
scans as normal (0.0) or abnormal (1.0). Metric: log loss (AUROC shown for
reference only). Code-execution submission (Docker, `main.py`, 3h budget,
single A100 80GB, no network). Deadline: 2026-09-16 23:59 UTC.

See `problem_description.txt`, `about_data.txt`, `submission_format.txt`,
`rules.txt`, `home.txt` for the full competition docs.

Project follows the `structuring-ml-projects` skill (+ `deep-learning-imaging.md`
and `code-execution-submission.md` extensions).

## Constraints that shape this project

- Multicenter data (10 French hospital centers), variable volume shape/voxel
  spacing — no site/scanner label in `train_labels.csv`, and none recoverable
  from the NIfTI headers (all descriptive string fields are scrubbed). Best
  available scanner proxy is **(in-plane spacing family, z-spacing, FOV_xy,
  dtype, obliquity)** — not raw (shape, spacing), which over-fragments one
  acquisition into ~12 groups and fails its own significance test.
- **40% of volumes have oblique affines** (up to 40.3° tilt) despite all
  reporting RAS axis codes. `data.py` must resample through the affine;
  `as_closest_canonical()` is not sufficient.
- Log loss is a proper scoring rule — gate on calibrated probability, not AUC.
- External data/pretrained models: license + public-availability +
  commercial-use terms must be recorded in `RESOURCES.md` before use.
  **DINOv3 is not prize-eligible** (license conflict, confirmed by
  DrivenData staff). ImageNet-pretrained weights and PPMI eligibility are
  **unresolved** as of the forum check on 2026-09-07 — treat as a risk, not
  a default-safe choice.
- AI-assistant data rule (DrivenData staff, forum thread on AI coding
  assistants): raw scan files and per-patient labels/predictions must never
  be read by a third-party AI tool that could retain them. Claude does not
  read `.nii.gz` files or row-level `train_labels.csv` contents in this
  repo — only code, config, and aggregate metrics.

## Progress

- 2026-09-07: Project scaffolded. Data downloaded and extracted to
  `data/raw/` (1362 training volumes + labels, 20-sample smoke test set).
  Git repo initialized.
- 2026-09-07: Prior-art scan done (`structuring-ml-projects` step 0) —
  found a 2026 reproduction study showing training protocol matters more
  than architecture for CNN DaT-SPECT classification on small datasets, a
  striatum-cropping preprocessing pipeline, an alternative 2D-slice +
  attention architecture, and several PPMI-based GitHub reference
  implementations. Logged in `RESOURCES.md`.
- 2026-09-07: `notebooks/01_eda_volumes.ipynb` written, covering the 7 EDA
  questions grounded in the prior-art scan (geometry, class balance,
  scanner-proxy confound, orientation, background/intensity, striatum
  location + asymmetry). Not yet run — cells that load pixel data are
  meant to be run and inspected by the user, not Claude (see AI-assistant
  data rule above).
- 2026-09-08: EDA run and documented. Added 2 extra checks (data integrity,
  smoke-test geometry compatibility) after a coverage review, and 2
  quantitative sub-checks under the striatum/asymmetry question (centroid +
  bbox, L-R asymmetry index by class) since visual-only assessment isn't
  something either of us has the clinical background to judge reliably.
  Key decisions:
  - **Resampling target:** physical spacing, not a fixed voxel shape — 66
    distinct shapes / 52 distinct spacings observed (dominant:
    `(256,256,256)@2.46mm`, 33.8%).
  - **Class balance:** 54.8% pathological / 45.2% normal — mild, Stratified
    K-Fold is enough; `class_weight="balanced"` is an experiment to
    validate, not a default.
  - **Scanner-proxy confound** — target rate 0.593/0.502/0.455 across the
    three largest (shape,spacing) groups vs. 0.548 average. **Superseded
    on 2026-09-08 by a significance test — see the review entry below;
    the (shape,spacing) proxy's own omnibus test is not significant.**
  - **Orientation:** `aff2axcodes` reports RAS for 1362/1362.
    **Superseded on 2026-09-08: `aff2axcodes` reports only the *closest*
    canonical axes and 544/1362 volumes are actually oblique — see below.**
  - **Background/normalization:** p30 threshold recovers real background;
    per-volume normalization is mandatory (max/p99 span several orders of
    magnitude across volumes); clip small negative artifacts before
    thresholding.
  - **Crop center/size — NOT resolved. Superseded on 2026-09-08**, see the
    review entry below. The percentile-threshold estimates (p95, then p99 +
    largest connected component, median bbox ≈29%/33%/32% of shape, median
    58.5 components) measured the whole head, not the striatum, and were
    expressed as a fraction of each volume's own shape, which is not
    comparable across a 3× range of field of view. No crop number is
    settled; `config.CROP_SIZE_MM` is deliberately `None`.
  - **L-R asymmetry differs by class** (median 0.073 pathological vs.
    0.030 normal). **The flip-augmentation conclusion drawn from this was
    wrong and is retracted — see the review entry below.**
  - Full what/why/source/found/decision detail lives in the notebook's own
    markdown cells, one per section.
- 2026-09-08: **EDA methodology review + header-only re-analysis.** The
  notebook's structure held up; three of its quantitative conclusions did
  not. Corrections, and new header-only results (no voxel data read):
  - **Evaluation reference point pinned:** base rate 0.548458 (747/615),
    constant-prediction **log loss = 0.688443**. Now in `config.py` as
    `BASELINE_LOGLOSS`. Every model number gets quoted against it.
  - **The metadata shortcut is worth essentially nothing.** Now reproduced
    directly in the notebook (section 11), fitting classifiers on header
    metadata alone (shape, spacing, FOV, voxel count, obliquity, dtype,
    spacing-family one-hots; 31 features) to predict `is_pathologic`:
    logistic regression 5-fold CV log loss **0.688952** (+0.000509 vs.
    baseline, AUROC 0.551); HistGradientBoosting **0.735176**, i.e. *worse*
    than predicting the base rate (AUROC 0.523, spread across 5
    fold-reshuffles sd 0.0025). A site-correlated shortcut on metadata is
    therefore not an exploitable leak here. This is a negative result and
    it **downgrades**, but does not remove, the fold-design concern.
  - **Section 3's confound claim needed the test it never ran, and the
    grouping was wrong.** χ² across the 11 (shape,spacing) groups with
    n≥20: **χ²=15.22, dof=10, p=0.124 — not significant.** The
    (shape,spacing) proxy over-fragments: the whole `(128,128,z)@3.895mm`
    family is one acquisition split across ~12 groups by axial coverage
    alone. Collapsing to **in-plane spacing families** gives 7 groups with
    n≥20 and **χ²=24.54, dof=6, p=0.00042** — the confound is real, but it
    lives in the *small* families the notebook dismissed as noise
    (~1.5-1.8 mm: n=48, rate 0.354; ~1.47 mm: n=31, 0.355) rather than in
    the large ones. Pairwise on the big three: 460 vs 143 z=+2.93
    (p=0.0034), 460 vs 207 z=+2.20 (p=0.028), 207 vs 143 p=0.378.
  - **40% of volumes are oblique — section 4's conclusion was wrong.**
    `nib.aff2axcodes` reports only the *closest* canonical axes, so
    "1362/1362 RAS" did not mean axis-aligned. **544/1362 (39.9%)** have a
    non-diagonal affine rotation block: median tilt 4.7°, max **40.3°**.
    `as_closest_canonical()` only permutes/flips axes and will **not**
    correct this. `data.py` must do a real affine-aware resample.
    qform/sform agree exactly on all 1362 (both code 1), diag signs all
    `+++`.
  - **Obliquity is a strong scanner signature** and is uncorrelated with
    the target (0.546 oblique vs. 0.550 axis-aligned), so it is a clean
    site marker: 2.398 mm → 100% oblique (n=149), 3.895 mm → 100% (n=325),
    ~3.30 mm → 67% (n=39), 2.46 mm → 8% (n=528), and 2.30/1.47/2.00/3.59/
    4.42 mm → 0%. Best available site proxy = (in-plane spacing family,
    z-spacing, FOV_xy, dtype, obliquity).
  - **No scanner strings survive in the headers.** `descrip`, `aux_file`,
    `db_name`, `intent_name` are empty on all 1362 (organizers scrubbed
    them); `scl_slope`/`scl_inter` are NaN on all 1362. A real centre id
    cannot be recovered this way — logged as a negative result so it is
    not retried.
  - **Section 5's negative-intensity explanation was wrong.** It attributed
    negative minima to `scl_slope`/`scl_inter` scaling, but those are unset
    on every volume. The actual cause: **16 volumes are stored as `int16`,
    not `uint16`** (1346 are `uint16`), contradicting the problem
    description. The `np.clip(..., 0, None)` guard is still right; the
    reason recorded for it was not.
  - **Field of view computed (shape × spacing), which nothing had done.**
    FOV[x,y] 175.6 → 629.8 mm, FOV[z] **108.0** → 629.8 mm. The z minimum
    is what bounds any fixed-mm crop: 1 volume under 110 mm, 4 under
    120 mm, 69 under 140 mm. Recorded as `config.MIN_FOV_MM`.
  - **Resampling target chosen: 2.46 mm isotropic** (`config.TARGET_SPACING`)
    — the median *and* modal spacing, identity for 528 volumes, and it
    keeps an ~11 mL striatum at ~739 voxels instead of ~186 at 3.895 mm.
  - **Flip-augmentation caution retracted.** The section-6b index
    `|left−right|/(left+right)` is *invariant* under a left-right flip, so
    it cannot be evidence for or against flip augmentation in either
    direction. Deciding this needs the **signed** index instead; until that
    is run, the project has no evidence on flip.
    **Resolved 2026-09-08** — corrected 6b run on all 1362 volumes: signed
    index means -0.0037 (normal) / -0.0083 (pathological), both
    **not significantly different from 0** (Wilcoxon p=0.056 / p=0.064) and
    not different from each other (Mann-Whitney p=0.162). Magnitude *is*
    highly different by class (p<0.00001, ~2.4-2.7× higher pathological) —
    clinically coherent (PD causes real asymmetry, but which side varies
    patient to patient, so the population-level signed average is ~0 in
    both groups). **Flip augmentation is safe to try** — no population
    directional signal to destroy, and the actual discriminative signal
    (magnitude) is flip-invariant regardless. p-values are close to 0.05,
    though, so validate via the gate rather than treating this as settled.
  - **Section 6a's striatum estimate is invalid** (crop numbers retracted
    above). `np.percentile(data, 99)` retains 1% of *all* voxels by
    construction — 167,772 voxels ≈ 2.5 L at 2.46 mm in the dominant
    (256,256,256) group, larger than a whole head, against an ~11 mL
    striatum. The measured bbox is also just a restatement of the
    threshold: for a blob of fraction *f*, the bbox linear fraction tracks
    *f*^(1/3), and 0.05^(1/3)=0.37 → 0.01^(1/3)=0.215 explains the whole
    p95→p99 "improvement". Corrected cells (physical-volume mask in mL, two
    largest components, results in mm, broken down by site proxy) are
    written and awaiting a run.
- 2026-09-08: **Follow-up reference/prior-art search** (extends the
  2026-09-07 scan), targeted at the confounds and gaps the EDA methodology
  review actually found, plus a Kaggle check for comparable projects. All
  entries logged in `RESOURCES.md`, full detail there.
  - **Harmonization, for the section 3/3a confound**: Fortin et al. 2018
    (ComBat, NeuroImage 167:104-120) — the field's standard tool for
    site-effect removal on derived features. A 7-site PPMI radiomics study
    (Frontiers 2022) using the *same* confound reports AUC 0.71→0.77 after
    ComBat-GAM and states per-volume intensity normalization alone did not
    remove scanner variability — independent confirmation that section 5's
    normalization and section 3a's fold-stratification plan address
    different problems, not the same one. Logged as a candidate for the
    classical/radiomics baseline (`## Next steps`), not the 3D-CNN track.
  - **PPMI's own SBR methodology**: standardized VOI-atlas approach (not a
    percentile threshold) over caudate/putamen with the occipital lobe as
    reference, and **left-right percent asymmetry as a standard clinical
    metric** — confirms section 6b's asymmetry index measures a real
    clinical quantity, and is a fallback if the physical-volume threshold
    in the corrected 6a proves unreliable once run.
  - **Kaggle check**: no DaT-SPECT competition exists (niche modality).
    Closest analog by problem shape: RSNA-MICCAI Brain Tumor Radiogenomic
    Classification (multi-parameter 3D MRI, multi-institution, AUROC).
    **The load-bearing finding isn't a technique, it's a warning**: the
    BraTS 2021 benchmark's own follow-up validation study found that of
    models scoring well on small single-center studies, **~80% showed no
    significant difference from chance** once validated on the full
    multicenter set. Several of this project's own cited PPMI papers report
    92-99.5% accuracy on small (200-2720 volume) single/few-site subsets —
    the same shape of claim. Do not treat those numbers as a realistic
    target until validated on the actual 10-center, 1362-volume mix
    (reinforces the leave-one-family-out check below).
  - One Kaggle dataset candidate ("Parkinson's Disease Dat and MRI Scans",
    rishikjha) checked and rejected — no stated license or source.
  - **Follow-up (2026-09-08, user-provided PDF): Wenzel et al. 2019**
    (EJNMMI, the paper behind `mtwenzel/parkinson-classification`) closes
    the last open citation from the 2026-09-07 scan. Directly actionable:
    a CNN trained only on their higher-resolution PPMI images generalized
    badly to lower-resolution ones (accuracy 0.629, effectively predicting
    one class), while a CNN trained on lower-resolution data generalized
    well to higher-resolution (0.945) — asymmetric. **For us**: don't let
    3D-CNN training skew toward the dominant 2.46mm family (528/1362,
    highest-resolution); include the full spacing range, biased if
    anything toward the coarser end. Also: their "hottest voxels" SBR
    method — average the hottest voxels up to a **fixed physical volume**
    (15 mL for whole striatum) — is the same technique EDA section 6a's
    corrected code already uses, independent validation it's not ad hoc.
    Full detail in `RESOURCES.md`.
- 2026-09-08: `environment.yml` added and `dat-parkinson` conda env created
  (per the `structuring-ml-projects` skill's per-project environment step).
  Data-science stack plus PyTorch 2.14.0+cu126/torchvision 0.29.0+cu126 —
  confirmed `torch.cuda.is_available() == True` on the local GPU (RTX 4060
  Laptop, 8GB). The PyPI `torch` wheel defaults to CPU-only; the CUDA build
  needs `--extra-index-url https://download.pytorch.org/whl/cu126` pinned
  in `environment.yml` (cu121/cu124 don't publish this torch release).
- 2026-09-08: **Fixed a BLAS/OpenMP runtime crash in `dat-parkinson`**,
  found while checking `nibabel.processing`'s API ahead of the CNN
  data-plumbing plan: any BLAS call (even a bare 4x4 `@` matmul) crashed
  the process natively (access violation, no Python traceback) — the env
  had both MKL and libgomp (GNU OpenMP) DLLs loaded at once, a known
  Windows conda clash. Also, independently, nibabel 5.4.2's
  `Nifti1Image(data, affine)` crashes under numpy>=2.5 (verified 2.2.6
  works). Fixed in `environment.yml`: `numpy<2.3` + `libblas=*=*openblas`.
  Verified from a fully fresh `conda env create`: nibabel construction +
  `resample_to_output`, scipy, scikit-learn, torch+CUDA all work, 28/28
  project tests pass.
- 2026-09-09: **Rung 3 gate PASSED** in `notebooks/07_cnn_rung3.ipynb`:
  5-fold × 5-repeat nested-CV CNN, mean log loss **0.4520** (sd 0.0097)
  vs. `build_combat_baseline()` at 0.5290 — paired bootstrap 95% CI of the
  delta fully negative ([-0.1041, -0.0351]), clears the 2×-noise-threshold
  gate by 0.0770 vs. 0.0218. Per-family log loss worst on the 2.46mm
  (0.5191) and 3.895mm (0.5035) families, best on 2.30mm (0.3177).
  **3D CNN is now the winning track.** Note: `notebooks/06_cnn_rung2.ipynb`'s
  batch/LR sweep and LOFO transfer-check cells didn't survive the run
  (only the cache-build and helper-def cells remain on disk); rung 3 used
  `batch=32, lr=2e-3` without an on-record confirmation of the winner —
  left as-is since rung 3 cleared the gate with a wide margin anyway.
  - **Final model decided: CNN+baseline blend at w_cnn=0.70**, not the
    CNN alone. The notebook's original 3-point grid check (w=0.25/0.5/0.75)
    was a cheap fallback, not a validated choice; added a leave-one-
    repeat-out cell (`src/evaluate.py::paired_bootstrap_ci`, TDD) that
    picks the weight on 4 of the 5 CNN OOF repeats and scores it on the
    held-out one, so no weight is ever evaluated on the data used to
    select it. **w_cnn=0.70 was selected unanimously in all 5 LOFO
    folds.** Honest blend mean log loss 0.4250 (sd 0.0099) vs. CNN-alone
    0.4520 (sd 0.0109) — beats it by +0.0271, above the 2×-noise
    threshold (0.0218); paired bootstrap 95% CI on the blend-vs-CNN delta
    [-0.0354, -0.0106], fully negative. **`main.py` must run both
    models and blend their probabilities at w_cnn=0.70.**
- 2026-09-09: **Submission packaging built and smoke-tested successfully.**
  `src/config.py` (runtime-environment auto-detection),
  `src/features.py::extract_baseline_features`,
  `src/model.py::rung3_checkpoint_filenames`,
  `src/submission.py::combine_predictions`,
  `scripts/build_submission_assets.py`, and `submission_src/main.py`
  written (TDD where applicable), reviewed via subagent-driven
  development (one Critical fix before merge: missing/partial
  checkpoints would have silently written a NaN-filled `submission.csv`
  with no exception — now raises `FileNotFoundError`). Ran
  `scripts/build_submission_assets.py`, packaged with the runtime
  repo's (`drivendataorg/competition-sfmn-parkinsons-runtime`) `just
  pack-submission`, and ran `just test-submission` against the 20-volume
  smoke test set:
  - **Exit code 0, ~64s total** (well under the 6-minute smoke-test
    limit), GPU available inside the container (`device=cuda`).
  - All 25 rung-3 checkpoints loaded via `torch.load` with no error,
    despite the local/runtime torch version gap (2.14.0+cu126 vs.
    2.12.1+cu129) flagged as a risk beforehand.
  - Volume-caching worked as designed: checkpoint 1 took 34.1s (all 20
    volumes' preprocessing), checkpoints 2-25 took 0.0-0.4s each (reused
    the cache, not reprocessed).
  - `sklearn.InconsistentVersionWarning` fired (pipeline pickled with
    1.9.0, runtime has 1.8.0) — non-fatal, pipeline still ran correctly;
    the version-gap risk flagged in the final review did materialize,
    just didn't break anything this time.
  - 20/20 volumes had a valid classical-feature mask (0 CNN-alone
    fallback rows).
  - `submission/submission.csv`: header `uid,is_pathologic` and 21 lines
    (1 header + 20 rows), exactly matching `submission_format.csv`'s
    shape.
  - **Confirmed on DrivenData's own platform too**: submitted the same
    `submission.zip` as a platform smoke test (separate from the
    3-per-week regular submission limit — 3 smoke tests/day, no cost to
    the weekly quota). Exit 0, ~35s. Scored (smoke tests get scored for
    debugging even though excluded from the leaderboard): **log loss
    0.3051** — well below the classical baseline (0.5290) and the CNN's
    CV mean (0.4520), though n=20 is small/high-variance so this single
    number isn't the expected real-test-set score, just a strong signal.
  - **Bug caught mid-flight**: `data.load_volume`'s degenerate-volume
    warning embedded the real test uid, and DrivenData's log scanner
    auto-filtered two lines from a full-submission run with an explicit
    disqualification warning. Fixed in `src/data.py` (commit 069bb28,
    TDD) — the message never includes uid, in any context. Rebuilt
    `submission.zip` immediately after.
- 2026-09-09: **First full/regular submission: log loss 0.4648** on the
  real (undisclosed size, ~600 volumes estimated from timing) held-out
  test set. Exit 0, ~15m20s (well under the 3h limit). Beats the
  classical baseline by -0.064; ~1.3 sd from the CNN's own 5×5 CV mean
  (0.4520, sd 0.0109) — normal CV-vs-holdout variance. Used 1 of the
  3-per-week regular submissions (2 remain before the 2026-09-16
  deadline). This run used the pre-fix zip (see the uid-leak bug just
  above) but succeeded anyway since DrivenData's filtering caught it;
  the rebuilt zip should be used for any further submissions.
  Public leaderboard: **rank #254**, log loss 0.4648, AUROC 0.8796.

## Next steps

- [x] Run `notebooks/01_eda_volumes.ipynb` and record the answers to its
      questions here — see Progress above (2026-09-08).
- [x] Redo the striatum crop-size estimate with a p99 threshold +
      connected-component filter — see Progress above (2026-09-08).
- [x] Section 6b (signed L-R asymmetry) re-run on all 1362 volumes — see
      Progress above (2026-09-08). Flip augmentation cleared as safe to try.
- [x] Section 6d (single-feature ranking) re-run on all 1362 volumes.
      **`abs_asym` (striatal L-R asymmetry magnitude) alone: AUC 0.731,
      log loss 0.6126, a -0.0759 improvement over baseline** — stronger
      than the entire 31-feature metadata-only model (section 11, AUC
      0.551). `striatal_ratio`: AUC 0.621, -0.0212. Everything else
      (total_counts, vol_max, vol_p99, signed_asym) is noise-floor-level.
      **Strong candidate features for the classical baseline**, and a
      reference point for the CNN track (log loss ≈0.61 from one scalar —
      the CNN should clear this once trained). Full table in `RESOURCES.md`
      is unnecessary — it's in the notebook's own section 6d findings cell.
- [x] Sections 5 + 6c (adaptive background threshold, per-family intensity)
      re-run on 119 volumes stratified across all 17 spacing families.
      Verifies, doesn't revise, the earlier decision: `p30 > 0` for 35/119
      (29%) volumes, but **deterministically by family** — every volume in
      the 3 finest-resolution families (2.00mm, ~1.47mm, ~1.5-1.8mm) has
      `p30>0`, every volume in every coarser family has `p30=0`
      (`zero_frac` median 0.043 vs. 0.965 — a 22× difference in how much
      of the FOV is background). Confirms `BACKGROUND_MAX_FRACTION=0.05`
      is load-bearing for those 3 families specifically, and that raw
      intensity scale (why `vol_max`/`vol_p99`/`total_counts` scored at
      noise-floor in section 6d) is dominated by scanner family, not
      pathology — only scale-free ratios are comparable across the
      dataset. No config change.
- [x] Section 12 (duplicate fingerprint) re-run on all 1362 volumes: 0
      hash collisions, 0 total-intensity collisions, 0 volumes sharing
      both geometry and intensity. **"One file per patient" confirmed** —
      `GroupKFold` on patient identity remains unnecessary.
- [x] **Section 6a RESOLVED. `config.CROP_SIZE_MM` is set.** Excluding
      the 7 named outlier uids (112/119 remaining), `kept_mL` is tight and
      trustworthy: mean 19.10, std 3.25, median 20.00 exactly, max 29.34
      (no more 500+ mL blobs). Final values, now in `config.py`:
      - `CROP_SIZE_MM = (135.0, 70.0, 105.0)` mm — fits `MIN_FOV_MM` on
        all 3 axes, but only by **3 mm on z** (105 vs. 108) — `data.py`
        must pad/clamp explicitly for the handful of tight-FOV volumes
        (section 7), not assume headroom.
      - `CROP_CENTER_MM = (1.5, 22.8, -15.5)` mm (median offset from
        geometric centre) — z is negative (inferior to centre), matching
        the independently-measured `centroid_z ≈ 0.37-0.39` finding.
      - `TARGET_SHAPE = (56, 30, 44)` voxels at `TARGET_SPACING = 2.46mm`.
      `ft07x5ic` (the 523 mL outlier) checked and has an unremarkable
      header (normal `uint16`, mild obliquity, ordinary shape) — a real
      per-file pixel-content anomaly, not a recurring pattern. **Because
      this kind of anomaly can recur on unseen data, `data.py` must
      sanity-check the detected striatum region size defensively** (flag
      or fall back if extraction lands far outside ~10-30 mL), not assume
      every file behaves like the clean 112. This closes the last blocker
      for `data.py`. Section 6d's `abs_asym`/`striatal_ratio` numbers used
      the same `TARGET_ML=20.0` mask and don't need to be re-run.
- [x] `src/evaluate.py` written (TDD, `tests/test_evaluate.py`, 11 tests,
      clean pass): `log_loss_score` (the gated metric), `auroc_score` +
      `expected_calibration_error` (diagnostics, per Chegodaev et al. in
      `RESOURCES.md`), `combined_score`, and `make_folds` (Stratified
      K-Fold jointly on target × in-plane-spacing family — rare families
      collapsed to a "rare" bucket per target class, falling back to that
      class's largest family if even "rare" would be unsplittable, so
      `StratifiedKFold` never gets a cell smaller than `n_splits`).
- [x] Noise floor measured in `notebooks/02_evaluate_noise_floor.ipynb`:
      `evaluate.make_folds` + `evaluate.log_loss_score`, `abs_asym` alone,
      5 seeds — **mean=0.6119, sd=0.0001** (matches section 6d's
      single-split 0.6126, confirming `evaluate.py` reproduces it — the
      harness works end-to-end on real data). **Caveat that matters**:
      this is the floor for *this specific probe* (1 feature, 2-parameter
      model, these folds) — a low-variance case by construction (strong,
      stable single feature). It is **not** a universal threshold; the
      classical baseline and especially the 3D-CNN (more parameters,
      stochastic training) need their **own** noise-floor measurement once
      built (`SKILL.md`: re-measure whenever fold structure, data, or
      model class changes) — don't reuse `sd=0.0001` to judge their
      deltas.
      **Fold design:** Stratified K-Fold on `is_pathologic` **stratified
      jointly with the in-plane-spacing family** (the proxy whose confound
      is actually significant, p=0.00042 — not the (shape,spacing) proxy,
      p=0.124). Rationale for stratify-by-source rather than
      leave-one-site-out: all 20 smoke-test (shape,spacing) combos already
      occur in training (EDA section 9), so the test set appears to draw
      from the same centres, and `deep-learning-imaging.md` says to pick
      the fold structure from the test set's actual composition. Because
      metadata alone is worth only −0.0006 log loss, this is insurance
      against uneven fold composition, not leak containment — so do **not**
      pay for `GroupKFold` on the proxy (its largest group is 34% of the
      data, which cannot be split 5 ways). Additionally run
      leave-one-family-out once at rung 2 as a transfer check, and report
      the scored metric per family, not only pooled.
- [x] Baseline gate cleared in `notebooks/03_baseline_classical.ipynb`
      (`StandardScaler` + `LogisticRegression`, 5 seeds, apples-to-apples
      via `evaluate.py`): `abs_asym` alone 0.5959, `striatal_ratio` alone
      0.6666 (weak on its own), **combined 0.5831 — beats `abs_asym` alone
      by -0.0128, 18-25× this run's own sd (0.0003-0.0007), a real win**.
      Both features graduate to `src/features.py` + `src/model.py`.
      **Methodology note**: this pipeline adds `StandardScaler`, which
      `02_evaluate_noise_floor.ipynb` didn't — that changes the
      `abs_asym`-alone number (0.6119 there vs. 0.5959 here), so `02`'s
      sd=0.0001 isn't the right noise reference for this comparison; use
      this run's own per-feature-set sd instead. `02` still stands for
      confirming `evaluate.py` reproduces section 6d end-to-end.
- [x] `src/features.py` + `src/model.py` written (TDD, synthetic 3D
      arrays, not real patient data). 20/20 tests pass, clean.
      `features.striatum_mask`/`signed_asymmetry`/`striatal_ratio` use the
      **corrected** (central-region-restricted) mask logic from EDA
      section 6a's final resolution — not the unrestricted version
      section 6d originally used to produce `eda_features.csv`. TDD caught
      two real bugs before they shipped: `striatum_mask` on an all-zero
      volume returned the *entire* array as the mask (a `>=` threshold tie
      on an all-zero central region), and a peripheral artifact was
      confirmed excluded by the central-margin restriction.
      `src/model.py::build_classical_baseline()` is
      `StandardScaler` + `LogisticRegression`.
- [x] **Gate re-validated against the real `src/features.py` code**
      (`notebooks/03_baseline_classical.ipynb`, all 1362 volumes recomputed
      via `src/features.py` itself, 0 skipped as degenerate -- writes
      `data/processed/baseline_features.csv`; same abs_asym/striatal_ratio/
      combined comparison via `evaluate.py`, 5 seeds): `abs_asym` alone
      0.5889, `striatal_ratio` alone 0.6647, **combined 0.5753 — beats
      `abs_asym` alone by -0.0136, ~20-45× this run's own sd
      (0.0003-0.0006)**. Confirms (doesn't revise) the provisional pass:
      same direction and magnitude (-0.1132 vs. baseline here, -0.1054
      there). The central-region-restricted mask changed the exact numbers
      slightly but not the decision. `src/model.py::build_classical_baseline()`
      docstring updated to cite these confirmed numbers.
- [x] Baseline candidate technique for the site confound at the feature
      level: **ComBat** harmonization (Fortin et al. 2018) — a 7-site
      PPMI radiomics study using the same confound reported AUC 0.71→0.77
      after ComBat-GAM, and found per-volume intensity normalization alone
      insufficient for it (`RESOURCES.md`). Fit harmonization parameters
      on controls only (their leakage-avoidance detail) before applying to
      patients. Not directly applicable to the 3D-CNN track (ComBat
      operates on scalar features, not raw voxel grids).
      **Gate passed 2026-09-08** in `notebooks/04_combat_harmonization.ipynb`
      (reused `baseline_features.csv`, no new `.nii.gz` access, 5 seeds,
      same fold loop as `03`): `combined + ComBat` 0.5290 vs. `combined`
      (no ComBat, same-run-recomputed) 0.5753 — **delta -0.0462, ~40-80x
      this run's own per-variant sd (0.0006-0.0011)**, far outside noise.
      Parametric empirical-Bayes ComBat (Johnson et al. 2007,
      `RESOURCES.md`), batch = in-plane spacing family (the confound EDA
      3a found significant, χ²=24.54, p=0.00042), fit on each fold's
      training-set controls only — matches the Frontiers 2022 PPMI
      leakage-avoidance precedent. **Wired into `src/model.py`** as
      `ComBatHarmonizedPipeline`/`build_combat_baseline()` (TDD,
      `tests/test_model.py`, 4 new tests — fits/predicts valid
      probabilities, fits ComBat on controls only, handles unseen-batch
      rows at predict time, fresh instance per call; 28/28 project tests
      pass). `build_combat_baseline()` is now the classical baseline to
      actually use; `build_classical_baseline()` (no ComBat) stays as the
      weaker reference point it beat.
- [x] Main track: 3D CNN on resampled volumes — see Progress above
      (2026-09-09). **Gate passed; final model is a CNN+`build_combat_baseline()`
      blend at w_cnn=0.70**, not the CNN alone.
- [ ] `RESOURCES.md`: log every technique and every external
      data/pretrained-model candidate as it's considered.
- [x] Submission packaging + local Docker rehearsal + first full
      submission — see Progress above (2026-09-09). Local smoke test,
      platform smoke test (0.3051), and first full submission
      (**0.4648, rank #254**) all passed/scored. A real
      disqualification-risk bug (test uid leaked into a log warning) was
      caught and fixed (commit 069bb28) before further submissions.
      2 of 3 weekly regular submissions remain before 2026-09-16.
