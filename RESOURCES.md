# Resources

Every technique and every external data/pretrained-model candidate used in
this project gets an entry here before it's used, per the
`structuring-ml-projects` skill.

## Papers

- **Booij et al. 1998** (J Nucl Med, PMID 9829575) — DAT SPECT imaging in
  healthy controls vs. Parkinson's disease.
  Why: clinical background on what DAT uptake asymmetry/reduction means.
- **Nazari et al. 2022** (EJNMMI, https://doi.org/10.1007/s00259-021-05569-9)
  — explainable AI / CNN classification of DAT SPECT for parkinsonian
  syndromes.
  Why: precedent for CNN-based classification on this exact modality.
- **Zhao et al. 2022** (EJNMMI, https://doi.org/10.1007/s00259-022-05804-x)
  — deep learning DAT imaging for parkinsonism differential diagnosis.
  Why: precedent for the 3D-CNN main-track approach.
- **Quan et al. 2019** (arXiv, https://arxiv.org/abs/1909.04142) — DaTscan
  SPECT image classification for Parkinson's disease.
  Why: additional prior-approach reference, cited by the organizers.
- **Chegodaev et al., "Revisiting CNN-Based Parkinson's Disease Classification
  from DaT-SPECT Images: The Role of Training Protocols"** (MDPI Electronics
  2026, 15(9):1883, CC BY, https://www.mdpi.com/2079-9292/15/9/1883) — full
  text read. Two PPMI-derived 2D-slice datasets (645 and 1528 images),
  4 CNN backbones (VGG16/ResNet50/InceptionV3/Xception), 3 protocols
  (baseline reproduction vs. optimized+transfer-learning vs.
  optimized-no-transfer-learning).
  Why, concrete takeaways for this project:
  - **Training protocol (preprocessing, weight decay, LR schedule) mattered
    more than architecture or ensembling.** A single optimized model matched
    or beat the literature's ensemble baseline (98.45%) — don't default to
    a fold ensemble for a score bump; validate it against the gate like
    anything else (step 6), same as this repo's own
    Financial_Stress_Prediction lesson on ensembling not being automatically
    beneficial.
  - **Concrete protocol values to start from**: Adam, weight decay (L2)
    1e-5, cosine annealing with warm restarts (T0=50, Tmult=1,
    eta_min=1e-6), model selection = best validation-metric checkpoint
    (not last epoch — already in `deep-learning-imaging.md`).
  - **Transfer learning (ImageNet) was empirically unreliable on the small
    dataset**: "preliminary experiments showed that pretrained ImageNet
    weights produced unstable convergence and, in some cases, inferior
    results compared to random initialization... pretrained feature priors
    likely did not align with SPECT intensity distributions." On the larger
    dataset its effect was inconsistent (helped VGG16, hurt Xception). This
    is an independent, non-license reason (beyond the eligibility question
    already logged below) to default to training from scratch and treat
    ImageNet init as an experiment to validate, not a default.
  - **Calibration/ECE was evaluated explicitly** (Guo et al. 2017 method,
    10 confidence bins) alongside accuracy/AUROC — directly relevant since
    our metric is log loss (a calibration-sensitive proper scoring rule);
    add an ECE/reliability-diagram check to `evaluate.py`'s diagnostics,
    not just log loss/AUROC.
  - **Grad-CAM was used as the qualitative sanity check**: confirmed the
    network attended to putamen/caudate (anatomically meaningful) rather
    than background — this is a concrete implementation of this skill's
    gate-step qualitative check (`SKILL.md` step 6) for this project;
    plan to run it once a model clears the quantitative gate.
  - **Preprocessing**: crop background, resize to a fixed size, center-crop
    to keep the central anatomical region (striatum) and reduce peripheral
    noise/spurious correlation from unrelated regions.
  - **Augmentation used in the reproduced baseline** (from the original
    Kurmi et al. protocol): central crop, **random horizontal flip**,
    rotation ±10°, brightness jitter [0.1, 1.5] — justified by the authors
    as "dopaminergic deficit patterns are clinically relevant regardless of
    hemispheric side." This **contradicts our earlier flip caution** and
    shows the literature itself disagrees on this point — treat flip as an
    experiment to validate via the gate (does it help or hurt our actual
    log loss?), not something to assume either way from first principles.
  - **Important scope limitation**: this entire study is 2D-slice-based
    (single or averaged axial slices), explicitly not 3D volumes — "future
    work will extend to 3D and multi-slice inputs" and no external
    multi-center validation was performed. Our competition gives full 3D
    volumes across 10 real centers, so this paper's protocol values are a
    starting point to adapt, not a directly reusable pipeline, and our
    multi-site generalization work has less direct precedent here.
  - **Table 1 of this paper is a literature survey** of ~15 prior DaT-SPECT
    classification methods (SVM/logistic-regression/CNN variants, 92-99%
    accuracy range). Two entries worth following up given more time:
    Martínez-Murcia et al. 2017 (a genuine **3D** CNN on PPMI, 95.5%
    accuracy — closer analog to our task than any 2D-slice paper) and
    Wenzel et al. 2019 (EJNMMI, "deep CNNs trained to be robust to variable
    image characteristics" — directly on-topic for our multi-scanner
    generalization concern; likely the paper behind the
    `mtwenzel/parkinson-classification` GitHub repo already listed below).
- **Boulkrinat, Taourirt & Saada, "Parkinson's disease detection based on 3D
  Convolutional Neural Network and DaTscan imaging"** (Procedia Computer
  Science 272, 2025, CC BY-NC-ND, https://www.sciencedirect.com/science/
  article/pii/S187705092503563X) — full text read. 730 PPMI DaTscan exams
  (355 HC, 375 PD scans), volumes 91×109×91.
  License note: CC BY-NC-ND restricts redistributing *the article itself*,
  not implementing the described method — methodology/algorithms are not
  copyrightable, only their specific expression is, so reimplementing this
  preprocessing pipeline and citing the paper is standard practice, distinct
  from the "external data/pretrained model" licensing question below.
  Why, concrete takeaways:
  - **Full preprocessing pipeline, in order**: (1) DICOM→NIfTI conversion
    (n/a for us, we already have NIfTI); (2) background intensity removal —
    threshold at the volume's 30th percentile, zero everything below (cites
    Martínez-Murcia et al. for improved CNN stability); (3) patch-wise
    Non-Local-Means denoising (`estimate_sigma` + `denoise_nl_means`,
    `patch_size=3`, `patch_distance=5`) — preserves edges/fine structure
    better than median/Gaussian filters; (4) Z-score normalization; (5)
    crop centered on the striatum, computed **proportionally to each
    volume's own shape** (not a fixed atlas coordinate) — e.g. their
    91×109×91 volumes cropped to 45×54×14.
    **Caveat for us**: their proportional crop is tuned to PPMI's fixed
    91×109×91 shape. Our volumes vary in both shape *and* voxel spacing
    across the 10 centers, so the crop needs to be defined in physical
    (mm) space after resampling to a common spacing, not as a fixed
    voxel-proportional box — confirm the striatum's typical physical
    location/size against our own EDA before fixing the crop size.
  - **A small from-scratch 3D CNN beat pretrained 3D architectures on a
    dataset our size.** Architecture: Conv3D(32,3×3×3)+MaxPool3D →
    Conv3D(64)+MaxPool3D → Conv3D(128)+MaxPool3D → GlobalMaxPool3D →
    Dense(64, ReLU)+Dropout(0.5) → Dense(1, sigmoid). Beat 3D
    ResNet-18/DenseNet121/VGG/I3D/S3D/MedicalNet (all pretrained or larger)
    on accuracy (99.53% vs. 84-98%) *and* training time (5 min vs.
    15-45 min). Independent confirmation (alongside the MDPI paper above
    and `PPMI_DL`) that a simple from-scratch 3D CNN is a strong, cheap
    starting architecture for this data scale — don't reach for a heavy
    pretrained 3D backbone by default.
  - **Training recipe**: 50/20/30 train/val/test split with class
    balancing; two-stage augmentation — *offline* (before training: random
    axis symmetries + Gaussian noise, 4× expansion of the training set) and
    *online* (during training: flip, noise, contrast); `class_weight=
    "balanced"`; Adam, LR 3e-4, binary cross-entropy, 20 epochs, batch
    size 8. Concrete starting point for our own `config.py` DL defaults,
    to be validated against our own noise floor rather than copied as-is.
  - Table 1 of this paper is a second, complementary literature survey
    (partially overlapping the MDPI paper's Table 1) — notably Nazari et
    al. 2022 (already cited above, 3D CNN + LRP explainability, 95.8% on
    200 PPMI scans, highlights the putamen) and Khachnaoui et al. 2023
    (pretrained EfficientNet-B0+MobileNet-V2 bilinear fusion, 99.14% on
    2720 PPMI DaTscans) as further reference points if more depth is
    needed later.
- **Multiclass PD-stage 3D-brain-image CNN study** (Springer JIIM, 2025,
  https://link.springer.com/article/10.1007/s10278-025-01402-z) — treats
  the volume as a sequence of 2D slices fed to a 2D CNN (ImageNet-
  pretrained) or a 3D CNN (Kinetics-400-pretrained), with attention over
  slice importance.
  Why: alternative architecture family (2D-slice + attention) to the
  volumetric 3D-CNN main track; both cited pretraining sources carry their
  own license questions (see `## Pretrained models & external data` below)
  and would need their own eligibility check before use, same as ImageNet.
- **Fortin et al. 2018, "Harmonization of cortical thickness measurements
  across scanners and sites"** (NeuroImage 167:104-120) — introduces
  ComBat harmonization (adapted from genomics batch-effect correction) to
  neuroimaging: removes additive/multiplicative site effects from
  image-derived features via empirical Bayes while preserving biological
  variance.
  Why: this project's own EDA (`01_eda_volumes.ipynb` section 3/3a) found a
  real target-rate confound tied to acquisition family, and section 11
  found it's not exploitable via metadata alone but per-volume z-score
  normalization (section 5) only addresses intensity *scale*, not a
  site-correlated shift in derived *features* — ComBat is the standard
  tool for the latter, candidate for the classical/radiomics baseline
  track (operates on scalar features, not raw voxel grids, so not
  directly applicable to the 3D-CNN track without a different approach).
- **"The impact of harmonization on radiomic features in Parkinson's
  disease and healthy controls: A multicenter study"** (Frontiers in
  Neuroscience, 2022, https://www.frontiersin.org/articles/10.3389/fnins.2022.1012287/full)
  — 7 PPMI sites (Philips/GE/Siemens, 1.5-3T), 88 radiomic features from
  bilateral caudate/putamen/thalamus, ComBat-GAM harmonization via
  NeuroHarmonize.
  Why: closest direct precedent for our own confound problem — same PPMI
  multi-site setup, explicitly reports **AUC 0.71 → 0.77 after
  harmonization**, and states plainly that "intensity normalization alone
  proved insufficient for eliminating scanner-related variability" —
  independent confirmation that section 5's per-volume normalization is
  necessary but not sufficient for the section 3/3a confound. Also notes a
  leakage-avoidance detail worth replicating: harmonization parameters
  fit on controls only, before applying to patients.
- **Johnson, Li & Rabinovic 2007, "Adjusting batch effects in microarray
  expression data using empirical Bayes methods"** (Biostatistics 8(1):
  118-127) — the original ComBat algorithm (genomics) that Fortin et al.
  2018 adapts to neuroimaging: standardize per-feature, estimate naive
  per-batch location/scale on the standardized data, shrink those toward
  an across-batch empirical-Bayes prior (method-of-moments normal/inverse-
  gamma), then adjust.
  Why: this is the actual algorithm implemented in
  `src/features.py::fit_combat`/`apply_combat` (parametric empirical
  Bayes, no other covariates besides batch) — Fortin's paper motivates
  *why* to use it here, this paper is *how* it's computed.
- **Wenzel et al. 2019, "Automatic classification of dopamine transporter
  SPECT: deep convolutional neural networks can be trained to be robust
  with respect to variable image characteristics"** (EJNMMI 46(13):2800-2811,
  DOI 10.1007/s00259-019-04502-5) — full text read (user-provided PDF,
  paywalled, no open-access copy found). 645 PPMI FP-CIT SPECT (207 HC,
  438 PD) plus an independent clinical sample of 298 patients (2 different
  reconstruction algorithms). Directly on-topic: this is the paper behind
  the `mtwenzel/parkinson-classification` GitHub repo already listed above.
  Why, concrete takeaways:
  - **A CNN is measurably more robust to heterogeneous image
    characteristics than semi-quantitative SBR, but only if trained on
    that heterogeneity — and the direction of training matters.** They
    simulated site variability via 18-mm Gaussian smoothing, then trained
    separately on original, smoothed, and mixed (both) data. SBR accuracy
    dropped significantly in the mixed setting (e.g. AAL-SBR 0.957→0.900,
    p=0.004); CNN accuracy did not (0.972 vs. 0.967, ns). Confirmed in the
    independent clinical sample (2 reconstruction algorithms) and by
    applying the PPMI-mixed-trained CNN directly to that clinical sample.
    **Cross-site test was asymmetric**: CNN trained on original
    (high-resolution) data generalized badly to smoothed data (accuracy
    0.629 — sensitivity 1.000/specificity 0.259, i.e. it just predicted
    the positive class); CNN trained on smoothed (low-resolution) data
    generalized well to original data (accuracy 0.945). **Actionable for
    us**: if the 3D-CNN track is trained on a subset skewed toward the
    dominant `2.46mm` family (528/1362, the highest-resolution group per
    EDA section 7), it risks generalizing badly to the coarser families
    (`3.895mm`, 325 volumes) — training should include the full spacing
    range actually present, not just the dominant/cleanest group, and if
    anything bias exposure toward the coarser end during training.
  - **Their "hottest voxels" (HV) semi-quantification is methodologically
    what this project's corrected EDA section 6a already does.** HV-SBR
    averages the hottest voxels within a large ROI up to a **fixed
    physical volume** (5/10/15 mL for unilateral caudate/putamen/whole
    striatum) rather than a fixed voxel count or percentile — this is an
    independent, published validation of the "keep the top-N voxels by
    physical mL" approach used in section 6a's corrected code, not an
    ad hoc invention. Confirms the ~15-25 mL target range used there is
    in the right ballpark (their whole-striatum HV target is 15 mL).
  - **Normalization detail worth comparing against section 5's per-volume
    z-score**: they scale voxel intensity to the 75th percentile of a
    reference region (whole brain excluding striata/thalamus/brainstem/
    ventricles), not a global z-score. A candidate refinement for
    `config.INTENSITY_NORM` if z-score normalization underperforms once
    validated.
  - **Third independent confirmation that ImageNet transfer learning
    underperforms on this modality** (after Chegodaev et al. and their own
    citation of Kim et al. 2018, who got only 0.844 accuracy via
    InceptionV3 transfer learning): validation loss rose while training
    loss converged in their own transfer-learning experiments —
    overfitting, attributed to a resolution/domain mismatch between
    natural images and low-resolution FP-CIT SPECT. Reinforces training
    from scratch as the default, ImageNet init as an experiment to
    validate, not assume.
  - **Architecture used**: 2D CNN on thick axial "slab" views through the
    striatum (2×2×12 mm³, not full 3D volumes) — 4 conv blocks
    (64/64/96/128 3×3 filters, batch norm, 2×2 max pool) → flatten → 1024
    dense → 2-unit softmax, 2,872,642 parameters, Adam, categorical
    cross-entropy, best-validation-checkpoint kept (same model-selection
    rule as `deep-learning-imaging.md`). **Implemented 2026-09-11** as
    `model.DatSlab2DCNN`/`data.load_slab` (roadmap item 6, architecture
    diversity, gated in `notebooks/24_slab2d_architecture_diversity.ipynb`)
    — two deliberate deviations logged in the class docstring
    (`AdaptiveAvgPool2d` instead of a fixed flatten size; single-logit
    `BCEWithLogitsLoss` instead of the 2-unit softmax head). **Geometry bug
    found and fixed same day**: the initial implementation took Wenzel's
    "12mm slab" as one literal 12mm-thick resampled voxel, which made
    `data.py::crop_or_pad`'s nearest-voxel rounding (±6mm at that spacing)
    large enough to miss the striatum outright — a fold-0 sanity check
    scoring barely above the base-rate baseline flagged it, and a
    synthetic-marker diagnostic confirmed the miss before any fix was
    attempted (systematic-debugging discipline). Fixed by resampling to
    2mm z-spacing and averaging 6 voxels (same 12mm total thickness,
    ±1mm rounding error) — see `config.py`'s comment above
    `SLAB_TARGET_SPACING` for the full diagnosis. **Gated result (same
    day, after the fix): NOT adopted — but a SECOND Opus review
    (2026-09-11) found the negative result is not decisive and should
    NOT be read as "this architecture doesn't work here."** 5×5
    nested-CV pooled log loss mean=0.6765 sd=0.0048 (worse than a single
    hand-crafted scalar, `abs_asym` alone at 0.5889 — the strongest tell
    that the crop mostly isn't landing on real striatal tissue); as a 7th
    ensemble member (diluted to a fixed, unlearned 1/7 weight via
    `submission.pool_logit_mean`, not tested as its own blend feature),
    `evaluate.paired_gate` gave delta=+0.0013, 95% CI=[+0.0001, +0.0025].
    **Root cause per the second review**: `CROP_CENTER_MM` is a
    *population-median* offset (notebooks/01 section 6a) with z sd
    ~32.5mm — a 12mm-thick window fixed at that one point contains real
    striatal tissue for only ~14-47% of subjects (vs. ~89% for the 3D
    track's 108mm-thick crop, which tolerates the same offset error by
    sheer thickness); the rounding-precision fix (±6mm→±1mm) was real but
    two orders of magnitude smaller than this actual error term, which is
    why the fold-0 sanity check barely moved after the fix (0.6786→0.6791).
    The tight gate CI reflects a *consistently uninformative* input, not a
    validated negative — this design presumes spatial normalization to a
    common frame (as Wenzel's own PPMI data has), which this pipeline
    never performs. **Untested and possibly valuable**: per-subject
    centering via `features.striatum_mask` instead of the fixed population
    median (never attempted — would need a real retrain to evaluate); also
    untested: whether the *production* 3D crop itself misses the striatum
    for a meaningful fraction of subjects (same root cause, smaller
    relative impact since the 3D crop is 9x thicker) — flagged as a
    higher-value audit than slab2d itself. Full second-review findings in
    `notebooks/24_slab2d_architecture_diversity.ipynb`'s reflection cell.
    A lighter-weight comparison point
    to Boulkrinat et al.'s from-scratch 3D CNN if the full 3D track proves
    too expensive within the 3h submission budget.
  - **Scope caveat**: their "multi-site variability" is *simulated*
    (post-hoc Gaussian smoothing of PPMI images), not real multi-center
    data with real scanner/reconstruction differences, and the clinical
    validation used only 2 reconstruction algorithms from 1 institution —
    weaker generalization evidence than our own 10-real-center dataset
    will provide once the CNN is actually trained and validated per family
    (`README.md` Next steps, leave-one-family-out check).
- **PPMI's own striatal binding ratio (SBR) methodology** (see "Baseline
  Neuroimaging Characteristics of the PPMI..." https://www.michaeljfox.org/publication/baseline-neuroimaging-characteristics-parkinsons-progression-marker-initiative-ppmi
  and the count-based SBR method, PMC6314989) — standardized VOI template
  (not a percentile threshold) over left/right caudate + putamen, occipital
  lobe as reference region, with **left-right percent asymmetry** reported
  as a standard derived clinical metric.
  Why: validates that this project's own asymmetry index (`01_eda_volumes.ipynb`
  section 6b) is measuring a clinically recognized quantity, not an
  invented proxy — but PPMI's atlas-based VOI placement is more principled
  than this project's physical-volume-threshold approximation (section 6a)
  and is a candidate to revisit if the threshold-based crop proves
  unreliable once actually run.

**Training/validation methodology for the CNN track** (Opus research pass,
2026-09-09, ahead of `docs/superpowers/specs/2026-09-09-cnn-train-rung23-design.md`):

- **Malladi et al. 2022** (arXiv:2205.10287, "On the SDEs and Scaling Rules
  for Adaptive Gradient Algorithms") — derives that Adam's step size should
  scale with **sqrt(batch ratio)**, not linearly (Goyal et al.'s SGD rule).
  Why: directly sets the LR when moving `config.BATCH_SIZE` off 8 (8→32
  implies LR 1e-3→2e-3, not 4e-3) — using the SGD linear rule here would
  have been wrong.
- **Goyal et al. 2017** (arXiv:1706.02677, "Accurate, Large Minibatch SGD")
  — the linear LR-scaling rule, SGD-specific, regime starts ~batch 256+.
  Why: logged mainly to record why it does *not* apply here (Adam, small
  batch range) — a wrong-but-tempting default.
- **Masters & Luschi 2018** (arXiv:1804.07612, "Revisiting Small Batch
  Training for Deep Neural Networks") — best results found in batch 2-32;
  the stable-LR range narrows as batch grows past that.
  Why: caps the batch increase for `DatCNN` at 32, not higher, independent
  of the sqrt-LR-rule argument.
- **Wu & He 2018** (arXiv:1803.08494, "Group Normalization") — documents
  BatchNorm's batch-statistics noise rising sharply below ~16 samples/batch.
  Why: `DatCNN` uses `BatchNorm3d` in every block; this is a second,
  independent reason (besides throughput) that `config.BATCH_SIZE=8` is a
  bad default and 32 is a better one.
- **Varoquaux 2018** (arXiv:1706.07581, NeuroImage, "Cross-validation
  failure: small sample sizes lead to large error bars") — the across-fold
  standard deviation of a CV metric systematically underestimates the true
  error bar, worse for nonlinear metrics and small N.
  Why: rules out using the 5-fold sd alone as the CNN's noise floor for the
  rung-3 gate decision (naively reusing the classical baseline's
  sd=0.0006-0.0011 approach); need a paired bootstrap + seed-repeats
  instead (see the rung 2/3 spec).
- **Bouthillier et al. 2021** (arXiv:2103.03098, MLSys, "Accounting for
  Variance in Machine Learning Benchmarks") — randomizing multiple sources
  of variation (init seed + fold seed together) at a fixed compute budget
  approaches the ideal variance estimator far better than repeating one
  source alone.
  Why: rung 3's 5-seed noise-floor repeats vary both the model-init seed
  and `make_folds`' `random_state` together, per this finding, not just one
  of them.
- **Nadeau & Bengio 2003** (Machine Learning 52, "Inference for the
  Generalization Error") — corrected resampled t-test variance estimator
  for a single CV run, inflating naive fold variance by
  `1/k + n_test/n_train`.
  Why: fallback variance estimate if a single CV run must be judged without
  time for seed-repeats.
- **RSNA / Radiology:AI, "A Guide to Cross-Validation for AI in Medical
  Imaging"** (PMC10388213) — leave-one-institution-out is typically
  pessimistic vs. k-fold; recommends nested CV (inner split for model
  selection/early stopping, outer fold purely held out) to avoid selection
  optimism.
  Why: motivated the nested-CV design in the rung 2/3 spec — the original
  draft picked the best epoch/checkpoint using the same fold it then scored
  the gate on, which is optimistically biased and not comparable to
  `build_combat_baseline()` (a LogisticRegression with no per-fold model
  selection at all).
- **The Impact of Scanner Domain Shift on DL Performance in Medical
  Imaging** (arXiv:2409.04368) and **Domain Generalization Mitigates
  Scanner-Induced Domain Shift** (J Imaging Inform Med,
  10.1007/s10278-026-02082-z) — both use a single dissimilar held-out
  center as a cheap proxy for a full leave-one-site-out CV, rather than
  iterating every site.
  Why: supports rung 2's single-family leave-one-family-out design (one
  family, not all 17) as a reasonable cheap probe, not an under-scoped one.
- **Wenzel et al. 2019** (already logged above) resolution-transfer
  asymmetry finding directly decided *which* family to hold out in rung 2:
  the 3.895mm family (100% oblique, the coarsest resolution, `config.py`'s
  own comment flags it as "interpolated UP, must be revisited if it
  underperforms"), not the largest 2.46mm family — holding out 2.46mm
  would test the transfer direction Wenzel et al. already found working
  well (0.945 accuracy), telling us nothing new.

- **Khachnaoui et al. 2023, "Enhanced Parkinson's Disease Diagnosis through
  Convolutional Neural Network Models Applied to SPECT DaTSCAN Images"**
  (EfficientNet-B0+MobileNet-V2 "bilinear fusion", 99.14% on 2720 PPMI
  slices) -- reviewed in depth 2026-09-11. "Bilinear fusion" is classical
  bilinear CNN pooling (Lin et al.): outer-product interaction of two
  backbones' feature maps at matching spatial resolution, sum-pooled,
  signed-sqrt + L2-normalized. Their literal method (embedding-level fusion
  of two ImageNet-pretrained backbones) does not transfer to this project:
  our 30 CNN members are independently initialized, so their penultimate
  embeddings live in mutually unaligned spaces, and no embedding is saved
  anywhere in this codebase. The scalar remnant of the idea (second-order
  interaction terms between the CNN and baseline blend scores, plus the
  classical baseline's own engineered features) is CV-tested in
  `notebooks/23_paired_gate_shipped_recipe_candidates.ipynb` (candidate 1).
  Why: closest sourced precedent for a feature-interaction fusion mechanism
  beyond this project's current 2-feature logistic blend.
- **Ithapu, Singh & Johnson 2015, "Randomized Deep Learning Methods for
  Clinical Trial Enrichment and Design in Alzheimer's Disease"** (Ch. 15,
  *Deep Learning for Medical Image Analysis*, Zhou/Greenspan/Shen eds.,
  Academic Press 2017) -- full chapter read 2026-09-11 (user-provided
  transcript). Proposes randomized deep networks (rDA/rDr): many weak
  learners trained on randomly-partitioned voxel blocks, combined via a
  ridge-regression fit rather than uniform averaging, to construct a
  minimum-variance-unbiased (MVUB) estimator (eq. 15.6-15.8) -- validated on
  ADNI for reducing AD clinical-trial sample sizes vs. an MKL baseline. The
  voxel-blocking architecture itself does not transfer (it's a workaround
  for fully-connected/autoencoder nets in a d>>n regime that this project's
  3D CNN already solves via convolutional weight sharing). The transferable
  piece is the combination mechanism: ridge/L2-regularized weighting across
  near-decorrelated ensemble members instead of uniform averaging. CV-tested
  at the 6-variant level (not the 30-member level, to avoid the same
  overfitting mechanism the theory warns about) in
  `notebooks/23_paired_gate_shipped_recipe_candidates.ipynb` (candidate 2).
  Why: only sourced precedent found for reweighting (vs. adding) ensemble
  members as a variance-reduction mechanism.
- **Budd et al. 2023, "Automated identification of uncertain cases in deep
  learning-based classification of dopamine transporter SPECT..."** (EJNMMI,
  DOI 10.1007/s00259-023-06566-w, code: github.com/ThomasBudd/dat_spect_ud)
  -- the most directly on-topic paper found in the 2026-09-11 research pass:
  real multi-site [123I]FP-CIT DAT-SPECT (not PPMI). Trains a 5-CNN ensemble
  plus two additional ensembles with asymmetric losses (high-sensitivity,
  high-specificity); disagreement between the two flags uncertain cases for
  review, catching 90% of misclassifications while flagging only 4.3% of
  cases -- beats sigmoid-thresholding, MC-dropout, and model-averaging as an
  uncertainty measure. Not free (2 new training runs). The free precursor --
  whether disagreement already present across this project's 30 existing CNN
  members (per-row logit sd) carries information a log-loss-scored blend can
  use -- is CV-tested in
  `notebooks/23_paired_gate_shipped_recipe_candidates.ipynb` (candidate 3),
  gated before any GPU time is spent on new asymmetric-loss ensembles.
  Why: only sourced technique found that targets prediction uncertainty
  specifically for DAT-SPECT classification, on real (not PPMI) multi-site
  data matching this project's own data shape.
- **Tajbakhsh et al. 2016, "Convolutional neural networks for medical image
  analysis: full training or fine tuning?"** (IEEE TMI 35(5):1299-1312) --
  found via a cerebral-microbleed-detection book chapter's reference list
  (Zhou/Greenspan/Shen eds. 2017, Ch. 6.4), reviewed 2026-09-11. The classic
  reference on when fine-tuning beats training from scratch in medical
  imaging.
  Why: a fourth independent source reinforcing this project's already-made
  decision to train the 3D CNN from scratch (alongside Chegodaev et al.,
  Boulkrinat et al., and Wenzel et al., all already logged above) -- no new
  action, logged for completeness per this skill's "every technique needs a
  source" rule.

## Comparable projects

- **mtwenzel/parkinson-classification** (GitHub) — fine-tunes Inception V3
  on PPMI SPECT scans, Colab-based.
  Why: worked example of a 2D transfer-learning approach on this exact
  data type; same ImageNet-license caveat applies to any weights reused.
- **IoBT-VISTEC/PPMI_DL** (GitHub) — deep learning + interpretability
  tutorial for 3D SPECT PD classification on PPMI.
  Why: closest public analog for the 3D-volumetric main track; review its
  preprocessing/architecture choices before designing ours from scratch.
- **vtn6/PPMI-Data-Mining** (GitHub) — reports 92.5% accuracy classifying
  PD vs. healthy control from DatSCAN imaging.
  Why: another prior-art accuracy anchor, classical-ML side.
- **RSNA_Knee_Abnormality_Detection** (this user's own prior project,
  github.com/Alherma7) — closest analog for project structure and lived
  pitfalls
  (CV leakage via scanner identity, geometric slice ordering); see
  `deep-learning-imaging.md`'s Common Mistakes for the specifics already
  folded into the skill.
- **drivendataorg/competition-winners** (GitHub org) — archive of winning
  code from past DrivenData competitions, including other medical-imaging
  code-execution challenges (e.g. VisioMel Challenge).
  Why: worth checking for submission-packaging patterns and how other
  winners structured a code-execution medical-imaging solution, even
  though none is DAT-SPECT-specific.
- **RSNA-MICCAI Brain Tumor Radiogenomic Classification** (Kaggle
  competition, https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification)
  — no DAT-SPECT Kaggle competition exists (checked 2026-09-08, niche
  modality), but this is the closest public analog by problem shape:
  binary classification (MGMT biomarker status) on multi-parameter 3D MRI
  (T1w/T1wCE/T2w/FLAIR), AUROC-scored, multi-institution (BraTS, 19
  institutions, different scanners/protocols, released as NIfTI — the
  same class of multi-site variability sections 3/3a/4 of our own EDA
  found empirically). Public EDA notebook: "Brain Tumor Radiogenomic
  Classification - EDA" (kaggle.com/tanlikesmath/brain-tumor-radiogenomic-classification-eda,
  not fetchable without a Kaggle session — link only, not reviewed).
  Solution repos (light detail, READMEs only — actual notebooks not
  reviewed): cedricsoares/kaggle-rsna-miccai-brain-tumor-radiogenomic-classification;
  MahmoudRabea13/RSNA-MICCAI-brain-tumor-radiogenomic-classification
  tried ViT3D/ResNet50/Xception/EfficientNet-B3, best result Xception
  AUC 0.638 (val) / 0.617 (test) — notably weak for a well-resourced
  Kaggle effort; SrLozano/Brain-Tumor-Radiogenomic-Classification
  (transfer learning).
  **The actual lesson, from the benchmark's own follow-up validation
  study** (Validation of MRI-Based Models to Predict MGMT Promoter
  Methylation in Gliomas: BraTS 2021 Radiogenomics Challenge, PMC9562637):
  when models that scored well on small single-center studies were
  externally validated on the challenge's larger multicenter set,
  **~80% showed no significant difference from chance (50%)** — the
  smaller studies' promising accuracy did not generalize. This matches
  the low AUCs (0.61-0.64) the solution repos above report despite
  competent architectures.
  Why: a direct, sourced caution against this project's own prior-art
  numbers — several PPMI papers logged in `## Papers` above report
  92-99.5% accuracy on small (200-2720 volume) single/few-site subsets,
  the same shape of claim this benchmark showed largely failed to
  reproduce at multicenter scale. Reinforces `SKILL.md` step 6's
  noise-floor discipline and this project's own leave-one-family-out
  check (`README.md` Next steps) — validate on the *actual* 10-center,
  1362-volume mix before trusting any literature accuracy number as a
  realistic target, and treat a small-sample high accuracy as a claim to
  verify, not a benchmark to match.
- **RSNA-ASNR-MICCAI BraTS 2021 benchmark** (arXiv:2107.02314) — the
  dataset behind the competition above: multimodal 3D brain MRI from 19
  institutions, different scanners/protocols, released as NIfTI.
  Why: the benchmark paper's own discussion of multi-site variability is
  a second, independent source (beyond the PPMI radiomics-harmonization
  paper above) for how the field handles the same class of problem this
  project's sections 3/3a/4 uncovered empirically.
- **"Parkinson's Disease Dat and MRI Scans"** (Kaggle dataset,
  kaggle.com/datasets/rishikjha/parkinsons-disease-dat-and-mri-scans) —
  checked 2026-09-08: license and original source are **not stated** on
  the dataset page. Logged as a negative result, not a candidate — do not
  use without a stated license, and note the competition's own rules
  already restrict external DAT-SPECT-like data to what can be publicly
  and freely verified (see External Data caveats below); an
  unattributed re-upload does not clear that bar regardless of license.

## Pretrained models & external data

Record every candidate here BEFORE running the experiment, not after —
license and commercial-use terms determine prize eligibility.

- **ImageNet-pretrained backbones (torchvision/timm)** — eligibility
  **UNRESOLVED**. Forum thread "Clarification on ImageNet pre-trained
  weights under Open Source / Commercial rules" (asked 2026-08-04, no
  staff answer as of 2026-09-07): the original ImageNet dataset license is
  non-commercial-research use, which may conflict with the MIT/commercial-use
  requirement for prize eligibility even though the framework code is
  Apache/BSD. Do not rely on this for a prize-eligible submission without
  an explicit staff answer or independent legal read.
- **PPMI (Parkinson's Progression Markers Initiative)** — eligibility
  **UNRESOLVED**. Forum thread "Is PPMI (DUA-gated) acceptable as external
  data for prize eligibility?" (asked 2026-08-04, no staff answer as of
  2026-09-07): PPMI is free to register for but gated behind a LONI IDA
  Data Use Agreement (redistribution-restricted, purpose-limited), which
  may not satisfy "freely and publicly available to all participants."
  Also unresolved whether models pretrained on PPMI by third parties
  inherit the restriction. Do not use for a prize-eligible submission
  without clarification.
- **DINOv3** — **NOT prize-eligible**. Confirmed by DrivenData staff
  (forum thread "Is DINOv3 allowed to use?", 2026-09-03-ish): its DINOv3
  Agreement license conflicts with the required MIT license for winning
  solutions. Avoid as a backbone if targeting prize eligibility.
