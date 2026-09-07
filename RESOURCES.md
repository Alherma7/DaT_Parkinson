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
