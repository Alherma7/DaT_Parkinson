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
  spacing — no site/scanner label in `train_labels.csv`; use (shape, spacing)
  combos as an EDA-level scanner proxy.
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

## Next steps

- [ ] Run `notebooks/01_eda_volumes.ipynb` and record the answers to its 7
      questions here (geometry target, striatum crop size, class balance,
      scanner-proxy confound, orientation, background threshold,
      asymmetry).
- [ ] Pin down the evaluation harness (`src/evaluate.py`): log loss +
      AUROC, Stratified K-Fold on `is_pathologic`.
- [ ] Baseline: handcrafted/radiomics features + classical model.
- [ ] Main track: 3D CNN on resampled volumes (training from scratch or a
      clearly-eligible pretrained backbone — see the ImageNet/PPMI caveat
      above).
- [ ] `RESOURCES.md`: log every technique and every external
      data/pretrained-model candidate as it's considered.
- [ ] Submission packaging + local Docker rehearsal once a model clears
      the gate.
