"""Submission entrypoint (docs/superpowers/specs/2026-09-09-submission-packaging-design.md).

Runs the 150-checkpoint CNN ensemble (6 production variants x 5 seeds x
5 folds, pooled in logit space) and the pre-fit ComBat classical
baseline, blends them via a fitted logistic-regression calibration
(BLEND_A/B/C, with a CNN-only FALLBACK_A1/C1 for degenerate-mask rows),
and writes submission.csv.

Never logs per-row information (uid next to a prediction, per-row
feature values) -- only aggregate counts and phase timings, per this
competition's submission checklist and this project's own
AI-assistant data rule.
"""
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import data
import dataset
import features
import model
import submission

DATA_DIR = Path("/code_execution/data")
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")
MODEL_ASSETS = Path(__file__).parent / "model_assets"
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
DEVICE = config.DEVICE if torch.cuda.is_available() else "cpu"


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


def run_classical_baseline(uids):
    """(abs_asym, striatal_ratio, inplane_family) per uid via
    features.extract_baseline_features, then the pre-fit pipeline's
    predict_proba. A uid with a degenerate mask (None) is simply absent
    from the returned dict -- submission.combine_predictions() falls
    back to the CNN alone for it.
    """
    import pickle
    with open(MODEL_ASSETS / "combat_baseline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    rows, valid_uids = [], []
    n_degenerate = 0
    for uid in uids:
        img = nib.load(str(NIFTI_DIR / f"{uid}.nii.gz"))
        volume = img.get_fdata()
        spacing = img.header.get_zooms()[:3]
        feat = features.extract_baseline_features(volume, spacing)
        if feat is None:
            n_degenerate += 1
            continue
        rows.append(feat)
        valid_uids.append(uid)
    print(f"classical baseline: {len(valid_uids)}/{len(uids)} volumes had a valid mask "
          f"({n_degenerate} degenerate -> CNN-alone fallback)")

    if not rows:
        return {}
    feat_df = pd.DataFrame(rows)
    X = feat_df[["abs_asym", "striatal_ratio"]].to_numpy()
    batch = feat_df["inplane_family"].to_numpy()
    probs = pipeline.predict_proba(X, batch)[:, 1]
    return dict(zip(valid_uids, probs.tolist()))


def main():
    assert NIFTI_DIR == config.NIFTI_DIR, (
        f"main.py's own NIFTI_DIR ({NIFTI_DIR}) disagrees with config.NIFTI_DIR "
        f"({config.NIFTI_DIR}) -- config.py's runtime-environment detection "
        "did not fire as expected."
    )

    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    uids = submission_format[config.UID_COLUMN].tolist()
    print(f"loaded submission_format.csv: {len(uids)} rows")

    cnn_probs = run_cnn_ensemble(uids)
    baseline_probs = run_classical_baseline(uids)
    predictions = submission.combine_predictions(
        uids, cnn_probs, baseline_probs,
        BLEND_A, BLEND_B, BLEND_C, FALLBACK_A1, FALLBACK_C1,
    )
    predictions = np.clip(predictions, *PROB_CLIP)

    out = submission_format.copy()
    out[config.TARGET_COLUMN] = predictions
    out.to_csv(WRITE_SUBMISSION_PATH, index=False)
    print(f"wrote {len(out)} predictions -> {WRITE_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
