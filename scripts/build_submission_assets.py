"""[RUN ME] -- not run by Claude. Reads real per-row training data
(data/processed/baseline_features.csv's labels) to fit the classical
baseline on 100% of the training set, then assembles submission_src/:
the fitted pipeline, the 25 rung-3 CNN checkpoints, and the src/ modules
main.py needs. Run this once before packaging submission.zip (see
docs/superpowers/specs/2026-09-09-submission-packaging-design.md), and
again any time config.py/data.py/model.py/features.py/dataset.py/
submission.py change.

Usage: python scripts/build_submission_assets.py
"""
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import config
import model

SUBMISSION_SRC = config.PROJECT_ROOT / "submission_src"
MODEL_ASSETS = SUBMISSION_SRC / "model_assets"
MODULES_TO_COPY = ["config.py", "data.py", "model.py", "features.py", "dataset.py", "submission.py"]


def fit_and_pickle_baseline():
    feat_df = pd.read_csv(config.DATA_PROCESSED / "baseline_features.csv")
    X = feat_df[["abs_asym", "striatal_ratio"]].to_numpy()
    y = feat_df[config.TARGET_COLUMN].to_numpy()
    batch = feat_df["inplane_family"].to_numpy()

    pipeline = model.build_combat_baseline()
    pipeline.fit(X, y, batch)

    MODEL_ASSETS.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_ASSETS / "combat_baseline.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"fit ComBat baseline on {len(feat_df)} rows, "
          f"{len(set(batch))} inplane_family batches -> {out_path}")


def copy_checkpoints():
    dest = MODEL_ASSETS / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    names = model.rung3_checkpoint_filenames()
    copied = 0
    for name in names:
        src_path = config.CHECKPOINT_DIR / name
        if not src_path.exists():
            raise FileNotFoundError(
                f"expected checkpoint not found: {src_path} -- "
                "did notebooks/07_cnn_rung3.ipynb finish its full 5x5 run?"
            )
        shutil.copy2(src_path, dest / name)
        copied += 1
    print(f"copied {copied}/{len(names)} rung-3 checkpoints -> {dest}")


def copy_modules():
    SUBMISSION_SRC.mkdir(parents=True, exist_ok=True)
    src_dir = config.PROJECT_ROOT / "src"
    for name in MODULES_TO_COPY:
        shutil.copy2(src_dir / name, SUBMISSION_SRC / name)
    print(f"copied {len(MODULES_TO_COPY)} modules -> {SUBMISSION_SRC}")


def main():
    copy_modules()
    copy_checkpoints()
    fit_and_pickle_baseline()
    print(f"\nsubmission_src/ ready at {SUBMISSION_SRC}. "
          "main.py is committed separately -- copy or symlink it in "
          "before running `just pack-submission`.")


if __name__ == "__main__":
    main()
