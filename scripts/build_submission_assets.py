"""[RUN ME] -- not run by Claude. Reads real per-row training data
(data/processed/baseline_features.csv's labels) to fit the classical
baseline on 100% of the training set, then assembles submission_src/:
the fitted pipeline, the production CNN checkpoints
(model.production_checkpoint_filenames() -- currently 25: 1 per-subject-
centered variant x 5 seeds x 5 folds), and the src/ modules main.py
needs. Run this once before packaging submission.zip (see
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

import numpy
import pandas as pd
import sklearn
import torch

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
    """Bug fixed 2026-09-13: this used to only ever ADD files, never remove
    ones no longer in model.production_checkpoint_filenames() -- when the
    composition shrank from 150 checkpoints/6 variants to 25/1, the old 150
    stayed behind in model_assets/checkpoints/ and got zipped into
    submission.zip too (harmless at inference, since main.py only loads the
    expected 25, but needless bloat). Wipe the destination first so it
    always matches the current composition exactly."""
    dest = MODEL_ASSETS / "checkpoints"
    if dest.exists():
        shutil.rmtree(dest)
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
    print(f"copied {copied}/{len(names)} production checkpoints "
          f"({len(model.PRODUCTION_VARIANT_PREFIXES)} variant(s)) -> {dest}")


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
    print(f"local versions: torch={torch.__version__}, sklearn={sklearn.__version__}, "
          f"numpy={numpy.__version__} -- diff these against the runtime repo's uv.lock "
          "(torch==2.12.1+cu129 there) before packing submission.zip.")
    main_py_status = "present" if (SUBMISSION_SRC / "main.py").exists() else "MISSING -- expected at submission_src/main.py"
    print(f"\nsubmission_src/ ready at {SUBMISSION_SRC} "
          f"(main.py: {main_py_status}).")


if __name__ == "__main__":
    main()
