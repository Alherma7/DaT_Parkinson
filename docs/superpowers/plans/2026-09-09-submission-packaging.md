# Submission Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything needed to produce `submission.zip` for the DaT
Parkinson's Challenge: a `config.py` fix so the shared preprocessing code
finds test data inside the competition container, a small set of new
pure functions (feature extraction, checkpoint-filename generation,
prediction blending) each with unit tests, a `[RUN ME]` asset-build
script, and `submission_src/main.py`.

**Architecture:** `submission_src/` is a flat-import staging directory
(the runtime repo's own convention) holding `main.py` plus verbatim
copies of five `src/*.py` modules and a `model_assets/` folder (25 CNN
checkpoints + one pickled classical pipeline). All new logic with real
branching/arithmetic is written as a pure, unit-tested function in
`src/` first; `main.py` and the build script are then thin orchestration
over already-tested pieces, since neither can be run or verified against
real data by Claude.

**Tech Stack:** Python 3.12, PyTorch, scikit-learn, nibabel, pandas,
pytest (matches `environment.yml` and the confirmed runtime
dependencies).

**Spec:** `docs/superpowers/specs/2026-09-09-submission-packaging-design.md`

## Global Constraints

- Final model: CNN ensemble blended with `model.build_combat_baseline()`
  at **`CNN_WEIGHT = 0.70`** (README.md, 2026-09-09 leave-one-repeat-out
  result). This exact constant is used in Task 6.
- Runtime container: no network access, 3-hour limit (smoke test 6
  minutes), read-only `data/` mount, root working directory
  `/code_execution/`. Confirmed pre-installed: `nibabel`, `scipy`,
  `scikit-learn`, `pandas`, `numpy`, `torch==2.12.1+cu129`,
  `torchvision`, `matplotlib` — no new package requests needed.
- Environment-detection root is exactly `Path("/code_execution")` (not
  configurable, not an env var — matches the actual container path).
- The 25 rung-3 checkpoints are named `rung3_seed{S}_fold{F}.pt` for
  `S in [42, 43, 44, 45, 46]`, `F in [0, 1, 2, 3, 4]`, already on disk in
  `checkpoints/`.
- No task in this plan reads real per-row patient data, real `.nii.gz`
  files, or real training labels. Every test uses synthetic data. Tasks
  5 and 6 produce scripts that touch real data only when the user runs
  them — never run or executed against real data by Claude.
- Follow existing repo conventions exactly: flat `import config` style
  (no package-relative imports), Google-less docstrings matching the
  terse rationale style already in `src/*.py`, `config.RANDOM_STATE` /
  `config.SEED` for any seeding.

---

## Task 1: `config.py` runtime-environment path detection

**Files:**
- Modify: `src/config.py:1-19` (imports through `CHECKPOINT_DIR.mkdir`)
- Test: `tests/test_config.py` (new)

**Interfaces:**
- Produces: `config._resolve_runtime_paths(code_execution_root: Path) -> tuple[bool, Path, Path | None]`,
  `config.RUNNING_IN_CODE_EXECUTION: bool`, `config.NIFTI_DIR: Path`,
  `config.SUBMISSION_FORMAT_PATH: Path | None`. Later tasks (2-6) rely on
  `config.NIFTI_DIR` resolving correctly inside `data.load_volume` — no
  other task calls `_resolve_runtime_paths` directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
"""Unit tests for src/config.py's runtime-environment path detection."""

import config


def test_resolve_runtime_paths_detects_existing_code_execution_root(tmp_path):
    fake_root = tmp_path / "code_execution"
    fake_root.mkdir()

    running, nifti_dir, submission_format_path = config._resolve_runtime_paths(fake_root)

    assert running is True
    assert nifti_dir == fake_root / "data" / "niftis"
    assert submission_format_path == fake_root / "data" / "submission_format.csv"


def test_resolve_runtime_paths_falls_back_when_root_is_missing(tmp_path):
    missing_root = tmp_path / "does_not_exist"

    running, nifti_dir, submission_format_path = config._resolve_runtime_paths(missing_root)

    assert running is False
    assert nifti_dir == config.DATA_RAW / "niftis"
    assert submission_format_path is None


def test_module_level_nifti_dir_defaults_to_local_layout():
    """Regression guard: on a normal dev machine (no /code_execution),
    the module-level NIFTI_DIR used by data.load_volume must still be the
    local training path, unchanged from before this task."""
    assert config.NIFTI_DIR == config.DATA_RAW / "niftis"
    assert config.RUNNING_IN_CODE_EXECUTION is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `tests/test_config.py` doesn't exist yet, or (once
created) `AttributeError: module 'config' has no attribute
'_resolve_runtime_paths'`.

- [ ] **Step 3: Implement the minimal change**

In `src/config.py`, replace:

```python
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
NIFTI_DIR = DATA_RAW / "niftis"
TRAIN_LABELS_PATH = DATA_RAW / "train_labels.csv"

SMOKE_TEST_DIR = DATA_RAW / "smoke_test"
SMOKE_TEST_NIFTI_DIR = SMOKE_TEST_DIR / "niftis"
SMOKE_TEST_SUBMISSION_FORMAT_PATH = SMOKE_TEST_DIR / "submission_format.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
```

with:

```python
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_LABELS_PATH = DATA_RAW / "train_labels.csv"

SMOKE_TEST_DIR = DATA_RAW / "smoke_test"
SMOKE_TEST_NIFTI_DIR = SMOKE_TEST_DIR / "niftis"
SMOKE_TEST_SUBMISSION_FORMAT_PATH = SMOKE_TEST_DIR / "submission_format.csv"


def _resolve_runtime_paths(code_execution_root):
    """(running_in_code_execution, nifti_dir, submission_format_path).

    If `code_execution_root` exists, this process is running inside the
    DrivenData competition container (docs/superpowers/specs/
    2026-09-09-submission-packaging-design.md) -- test volumes and the
    submission format live under it, not under this repo's data/raw/.
    Otherwise, fall back to the local training layout (submission_format_path
    is None locally; nothing needs it outside the container).
    """
    if code_execution_root.exists():
        return (True, code_execution_root / "data" / "niftis",
                code_execution_root / "data" / "submission_format.csv")
    return False, DATA_RAW / "niftis", None


RUNNING_IN_CODE_EXECUTION, NIFTI_DIR, SUBMISSION_FORMAT_PATH = (
    _resolve_runtime_paths(Path("/code_execution"))
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
if not RUNNING_IN_CODE_EXECUTION:
    # In the competition container this directory is unused (checkpoints
    # ship pre-copied into submission_src/model_assets/) and the
    # filesystem outside /code_execution may not be writable -- skip the
    # mkdir there rather than risk a crash at import time.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `python -m pytest tests/ -q --ignore=tests/test_cache.py --ignore=tests/test_data.py --ignore=tests/test_dataset.py`
Expected: all passing (the three ignored files need `nibabel`, not
installed in this dev shell — pre-existing, unrelated to this change).

- [ ] **Step 6: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "Add runtime-environment path detection to config.py

Submission packaging spec (docs/superpowers/specs/2026-09-09-submission-packaging-design.md):
data.load_volume must find test niftis under /code_execution/data
inside the competition container instead of data/raw/. Also guards
CHECKPOINT_DIR.mkdir so it never runs inside the (possibly read-only)
container."
```

---

## Task 2: `features.py::extract_baseline_features`

**Files:**
- Modify: `src/features.py` (add function near the bottom, after `striatal_ratio`)
- Test: `tests/test_features.py` (extend)

**Interfaces:**
- Consumes: `features.striatum_mask(volume, spacing, target_ml)`,
  `features.signed_asymmetry(volume, mask, spacing)`,
  `features.striatal_ratio(volume, mask)`,
  `features.inplane_family(spacing_x)` — all already defined, unchanged.
- Produces: `features.extract_baseline_features(volume, spacing, target_ml=20.0) -> dict | None`
  with keys `"abs_asym"`, `"striatal_ratio"`, `"inplane_family"`. Task 6
  (`main.py`) calls this once per test volume.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features.py`, after the `inplane_family` section:

```python
# --- extract_baseline_features -----------------------------------------------

def test_extract_baseline_features_returns_all_three_keys():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)
    volume[5:8, 13:17, 8:12] = 1000.0
    volume[22:25, 13:17, 8:12] = 500.0  # asymmetric on purpose

    result = features.extract_baseline_features(volume, spacing, target_ml=0.5)

    assert result is not None
    assert set(result.keys()) == {"abs_asym", "striatal_ratio", "inplane_family"}
    assert result["abs_asym"] > 0  # asymmetric blobs -> nonzero
    assert result["inplane_family"] == "2.00"


def test_extract_baseline_features_returns_none_for_empty_volume():
    volume = _blank_volume()
    spacing = (2.0, 2.0, 2.0)

    result = features.extract_baseline_features(volume, spacing, target_ml=0.5)

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py -k extract_baseline_features -v`
Expected: FAIL — `AttributeError: module 'features' has no attribute
'extract_baseline_features'`.

- [ ] **Step 3: Implement the minimal change**

Add to `src/features.py`, after the `striatal_ratio` function:

```python
def extract_baseline_features(volume, spacing, target_ml=20.0):
    """(abs_asym, striatal_ratio, inplane_family) for one already-loaded
    volume, or None if the striatum mask is degenerate (mirrors
    notebooks/03_baseline_classical.ipynb's skip condition). A caller
    that gets None for a row should treat that feature as missing, not
    silently zero-fill it.
    """
    mask = striatum_mask(volume, spacing, target_ml=target_ml)
    if mask is None:
        return None
    signed = signed_asymmetry(volume, mask, spacing)
    if signed is None:
        return None
    return {
        "abs_asym": abs(signed),
        "striatal_ratio": striatal_ratio(volume, mask),
        "inplane_family": inplane_family(spacing[0]),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py -v`
Expected: all passed (existing `test_features.py` tests plus the 2 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "Add features.extract_baseline_features for submission inference

Reuses striatum_mask/signed_asymmetry/striatal_ratio/inplane_family so
notebooks/03's feature computation isn't inlined a third time in
submission_src/main.py (docs/superpowers/specs/2026-09-09-submission-packaging-design.md)."
```

---

## Task 3: `model.py::rung3_checkpoint_filenames`

**Files:**
- Modify: `src/model.py` (add function near `build_model`)
- Test: `tests/test_model.py` (extend)

**Interfaces:**
- Produces: `model.rung3_checkpoint_filenames(seeds=range(42, 47), n_folds=5) -> list[str]`.
  Task 5 (`scripts/build_submission_assets.py`) calls this with defaults
  to know which 25 files to copy from `checkpoints/`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py`, after the `build_model`/`predict` section:

```python
# --- rung3_checkpoint_filenames ------------------------------------------

def test_rung3_checkpoint_filenames_matches_the_25_files_on_disk():
    names = model_module.rung3_checkpoint_filenames()

    assert len(names) == 25
    assert names[0] == "rung3_seed42_fold0.pt"
    assert names[-1] == "rung3_seed46_fold4.pt"
    assert "rung3_seed44_fold2.pt" in names


def test_rung3_checkpoint_filenames_respects_custom_seeds_and_folds():
    names = model_module.rung3_checkpoint_filenames(seeds=[1, 2], n_folds=3)

    assert names == [
        "rung3_seed1_fold0.pt", "rung3_seed1_fold1.pt", "rung3_seed1_fold2.pt",
        "rung3_seed2_fold0.pt", "rung3_seed2_fold1.pt", "rung3_seed2_fold2.pt",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -k rung3_checkpoint_filenames -v`
Expected: FAIL — `AttributeError: module 'model' has no attribute
'rung3_checkpoint_filenames'`.

- [ ] **Step 3: Implement the minimal code**

Add to `src/model.py`, after `build_model`:

```python
def rung3_checkpoint_filenames(seeds=range(config.SEED, config.SEED + 5), n_folds=config.N_FOLDS):
    """Filenames of the rung-3 nested-CV checkpoints
    (notebooks/07_cnn_rung3.ipynb), e.g. "rung3_seed42_fold0.pt" --
    the naming convention that notebook's own torch.save calls already
    use. Kept here so scripts/build_submission_assets.py names the files
    it copies the same way instead of re-deriving the pattern.
    """
    return [f"rung3_seed{seed}_fold{fold}.pt" for seed in seeds for fold in range(n_folds)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Add model.rung3_checkpoint_filenames for submission asset packaging

Single source of truth for the 25 rung-3 checkpoint filenames
(docs/superpowers/specs/2026-09-09-submission-packaging-design.md),
so scripts/build_submission_assets.py doesn't re-derive the naming
pattern separately from notebooks/07_cnn_rung3.ipynb."
```

---

## Task 4: `src/submission.py::combine_predictions`

**Files:**
- Create: `src/submission.py`
- Test: `tests/test_submission.py` (new)

**Interfaces:**
- Produces: `submission.combine_predictions(uids: list[str], cnn_probs: dict[str, float], baseline_probs: dict[str, float], cnn_weight: float) -> list[float]`.
  Task 6 (`main.py`) calls this once, after computing both probability
  dicts, to get the final per-row prediction in `submission_format.csv`'s
  row order.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_submission.py`:

```python
"""Unit tests for src/submission.py -- the submission.zip inference
entrypoint's testable logic. No real data; uids here are arbitrary
strings, not real patient identifiers.
"""

import pytest

import submission


def test_combine_predictions_blends_when_both_probabilities_present():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8, "b": 0.2}
    baseline_probs = {"a": 0.6, "b": 0.4}

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=0.70)

    assert result[0] == pytest.approx(0.70 * 0.8 + 0.30 * 0.6)
    assert result[1] == pytest.approx(0.70 * 0.2 + 0.30 * 0.4)


def test_combine_predictions_falls_back_to_cnn_alone_when_baseline_missing():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8, "b": 0.2}
    baseline_probs = {"a": 0.6}  # "b" has no classical features (degenerate mask)

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=0.70)

    assert result[1] == pytest.approx(0.2)  # CNN alone, not blended with anything


def test_combine_predictions_preserves_uids_order_regardless_of_dict_order():
    uids = ["z", "a", "m"]
    cnn_probs = {"a": 0.1, "m": 0.5, "z": 0.9}  # inserted in a different order than uids
    baseline_probs = {"a": 0.1, "m": 0.5, "z": 0.9}

    result = submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=1.0)

    assert result == [0.9, 0.1, 0.5]


def test_combine_predictions_raises_if_a_uid_is_missing_from_cnn_probs():
    uids = ["a", "b"]
    cnn_probs = {"a": 0.8}  # "b" missing -- every test uid must have a CNN prediction
    baseline_probs = {}

    with pytest.raises(KeyError):
        submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight=0.70)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_submission.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'submission'`.

- [ ] **Step 3: Implement the minimal code**

Create `src/submission.py`:

```python
"""Pure, unit-testable logic shared by submission_src/main.py. Kept here
(not inlined in main.py) so it's covered by pytest -- main.py itself
can't be run or verified against real data by Claude
(docs/superpowers/specs/2026-09-09-submission-packaging-design.md).
"""


def combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight):
    """Blend per-uid CNN and classical-baseline probabilities into the
    submission's final prediction, in `uids`' exact order.

    `cnn_probs` and `baseline_probs` are dicts keyed by uid. Every uid in
    `uids` must have a CNN prediction (raises KeyError otherwise -- the
    CNN ensemble runs on every test volume unconditionally). A uid
    missing from `baseline_probs` (features.extract_baseline_features
    returned None for it -- a degenerate striatum mask) falls back to
    the CNN probability alone rather than crashing or dropping the row.

    Returns a list of floats, same length and order as `uids`.
    """
    predictions = []
    for uid in uids:
        cnn_p = cnn_probs[uid]
        if uid in baseline_probs:
            predictions.append(cnn_weight * cnn_p + (1 - cnn_weight) * baseline_probs[uid])
        else:
            predictions.append(cnn_p)
    return predictions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_submission.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q --ignore=tests/test_cache.py --ignore=tests/test_data.py --ignore=tests/test_dataset.py`
Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add src/submission.py tests/test_submission.py
git commit -m "Add submission.combine_predictions for the CNN+baseline blend

Pure function extracted so the w_cnn=0.70 blend-with-fallback logic
main.py needs is unit tested, since main.py itself can't be run against
real data by Claude (docs/superpowers/specs/2026-09-09-submission-packaging-design.md)."
```

---

## Task 5: `scripts/build_submission_assets.py`

**Files:**
- Create: `scripts/build_submission_assets.py`

**Interfaces:**
- Consumes: `model.build_combat_baseline()`, `model.rung3_checkpoint_filenames()`,
  `config.DATA_PROCESSED`, `config.CHECKPOINT_DIR`, `config.PROJECT_ROOT`,
  `config.UID_COLUMN`, `config.TARGET_COLUMN`.
- Produces (on disk, when the user runs it): `submission_src/model_assets/combat_baseline.pkl`,
  `submission_src/model_assets/checkpoints/*.pt` (25 files),
  `submission_src/{config,data,model,features,dataset,submission}.py`.
  Task 6 (`main.py`) depends on all of these existing before it can run.

Not TDD: this script's only logic (which files to copy, which pipeline
to fit) is either already covered by Tasks 1-4's tests
(`rung3_checkpoint_filenames`) or is file I/O with no branching worth
unit testing in isolation. It touches real per-row training data
(`baseline_features.csv`), so per the AI-assistant data rule the user
runs it, not Claude.

- [ ] **Step 1: Write the script**

Create `scripts/build_submission_assets.py`:

```python
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
```

- [ ] **Step 2: Syntax-check the script (no real data touched)**

Run: `python -m py_compile scripts/build_submission_assets.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Update `.gitignore`**

`.gitignore` already ignores `checkpoints/`, `models/`, `outputs/`,
`submission/` (model artifacts) but not `submission_src/` — the
directory this script populates with copied `.py` modules, 25 copied
`.pt` checkpoints, and a pickled pipeline. Without this, running the
script leaves all of that untracked and one `git add -A` away from
committing large binaries. `submission_src/main.py` is the one file in
that directory that IS meant to be tracked (it's hand-written source,
not a generated copy), so ignore everything else in the directory
individually rather than the whole directory.

Add this block to `.gitignore`, after the existing `# Model artifacts`
block:

```
# Generated submission assets (scripts/build_submission_assets.py output) --
# submission_src/main.py is hand-written and stays tracked; everything
# else here is a copy or a binary artifact, regenerated on demand.
submission_src/model_assets/
submission_src/config.py
submission_src/data.py
submission_src/model.py
submission_src/features.py
submission_src/dataset.py
submission_src/submission.py
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_submission_assets.py .gitignore
git commit -m "Add scripts/build_submission_assets.py to package submission_src/

[RUN ME] -- fits the classical baseline on 100% of training data and
copies checkpoints/modules into submission_src/. Not run by Claude
(docs/superpowers/specs/2026-09-09-submission-packaging-design.md).
Also updates .gitignore -- submission_src/ held nothing before this
task, so its generated contents (copied modules, checkpoints, the
pickled pipeline) were not yet excluded."
```

---

## Task 6: `submission_src/main.py`

**Files:**
- Create: `submission_src/main.py`

**Interfaces:**
- Consumes: `data.load_volume(uid)` (via `dataset.DatParkinsonDataset`),
  `model.build_model()`, `model.predict(model, x)`,
  `features.extract_baseline_features(volume, spacing, target_ml)`,
  `submission.combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight)`,
  `config.NIFTI_DIR`, `config.BATCH_SIZE`, `config.DEVICE`,
  `config.UID_COLUMN`, `config.TARGET_COLUMN`. All copied into
  `submission_src/` by Task 5's script before this file is used for real.

Not TDD, matching Task 5's reasoning: `main.py` is thin orchestration
over functions Tasks 1-4 already unit test; it can only be exercised
end-to-end via the Docker rehearsal the user runs (spec's Definition of
done), never by Claude.

- [ ] **Step 1: Write the script**

Create `submission_src/main.py`:

```python
"""Submission entrypoint (docs/superpowers/specs/2026-09-09-submission-packaging-design.md).

Runs the rung-3 CNN ensemble (25 checkpoints) and the pre-fit ComBat
classical baseline, blends them at CNN_WEIGHT, and writes submission.csv.

Never logs per-row information (uid next to a prediction, per-row
feature values) -- only aggregate counts and phase timings, per this
competition's submission checklist and this project's own
AI-assistant data rule.
"""
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

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
CNN_WEIGHT = 0.70  # README.md, 2026-09-09 leave-one-repeat-out result


def run_cnn_ensemble(uids):
    """Average sigmoid probability across all 25 rung-3 checkpoints.

    Each test volume's preprocessing (resample/crop/normalize --
    data.load_volume, the expensive part) runs at most ONCE per uid and
    is cached in memory (a plain dict, not the disk-backed
    cache.CachedVolumeStore -- inference is a single session, nothing to
    persist across runs) so all 25 checkpoints' passes reuse it instead
    of repeating it 25x per volume. Returns {uid: probability}.
    """
    checkpoint_dir = MODEL_ASSETS / "checkpoints"
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt"))
    print(f"CNN ensemble: {len(checkpoint_paths)} checkpoints, {len(uids)} test volumes")

    volume_cache = {}

    def cached_load_volume(uid):
        if uid not in volume_cache:
            volume_cache[uid] = data.load_volume(uid)
        return volume_cache[uid]

    ds = dataset.DatParkinsonDataset(uids, load_fn=cached_load_volume)
    summed = np.zeros(len(uids))
    for i, checkpoint_path in enumerate(checkpoint_paths):
        start = time.time()
        net = model.build_model().to(config.DEVICE)
        net.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        loader = torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, num_workers=0)
        probs = []
        for x, _ in loader:
            probs.append(model.predict(net, x))
        summed += np.concatenate(probs)
        print(f"  checkpoint {i + 1}/{len(checkpoint_paths)} done in {time.time() - start:.1f}s")

    averaged = summed / len(checkpoint_paths)
    return dict(zip(uids, averaged.tolist()))


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
    predictions = submission.combine_predictions(uids, cnn_probs, baseline_probs, CNN_WEIGHT)

    out = submission_format.copy()
    out[config.TARGET_COLUMN] = predictions
    out.to_csv(WRITE_SUBMISSION_PATH, index=False)
    print(f"wrote {len(out)} predictions -> {WRITE_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check the script (no real data touched)**

Run: `python -m py_compile submission_src/main.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add submission_src/main.py
git commit -m "Add submission_src/main.py: CNN ensemble + ComBat blend entrypoint

Not run by Claude -- needs submission_src/model_assets/ (scripts/build_submission_assets.py)
and real test data. User verifies via the runtime repo's
just pack-submission / just test-submission against the smoke-test data
(docs/superpowers/specs/2026-09-09-submission-packaging-design.md)."
```

---

## After this plan

Per the spec's Definition of done, the remaining steps are the user's,
not another implementation task:
1. Run `scripts/build_submission_assets.py`.
2. Copy/symlink `submission_src/main.py` isn't needed — it's already
   inside `submission_src/`; just confirm `submission_src/` contains
   `main.py` + the 6 copied modules + `model_assets/` before packaging.
3. Clone `https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime`,
   download the smoke-test data into its `data-demo/`, run `just pull`,
   `just pack-submission` (pointed at this repo's `submission_src/`),
   then `just test-submission`.
4. Report back: did it finish inside 6 minutes, did `submission.csv`
   match `submission_format.csv`'s shape/columns, any errors — aggregate
   facts only, never row-level output.
5. Update `README.md` Next steps with the result.
