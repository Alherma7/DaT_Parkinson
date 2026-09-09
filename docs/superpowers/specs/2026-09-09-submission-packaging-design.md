# Submission packaging: `main.py`, `config.py` env-detection, asset build script

**Decision this feeds**: the DaT Parkinson's Challenge (DrivenData, deadline
2026-09-16) requires a `submission.zip` containing a `main.py` entrypoint
that runs offline (no network) in a Docker container — a single A100 GPU,
24 vCPUs, 220GB RAM, 3-hour limit (smoke test 6 minutes) — reads test
`.nii.gz` volumes + `submission_format.csv` from a read-only `data/`
mount, and writes `submission.csv`. This is the first submission this
project has made; nothing in `submission_src/` exists yet.

**Final model** (`README.md`, 2026-09-09): a blend of the rung-3 CNN
ensemble and `model.build_combat_baseline()` at **w_cnn=0.70**, decided
via leave-one-repeat-out validation (`notebooks/07_cnn_rung3.ipynb`).

**Runtime environment** (`https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime`,
checked 2026-09-09): `runtime/pyproject.toml` confirms `nibabel`, `scipy`,
`scikit-learn`, `pandas`, `numpy`, `torch==2.12.1+cu129`, `torchvision`,
`matplotlib` are all pre-installed — no package request needed. The
example submission (`examples/minimal/main.py`) confirms the contract:
`main.py` at the root of `submission_src/`, fixed paths
`/code_execution/data/niftis/{uid}.nii.gz` and
`/code_execution/data/submission_format.csv`, output written to
`submission.csv` in the working directory. Local testing uses `just
pack-submission` (zips `submission_src/`) then `just test-submission`
(runs it in the real Docker image, mounting `data-demo/` or a
user-specified `DATA_DIR`).

## Architecture

`submission_src/` (new directory, the runtime repo's own staging
convention) holds everything `submission.zip` needs:

```
submission_src/
├── main.py                  # entrypoint (hand-written, no data access to write it)
├── config.py                # copy of src/config.py + env-detection (see below)
├── data.py                  # copy of src/data.py, unchanged
├── model.py                 # copy of src/model.py, unchanged
├── features.py              # copy of src/features.py + one new function
├── dataset.py                # copy of src/dataset.py, unchanged
├── submission.py             # copy of src/submission.py, holds combine_predictions
└── model_assets/
    ├── checkpoints/
    │   └── rung3_seed{42-46}_fold{0-4}.pt   # 25 files, copied as-is
    └── combat_baseline.pkl   # ComBatHarmonizedPipeline, pre-fit + pickled
```

Flat imports (`import config`, `import features`, ...) match the existing
`src/*.py` convention exactly, so those five files are copied verbatim —
no import rewrites, no risk of the copy silently drifting from the
tested originals beyond the one deliberate `config.py` change below.

`cache.py`, `evaluate.py`, `train.py` are not needed at inference (no
multi-epoch reuse, no fold splitting, no loss computation) and are not
copied.

## `config.py`: environment auto-detection

Two problems, both in the existing module-level code:

1. `data.load_volume` reads `config.NIFTI_DIR` internally, hardcoded to
   `DATA_RAW / "niftis"` (the local training path). In the runtime this
   must resolve to `/code_execution/data/niftis` instead.
2. `CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)` runs at import
   time. In the runtime container this could hit a permission error, or
   silently create an unwanted directory outside `/code_execution` — a
   crash risk that has nothing to do with inference.

Fix, gated on a single check (`Path("/code_execution").exists()`) so the
byte-identical file works in both places — critical because
`TARGET_SPACING`/`CROP_SIZE_MM`/`CROP_CENTER_MM`/`TARGET_SHAPE`/
`BACKGROUND_PERCENTILE`/`BACKGROUND_MAX_FRACTION` must never diverge
between training and inference (`data.py`'s own docstring rule):

```python
_CODE_EXECUTION_ROOT = Path("/code_execution")
RUNNING_IN_CODE_EXECUTION = _CODE_EXECUTION_ROOT.exists()

if RUNNING_IN_CODE_EXECUTION:
    NIFTI_DIR = _CODE_EXECUTION_ROOT / "data" / "niftis"
    SUBMISSION_FORMAT_PATH = _CODE_EXECUTION_ROOT / "data" / "submission_format.csv"
else:
    NIFTI_DIR = DATA_RAW / "niftis"
    SUBMISSION_FORMAT_PATH = None

...

if not RUNNING_IN_CODE_EXECUTION:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
```

The detection itself (`RUNNING_IN_CODE_EXECUTION`, and the two path
variables) must be derivable from an injectable root, not the literal
`Path("/code_execution")`, so `tests/test_config.py` can exercise both
branches with a `tmp_path`-backed fake root rather than depending on
`/code_execution` actually existing on the test machine. Refactor as a
small pure function:

```python
def _resolve_runtime_paths(code_execution_root):
    if code_execution_root.exists():
        return True, code_execution_root / "data" / "niftis", code_execution_root / "data" / "submission_format.csv"
    return False, DATA_RAW / "niftis", None

RUNNING_IN_CODE_EXECUTION, NIFTI_DIR, SUBMISSION_FORMAT_PATH = _resolve_runtime_paths(Path("/code_execution"))
```

**Testing (`tests/test_config.py`, new, no real data):**
1. `_resolve_runtime_paths` with a `tmp_path` that exists returns
   `RUNNING_IN_CODE_EXECUTION=True` and the two paths nested under it.
2. `_resolve_runtime_paths` with a non-existent path returns
   `RUNNING_IN_CODE_EXECUTION=False`, `NIFTI_DIR` under `DATA_RAW`, and
   `SUBMISSION_FORMAT_PATH=None`.

## `features.py`: `extract_baseline_features()`

`notebooks/03_baseline_classical.ipynb`'s cell 1 already computes
`abs_asym`/`striatal_ratio`/`inplane_family` from a loaded volume +
spacing via `features.striatum_mask` → `signed_asymmetry`/
`striatal_ratio`, inline, not as a reusable function. `main.py` needs
the identical computation for every test volume; inlining it a third
time (after the notebook and the never-shipped EDA version) is exactly
the duplication `inplane_family`'s own docstring already flagged once.

```python
def extract_baseline_features(volume, spacing, target_ml=20.0):
    """(abs_asym, striatal_ratio, inplane_family) for one volume, or None
    if the striatum mask is degenerate (mirrors
    notebooks/03_baseline_classical.ipynb's skip condition -- a row this
    happens to should be treated as a missing feature by the caller, not
    silently zero-filled).
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

**Testing (`tests/test_features.py`, extend, synthetic volumes only):**
1. A volume with a clear off-center bright blob returns a dict with all
   three keys and plausible values (same shape of test the existing
   `striatum_mask`/`signed_asymmetry` tests already use).
2. A degenerate (all-zero) volume returns `None`.

If `main.py` hits a `None` here for a test volume (should be rare —
notebook 03 skipped 0 volumes out of 1362 at the same `target_ml`, but
the test set could differ), the design's fallback is: use the CNN
probability alone for that row (skip the blend, log the uid count but
never the uid itself) rather than crash the whole run over one volume.

## `scripts/build_submission_assets.py` — `[RUN ME]`, not run by Claude

Real per-row training data access (`baseline_features.csv`'s labels) to
fit the classical pipeline — same rule as the notebooks. Not a notebook
because it has no cells to inspect/paste results into; it's pure
packaging. Responsibilities:

1. Fit `model.build_combat_baseline()` on **100% of labeled training
   data** (`X=[abs_asym, striatal_ratio]`, `y=is_pathologic`,
   `batch=inplane_family`, from `baseline_features.csv`) — not refit
   inside the container at inference time, so no training labels are
   bundled into `submission.zip` and inference has one less moving part.
2. Pickle the fitted pipeline to `submission_src/model_assets/combat_baseline.pkl`.
3. Copy the 25 `checkpoints/rung3_seed{42-46}_fold{0-4}.pt` files to
   `submission_src/model_assets/checkpoints/` (file copy, not a
   per-row read — same as the existing `torch.save` calls already did).
4. Copy `src/config.py`, `src/data.py`, `src/model.py`, `src/features.py`,
   `src/dataset.py` into `submission_src/` (after `config.py`'s
   env-detection change and `features.py`'s new function land).
5. Print only aggregate confirmation (file counts, pipeline `classes_`
   sanity check) — no row-level output, matching the notebooks'
   AI-assistant data rule.

## `main.py`

```python
DATA_DIR = Path("/code_execution/data")
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")
MODEL_ASSETS = Path(__file__).parent / "model_assets"
CNN_WEIGHT = 0.70  # README.md, 2026-09-09 leave-one-repeat-out result
```

Note `main.py` hardcodes `/code_execution/data` directly (matching the
DrivenData example) rather than importing `config.NIFTI_DIR` — the two
must still agree, but keeping `main.py`'s own top-level paths explicit
means a reader can see the whole I/O contract without opening
`config.py`, and it's one less thing that depends on `config.py`'s
env-detection actually firing correctly. `data.load_volume` (used
indirectly via `dataset.DatParkinsonDataset`) still goes through
`config.NIFTI_DIR`, so the two paths are asserted equal at the top of
`main()` as a cheap sanity check, not just assumed to agree.

Steps:
1. Read `submission_format.csv` — this fixes the uid order the output
   must match exactly.
2. **CNN ensemble**: for each of the 25 checkpoints, load `DatCNN`,
   `load_state_dict`, run batched inference (`DatParkinsonDataset` +
   `DataLoader`, `batch_size=config.BATCH_SIZE`, `num_workers=0` — same
   rule as training: a `CachedVolumeStore` isn't used here so
   `num_workers` could safely be >0, but inference is I/O-light enough
   (one pass, no repeat access) that the extra complexity isn't worth
   it) over every test uid, accumulate sigmoid probabilities, discard
   the model. Average the 25 probability vectors.
3. **Classical baseline**: for each test uid, load the volume once more
   (already-resampled-for-CNN array isn't reusable here — the classical
   features need the *original* volume/spacing, not the CNN's
   resampled/cropped one), call `features.extract_baseline_features`;
   `None` rows fall back to the CNN-alone probability (see above).
   Batch the valid rows through the pickled pipeline's `predict_proba`.
4. **Blend**: `CNN_WEIGHT * cnn_prob + (1 - CNN_WEIGHT) * baseline_prob`
   per row (CNN-alone for the rare `None`-feature rows).
5. Write `submission.csv` with columns `uid, is_pathologic`, in
   `submission_format.csv`'s row order (`.set_index("uid").loc[...]` or
   equivalent — never re-sort, never rely on dict ordering across a
   `set`).
6. Logging: elapsed time per phase (cache/load, each checkpoint's pass,
   classical features, write) and row counts only — never a uid next to
   its prediction, never per-row feature values (the checklist's own
   "does not print or log any information about the test dataset" rule,
   and this project's standing AI-assistant data rule).

**Testing**: `main.py` itself is thin orchestration over already-tested
functions (`data.load_volume`, `model.predict`, `features.extract_baseline_features`,
the pickled pipeline's `predict_proba`) and is only exercised end-to-end
via the Docker rehearsal — not unit-tested in isolation. Anything with
real logic (the env-detection, the new feature-extraction function) is
tested above, before it's copied into `submission_src/`.

## Definition of done

- `pytest tests/` passes (existing tests + new `test_config.py` +
  extended `test_features.py`), synthetic data only.
- `config.py`'s env-detection and `features.py`'s
  `extract_baseline_features` are the only changes to `src/*.py` —
  everything else is copied verbatim by the build script.
- `scripts/build_submission_assets.py` written, `[RUN ME]`-marked, not
  run by Claude.
- `main.py` written; not run by Claude (real data + Docker).
- User runs `scripts/build_submission_assets.py`, then the runtime
  repo's `just pack-submission` + `just test-submission` against the
  downloaded smoke-test data, and reports back: did it complete inside
  6 minutes, did `submission.csv` match `submission_format.csv`'s shape/
  columns, any errors. **Not attempted in this pass**: deciding what to
  do if the 25-checkpoint ensemble blows the time budget (the fallback —
  fewer checkpoints — is a one-line change, made only if the smoke test
  actually shows it's needed).
- `README.md` Next steps updated once the smoke test result comes back.
