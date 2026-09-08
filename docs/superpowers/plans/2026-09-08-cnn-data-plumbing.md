# CNN Data Plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and unit-test the deterministic data pipeline (`src/data.py`), the `Dataset` wrapper (`src/dataset.py`), and a first from-scratch 3D CNN architecture (`src/model.py` additions), then write a rung-0/1 smoke-test notebook -- the plumbing for the project's 3D-CNN main track, stopping before any full training loop.

**Architecture:** `data.py::load_volume(uid)` composes three separately-tested, deterministic steps (affine-aware resample -> fixed-physical crop/pad -> background-aware per-volume z-score) into one `(1, 56, 30, 44)` float32 array. `dataset.py::DatParkinsonDataset` is a thin `torch.utils.data.Dataset` wrapping `load_volume`. `model.py` gains `DatCNN` (a small `Conv3d`/`BatchNorm3d`/`ReLU`/`MaxPool3d` stack with He init), `build_model()`, and `predict()`.

**Tech Stack:** Python 3.12, PyTorch 2.14.0+cu126, nibabel 5.4.2, numpy<2.3, scipy, pytest. All code and tests run in the `dat-parkinson` conda env.

**Spec:** `docs/superpowers/specs/2026-09-08-cnn-data-plumbing-design.md`

## Global Constraints

- TDD throughout (`structuring-ml-projects` SKILL.md step 5): every `data.py`/`dataset.py`/`model.py` function gets a failing test written first, on tiny synthetic arrays -- never real patient data (the project's AI-assistant data rule, `README.md`).
- `notebooks/05_cnn_smoke_test.ipynb`'s cells that load real `.nii.gz` volumes and labels must be marked **[RUN ME]** and are never executed by the plan's implementer (Claude) -- only written.
- Run tests with: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/ -q` (the project's own conda env; do not use a bare `python`/`pytest` on PATH).
- Every new function gets a one-paragraph docstring citing its source/rationale per this project's existing style (see `src/features.py`, `src/evaluate.py`).
- `config.py` already has every constant this plan needs (`TARGET_SPACING`, `CROP_SIZE_MM`, `CROP_CENTER_MM`, `TARGET_SHAPE`, `BACKGROUND_PERCENTILE`, `BACKGROUND_MAX_FRACTION`, `NIFTI_DIR`, `TRAIN_LABELS_PATH`, `SEED`, `DEVICE`) -- do not re-derive or change any of them.

---

## Task 1: `resample_to_spacing` in `src/data.py`

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `resample_to_spacing(volume: np.ndarray, affine: np.ndarray, target_spacing: tuple[float, float, float], order: int = 1) -> tuple[np.ndarray, np.ndarray]` -- returns `(resampled_volume, resampled_affine)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
"""Unit tests for src/data.py, using small synthetic 3D arrays -- never
real patient volumes, per the project's AI-assistant data rule.
"""

import numpy as np
import pytest

import data


def _oblique_affine(spacing=2.0, tilt_deg=15.0, offset=(-10.0, -10.0, -10.0)):
    """A synthetic oblique affine (rotation around z), matching the shape
    of the real confound EDA found (`README.md`: 40% of volumes oblique,
    up to 40.3 deg) -- `as_closest_canonical()` would not correct this.
    """
    theta = np.radians(tilt_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    affine = np.eye(4)
    affine[:3, :3] = rot * spacing
    affine[:3, 3] = offset
    return affine


def test_resample_to_spacing_changes_voxel_count_for_oblique_affine():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[8:12, 8:12, 8:12] = 500.0
    affine = _oblique_affine(spacing=2.0, tilt_deg=15.0)

    resampled, new_affine = data.resample_to_spacing(volume, affine, (1.0, 1.0, 1.0))

    # 2mm voxels resampled to 1mm voxels: roughly double the extent per axis.
    assert resampled.shape[0] > volume.shape[0]
    assert new_affine.shape == (4, 4)


def test_resample_to_spacing_preserves_signal_for_identity_affine():
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[4:6, 4:6, 4:6] = 1000.0
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0

    resampled, _ = data.resample_to_spacing(volume, affine, (2.0, 2.0, 2.0))

    assert resampled.shape == volume.shape
    assert resampled.max() == pytest.approx(1000.0, rel=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 3: Write the implementation**

```python
# src/data.py
"""Deterministic geometry/intensity preprocessing for DaT-SPECT volumes
(the CNN track). Every function here is the same one used at training and
inference time -- never reimplemented in the inference path
(`deep-learning-imaging.md`'s "Common mistakes").

Unlike `features.py` (the classical baseline's adaptive per-volume
striatum mask), this pipeline resamples to a fixed physical spacing and
crops/pads to a fixed physical box every time -- the standard pattern for
feeding a CNN a consistent input geometry.
"""

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np


def resample_to_spacing(volume, affine, target_spacing, order=1):
    """Resample `volume` (+ its affine) to `target_spacing` mm, through
    the full affine -- not just axis permutation. `nibabel.processing
    .resample_to_output` is nibabel's own affine-aware resampler, which is
    what EDA's finding requires (40% of volumes are oblique, up to 40.3
    deg; `as_closest_canonical()` does not correct this -- README.md).
    Its output is RAS-aligned, so no separate reorientation step is
    needed. `order=1` (trilinear) by default; medical volumes should not
    use the default `order=3` spline, which can ring/overshoot near sharp
    edges.

    Returns `(resampled_volume, resampled_affine)`.
    """
    img = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    out = nibproc.resample_to_output(img, voxel_sizes=target_spacing, order=order)
    return out.get_fdata(), out.affine
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "Add data.py::resample_to_spacing (affine-aware, TDD)"
```

---

## Task 2: `crop_or_pad` in `src/data.py`

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing from Task 1 (takes `spacing` directly, not an affine, so it stays independently testable).
- Produces: `crop_or_pad(volume: np.ndarray, spacing: tuple[float, float, float], center_mm: tuple[float, float, float], target_shape: tuple[int, int, int]) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_data.py

def test_crop_or_pad_extracts_target_shape_centered_at_offset():
    volume = np.zeros((40, 40, 40), dtype=np.float32)
    volume[18:22, 18:22, 18:22] = 777.0  # centered at voxel (19.5,19.5,19.5)

    cropped = data.crop_or_pad(volume, spacing=(1.0, 1.0, 1.0),
                                center_mm=(0.0, 0.0, 0.0), target_shape=(10, 10, 10))

    assert cropped.shape == (10, 10, 10)
    assert cropped.max() == pytest.approx(777.0)


def test_crop_or_pad_respects_center_mm_offset():
    volume = np.zeros((40, 40, 40), dtype=np.float32)
    volume[28:32, 18:22, 18:22] = 777.0  # offset +10 voxels on axis 0

    cropped = data.crop_or_pad(volume, spacing=(1.0, 1.0, 1.0),
                                center_mm=(10.0, 0.0, 0.0), target_shape=(10, 10, 10))

    assert cropped.max() == pytest.approx(777.0)
    # Signal should now land near the center of the *cropped* volume.
    assert cropped[4:6, 4:6, 4:6].max() == pytest.approx(777.0)


def test_crop_or_pad_zero_pads_when_target_exceeds_volume():
    """Tight-FOV case: `config.py` documents volumes where TARGET_SHAPE's
    z-extent fits MIN_FOV_MM by only 3mm -- crop_or_pad must pad, not
    crash or silently clip, when the source volume is smaller than the
    target shape on an axis.
    """
    volume = np.full((56, 30, 20), 10.0, dtype=np.float32)

    padded = data.crop_or_pad(volume, spacing=(1.0, 1.0, 1.0),
                               center_mm=(0.0, 0.0, 0.0), target_shape=(56, 30, 44))

    assert padded.shape == (56, 30, 44)
    assert (padded[:, :, 0] == 0.0).all()  # padded edge
    assert (padded[:, :, 22] == 10.0).all()  # original data preserved near center
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v -k crop_or_pad`
Expected: FAIL with `AttributeError: module 'data' has no attribute 'crop_or_pad'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/data.py

def crop_or_pad(volume, spacing, center_mm, target_shape):
    """Crop or zero-pad `volume` (already resampled to `spacing`) to
    `target_shape` voxels, centered at the volume's geometric center
    offset by `center_mm` (mm, RAS axes -- same convention as
    `config.CROP_CENTER_MM`). Zero-pads on any axis where `target_shape`
    extends past the volume's edge (the tight-FOV volumes noted in
    `config.py`: the crop fits `MIN_FOV_MM` by only 3mm on z).
    """
    volume = np.asarray(volume)
    out = np.zeros(target_shape, dtype=volume.dtype)
    src_slices, dst_slices = [], []
    for axis in range(3):
        center_vox = (volume.shape[axis] - 1) / 2.0 + center_mm[axis] / spacing[axis]
        start = int(round(center_vox - target_shape[axis] / 2.0))
        end = start + target_shape[axis]
        src_start, src_end = max(start, 0), min(end, volume.shape[axis])
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v -k crop_or_pad`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "Add data.py::crop_or_pad (fixed physical box, TDD)"
```

---

## Task 3: `normalize_intensity` in `src/data.py`

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `normalize_intensity(volume: np.ndarray, background_percentile: float = config.BACKGROUND_PERCENTILE, background_max_fraction: float = config.BACKGROUND_MAX_FRACTION) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_data.py

def test_normalize_intensity_zero_means_the_foreground():
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[:5] = 100.0   # foreground half
    volume[5:] = 0.0     # background half

    normalized = data.normalize_intensity(volume, background_percentile=30,
                                           background_max_fraction=0.05)

    # Foreground (>threshold) voxels should be ~zero-mean.
    assert normalized[:5].mean() == pytest.approx(0.0, abs=1e-4)


def test_normalize_intensity_clips_negative_values_first():
    """A handful of volumes are stored as int16, not uint16, and can carry
    small negative artifacts (EDA section 5, `features.py`'s same
    np.clip guard)."""
    volume = np.array([[[-5.0, 10.0], [20.0, 30.0]]], dtype=np.float32)

    normalized = data.normalize_intensity(volume)

    assert np.isfinite(normalized).all()


def test_normalize_intensity_handles_all_zero_volume():
    volume = np.zeros((5, 5, 5), dtype=np.float32)

    normalized = data.normalize_intensity(volume)

    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, volume)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v -k normalize_intensity`
Expected: FAIL with `AttributeError: module 'data' has no attribute 'normalize_intensity'`

- [ ] **Step 3: Write the implementation**

```python
# add near the top of src/data.py, after the existing imports
import config


# append to src/data.py

def normalize_intensity(volume, background_percentile=config.BACKGROUND_PERCENTILE,
                         background_max_fraction=config.BACKGROUND_MAX_FRACTION):
    """Background-aware per-volume z-score, computed on the *cropped*
    volume. `np.clip(volume, 0, None)` first (16 volumes are stored as
    int16, not uint16 -- EDA section 5, same guard as `features.py`).
    Background threshold = min(percentile(volume, background_percentile),
    background_max_fraction * volume.max()) -- the EDA-validated rule
    already in `config.py` (adaptive, since a flat percentile cuts into
    brain on tight-FOV volumes). z-score uses the foreground
    (above-threshold) voxels' mean/std, applied to the whole volume.
    """
    volume = np.clip(volume, 0, None).astype(np.float32)
    if volume.max() <= 0:
        return volume  # degenerate all-zero volume; nothing to normalize
    threshold = min(np.percentile(volume, background_percentile),
                     background_max_fraction * volume.max())
    foreground = volume[volume > threshold]
    if foreground.size == 0:
        foreground = volume.ravel()
    mean = foreground.mean()
    std = foreground.std()
    if std <= 0:
        std = 1.0
    return (volume - mean) / std
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v -k normalize_intensity`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "Add data.py::normalize_intensity (background-aware z-score, TDD)"
```

---

## Task 4: `load_volume` in `src/data.py`

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `resample_to_spacing` (Task 1), `crop_or_pad` (Task 2), `normalize_intensity` (Task 3); `config.NIFTI_DIR`, `config.TARGET_SPACING`, `config.CROP_CENTER_MM`, `config.TARGET_SHAPE`.
- Produces: `load_volume(uid: str) -> np.ndarray` -- shape `(1, *config.TARGET_SHAPE)`, dtype `float32`. This is the one function both `dataset.py` (Task 5) and any future inference path call.

- [ ] **Step 1: Write the failing test**

Writes a synthetic `.nii.gz` to `tmp_path` (never a real patient file) and points `config.NIFTI_DIR` at it via `monkeypatch`.

```python
# append to tests/test_data.py

def test_load_volume_returns_target_shape_with_channel_dim(tmp_path, monkeypatch):
    import nibabel as nib

    volume = np.zeros((60, 60, 40), dtype=np.float32)
    volume[25:35, 25:35, 15:25] = 500.0
    affine = np.eye(4) * 2.46
    affine[3, 3] = 1.0
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "synthetic_uid.nii.gz")
    monkeypatch.setattr(data.config, "NIFTI_DIR", tmp_path)

    out = data.load_volume("synthetic_uid")

    assert out.shape == (1, *data.config.TARGET_SHAPE)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v -k load_volume`
Expected: FAIL with `AttributeError: module 'data' has no attribute 'load_volume'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/data.py

def load_volume(uid):
    """Load, resample, crop, and normalize the volume for `uid`. Returns
    a `(1, *config.TARGET_SHAPE)` float32 array -- the single function
    both training and inference call, never reimplemented in the
    inference path.
    """
    path = config.NIFTI_DIR / f"{uid}.nii.gz"
    img = nib.load(str(path))
    resampled, _ = resample_to_spacing(img.get_fdata(), img.affine, config.TARGET_SPACING)
    cropped = crop_or_pad(resampled, config.TARGET_SPACING, config.CROP_CENTER_MM,
                           config.TARGET_SHAPE)
    normalized = normalize_intensity(cropped)
    return normalized[np.newaxis, ...].astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_data.py -v`
Expected: PASS (all tests in the file, 9 total)

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "Add data.py::load_volume, composing the preprocessing pipeline"
```

---

## Task 5: `DatParkinsonDataset` in `src/dataset.py`

**Files:**
- Create: `src/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `data.load_volume(uid) -> np.ndarray` (Task 4) -- imported as `import data`, so tests can `monkeypatch.setattr(data, "load_volume", fake)`.
- Produces: `DatParkinsonDataset(uids, labels=None)`, a `torch.utils.data.Dataset`. `__getitem__` returns `(tensor, label)` when `labels` is given, `(tensor, uid)` when `labels=None` (inference).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
"""Unit tests for src/dataset.py. Uses a monkeypatched data.load_volume
returning synthetic arrays -- never real patient data.
"""

import numpy as np
import pytest
import torch

import data
import dataset


def _fake_load_volume(uid):
    rng = np.random.RandomState(abs(hash(uid)) % (2**31))
    return rng.normal(size=(1, 56, 30, 44)).astype(np.float32)


def test_dataset_with_labels_returns_tensor_and_label(monkeypatch):
    monkeypatch.setattr(data, "load_volume", _fake_load_volume)
    ds = dataset.DatParkinsonDataset(uids=["a", "b", "c"], labels=[0, 1, 0])

    tensor, label = ds[1]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 56, 30, 44)
    assert tensor.dtype == torch.float32
    assert label == pytest.approx(1.0)
    assert isinstance(label, torch.Tensor)


def test_dataset_without_labels_returns_tensor_and_uid(monkeypatch):
    monkeypatch.setattr(data, "load_volume", _fake_load_volume)
    ds = dataset.DatParkinsonDataset(uids=["x", "y"])

    tensor, uid = ds[0]

    assert isinstance(tensor, torch.Tensor)
    assert uid == "x"


def test_dataset_length_matches_uids():
    ds = dataset.DatParkinsonDataset(uids=["a", "b", "c", "d"])

    assert len(ds) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataset'`

- [ ] **Step 3: Write the implementation**

```python
# src/dataset.py
"""torch.utils.data.Dataset wrapper for DaT-SPECT volumes. Pure glue --
index -> data.py::load_volume -> tensor -- per `deep-learning-imaging.md`'s
dataset.py rule: test its output contract, not its content.
"""

import torch
from torch.utils.data import Dataset

import data


class DatParkinsonDataset(Dataset):
    """`uids[i]` -> `data.load_volume(uids[i])` as a float32 tensor.
    With `labels` given, returns `(tensor, label)`; with `labels=None`
    (inference), returns `(tensor, uid)`.
    """

    def __init__(self, uids, labels=None):
        self.uids = list(uids)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        tensor = torch.from_numpy(data.load_volume(uid))
        if self.labels is None:
            return tensor, uid
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_dataset.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "Add dataset.py::DatParkinsonDataset (TDD)"
```

---

## Task 6: `DatCNN`, `build_model`, `predict` in `src/model.py`

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `DatCNN(nn.Module)`, `build_model() -> DatCNN`, `predict(model: DatCNN, x: torch.Tensor) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py

import torch


# --- DatCNN / build_model / predict ------------------------------------------

def test_build_model_forward_pass_shape_and_finiteness():
    model = model_module.build_model()
    x = torch.randn(4, 1, 56, 30, 44)

    out = model(x)

    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_build_model_returns_a_new_instance_each_call():
    first = model_module.build_model()
    second = model_module.build_model()

    assert first is not second


def test_predict_returns_probabilities_in_zero_one():
    model = model_module.build_model()
    x = torch.randn(3, 1, 56, 30, 44)

    proba = model_module.predict(model, x)

    assert proba.shape == (3,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_datcnn_can_overfit_a_tiny_batch():
    """Rung-0-style sanity check (deep-learning-imaging.md): the
    architecture itself must be able to drive loss to near-zero on a
    handful of samples -- catches a frozen-parameter or shape bug that a
    single forward pass would not."""
    torch.manual_seed(0)
    model = model_module.build_model()
    x = torch.randn(4, 1, 56, 30, 44)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.1
```

Note: the existing `tests/test_model.py` imports the module as `import model` and uses `model.build_classical_baseline()`. Since this task's tests use `torch.nn.Module` naming that would collide with the `model` module name inside test functions, import the module under an alias at the top of the file instead of adding a second bare `import model`:

```python
# at the top of tests/test_model.py, alongside the existing imports
import model as model_module
```

(Keep the existing `import model` used by the classical-baseline tests as-is; just add the alias import for the new tests to use.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_model.py -v -k "build_model or predict or datcnn"`
Expected: FAIL with `AttributeError: module 'model' has no attribute 'build_model'`

- [ ] **Step 3: Write the implementation**

```python
# add near the top of src/model.py, alongside the existing imports
import torch
import torch.nn as nn


# append to src/model.py

class DatCNN(nn.Module):
    """Small 3D CNN, trained from scratch (no pretrained backbone --
    RESOURCES.md has three independent findings against ImageNet transfer
    on this modality, and DINOv3/PPMI eligibility is unresolved/ineligible;
    see docs/superpowers/specs/2026-09-08-cnn-data-plumbing-design.md).
    He init / BatchNorm / dropout / Adam, no LR schedule (Geron Ch.11
    default DNN config; matches config.LR_SCHEDULE = None). Input is
    `config.TARGET_SHAPE` = (56, 30, 44), 1 channel. Returns a raw logit
    (not a sigmoid output) -- use with `BCEWithLogitsLoss`.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.BatchNorm3d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


def build_model():
    """`DatCNN` factory -- fresh instance each call, mirroring
    `build_classical_baseline()`'s existing convention."""
    return DatCNN()


def predict(model, x):
    """Forward pass -> sigmoid -> numpy probabilities -- the deep-learning
    analogue of `predict_proba` on the classical pipelines. Puts the model
    in eval mode (disables dropout/BatchNorm training behavior) and runs
    without gradient tracking.
    """
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_model.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Run the full project test suite**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/ -q`
Expected: PASS, all tests (28 existing + this plan's new tests)

- [ ] **Step 6: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Add DatCNN/build_model/predict: first 3D-CNN architecture (TDD)"
```

---

## Task 7: `notebooks/05_cnn_smoke_test.ipynb` (rung 0/1, [RUN ME])

**Files:**
- Create: `notebooks/05_cnn_smoke_test.ipynb`

**Interfaces:**
- Consumes: `data.load_volume` (Task 4, via `dataset.DatParkinsonDataset`), `dataset.DatParkinsonDataset` (Task 5), `model.build_model` (Task 6), `config.TRAIN_LABELS_PATH`, `config.UID_COLUMN`, `config.TARGET_COLUMN`, `evaluate.make_folds`.
- Produces: nothing consumed by later tasks -- this is the terminal deliverable of this plan. Not executed by Claude; the user runs it and reports back the printed shapes/loss curve, per the AI-assistant data rule.

This task has no TDD cycle (it is a notebook the user runs, not library code) -- write the cells directly.

- [ ] **Step 1: Write the notebook**

Structure, following this project's existing notebook conventions (`notebooks/01_eda_volumes.ipynb` through `04_combat_harmonization.ipynb`): a markdown intro cell stating purpose/decision/data-handling per the AI-assistant rule, then the `[RUN ME]` code cell(s), then a markdown findings cell with blanks for the user's results.

Markdown intro cell:

```markdown
# 05 — CNN smoke test: does the data/model plumbing run end-to-end and can it overfit a tiny batch?

**Decision this feeds (`structuring-ml-projects` SKILL.md, cheap-proxy
ladder, `deep-learning-imaging.md`):** rung 0 (mandatory before any
longer run) — right shapes/dtypes end to end, and the model can overfit
a tiny labeled subset to near-zero loss. If rung 0 passes, rung 1 (a
10-20% stratified subset, one fold) checks the loss curve is sane. Note:
**this notebook does not decide anything about the model's real
performance** — only rung 3 (a full CV run, not built yet) clears the
gate against the classical baseline (`model.build_combat_baseline()`,
0.5290 log loss). See
`docs/superpowers/specs/2026-09-08-cnn-data-plumbing-design.md`.

**Data handling:** this notebook loads real `.nii.gz` volumes and
row-level labels, so per the AI-assistant data rule (`README.md`) it is
**[RUN ME]** — run it yourself, share back only the printed shapes and
loss values, not any per-row output.
```

`[RUN ME]` code cell (rung 0):

```python
# [RUN ME] -- loads real pixel data + row-level labels.
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import torch

import config
import dataset
import model

torch.manual_seed(config.SEED)

labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
rng = np.random.RandomState(config.SEED)
smoke_idx = rng.choice(len(labels_df), size=24, replace=False)
smoke_df = labels_df.iloc[smoke_idx]

smoke_ds = dataset.DatParkinsonDataset(
    uids=smoke_df[config.UID_COLUMN].tolist(),
    labels=smoke_df[config.TARGET_COLUMN].tolist(),
)
loader = torch.utils.data.DataLoader(smoke_ds, batch_size=8, shuffle=True)

net = model.build_model()
opt = torch.optim.Adam(net.parameters(), lr=config.LR)
loss_fn = torch.nn.BCEWithLogitsLoss()

print(f"rung 0: {len(smoke_ds)} labeled volumes, checking shapes + overfit")
for x, y in loader:
    print("batch shapes:", x.shape, x.dtype, y.shape, y.dtype)
    break

losses = []
for epoch in range(30):
    epoch_loss = 0.0
    for x, y in loader:
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * x.shape[0]
    losses.append(epoch_loss / len(smoke_ds))

print("rung 0 loss curve (first/mid/last):", losses[0], losses[len(losses) // 2], losses[-1])
```

`[RUN ME]` code cell (rung 1, only run if rung 0's last loss is near zero):

```python
# [RUN ME] -- only run after rung 0 above shows the loss reaching near
# zero. Loads a larger real subset + labels.
import evaluate

# inplane_family isn't in train_labels.csv itself (it's derived from NIfTI
# headers) -- reuse notebooks/03's baseline_features.csv, which already
# has it per uid, rather than re-deriving it here.
family_df = pd.read_csv(config.DATA_PROCESSED / "baseline_features.csv")[
    [config.UID_COLUMN, "inplane_family"]
]
labeled_df = labels_df.merge(family_df, on=config.UID_COLUMN, how="inner")

rng2 = np.random.RandomState(config.SEED)
subset_idx = rng2.choice(len(labeled_df), size=int(0.15 * len(labeled_df)), replace=False)
subset_df = labeled_df.iloc[subset_idx].reset_index(drop=True)

folds = evaluate.make_folds(
    subset_df[config.TARGET_COLUMN].to_numpy(),
    subset_df["inplane_family"].to_numpy(),
    n_splits=config.N_FOLDS, random_state=config.SEED,
)
train_idx, val_idx = folds[0]

train_ds = dataset.DatParkinsonDataset(
    uids=subset_df.iloc[train_idx][config.UID_COLUMN].tolist(),
    labels=subset_df.iloc[train_idx][config.TARGET_COLUMN].tolist(),
)
val_ds = dataset.DatParkinsonDataset(
    uids=subset_df.iloc[val_idx][config.UID_COLUMN].tolist(),
    labels=subset_df.iloc[val_idx][config.TARGET_COLUMN].tolist(),
)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE)

net = model.build_model()
opt = torch.optim.Adam(net.parameters(), lr=config.LR)
loss_fn = torch.nn.BCEWithLogitsLoss()

print(f"rung 1: train={len(train_ds)}, val={len(val_ds)}, {config.EPOCHS} epochs, 1 fold")
for epoch in range(config.EPOCHS):
    net.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()

    net.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            val_probs.append(torch.sigmoid(net(x)).numpy())
            val_labels.append(y.numpy())
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)
    val_loss = evaluate.log_loss_score(val_labels, val_probs)
    if epoch % 5 == 0 or epoch == config.EPOCHS - 1:
        print(f"epoch {epoch}: val log loss = {val_loss:.4f}")

print(f"rung 1 final val log loss: {val_loss:.4f} "
      f"(reference only -- classical build_combat_baseline() = 0.5290; "
      f"this is NOT a gate decision)")
```

Markdown findings cell (template, filled in by the user after running):

```markdown
**What we're looking for:** does the plumbing run end to end with the
right shapes, and can the model overfit a tiny labeled batch (rung 0)?
If so, is the rung-1 loss curve sane (not diverging, not stuck at the
base-rate loss)?

**What we found:** *(paste the printed batch shapes, rung-0 loss curve,
and rung-1 final val log loss here after running the cells above)*

**Decision / next step:** *(if rung 0 fails to overfit, that's a bug in
data.py/dataset.py/model.py to fix before anything else. If rung 0
passes, the follow-up spec — full train.py, augmentation, rung 2/3 — is
next. Rung 1's number is a proxy only; it does not clear the gate.)*
```

- [ ] **Step 2: Sanity-check the notebook's non-data-touching parts**

The notebook cannot be run (it loads real patient data). Instead, verify the import lines and function names it references actually exist:

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -c "import sys; sys.path.insert(0, 'src'); import config, dataset, model, evaluate; assert hasattr(model, 'build_model'); assert hasattr(dataset, 'DatParkinsonDataset'); assert hasattr(config, 'SEED') and hasattr(config, 'UID_COLUMN'); print('all referenced names exist')"`
Expected: `all referenced names exist`

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_cnn_smoke_test.ipynb
git commit -m "Add notebooks/05_cnn_smoke_test.ipynb (rung 0/1, RUN ME)"
```

---

## Definition of done

- [ ] `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/ -q` passes with all tests (28 pre-existing + this plan's new tests across `test_data.py`, `test_dataset.py`, `test_model.py`).
- [ ] `notebooks/05_cnn_smoke_test.ipynb` exists, is `[RUN ME]`-marked, and was never executed by Claude.
- [ ] `README.md` is **not** updated by this plan with any rung-0/1 result numbers -- that happens after the user runs the notebook and reports the numbers back (per the AI-assistant data rule and the notebook-graduation rule: this code stays notebook-referenced, not wired into any production path, until a follow-up plan's rung 3 clears the gate).
