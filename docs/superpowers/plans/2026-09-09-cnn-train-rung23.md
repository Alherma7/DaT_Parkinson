# CNN train.py + rung 2/3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, tested training loop (`src/train.py`), a shared
disk-persisted preprocessing cache (`src/cache.py`), and the two `[RUN ME]`
notebooks (rung 2: transfer check + batch/LR pick + timing; rung 3: nested
5-fold CV, repeated 5x, gated against the classical baseline) that decide
whether the 3D CNN graduates past the classical baseline.

**Architecture:** `src/train.py::train_one_fold` is a pure, synthetic-data-
testable training loop (early stopping, AMP, optional scheduler/seed) that
never touches disk. `src/cache.py::CachedVolumeStore` is a pure,
dependency-injected uid→array cache, disk-persisted, that both notebooks
build once and share across every CV fold. `src/dataset.py` gains an
injectable `load_fn` so `DatParkinsonDataset` can be backed by the cache.
`src/features.py` gains `inplane_family()`, promoted from notebook-local
duplication since it now has three `src/`-level consumers. Real-data
orchestration (nested CV splitting, checkpoint persistence, the gate
decision) lives in the two notebooks, per the project's AI-assistant data
rule and existing `deep-learning-imaging.md` convention.

**Tech Stack:** Python 3.12, PyTorch 2.14.0+cu126, `torch.amp` (not the
deprecated `torch.cuda.amp`), numpy, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-cnn-train-rung23-design.md`

## Global Constraints

- Run all Python/pytest commands with
  `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe"` (the project's
  conda env), from the repo root `C:\Users\alher\Desktop\DaT_Parkinson`.
- **AI-assistant data rule**: never read/open/run any cell that loads real
  `.nii.gz` volumes or row-level `train_labels.csv`/`baseline_features.csv`
  contents. All new tests use synthetic data only. Both notebooks are
  `[RUN ME]` — write them, verify only that the names/imports they
  reference exist (Tasks 5 and 6's Step 2), never execute their data cells.
- Use `torch.amp.GradScaler(device_type, enabled=...)` and
  `torch.autocast(device_type=..., enabled=...)` — the `torch.cuda.amp.*`
  spellings are deprecated on this project's torch version.
- Reuse existing config constants (`config.PATIENCE = 10`,
  `config.USE_AMP = True`, `config.BATCH_SIZE`, `config.LR`,
  `config.WEIGHT_DECAY`, `config.EPOCHS`, `config.N_FOLDS`,
  `config.SEED`, `config.DEVICE`, `config.CHECKPOINT_DIR`,
  `config.DATA_PROCESSED`) — do not add new config constants for things
  these already cover.
- No new third-party dependencies — everything needed
  (`torch`, `numpy`, `pandas`, `scikit-learn`) is already in
  `environment.yml`.
- `LR_SCHEDULE` stays `None` for this plan's real runs (the notebook
  `scheduler=` argument exists in `train_one_fold` for testability, but
  rung 2/3 pass `scheduler=None`) — see the spec's "Batch size and
  learning rate" section for why not to change both LR schedule and
  batch size at once.

---

### Task 1: `src/cache.py` — `CachedVolumeStore`

**Files:**
- Create: `src/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: nothing project-specific — takes an injected `load_fn`
  (default `data.load_volume`, from the existing `src/data.py`).
- Produces: `CachedVolumeStore(uids, cache_dir, config_fingerprint, load_fn=None)`
  with `.get(uid) -> np.ndarray` and `__len__`. Consumed by Task 2's
  `DatParkinsonDataset(..., load_fn=...)` and by both notebooks (Tasks 5-6).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache.py`:

```python
"""Unit tests for src/cache.py. Synthetic load_fn only -- never real
patient data.
"""

import numpy as np

from cache import CachedVolumeStore

SHAPE = (2, 3, 4)


def _value_for(uid):
    return (abs(hash(uid)) % 100) / 10.0


def make_load_fn(calls):
    def load_fn(uid):
        calls.append(uid)
        return np.full(SHAPE, _value_for(uid), dtype=np.float32)
    return load_fn


def test_build_calls_load_fn_once_per_uid_then_never_again(tmp_path):
    uids = ["a", "b", "c"]
    calls = []
    store = CachedVolumeStore(uids, tmp_path / "cache", {"v": 1}, load_fn=make_load_fn(calls))

    assert sorted(calls) == sorted(uids)

    calls.clear()
    for uid in uids:
        store.get(uid)
        store.get(uid)
    assert calls == []


def test_returns_correct_volume_per_uid(tmp_path):
    uids = ["a", "b", "c"]
    store = CachedVolumeStore(uids, tmp_path / "cache", {"v": 1}, load_fn=make_load_fn([]))

    for uid in ["c", "a", "b", "a"]:
        expected = np.full(SHAPE, _value_for(uid), dtype=np.float32)
        np.testing.assert_array_equal(store.get(uid), expected)


def test_persists_across_instances_without_recalling_load_fn(tmp_path):
    uids = ["a", "b"]
    cache_dir = tmp_path / "cache"
    first = CachedVolumeStore(uids, cache_dir, {"v": 1}, load_fn=make_load_fn([]))

    def raising_load_fn(uid):
        raise AssertionError("load_fn should not be called when the disk cache is valid")

    second = CachedVolumeStore(uids, cache_dir, {"v": 1}, load_fn=raising_load_fn)
    for uid in uids:
        np.testing.assert_array_equal(second.get(uid), first.get(uid))


def test_config_fingerprint_mismatch_triggers_rebuild(tmp_path):
    uids = ["a", "b"]
    cache_dir = tmp_path / "cache"
    CachedVolumeStore(uids, cache_dir, {"shape": (56, 30, 44)}, load_fn=make_load_fn([]))

    calls_second = []
    CachedVolumeStore(uids, cache_dir, {"shape": (56, 30, 45)}, load_fn=make_load_fn(calls_second))
    assert sorted(calls_second) == sorted(uids)


def test_tuple_and_list_fingerprints_compare_equal(tmp_path):
    # JSON round-trips a tuple fingerprint value into a list -- the
    # comparison must normalize both sides, or the cache would silently
    # rebuild on every single instantiation.
    uids = ["a", "b"]
    cache_dir = tmp_path / "cache"
    CachedVolumeStore(uids, cache_dir, {"shape": (56, 30, 44)}, load_fn=make_load_fn([]))

    calls_second = []
    CachedVolumeStore(uids, cache_dir, {"shape": [56, 30, 44]}, load_fn=make_load_fn(calls_second))
    assert calls_second == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_cache.py -v`
Expected: FAIL (collection error) with `ModuleNotFoundError: No module named 'cache'`

- [ ] **Step 3: Write the implementation**

Create `src/cache.py`:

```python
"""Disk-persisted, uid-keyed cache for data.load_volume, shared across CV
folds so each volume is preprocessed at most once per notebook session
instead of once per fold (or once per fold per seed-repeat) -- see
docs/superpowers/specs/2026-09-09-cnn-train-rung23-design.md.
"""

import json
from pathlib import Path

import numpy as np

import data


class CachedVolumeStore:
    """uid -> data.load_volume(uid), computed at most once per uid and
    persisted to disk so a kernel restart doesn't repeat preprocessing.
    Rebuilds automatically if `config_fingerprint` doesn't match what's on
    disk (e.g. TARGET_SHAPE changed) or the uid set differs. `load_fn` is
    injected (default `data.load_volume`) so this is testable against a
    fake loader returning synthetic arrays -- never real patient data in
    tests.
    """

    def __init__(self, uids, cache_dir, config_fingerprint, load_fn=None):
        self.load_fn = load_fn or data.load_volume
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._array_path = self.cache_dir / "volumes.npy"
        self._index_path = self.cache_dir / "index.json"
        self._fingerprint_path = self.cache_dir / "fingerprint.json"
        self._uids = list(uids)
        # Round-trip through JSON now so later comparisons are apples to
        # apples (a tuple config value becomes a list on disk either way).
        self._config_fingerprint = json.loads(json.dumps(config_fingerprint))
        self._uid_to_row = {}
        self._array = None
        if self._on_disk_matches():
            self._load_from_disk()
        else:
            self._build()

    def _on_disk_matches(self):
        if not (self._array_path.exists() and self._index_path.exists()
                and self._fingerprint_path.exists()):
            return False
        with open(self._fingerprint_path) as f:
            disk_fp = json.load(f)
        if disk_fp != self._config_fingerprint:
            return False
        with open(self._index_path) as f:
            disk_index = json.load(f)
        return set(disk_index.keys()) == set(self._uids)

    def _load_from_disk(self):
        with open(self._index_path) as f:
            self._uid_to_row = json.load(f)
        self._array = np.load(self._array_path)

    def _build(self):
        first = self.load_fn(self._uids[0])
        array = np.empty((len(self._uids), *first.shape), dtype=np.float32)
        array[0] = first
        uid_to_row = {self._uids[0]: 0}
        for i, uid in enumerate(self._uids[1:], start=1):
            array[i] = self.load_fn(uid)
            uid_to_row[uid] = i
        np.save(self._array_path, array)
        with open(self._index_path, "w") as f:
            json.dump(uid_to_row, f)
        with open(self._fingerprint_path, "w") as f:
            json.dump(self._config_fingerprint, f)
        self._uid_to_row = uid_to_row
        self._array = array

    def get(self, uid):
        return self._array[self._uid_to_row[uid]].copy()

    def __len__(self):
        return len(self._uid_to_row)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_cache.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cache.py tests/test_cache.py
git commit -m "Add src/cache.py::CachedVolumeStore (TDD)"
```

---

### Task 2: `src/dataset.py` — injectable `load_fn`

**Files:**
- Modify: `src/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DatParkinsonDataset(uids, labels=None, load_fn=None)` —
  `load_fn` defaults to `data.load_volume` (unchanged default behavior);
  consumed by both notebooks passing `load_fn=volume_cache.get`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dataset.py` (below the existing tests, keep the
existing `_fake_load_volume` helper and imports as-is):

```python
def test_dataset_uses_injected_load_fn_instead_of_data_load_volume(monkeypatch):
    def raising_load_volume(uid):
        raise AssertionError("data.load_volume should not be called when load_fn is injected")
    monkeypatch.setattr(data, "load_volume", raising_load_volume)

    ds = dataset.DatParkinsonDataset(uids=["a", "b"], labels=[0, 1], load_fn=_fake_load_volume)
    tensor, label = ds[0]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, *config.TARGET_SHAPE)
    assert label == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_dataset.py::test_dataset_uses_injected_load_fn_instead_of_data_load_volume -v`
Expected: FAIL with `TypeError: DatParkinsonDataset.__init__() got an unexpected keyword argument 'load_fn'`

- [ ] **Step 3: Write the implementation**

Replace `src/dataset.py` in full:

```python
"""torch.utils.data.Dataset wrapper for DaT-SPECT volumes. Pure glue --
index -> load_fn -> tensor -- per `deep-learning-imaging.md`'s dataset.py
rule: test its output contract, not its content.
"""

import torch
from torch.utils.data import Dataset

import data


class DatParkinsonDataset(Dataset):
    """`uids[i]` -> `load_fn(uids[i])` (default `data.load_volume`) as a
    float32 tensor. With `labels` given, returns `(tensor, label)`; with
    `labels=None` (inference), returns `(tensor, uid)`. `load_fn` is
    injectable so a caller can pass a shared `cache.CachedVolumeStore.get`
    instead of re-reading/re-resampling from disk on every access
    (`docs/superpowers/specs/2026-09-09-cnn-train-rung23-design.md`).
    """

    def __init__(self, uids, labels=None, load_fn=None):
        self.uids = list(uids)
        self.labels = None if labels is None else list(labels)
        self.load_fn = load_fn or data.load_volume

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        tensor = torch.from_numpy(self.load_fn(uid))
        if self.labels is None:
            return tensor, uid
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_dataset.py -v`
Expected: PASS (4 tests — the 3 pre-existing plus the new one)

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "Add injectable load_fn to DatParkinsonDataset (TDD)"
```

---

### Task 3: `src/features.py` — `inplane_family()`

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `features.inplane_family(spacing_x: float) -> str`. Consumed
  by both notebooks (Tasks 5-6) for the LOFO family split and CV
  stratification; `notebooks/03_baseline_classical.ipynb`'s own inline
  copy is left untouched (its recorded results don't need re-running).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features.py` (keep existing imports/tests as-is):

```python
def test_inplane_family_buckets_known_spacings():
    assert features.inplane_family(2.46) == "2.46"
    assert features.inplane_family(3.895) == "3.895"
    assert features.inplane_family(1.47) == "~1.47"


def test_inplane_family_falls_back_for_unbucketed_spacing():
    assert features.inplane_family(3.123) == "other(3.123)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_features.py::test_inplane_family_buckets_known_spacings tests/test_features.py::test_inplane_family_falls_back_for_unbucketed_spacing -v`
Expected: FAIL with `AttributeError: module 'features' has no attribute 'inplane_family'`

- [ ] **Step 3: Write the implementation**

Add to the end of `src/features.py` (after `striatal_ratio`):

```python
def inplane_family(spacing_x):
    """Buckets an in-plane voxel spacing (mm, `img.header.get_zooms()[0]`)
    into the same named families used throughout this project's fold
    design (`evaluate.make_folds`), ComBat batching, and leave-one-family-
    out checks (EDA section 3a, `notebooks/01_eda_volumes.ipynb`). Kept in
    sync with `notebooks/03_baseline_classical.ipynb`'s own inline copy --
    that notebook's already-recorded results are untouched, but new code
    (rung 2/3) uses this version instead of duplicating it again.
    """
    for lo, hi, name in [(1.40, 1.50, "~1.47"), (1.50, 1.80, "~1.5-1.8"),
                         (1.99, 2.01, "2.00"), (2.29, 2.31, "2.30"),
                         (2.39, 2.41, "2.398"), (2.45, 2.47, "2.46"),
                         (3.28, 3.32, "~3.30"), (3.58, 3.60, "3.591"),
                         (3.88, 3.90, "3.895"), (4.41, 4.43, "4.42")]:
        if lo <= spacing_x < hi:
            return name
    return f"other({spacing_x:.3f})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_features.py -v`
Expected: PASS (13 tests — the 11 pre-existing plus 2 new)

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "Add features.inplane_family, promoted from notebook 03 (TDD)"
```

---

### Task 4: `src/train.py` — `train_one_fold`

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: nothing project-specific — takes `model` (any `nn.Module`
  already on `device`), `train_loader`/`val_loader`
  (`torch.utils.data.DataLoader`), `optimizer`, `loss_fn`.
- Produces: `train_one_fold(model, train_loader, val_loader, optimizer,
  loss_fn, epochs, patience, device, use_amp, scheduler=None, seed=None)
  -> (best_state_dict: dict, history: {"train_loss": list[float],
  "val_loss": list[float]})`. Consumed by both notebooks' `train_and_score_nested`
  helper (Tasks 5-6).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train.py`:

```python
"""Unit tests for src/train.py::train_one_fold. Synthetic tiny model +
random tensors only -- never real patient data. Uses two test doubles:
`ScriptedOptimizer` (sets model weights to a scripted per-epoch marker
value, ignoring real gradients) and `ScriptedValLoss` (returns a real
BCE loss during model.train() but a scripted per-epoch value during
model.eval()) so the exact best-epoch/early-stopping behavior can be
asserted without depending on real gradient-descent dynamics.
"""

from unittest import mock

import pytest
import torch

import train


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class ScriptedOptimizer:
    """Stub optimizer: .step() sets the model's weight/bias to the next
    entry of a scripted sequence, ignoring real gradients."""

    def __init__(self, model, weight_sequence):
        self.model = model
        self.weight_sequence = weight_sequence
        self.call_count = 0

    def zero_grad(self):
        pass

    def step(self):
        idx = min(self.call_count, len(self.weight_sequence) - 1)
        value = self.weight_sequence[idx]
        with torch.no_grad():
            self.model.linear.weight.fill_(value)
            self.model.linear.bias.fill_(value)
        self.call_count += 1


class ScriptedValLoss:
    """Real BCE loss while `model.training` is True; a scripted, fixed
    per-call value (one call per epoch, given a single-batch val_loader)
    while `model.training` is False."""

    def __init__(self, model, val_sequence):
        self.model = model
        self.val_sequence = val_sequence
        self.val_call_index = 0
        self._base_loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, y_pred, y_true):
        if self.model.training:
            return self._base_loss(y_pred, y_true)
        value = self.val_sequence[min(self.val_call_index, len(self.val_sequence) - 1)]
        self.val_call_index += 1
        return torch.tensor(value) + 0.0 * y_pred.sum()


def _make_loaders(n_train=8, n_val=4, batch_size=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x_train = torch.randn(n_train, 4, generator=gen)
    y_train = torch.randint(0, 2, (n_train,), generator=gen).float()
    x_val = torch.randn(n_val, 4, generator=gen)
    y_val = torch.randint(0, 2, (n_val,), generator=gen).float()
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    val_ds = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=n_val)
    return train_loader, val_loader


def test_returns_state_dict_and_bounded_history():
    net = TinyNet()
    train_loader, val_loader = _make_loaders()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state, history = train.train_one_fold(
        net, train_loader, val_loader, optimizer, loss_fn,
        epochs=5, patience=10, device=torch.device("cpu"), use_amp=False,
    )

    assert set(best_state.keys()) == set(net.state_dict().keys())
    assert len(history["val_loss"]) <= 5
    assert len(history["train_loss"]) == len(history["val_loss"])


def test_early_stopping_stops_at_best_epoch_plus_patience():
    net = TinyNet()
    train_loader, val_loader = _make_loaders()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    # best at epoch index 1 (0.5); flat (no improvement) for every epoch after.
    loss_fn = ScriptedValLoss(net, val_sequence=[1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    _, history = train.train_one_fold(
        net, train_loader, val_loader, optimizer, loss_fn,
        epochs=20, patience=3, device=torch.device("cpu"), use_amp=False,
    )

    # best epoch index 1, patience 3 -> runs epochs 0,1,2,3,4 (5 total) then stops
    assert len(history["val_loss"]) == 5


def test_best_state_dict_is_not_the_final_epoch_when_final_is_worse():
    net = TinyNet()
    train_loader, val_loader = _make_loaders(n_train=4, batch_size=4, n_val=4)
    optimizer = ScriptedOptimizer(net, weight_sequence=[0.1, 0.2, 0.3, 0.4, 0.5])
    loss_fn = ScriptedValLoss(net, val_sequence=[1.0, 0.2, 0.9, 0.9, 0.9])  # best at epoch index 1

    best_state, history = train.train_one_fold(
        net, train_loader, val_loader, optimizer, loss_fn,
        epochs=5, patience=10, device=torch.device("cpu"), use_amp=False,
    )

    assert len(history["val_loss"]) == 5  # patience never triggered, ran all 5 epochs
    assert torch.allclose(best_state["linear.weight"], torch.full((1, 4), 0.2))
    assert torch.allclose(best_state["linear.bias"], torch.full((1,), 0.2))
    # not the final epoch's weights (still on `net` after training finishes)
    assert torch.allclose(net.state_dict()["linear.weight"], torch.full((1, 4), 0.5))


def test_same_seed_gives_identical_history_different_seed_differs():
    def run(seed):
        torch.manual_seed(0)
        net = TinyNet()
        train_loader, val_loader = _make_loaders(seed=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        _, history = train.train_one_fold(
            net, train_loader, val_loader, optimizer, loss_fn,
            epochs=3, patience=10, device=torch.device("cpu"), use_amp=False, seed=seed,
        )
        return history

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_scheduler_is_stepped_once_per_epoch():
    net = TinyNet()
    train_loader, val_loader = _make_loaders()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scheduler = mock.MagicMock()

    _, history = train.train_one_fold(
        net, train_loader, val_loader, optimizer, loss_fn,
        epochs=4, patience=10, device=torch.device("cpu"), use_amp=False,
        scheduler=scheduler,
    )

    assert scheduler.step.call_count == len(history["train_loss"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_train.py -v`
Expected: FAIL (collection error) with `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Write the implementation**

Create `src/train.py`:

```python
"""Training loop for a binary classifier (DatCNN or any nn.Module) with
early stopping on validation loss. Pure -- no file I/O; the caller
persists checkpoints and reloads the returned best_state_dict before any
inference pass. See
docs/superpowers/specs/2026-09-09-cnn-train-rung23-design.md.
"""

import copy

import torch


def train_one_fold(model, train_loader, val_loader, optimizer, loss_fn,
                    epochs, patience, device, use_amp,
                    scheduler=None, seed=None):
    """Trains `model` (already on `device`) for up to `epochs` epochs,
    stopping early if validation loss hasn't improved in `patience`
    epochs. Returns `(best_state_dict, history)` where `history` =
    `{"train_loss": [...], "val_loss": [...]}` (one entry per epoch
    actually run). `best_state_dict` is a deep copy of the model's state
    at its lowest-validation-loss epoch, not the final epoch's weights.

    `seed`, if given, reseeds `torch.manual_seed` and each loader's own
    `generator` (if it has one) at the start of the call, so a single
    fold is reproducible in isolation rather than depending on how many
    folds ran earlier in the same process.

    `scheduler`, if given, has `.step()` called once per epoch, after the
    optimizer step.

    Validation loss is computed outside autocast (full precision) so AMP
    doesn't add noise to the early-stopping signal, and is a stopping
    signal only -- the number to report/gate against is always recomputed
    from the reloaded best checkpoint's actual predictions via
    `evaluate.log_loss_score`, not this raw BCE value (they can diverge
    on saturated predictions: BCEWithLogitsLoss clamps its internal log
    at -100, sklearn's log_loss clips probabilities at machine epsilon).
    """
    if seed is not None:
        torch.manual_seed(seed)
        for loader in (train_loader, val_loader):
            if loader.generator is not None:
                loader.generator.manual_seed(seed)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device_type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * x.shape[0]
            train_n += x.shape[0]
        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                val_loss_sum += loss.item() * x.shape[0]
                val_n += x.shape[0]

        history["train_loss"].append(train_loss_sum / train_n)
        val_loss = val_loss_sum / val_n
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return best_state, history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/test_train.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full project test suite**

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/ -q`
Expected: `59 passed` (46 pre-existing + 5 `test_cache.py` + 1 `test_dataset.py` addition + 2 `test_features.py` additions + 5 `test_train.py`)

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Add src/train.py::train_one_fold (TDD)"
```

---

### Task 5: `notebooks/06_cnn_rung2.ipynb` — transfer check + batch/LR pick + timing

**Files:**
- Create: `notebooks/06_cnn_rung2.ipynb`

**Interfaces:**
- Consumes: `cache.CachedVolumeStore` (Task 1), `dataset.DatParkinsonDataset(..., load_fn=...)`
  (Task 2), `features.inplane_family` (Task 3, indirectly — via
  `baseline_features.csv`'s already-materialized `inplane_family` column,
  not called directly in this notebook), `train.train_one_fold` (Task 4),
  `model.build_model`/`model.predict`/`model.build_combat_baseline`,
  `evaluate.make_folds`/`evaluate.log_loss_score`, `config.*`.
- Produces: `checkpoints/rung2_fold0_batch{b}_lr{lr}.pt`,
  `checkpoints/rung2_lofo_<family>.pt`,
  `data/processed/rung2_lofo_<family>_cnn_probs.npy`,
  `data/processed/volume_cache/` (the on-disk cache, also consumed by
  Task 6). Nothing here is imported by later tasks in Python — it's a
  notebook, not a library.

This task has no TDD cycle (it's a notebook the user runs, not library
code) — write the cells directly.

- [ ] **Step 1: Write the notebook**

Markdown intro cell:

```markdown
# 06 — CNN rung 2: full-data transfer check, batch/LR pick, timing

**Decision this feeds** (`docs/superpowers/specs/2026-09-09-cnn-train-rung23-design.md`):
measures wall-clock time for one full-scale nested-CV fold (needed to
budget rung 3's 5-fold × 5-repeat run against the 2026-09-16 deadline),
picks a batch size/learning rate empirically among three candidates, and
runs one leave-one-family-out transfer check (holding out the 3.895mm
family, not the largest 2.46mm family — see the spec for why). **This
notebook does not decide the gate** — only `notebooks/07_cnn_rung3.ipynb`'s
full nested CV, scored against `model.build_combat_baseline()` = 0.5290
log loss, does that.

**Nested cross-validation**: every training run below splits its training
portion again, 90/10, to pick the early-stopping epoch — the row(s) being
scored (the outer test fold, or the held-out family) are never used for
training or stopping. This avoids the optimism bias of scoring on the
same data used for model selection.

**Data handling**: this notebook loads real `.nii.gz` volumes and
row-level labels throughout, so per the AI-assistant data rule
(`README.md`) it is **[RUN ME]** — run it yourself, share back only the
printed aggregate numbers (timings, log loss values), not any per-row
output.
```

`[RUN ME]` code cell 1 (setup + shared cache):

```python
# [RUN ME] -- loads real pixel data + row-level labels. Builds the shared
# on-disk volume cache reused by every training run below and in
# notebooks/07_cnn_rung3.ipynb.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import torch

import cache
import config
import dataset
import evaluate
import model
import train as train_mod

labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
family_df = pd.read_csv(config.DATA_PROCESSED / "baseline_features.csv")[
    [config.UID_COLUMN, "inplane_family"]
]
labeled_df = labels_df.merge(family_df, on=config.UID_COLUMN, how="inner").reset_index(drop=True)
print("family counts:\n", labeled_df["inplane_family"].value_counts())

uids = labeled_df[config.UID_COLUMN].tolist()
labels = labeled_df[config.TARGET_COLUMN].tolist()
families = labeled_df["inplane_family"].tolist()

config_fingerprint = {
    "TARGET_SPACING": config.TARGET_SPACING,
    "CROP_SIZE_MM": config.CROP_SIZE_MM,
    "CROP_CENTER_MM": config.CROP_CENTER_MM,
    "TARGET_SHAPE": config.TARGET_SHAPE,
    "BACKGROUND_PERCENTILE": config.BACKGROUND_PERCENTILE,
    "BACKGROUND_MAX_FRACTION": config.BACKGROUND_MAX_FRACTION,
}

cache_start = time.time()
volume_cache = cache.CachedVolumeStore(
    uids, cache_dir=config.DATA_PROCESSED / "volume_cache",
    config_fingerprint=config_fingerprint,
)
cache_build_seconds = time.time() - cache_start
print(f"cache build: {cache_build_seconds:.1f}s for {len(uids)} volumes "
      f"({cache_build_seconds / len(uids) * 1000:.1f} ms/volume) -- "
      f"this cost is paid once per notebook session, not once per fold.")
```

`[RUN ME]` code cell 2 (nested-fold training+scoring helper):

```python
# [RUN ME] (no data access itself -- defines a function used by the
# [RUN ME] cells below). num_workers=0 always: a cached dataset must not
# be handed to multiple DataLoader worker processes, each of which would
# rebuild its own copy of the cache and silently multiply wall-clock time.
def train_and_score_nested(train_uids, train_labels, train_family,
                            outer_uids, batch_size, lr, seed,
                            epochs=config.EPOCHS, patience=config.PATIENCE,
                            inner_splits=10):
    """Splits (train_uids, train_labels, train_family) 90/10 (stratified,
    `inner_splits` folds, fold 0) for early stopping, trains DatCNN,
    reloads the best checkpoint, and predicts on outer_uids -- which are
    never used for training or stopping. Returns
    (outer_probs, history, best_state); outer_probs is aligned to
    outer_uids' order (the outer loader is never shuffled)."""
    inner_train_idx, inner_val_idx = evaluate.make_folds(
        train_labels, train_family, n_splits=inner_splits, random_state=seed
    )[0]

    def subset(idxs):
        return ([train_uids[i] for i in idxs], [train_labels[i] for i in idxs])

    inner_train_uids, inner_train_labels = subset(inner_train_idx)
    inner_val_uids, inner_val_labels = subset(inner_val_idx)

    inner_train_ds = dataset.DatParkinsonDataset(inner_train_uids, inner_train_labels, load_fn=volume_cache.get)
    inner_val_ds = dataset.DatParkinsonDataset(inner_val_uids, inner_val_labels, load_fn=volume_cache.get)
    inner_train_loader = torch.utils.data.DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    inner_val_loader = torch.utils.data.DataLoader(inner_val_ds, batch_size=batch_size, num_workers=0)

    net = model.build_model().to(config.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state, history = train_mod.train_one_fold(
        net, inner_train_loader, inner_val_loader, optimizer, loss_fn,
        epochs=epochs, patience=patience, device=config.DEVICE,
        use_amp=config.USE_AMP, seed=seed,
    )
    net.load_state_dict(best_state)

    outer_ds = dataset.DatParkinsonDataset(outer_uids, load_fn=volume_cache.get)
    outer_loader = torch.utils.data.DataLoader(outer_ds, batch_size=batch_size, num_workers=0)
    outer_probs = []
    for x, _ in outer_loader:
        outer_probs.append(model.predict(net, x))
    outer_probs = np.concatenate(outer_probs)

    return outer_probs, history, best_state
```

`[RUN ME]` code cell 3 (batch/LR mini-experiment on outer fold 0):

```python
# [RUN ME] -- batch size / LR mini-experiment on outer fold 0 (Malladi et
# al. 2022 sqrt-LR-scaling candidate vs. an unscaled control vs. the
# rung-0/1 continuity control). Cheap at a warm cache -- RESOURCES.md.
outer_folds = evaluate.make_folds(np.array(labels), np.array(families),
                                   n_splits=config.N_FOLDS, random_state=config.SEED)
fold0_train_idx, fold0_test_idx = outer_folds[0]
fold0_train_uids = [uids[i] for i in fold0_train_idx]
fold0_train_labels = [labels[i] for i in fold0_train_idx]
fold0_train_family = [families[i] for i in fold0_train_idx]
fold0_test_uids = [uids[i] for i in fold0_test_idx]
fold0_test_labels = np.array([labels[i] for i in fold0_test_idx])

candidates = [(8, 1e-3, "rung0/1 continuity"), (32, 2e-3, "sqrt-LR-scaled"), (32, 1e-3, "unscaled control")]
batch_lr_results = {}
for batch_size, lr, tag in candidates:
    start = time.time()
    probs, history, best_state = train_and_score_nested(
        fold0_train_uids, fold0_train_labels, fold0_train_family,
        fold0_test_uids, batch_size=batch_size, lr=lr, seed=config.SEED,
    )
    elapsed = time.time() - start
    score = evaluate.log_loss_score(fold0_test_labels, probs)
    batch_lr_results[(batch_size, lr)] = {"tag": tag, "history": history, "score": score, "seconds": elapsed}
    torch.save(best_state, config.CHECKPOINT_DIR / f"rung2_fold0_batch{batch_size}_lr{lr:.0e}.pt")
    print(f"batch={batch_size:>3} lr={lr:.0e} ({tag:<20}): "
          f"inner-val best={min(history['val_loss']):.4f}, outer log loss={score:.4f}, "
          f"{elapsed:.1f}s ({elapsed / len(history['val_loss']):.2f}s/epoch)")

winner = min(batch_lr_results, key=lambda k: min(batch_lr_results[k]["history"]["val_loss"]))
print(f"\nwinner (lowest inner-validation log loss): batch={winner[0]}, lr={winner[1]:.0e}")
print("This run's winner IS the first of rung 3's 5 seed-repeats (same seed, same fold 0) -- "
      "notebooks/07_cnn_rung3.ipynb reuses this pairing rather than duplicating it.")
```

`[RUN ME]` code cell 4 (leave-one-family-out on 3.895mm):

```python
# [RUN ME] -- leave-one-family-out transfer check. Holds out the 3.895mm
# family (not the largest family, 2.46mm) -- Wenzel et al. 2019 found the
# OTHER transfer direction (fine->coarse) already works well, so holding
# out 2.46mm would prove nothing (RESOURCES.md).
lofo_family = "3.895"
assert lofo_family in labeled_df["inplane_family"].unique(), \
    f"{lofo_family} not found -- check baseline_features.csv's inplane_family values"

held_out_mask = labeled_df["inplane_family"] == lofo_family
retained_mask = ~held_out_mask

retained_uids = labeled_df.loc[retained_mask, config.UID_COLUMN].tolist()
retained_labels = labeled_df.loc[retained_mask, config.TARGET_COLUMN].tolist()
retained_family = labeled_df.loc[retained_mask, "inplane_family"].tolist()
held_out_uids = labeled_df.loc[held_out_mask, config.UID_COLUMN].tolist()
held_out_labels = np.array(labeled_df.loc[held_out_mask, config.TARGET_COLUMN].tolist())

batch_size, lr = winner
lofo_probs, lofo_history, lofo_state = train_and_score_nested(
    retained_uids, retained_labels, retained_family,
    held_out_uids, batch_size=batch_size, lr=lr, seed=config.SEED,
)
torch.save(lofo_state, config.CHECKPOINT_DIR / f"rung2_lofo_{lofo_family.replace('.', '_')}.pt")

# Three reference numbers on the SAME held-out rows -- a CNN number alone
# is uninterpretable, since the family's own base rate differs from the
# global 0.548458 (README.md's pairwise family tests).
family_base_rate = held_out_labels.mean()
base_rate_preds = np.full_like(held_out_labels, family_base_rate, dtype=float)
family_base_rate_logloss = evaluate.log_loss_score(held_out_labels, base_rate_preds)

baseline_feat_df = pd.read_csv(config.DATA_PROCESSED / "baseline_features.csv")
baseline_X_all = baseline_feat_df[["abs_asym", "striatal_ratio"]].to_numpy()
baseline_y_all = baseline_feat_df[config.TARGET_COLUMN].to_numpy()
baseline_family_all = baseline_feat_df["inplane_family"].to_numpy()

retained_baseline_mask = baseline_family_all != lofo_family
baseline_pipeline = model.build_combat_baseline()
baseline_pipeline.fit(baseline_X_all[retained_baseline_mask], baseline_y_all[retained_baseline_mask],
                       baseline_family_all[retained_baseline_mask])
held_out_baseline_mask = baseline_family_all == lofo_family
baseline_lofo_probs = baseline_pipeline.predict_proba(
    baseline_X_all[held_out_baseline_mask], baseline_family_all[held_out_baseline_mask])[:, 1]
baseline_lofo_logloss = evaluate.log_loss_score(baseline_y_all[held_out_baseline_mask], baseline_lofo_probs)

cnn_lofo_logloss = evaluate.log_loss_score(held_out_labels, lofo_probs)

print(f"LOFO family={lofo_family} (n={int(held_out_mask.sum())}):")
print(f"  family's own base-rate log loss: {family_base_rate_logloss:.4f} (base rate={family_base_rate:.3f})")
print(f"  build_combat_baseline() on these rows: {baseline_lofo_logloss:.4f}")
print(f"  CNN (nested, trained on retained families): {cnn_lofo_logloss:.4f}")

np.save(config.DATA_PROCESSED / f"rung2_lofo_{lofo_family.replace('.', '_')}_cnn_probs.npy", lofo_probs)
```

Markdown findings cell (template, filled in by the user after running):

```markdown
**What we're looking for:** how long does one full-scale nested-CV fold
take (to budget rung 3), which (batch, LR) candidate wins, and does the
CNN transfer to a held-out acquisition family (3.895mm) better than the
family's own base rate and better than the classical baseline restricted
to those same rows?

**What we found:** *(paste: cache-build seconds and ms/volume; the three
batch/LR candidates' inner-val best loss, outer fold log loss, and
seconds/epoch; the winner; the three LOFO reference numbers)*

**Decision / next step:** *(confirm the winning (batch, LR) to use in
notebooks/07_cnn_rung3.ipynb; note whether the LOFO CNN number beats both
its family's own base rate AND the classical baseline restricted to that
family -- if it doesn't beat the base rate, something is wrong with the
CNN pipeline at scale, worth investigating before running rung 3's full
5x5 CV)*
```

- [ ] **Step 2: Sanity-check the notebook's non-data-touching parts**

The notebook cannot be run (it loads real patient data). Instead, verify
the import lines and function/attribute names it references actually
exist:

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -c "import sys; sys.path.insert(0, 'src'); import cache, config, dataset, evaluate, model, train; assert hasattr(cache, 'CachedVolumeStore'); assert hasattr(dataset, 'DatParkinsonDataset'); assert hasattr(model, 'build_model') and hasattr(model, 'predict') and hasattr(model, 'build_combat_baseline'); assert hasattr(evaluate, 'make_folds') and hasattr(evaluate, 'log_loss_score'); assert hasattr(train, 'train_one_fold'); assert hasattr(config, 'DATA_PROCESSED') and hasattr(config, 'CHECKPOINT_DIR') and hasattr(config, 'N_FOLDS') and hasattr(config, 'SEED') and hasattr(config, 'DEVICE') and hasattr(config, 'USE_AMP') and hasattr(config, 'WEIGHT_DECAY') and hasattr(config, 'EPOCHS') and hasattr(config, 'PATIENCE'); print('all referenced names exist')"`
Expected: `all referenced names exist`

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_cnn_rung2.ipynb
git commit -m "Add notebooks/06_cnn_rung2.ipynb (rung 2, RUN ME)"
```

---

### Task 6: `notebooks/07_cnn_rung3.ipynb` — full nested CV, the gate decision

**Files:**
- Create: `notebooks/07_cnn_rung3.ipynb`

**Interfaces:**
- Consumes: same modules as Task 5 (`cache`, `dataset`, `evaluate`,
  `model`, `train`, `config`) — this notebook is self-contained (does not
  assume notebook 06's kernel is still warm) and re-derives its own
  `train_and_score_nested` helper and `CachedVolumeStore`.
- Produces: `checkpoints/rung3_seed{s}_fold{i}.pt`,
  `data/processed/rung3_oof_seed{s}.npy`,
  `data/processed/baseline_oof_seed_match.npy`. Terminal deliverable of
  this plan.

This task has no TDD cycle — write the cells directly.

- [ ] **Step 1: Write the notebook**

Markdown intro cell:

```markdown
# 07 — CNN rung 3: full nested CV, the gate decision

**Decision this feeds** (`docs/superpowers/specs/2026-09-09-cnn-train-rung23-design.md`):
the actual gate — does the CNN beat `model.build_combat_baseline()`
(0.5290 log loss) by more than noise? 5-fold CV, repeated 5x (varying
both the model-init seed and the fold-split seed together, per
Bouthillier et al. 2021 — `RESOURCES.md`), nested per fold (an inner
90/10 split picks the early-stopping epoch; the outer fold is never used
for training or stopping — see notebook 06's intro for why). Gate rule,
decided in the spec *before* seeing this run's numbers: the CNN
graduates only if (a) a paired bootstrap 95% CI on the delta vs. 0.5290
excludes zero, **and** (b) the 5-repeat mean beats 0.5290 by more than 2x
the larger of the two runs' standard deviations.

**Batch size / LR**: uses the winner from `notebooks/06_cnn_rung2.ipynb`'s
mini-experiment — update the `batch_size, lr = ...` line below if that
notebook's winner differs from the placeholder here.

**Data handling**: this notebook loads real `.nii.gz` volumes and
row-level labels throughout, so per the AI-assistant data rule
(`README.md`) it is **[RUN ME]** — run it yourself, share back only the
printed aggregate numbers, not any per-row output.
```

`[RUN ME]` code cell 1 (setup + shared cache — mirrors notebook 06's cell 1
exactly, since this notebook must be runnable on its own):

```python
# [RUN ME] -- loads real pixel data + row-level labels. Rebuilds (or
# reuses, if this session is still warm from notebooks/06) the shared
# on-disk volume cache.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import torch

import cache
import config
import dataset
import evaluate
import model
import train as train_mod

labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
family_df = pd.read_csv(config.DATA_PROCESSED / "baseline_features.csv")[
    [config.UID_COLUMN, "inplane_family"]
]
labeled_df = labels_df.merge(family_df, on=config.UID_COLUMN, how="inner").reset_index(drop=True)

uids = labeled_df[config.UID_COLUMN].tolist()
labels = labeled_df[config.TARGET_COLUMN].tolist()
families = labeled_df["inplane_family"].tolist()

config_fingerprint = {
    "TARGET_SPACING": config.TARGET_SPACING,
    "CROP_SIZE_MM": config.CROP_SIZE_MM,
    "CROP_CENTER_MM": config.CROP_CENTER_MM,
    "TARGET_SHAPE": config.TARGET_SHAPE,
    "BACKGROUND_PERCENTILE": config.BACKGROUND_PERCENTILE,
    "BACKGROUND_MAX_FRACTION": config.BACKGROUND_MAX_FRACTION,
}

cache_start = time.time()
volume_cache = cache.CachedVolumeStore(
    uids, cache_dir=config.DATA_PROCESSED / "volume_cache",
    config_fingerprint=config_fingerprint,
)
print(f"cache ready: {time.time() - cache_start:.1f}s for {len(uids)} volumes "
      f"(0s if notebook 06 already built it this session).")
```

`[RUN ME]` code cell 2 (nested-fold helper — identical to notebook 06's
cell 2; duplicated intentionally so this notebook is runnable
independently, per the spec's cache-placement section):

```python
# [RUN ME] (no data access itself). num_workers=0 always -- see notebook
# 06's cell 2 for why.
def train_and_score_nested(train_uids, train_labels, train_family,
                            outer_uids, batch_size, lr, seed,
                            epochs=config.EPOCHS, patience=config.PATIENCE,
                            inner_splits=10):
    """See notebooks/06_cnn_rung2.ipynb's identical helper for the full
    docstring."""
    inner_train_idx, inner_val_idx = evaluate.make_folds(
        train_labels, train_family, n_splits=inner_splits, random_state=seed
    )[0]

    def subset(idxs):
        return ([train_uids[i] for i in idxs], [train_labels[i] for i in idxs])

    inner_train_uids, inner_train_labels = subset(inner_train_idx)
    inner_val_uids, inner_val_labels = subset(inner_val_idx)

    inner_train_ds = dataset.DatParkinsonDataset(inner_train_uids, inner_train_labels, load_fn=volume_cache.get)
    inner_val_ds = dataset.DatParkinsonDataset(inner_val_uids, inner_val_labels, load_fn=volume_cache.get)
    inner_train_loader = torch.utils.data.DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    inner_val_loader = torch.utils.data.DataLoader(inner_val_ds, batch_size=batch_size, num_workers=0)

    net = model.build_model().to(config.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state, history = train_mod.train_one_fold(
        net, inner_train_loader, inner_val_loader, optimizer, loss_fn,
        epochs=epochs, patience=patience, device=config.DEVICE,
        use_amp=config.USE_AMP, seed=seed,
    )
    net.load_state_dict(best_state)

    outer_ds = dataset.DatParkinsonDataset(outer_uids, load_fn=volume_cache.get)
    outer_loader = torch.utils.data.DataLoader(outer_ds, batch_size=batch_size, num_workers=0)
    outer_probs = []
    for x, _ in outer_loader:
        outer_probs.append(model.predict(net, x))
    return np.concatenate(outer_probs), history, best_state
```

`[RUN ME]` code cell 3 (5-fold × 5-repeat nested CV):

```python
# [RUN ME] -- full 5-fold nested CV, repeated 5x (model-init seed and
# fold-split seed varied together, per Bouthillier et al. 2021 --
# RESOURCES.md) for the rung-3 gate decision.
batch_size, lr = 32, 2e-3  # UPDATE to notebook 06's actual winner if different
N_REPEATS = 5
oof_repeats = []

for repeat_seed in range(config.SEED, config.SEED + N_REPEATS):
    outer_folds = evaluate.make_folds(np.array(labels), np.array(families),
                                       n_splits=config.N_FOLDS, random_state=repeat_seed)
    oof_probs = np.zeros(len(uids))
    for fold_i, (train_idx, test_idx) in enumerate(outer_folds):
        # NOTE: seed=config.SEED, fold 0 here repeats notebooks/06's
        # winning-candidate fold-0 training almost exactly (same split,
        # same seed) -- ~30s of GPU at a warm cache, not worth the added
        # complexity of passing state between notebook sessions to avoid.
        fold_train_uids = [uids[i] for i in train_idx]
        fold_train_labels = [labels[i] for i in train_idx]
        fold_train_family = [families[i] for i in train_idx]
        fold_test_uids = [uids[i] for i in test_idx]

        probs, history, best_state = train_and_score_nested(
            fold_train_uids, fold_train_labels, fold_train_family,
            fold_test_uids, batch_size=batch_size, lr=lr, seed=repeat_seed,
        )
        oof_probs[test_idx] = probs
        torch.save(best_state, config.CHECKPOINT_DIR / f"rung3_seed{repeat_seed}_fold{fold_i}.pt")
        fold_score = evaluate.log_loss_score(np.array(labels)[test_idx], probs)
        print(f"  seed={repeat_seed} fold={fold_i}: {len(history['val_loss'])} epochs, "
              f"outer fold log loss={fold_score:.4f}")

    repeat_logloss = evaluate.log_loss_score(np.array(labels), oof_probs)
    oof_repeats.append(oof_probs)
    print(f"seed={repeat_seed} pooled OOF log loss: {repeat_logloss:.4f}")
    np.save(config.DATA_PROCESSED / f"rung3_oof_seed{repeat_seed}.npy", oof_probs)

repeat_scores = np.array([evaluate.log_loss_score(np.array(labels), oof) for oof in oof_repeats])
print(f"\n{N_REPEATS}-repeat CNN pooled log loss: mean={repeat_scores.mean():.4f}, sd={repeat_scores.std():.4f}")
print("classical build_combat_baseline() (notebooks/04, README.md): 0.5290")
```

`[RUN ME]` code cell 4 (same-split baseline OOF, for the paired bootstrap):

```python
# [RUN ME] -- (re)compute build_combat_baseline()'s OOF predictions on the
# SAME fold split as the CNN's first CV repeat (seed=config.SEED), so the
# paired bootstrap below compares the two models row-for-row on identical
# splits. CPU-only, seconds.
baseline_feat_df = pd.read_csv(config.DATA_PROCESSED / "baseline_features.csv")
baseline_feat_df = baseline_feat_df.set_index(config.UID_COLUMN).loc[uids].reset_index()
baseline_X = baseline_feat_df[["abs_asym", "striatal_ratio"]].to_numpy()
baseline_y = baseline_feat_df[config.TARGET_COLUMN].to_numpy()
baseline_family = baseline_feat_df["inplane_family"].to_numpy()

baseline_oof = np.zeros(len(uids))
baseline_folds = evaluate.make_folds(baseline_y, baseline_family, n_splits=config.N_FOLDS, random_state=config.SEED)
for train_idx, test_idx in baseline_folds:
    pipeline = model.build_combat_baseline()
    pipeline.fit(baseline_X[train_idx], baseline_y[train_idx], baseline_family[train_idx])
    baseline_oof[test_idx] = pipeline.predict_proba(baseline_X[test_idx], baseline_family[test_idx])[:, 1]

baseline_pooled_logloss = evaluate.log_loss_score(baseline_y, baseline_oof)
print(f"baseline OOF pooled log loss (this split, seed={config.SEED}): {baseline_pooled_logloss:.4f} "
      f"(RESOURCES.md/README.md's recorded 5-seed mean: 0.5290)")
np.save(config.DATA_PROCESSED / "baseline_oof_seed_match.npy", baseline_oof)
```

`[RUN ME]` code cell 5 (paired bootstrap + gate rule + per-family breakdown):

```python
# [RUN ME] -- paired bootstrap (Varoquaux 2018 -- RESOURCES.md) on the
# CNN's first-repeat OOF vs. the same-split baseline OOF, plus the
# rung-3 gate rule decided in the spec BEFORE seeing this number.
CLASSICAL_BASELINE_LOGLOSS = 0.5290  # build_combat_baseline(), README.md, 5-seed mean
BASELINE_SD_RECORDED = 0.0011  # notebooks/04's recorded per-variant sd upper bound

cnn_oof_for_pairing = oof_repeats[0]  # same seed/split as baseline_oof above
y_true = np.array(labels)

rng = np.random.RandomState(config.SEED)
n = len(y_true)
n_bootstrap = 1000
deltas = np.empty(n_bootstrap)
for b in range(n_bootstrap):
    idx = rng.randint(0, n, size=n)
    cnn_ll = evaluate.log_loss_score(y_true[idx], cnn_oof_for_pairing[idx])
    baseline_ll = evaluate.log_loss_score(y_true[idx], baseline_oof[idx])
    deltas[b] = cnn_ll - baseline_ll  # negative = CNN better

ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
ci_excludes_zero_favoring_cnn = ci_high < 0

repeat_mean = repeat_scores.mean()
repeat_sd = repeat_scores.std()
noise_threshold = 2 * max(repeat_sd, BASELINE_SD_RECORDED)
mean_beats_baseline_by = CLASSICAL_BASELINE_LOGLOSS - repeat_mean

gate_passed = ci_excludes_zero_favoring_cnn and (mean_beats_baseline_by > noise_threshold)

print(f"paired bootstrap delta (CNN - baseline), 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
print(f"5-repeat mean={repeat_mean:.4f}, sd={repeat_sd:.4f}; "
      f"beats {CLASSICAL_BASELINE_LOGLOSS} by {mean_beats_baseline_by:+.4f} "
      f"(2x max-sd noise threshold = {noise_threshold:.4f})")
print(f"GATE {'PASSED' if gate_passed else 'NOT PASSED'}: "
      f"{'CNN graduates past rung 3.' if gate_passed else 'fallback applies -- see spec Definition of done.'}")

family_arr = np.array(families)
print("\nper-family CNN log loss (first repeat's pooled OOF):")
for fam in sorted(set(families)):
    mask = family_arr == fam
    if mask.sum() < 5:
        continue
    print(f"  {fam:<12} n={int(mask.sum()):>4}  log loss={evaluate.log_loss_score(y_true[mask], cnn_oof_for_pairing[mask]):.4f}")
```

`[RUN ME]` code cell 6 (fallback blend check, evaluated regardless of the
gate outcome since it's nearly free):

```python
# [RUN ME] -- fallback check (spec's Definition of done): a CNN+ComBat
# probability blend, evaluated regardless of the gate outcome above since
# it costs no GPU.
for w in [0.25, 0.5, 0.75]:
    blend = w * cnn_oof_for_pairing + (1 - w) * baseline_oof
    blend_ll = evaluate.log_loss_score(y_true, blend)
    print(f"blend w_cnn={w}: log loss={blend_ll:.4f}")
```

Markdown findings cell (template, filled in by the user after running):

```markdown
**What we're looking for:** does the CNN's 5-fold, 5-repeat nested-CV
pooled log loss beat `build_combat_baseline()` (0.5290) by more than
noise, per the gate rule fixed in the spec before this run?

**What we found:** *(paste: per-repeat pooled log loss and the 5-repeat
mean/sd; the paired-bootstrap 95% CI; the GATE PASSED/NOT PASSED line;
the per-family breakdown; the three blend log-loss numbers)*

**Decision / next step:** *(if the gate passed: the CNN becomes the
submission candidate -- next is submission packaging + local Docker
rehearsal, README.md's remaining Next step. If not: ship
build_combat_baseline() as the submission, note whether any blend weight
beat 0.5290 outright, and update README.md/RESOURCES.md with this
negative result, per the project's standing rule to log negative results
just like positive ones.)*
```

- [ ] **Step 2: Sanity-check the notebook's non-data-touching parts**

The notebook cannot be run (it loads real patient data). Instead, verify
the import lines and function/attribute names it references actually
exist (same check as Task 5's Step 2, since this notebook imports the
same modules):

Run: `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -c "import sys; sys.path.insert(0, 'src'); import cache, config, dataset, evaluate, model, train; assert hasattr(cache, 'CachedVolumeStore'); assert hasattr(dataset, 'DatParkinsonDataset'); assert hasattr(model, 'build_model') and hasattr(model, 'predict') and hasattr(model, 'build_combat_baseline'); assert hasattr(evaluate, 'make_folds') and hasattr(evaluate, 'log_loss_score'); assert hasattr(train, 'train_one_fold'); print('all referenced names exist')"`
Expected: `all referenced names exist`

- [ ] **Step 3: Commit**

```bash
git add notebooks/07_cnn_rung3.ipynb
git commit -m "Add notebooks/07_cnn_rung3.ipynb (rung 3, RUN ME)"
```

---

## Definition of done

- [ ] `"/c/Users/alher/anaconda3/envs/dat-parkinson/python.exe" -m pytest tests/ -q` passes with all tests: 46 pre-existing + 5 (`test_cache.py`) + 1 (`test_dataset.py` addition) + 2 (`test_features.py` addition) + 5 (`test_train.py`) = **59 tests**.
- [ ] `notebooks/06_cnn_rung2.ipynb` and `notebooks/07_cnn_rung3.ipynb` exist, are `[RUN ME]`-marked, and were never executed by Claude (Tasks 5-6's Step 2 sanity checks substitute for execution).
- [ ] `README.md` is **not** updated by this plan with any rung-2/3 result numbers — that happens after the user runs both notebooks and reports the numbers back (AI-assistant data rule + notebook-graduation rule).
