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
    tests. `was_reused` (bool) records which happened, so a caller can
    print/log whether this session paid the full preprocessing cost
    again without guessing from wall-clock time alone.
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
        self.was_reused = self._on_disk_matches()
        if self.was_reused:
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
        # Delete stale fingerprint at the start of rebuild. If this rebuild is
        # interrupted, the missing fingerprint.json will cause _on_disk_matches()
        # to correctly return False on next instantiation, preventing silent
        # cache corruption from incomplete writes.
        self._fingerprint_path.unlink(missing_ok=True)
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
