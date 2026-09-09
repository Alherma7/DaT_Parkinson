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
