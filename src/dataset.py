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
