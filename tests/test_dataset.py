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
