"""Unit tests for src/train.py::train_one_fold. Synthetic tiny model +
random tensors only -- never real patient data. Uses two test doubles:
`ScriptedOptimizer` (sets model weights to a scripted per-epoch marker
value, ignoring real gradients) and `ScriptedValLoss` (returns a real
BCE loss during model.train() but a scripted per-epoch value during
model.eval()) so the exact best-epoch/early-stopping behavior can be
asserted without depending on real gradient-descent dynamics.
"""

from unittest import mock

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


def test_seed_reseeds_an_explicit_loader_generator():
    def run():
        net = TinyNet()
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(8, 4, generator=gen)
        y = torch.randint(0, 2, (8,), generator=gen).float()
        train_ds = torch.utils.data.TensorDataset(x, y)
        val_ds = torch.utils.data.TensorDataset(x[:4], y[:4])
        loader_gen = torch.Generator()  # NOT seeded here -- train_one_fold must seed it
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, generator=loader_gen)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        _, history = train.train_one_fold(
            net, train_loader, val_loader, optimizer, loss_fn,
            epochs=3, patience=10, device=torch.device("cpu"), use_amp=False, seed=42,
        )
        return history

    torch.manual_seed(0)
    history_a = run()
    torch.manual_seed(0)
    history_b = run()
    assert history_a == history_b
