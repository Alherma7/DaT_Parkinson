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
