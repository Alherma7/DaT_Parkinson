"""Unit tests for src/model.py."""

import numpy as np

import config
import features
import model
import model as model_module


def test_build_classical_baseline_fits_and_predicts_valid_probabilities():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 2))
    y = rng.binomial(1, 0.5, size=40)

    pipeline = model.build_classical_baseline()
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]

    assert proba.shape == (40,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_build_classical_baseline_returns_a_new_instance_each_call():
    first = model.build_classical_baseline()
    second = model.build_classical_baseline()

    assert first is not second


# --- build_combat_baseline ---------------------------------------------------

def _batch_shifted_Xy(rng, n_per_batch=20, shift=4.0):
    """Two batches with a strong additive shift on feature 0 that is also
    correlated with the label, so an unharmonized model partly latches
    onto batch identity instead of the real signal -- ComBat should still
    fit cleanly and produce valid probabilities either way."""
    a = rng.normal(loc=0.0, scale=0.5, size=(n_per_batch, 2))
    b = rng.normal(loc=0.0, scale=0.5, size=(n_per_batch, 2))
    b[:, 0] += shift
    X = np.vstack([a, b])
    batch = np.array(["A"] * n_per_batch + ["B"] * n_per_batch)
    y = rng.binomial(1, 0.5, size=2 * n_per_batch)
    return X, y, batch


def test_build_combat_baseline_fits_and_predicts_valid_probabilities():
    rng = np.random.RandomState(0)
    X, y, batch = _batch_shifted_Xy(rng)

    pipeline = model.build_combat_baseline()
    pipeline.fit(X, y, batch)
    proba = pipeline.predict_proba(X, batch)[:, 1]

    assert proba.shape == (40,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_build_combat_baseline_fits_combat_on_controls_only():
    """Regression guard for the leakage-avoidance rule (RESOURCES.md /
    notebooks/04_combat_harmonization.ipynb): harmonization parameters must
    come from the training fold's label=0 rows, not the full training set.
    """
    rng = np.random.RandomState(1)
    X, y, batch = _batch_shifted_Xy(rng)

    pipeline = model.build_combat_baseline()
    pipeline.fit(X, y, batch)

    controls = y == 0
    expected_params = features.fit_combat(X[controls], batch[controls])
    np.testing.assert_array_equal(
        pipeline._combat_params.grand_mean, expected_params.grand_mean)


def test_build_combat_baseline_predict_requires_batch_labels_for_new_rows():
    rng = np.random.RandomState(2)
    X, y, batch = _batch_shifted_Xy(rng)
    pipeline = model.build_combat_baseline()
    pipeline.fit(X, y, batch)

    new_X = np.array([[1.0, 1.0], [5.0, 0.5]])
    new_batch = np.array(["A", "B"])
    proba = pipeline.predict_proba(new_X, new_batch)[:, 1]

    assert proba.shape == (2,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_build_combat_baseline_returns_a_new_instance_each_call():
    first = model.build_combat_baseline()
    second = model.build_combat_baseline()

    assert first is not second


import torch


# --- DatCNN / build_model / predict ------------------------------------------

def test_build_model_forward_pass_shape_and_finiteness():
    model = model_module.build_model()
    x = torch.randn(4, 1, *config.TARGET_SHAPE)

    out = model(x)

    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_build_model_returns_a_new_instance_each_call():
    first = model_module.build_model()
    second = model_module.build_model()

    assert first is not second


def test_predict_returns_probabilities_in_zero_one():
    model = model_module.build_model()
    x = torch.randn(3, 1, *config.TARGET_SHAPE)

    proba = model_module.predict(model, x)

    assert proba.shape == (3,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_restores_prior_training_eval_state():
    """Regression guard: predict() must not permanently flip the model into
    eval mode -- a future train.py calling predict() for per-epoch
    validation would otherwise silently disable dropout/BatchNorm training
    behavior for the rest of training."""
    x = torch.randn(2, 1, *config.TARGET_SHAPE)

    net = model_module.build_model()
    net.train()
    model_module.predict(net, x)
    assert net.training is True

    net.eval()
    model_module.predict(net, x)
    assert net.training is False


def test_datcnn_can_overfit_a_tiny_batch():
    """Rung-0-style sanity check (deep-learning-imaging.md): the
    architecture itself must be able to drive loss to near-zero on a
    handful of samples -- catches a frozen-parameter or shape bug that a
    single forward pass would not."""
    torch.manual_seed(0)
    model = model_module.build_model()
    x = torch.randn(4, 1, *config.TARGET_SHAPE)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.1


# --- DatSlab2DCNN / build_slab_model (roadmap item 6) -----------------------

def test_build_slab_model_forward_pass_shape_and_finiteness():
    net = model_module.build_slab_model()
    x = torch.randn(4, 1, *config.SLAB_TARGET_SHAPE[:2])

    out = net(x)

    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_build_slab_model_returns_a_new_instance_each_call():
    first = model_module.build_slab_model()
    second = model_module.build_slab_model()

    assert first is not second


def test_datslab2dcnn_can_overfit_a_tiny_batch():
    """Rung-0-style sanity check, same as test_datcnn_can_overfit_a_tiny_batch."""
    torch.manual_seed(0)
    net = model_module.build_slab_model()
    x = torch.randn(4, 1, *config.SLAB_TARGET_SHAPE[:2])
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.1


def test_production_variant_build_fns_covers_every_current_prefix_plus_slab2d():
    for prefix in model_module.PRODUCTION_VARIANT_PREFIXES:
        assert model_module.PRODUCTION_VARIANT_BUILD_FNS[prefix] is model_module.build_model
    assert model_module.PRODUCTION_VARIANT_BUILD_FNS["slab2d"] is model_module.build_slab_model


def test_slab2d_not_yet_in_production_variant_prefixes():
    """Regression guard: adding PRODUCTION_VARIANT_BUILD_FNS must not
    silently add "slab2d" to the shipped composition -- that only happens
    once notebooks/24_slab2d_architecture_diversity.ipynb's gate clears.
    """
    assert "slab2d" not in model_module.PRODUCTION_VARIANT_PREFIXES


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


# --- variant_checkpoint_filenames / production_checkpoint_filenames --------

def test_variant_checkpoint_filenames_matches_pattern_for_a_non_rung3_prefix():
    names = model_module.variant_checkpoint_filenames("rung4_augment")

    assert len(names) == 25
    assert names[0] == "rung4_augment_seed42_fold0.pt"
    assert names[-1] == "rung4_augment_seed46_fold4.pt"


def test_production_checkpoint_filenames_returns_25_names_across_1_variant():
    names = model_module.production_checkpoint_filenames()

    assert len(names) == 25
    assert len(set(names)) == 25  # no duplicates
    assert len(model_module.PRODUCTION_VARIANT_PREFIXES) == 1
    for name in names:
        assert any(name.startswith(prefix + "_seed") for prefix in model_module.PRODUCTION_VARIANT_PREFIXES)
