"""Feature engineering for the classical baseline (DaT-SPECT volumes).

Every function here takes an already-loaded volume array (plus its voxel
spacing) rather than a file path, so they stay testable with small
synthetic arrays instead of real patient scans.

Sources (RESOURCES.md): the physical-volume + two-largest-components mask
mirrors Wenzel et al. 2019's "hottest voxels" SBR method; the
central-region restriction and signed L-R asymmetry were validated
against a real failure mode in `notebooks/01_eda_volumes.ipynb` sections
6a/6b (an unrestricted mask occasionally latched onto a peripheral
artifact instead of the striatum).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


def striatum_mask(volume, spacing, target_ml=20.0, central_margin=0.15):
    """Boolean mask of the striatum: the two largest connected components
    among the top `target_ml` millilitres of brightest voxels, restricted
    to the central `1 - 2 * central_margin` fraction of each axis (the
    striatum is not near the volume's outer edge -- see EDA section 6a
    for why this restriction is necessary).

    Returns None if the volume/mask is empty or degenerate.
    """
    volume = np.clip(volume, 0, None)
    shape = np.asarray(volume.shape)
    lo = (shape * central_margin).astype(int)
    hi = shape - lo
    central = np.zeros(volume.shape, dtype=bool)
    central[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    restricted = np.where(central, volume, 0.0)
    if restricted.max() <= 0:
        return None  # no positive signal in the central region at all

    voxel_ml = float(np.prod(spacing)) / 1000.0
    n_keep = max(int(round(target_ml / voxel_ml)), 1)
    if n_keep >= int(central.sum()):
        return None

    threshold = np.partition(restricted.ravel(), -n_keep)[-n_keep]
    mask = restricted >= threshold
    if not mask.any():
        return None

    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return None
    sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
    keep = np.argsort(sizes)[::-1][:2] + 1
    return np.isin(labeled, keep)


def striatum_center_mm(volume, spacing, target_ml=20.0):
    """Striatum centroid, mm offset from the volume's geometric center on
    RAS axes -- same convention as `config.CROP_CENTER_MM` and the
    inverse of `data.crop_or_pad`'s own convention (`center_vox =
    (shape-1)/2 + center_mm/spacing`), so feeding this back into
    `crop_or_pad` as `center_mm` recenters the crop on the same striatum
    (verified in `tests/test_features.py`, and by the striatum-crop-
    coverage investigation's 6th Opus review -- project memory).

    Returns None if `striatum_mask` is degenerate -- a caller (e.g.
    `data.load_volume`'s per-subject "auto" centering mode) must supply a
    fallback constant, never crop around an unmeasured (0, 0, 0).
    """
    mask = striatum_mask(volume, spacing, target_ml=target_ml)
    if mask is None:
        return None
    idx = np.argwhere(mask)
    center_vox = (np.asarray(volume.shape, dtype=float) - 1) / 2.0
    centroid_vox = idx.mean(axis=0)
    return tuple((centroid_vox - center_vox) * np.asarray(spacing, dtype=float))


def signed_asymmetry(volume, mask, spacing):
    """Signed left-right asymmetry `(left - right) / (left + right)` of
    the masked signal, split at the intensity-weighted midline along
    axis 0. In RAS orientation, axis 0 increases toward Right, so low
    indices are anatomical Left.

    Returns None if the mask has no positive signal.
    """
    volume = np.clip(volume, 0, None)
    signal = np.where(mask, volume, 0.0)
    weights = signal.sum(axis=(1, 2))
    total_weight = weights.sum()
    if total_weight <= 0:
        return None

    xs = np.arange(volume.shape[0], dtype=float)
    midline = float((xs * weights).sum() / total_weight)

    left = float(weights[xs < midline].sum())
    right = float(weights[xs > midline].sum())
    total = left + right
    if total <= 0:
        return None
    return (left - right) / total


@dataclass
class ComBatParams:
    """Fitted ComBat parameters (Johnson et al. 2007), one grand mean/pooled
    variance per feature plus one (gamma_star, delta_star) location/scale
    pair per batch. `batch_map` records which original batch labels were
    collapsed into which fitted batch (including `"rare"`); `default_batch`
    is the fallback for a batch label never seen during `fit_combat`.
    """
    grand_mean: np.ndarray
    var_pooled: np.ndarray
    batch_gamma: dict = field(default_factory=dict)
    batch_delta: dict = field(default_factory=dict)
    batch_map: dict = field(default_factory=dict)
    default_batch: object = None
    no_op: bool = False


def _collapse_rare_batches(batch, min_size):
    """Relabel batches with fewer than `min_size` members as `"rare"`. If
    the combined rare bucket would itself still be smaller than `min_size`,
    fold it into the largest batch instead of leaving an unusable residual.

    Returns (collapsed_labels, batch_map) where `batch_map` maps every
    original label to the label it was collapsed into (identity if kept).
    """
    batch = np.asarray(batch, dtype=object)
    values, counts = np.unique(batch, return_counts=True)
    batch_map = {v: v for v in values}
    rare_values = values[counts < min_size]
    if len(rare_values) == 0:
        return batch.copy(), batch_map

    rare_mask = np.isin(batch, rare_values)
    collapsed = batch.copy()
    if rare_mask.sum() >= min_size:
        collapsed[rare_mask] = "rare"
        for v in rare_values:
            batch_map[v] = "rare"
    else:
        largest = values[np.argmax(counts)]
        collapsed[rare_mask] = largest
        for v in rare_values:
            batch_map[v] = largest
    return collapsed, batch_map


def _eb_batch_params(Zb, gamma_hat, gamma_bar, tau2, a_prior, b_prior,
                      conv=1e-4, max_iter=100):
    """Iterative empirical-Bayes point estimates (gamma_star, delta_star)
    for one batch, per Johnson et al. 2007's `it.sol` (sva package): shrink
    the naive per-batch mean/variance toward the across-batch prior.
    """
    n = Zb.shape[0]
    g_old = gamma_hat.copy()
    d_old = Zb.var(axis=0, ddof=1) if n > 1 else np.full(Zb.shape[1], 1e-6)
    for _ in range(max_iter):
        g_new = (tau2 * n * gamma_hat + d_old * gamma_bar) / (tau2 * n + d_old)
        sum2 = ((Zb - g_new) ** 2).sum(axis=0)
        d_new = (0.5 * sum2 + b_prior) / (n / 2.0 + a_prior - 1.0)
        d_new = np.where(d_new <= 0, 1e-6, d_new)
        denom_g = np.where(np.abs(g_old) < 1e-8, 1e-8, np.abs(g_old))
        denom_d = np.where(np.abs(d_old) < 1e-8, 1e-8, np.abs(d_old))
        change = max(np.max(np.abs(g_new - g_old) / denom_g),
                     np.max(np.abs(d_new - d_old) / denom_d))
        g_old, d_old = g_new, d_new
        if change < conv:
            break
    return g_old, d_old


def fit_combat(X, batch, min_batch_size=3):
    """Fit parametric empirical-Bayes ComBat (Johnson et al. 2007) removing
    additive/multiplicative batch effects from scalar features, no other
    covariates. `X` is (n_samples, n_features); `batch` labels each row
    (here, the in-plane spacing family -- the confound EDA section 3a found
    significant, p=0.00042). Batches smaller than `min_batch_size` are
    collapsed (see `_collapse_rare_batches`) so variance estimation never
    runs on too few samples.

    Fit this on a *training fold's controls only* (label=0 rows) -- per
    the harmonization leakage-avoidance precedent in RESOURCES.md (the
    analogous PPMI radiomics study) -- then `apply_combat` the result to
    every row (both classes, train and test) with the same batch identity.
    """
    X = np.asarray(X, dtype=float)
    collapsed, batch_map = _collapse_rare_batches(batch, min_batch_size)
    labels = np.unique(collapsed)
    default_batch = labels[np.argmax([np.sum(collapsed == l) for l in labels])]

    if len(labels) < 2:
        return ComBatParams(
            grand_mean=np.zeros(X.shape[1]), var_pooled=np.ones(X.shape[1]),
            batch_map=batch_map, default_batch=default_batch, no_op=True)

    grand_mean = X.mean(axis=0)
    batch_means = {l: X[collapsed == l].mean(axis=0) for l in labels}
    residual = X - np.vstack([batch_means[l] for l in collapsed])
    var_pooled = residual.var(axis=0, ddof=0)
    var_pooled = np.where(var_pooled <= 0, 1e-6, var_pooled)

    Z = (X - grand_mean) / np.sqrt(var_pooled)

    gamma_hat, delta_hat, Zb_by_label = {}, {}, {}
    for l in labels:
        Zb = Z[collapsed == l]
        Zb_by_label[l] = Zb
        gamma_hat[l] = Zb.mean(axis=0)
        delta_hat[l] = Zb.var(axis=0, ddof=1) if len(Zb) > 1 else np.full(X.shape[1], 1e-6)

    gamma_hat_mat = np.vstack([gamma_hat[l] for l in labels])
    delta_hat_mat = np.vstack([delta_hat[l] for l in labels])

    gamma_bar = gamma_hat_mat.mean(axis=0)
    tau2 = gamma_hat_mat.var(axis=0, ddof=1) if len(labels) > 1 else np.full(X.shape[1], 1e-6)
    tau2 = np.where(tau2 <= 0, 1e-6, tau2)

    m = delta_hat_mat.mean(axis=0)
    s2 = delta_hat_mat.var(axis=0, ddof=1) if len(labels) > 1 else np.full(X.shape[1], 1e-6)
    s2 = np.where(s2 <= 0, 1e-6, s2)
    a_prior = (2 * s2 + m ** 2) / s2
    b_prior = (m * s2 + m ** 3) / s2

    batch_gamma, batch_delta = {}, {}
    for l in labels:
        g_star, d_star = _eb_batch_params(
            Zb_by_label[l], gamma_hat[l], gamma_bar, tau2, a_prior, b_prior)
        batch_gamma[l] = g_star
        batch_delta[l] = d_star

    return ComBatParams(grand_mean=grand_mean, var_pooled=var_pooled,
                         batch_gamma=batch_gamma, batch_delta=batch_delta,
                         batch_map=batch_map, default_batch=default_batch)


def apply_combat(X, batch, params):
    """Apply fitted `ComBatParams` to `X` (any rows -- both classes, train
    or test). A batch label collapsed during `fit_combat` (including one
    never seen at fit time) uses `params.default_batch`'s parameters via
    `params.batch_map`.
    """
    X = np.asarray(X, dtype=float)
    if params.no_op:
        return X.copy()

    batch = np.asarray(batch, dtype=object)
    Z = (X - params.grand_mean) / np.sqrt(params.var_pooled)
    adjusted = np.empty_like(Z)
    for original_label in np.unique(batch):
        key = params.batch_map.get(original_label, params.default_batch)
        rows = batch == original_label
        adjusted[rows] = ((Z[rows] - params.batch_gamma[key])
                           / np.sqrt(params.batch_delta[key]))
    return adjusted * np.sqrt(params.var_pooled) + params.grand_mean


def striatal_ratio(volume, mask):
    """Mean intensity inside `mask` relative to the mean of all
    positive-intensity voxels in the volume -- scale-free, comparable
    across volumes with very different absolute intensity scale (EDA
    section 5/6c found raw intensity scale varies by orders of magnitude
    across scanner families).
    """
    volume = np.clip(volume, 0, None)
    masked_mean = float(volume[mask].mean())
    positive = volume[volume > 0]
    background_mean = float(positive.mean()) if positive.size else 0.0
    return masked_mean / (background_mean + 1e-9)


def extract_baseline_features(volume, spacing, target_ml=20.0):
    """(abs_asym, striatal_ratio, inplane_family) for one already-loaded
    volume, or None if the striatum mask is degenerate (mirrors
    notebooks/03_baseline_classical.ipynb's skip condition). A caller
    that gets None for a row should treat that feature as missing, not
    silently zero-fill it.
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
