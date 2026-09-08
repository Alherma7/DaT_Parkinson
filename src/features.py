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
