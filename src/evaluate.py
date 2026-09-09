"""Evaluation metric(s) and cross-validation fold design for this project."""

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config


def log_loss_score(y_true, y_pred_proba):
    """Competition metric: -(y*log(p) + (1-y)*log(1-p)), averaged over rows."""
    return float(log_loss(y_true, y_pred_proba, labels=[0, 1]))


def auroc_score(y_true, y_pred_proba):
    """AUROC — reference/diagnostic only, does not affect the gate."""
    return float(roc_auc_score(y_true, y_pred_proba))


def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """ECE (Guo et al. 2017): weighted-average |accuracy - confidence| per bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (y_pred_proba > lo) & (y_pred_proba <= hi)
        if lo == 0.0:
            in_bin |= y_pred_proba == 0.0
        if not in_bin.any():
            continue
        bin_confidence = y_pred_proba[in_bin].mean()
        bin_accuracy = y_true[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def combined_score(y_true, y_pred_proba, n_bins=10):
    """Gated metric (log loss) plus diagnostics (AUROC, ECE)."""
    return {
        "log_loss": log_loss_score(y_true, y_pred_proba),
        "auroc": auroc_score(y_true, y_pred_proba),
        "ece": expected_calibration_error(y_true, y_pred_proba, n_bins=n_bins),
    }


def _collapse_rare_families(target, family, min_size):
    """Relabel families with fewer than min_size members (within a target
    class) as 'rare', so StratifiedKFold always has enough members per
    stratification cell. If the combined 'rare' bucket for a target class
    would itself still be smaller than min_size, fold it into that target
    class's largest family instead of leaving an unsplittable residual.
    """
    target = np.asarray(target)
    family = np.asarray(family, dtype=object)
    collapsed = family.copy()
    for t in np.unique(target):
        mask = target == t
        values, counts = np.unique(collapsed[mask], return_counts=True)
        rare_values = values[counts < min_size]
        if len(rare_values) == 0:
            continue
        rare_mask = mask & np.isin(collapsed, rare_values)
        if rare_mask.sum() >= min_size:
            collapsed[rare_mask] = "rare"
        else:
            largest_family = values[np.argmax(counts)]
            collapsed[rare_mask] = largest_family
    return collapsed


def paired_bootstrap_ci(y_true, probs_a, probs_b, seed, n_bootstrap=1000):
    """95% CI (percentile method) of log_loss(a) - log_loss(b) under paired
    bootstrap resampling of rows (Varoquaux 2018). Negative values favor a.

    Returns (ci_low, ci_high).
    """
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)

    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[b] = (log_loss_score(y_true[idx], probs_a[idx])
                     - log_loss_score(y_true[idx], probs_b[idx]))

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def make_folds(target, family, n_splits=config.N_FOLDS, random_state=config.RANDOM_STATE):
    """Stratified K-Fold jointly on target and in-plane-spacing family.

    Rare (target, family) combinations (fewer than n_splits members within
    their target class) are collapsed into a single 'rare' bucket first, so
    every stratification cell has enough members for StratifiedKFold to
    split it n_splits ways.

    Returns a list of (train_idx, test_idx) index arrays, one per fold.
    """
    target = np.asarray(target)
    collapsed_family = _collapse_rare_families(target, family, min_size=n_splits)
    combined_key = np.array(
        [f"{t}_{f}" for t, f in zip(target, collapsed_family)]
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(target)), combined_key))
