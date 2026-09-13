"""Evaluation metric(s) and cross-validation fold design for this project."""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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


def paired_bootstrap_ci(y_true, probs_a, probs_b, seed, n_bootstrap=20000):
    """95% CI (percentile method) of log_loss(a) - log_loss(b) under paired
    bootstrap resampling of rows (Varoquaux 2018). Negative values favor a.

    `n_bootstrap` default raised from 1000 to 20000 (2026-09-13, Opus
    review of notebook 25's op07 near-miss): the Monte Carlo standard
    error of a percentile estimated from B draws is
    ~sqrt(.975*.025/B)/f(x_.975), which at B=1000 (~0.00024 for a
    bootstrap sd around 0.003, this project's typical scale) can be
    larger than the margin a gate is deciding on -- op07's CI upper bound
    sat 0.0001 above zero, well inside that noise. Raising to 20000 costs
    seconds (pure numpy, no model fitting) and shrinks that MC error by
    ~4.5x. This does not retroactively change any past gate verdict
    (those remain as documented, computed at n=1000); it only raises the
    precision floor for gates run from here on.

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


def paired_repeat_gate(deltas, confidence=0.95):
    """Rung-4 gate statistic: is a candidate's per-repeat delta
    (candidate_log_loss - baseline_log_loss, one value per CV repeat,
    same fold/init seeds on both sides) reliably negative?

    Replaces the rung-4 gate originally used in notebooks 09-12, which
    had two problems an Opus review (2026-09-10, see project memory)
    caught: (1) it took `paired_bootstrap_ci` on a single arbitrarily-
    chosen repeat instead of aggregating across repeats, so its verdict
    could -- and for two of the four experiments, did -- disagree in
    sign with the actual multi-repeat mean; (2) its "2x noise threshold"
    compared a mean-of-N-repeats against 2x a single repeat's standard
    deviation, instead of the standard error of that mean (sd/sqrt(n)),
    making the bar 4.5-7.7 sigma in practice -- effectively unclearable
    by a real but modest effect at n=5 repeats.

    This is a one-sample paired t-interval on `deltas` directly (Student's
    t, not bootstrap, appropriate for the small repeat counts -- typically
    5-30 -- these gates run at). The gate passes when the whole CI is
    negative (candidate beats baseline on every plausible mean under this
    sample).

    Returns a dict: `mean`, `sd`, `ci_low`, `ci_high`, `passed`.
    """
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n)
    t_crit = float(stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)) if n > 1 else 0.0
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem
    return {
        "mean": mean,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "passed": bool(ci_high < 0),
    }


def compute_pos_weight(labels):
    """`torch.nn.BCEWithLogitsLoss`'s `pos_weight` for `class_weight=
    "balanced"`: n_negative / n_positive, computed from the actual
    labels passed in (a fold's inner-train split, not a fixed global
    constant -- class balance can vary slightly per fold).
    """
    labels = np.asarray(labels)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float(n_neg / n_pos)


def family_oversample_weights(family, boosted_families, boost_factor):
    """Per-sample weight for `torch.utils.data.WeightedRandomSampler`:
    `boost_factor` for rows whose `inplane_family` is in
    `boosted_families`, 1.0 for every other row.

    Targets specific underperforming families directly (rung 3's own
    per-family log loss breakdown, README.md) rather than a continuous
    function of every family's noisy per-family score -- several
    families have n<10, too few to trust a smooth weighting by their own
    log loss.
    """
    family = np.asarray(family, dtype=object)
    return np.where(np.isin(family, list(boosted_families)), boost_factor, 1.0)


def oof_predict(X, y, folds, regularized=False, cs=np.logspace(-2, 2, 9), inner_cv=5):
    """Honest row-wise-CV out-of-fold prediction: for each `(train_idx,
    test_idx)` in `folds`, fit a logistic-regression classifier on the
    train rows and predict the held-out rows, reassembled into one array
    in original row order. Promoted from
    `notebooks/23_paired_gate_shipped_recipe_candidates.ipynb` (2026-09-11)
    -- `y` is now an explicit parameter (was an implicit notebook global).

    `regularized=True` uses `LogisticRegressionCV` (L2 via `l1_ratios=(0.0,)`,
    `cs` grid, `inner_cv` folds tuned *inside* each outer fold's train rows
    only -- a proper nested design, never selects the penalty on rows it's
    then scored on). Use this for a candidate feature matrix whose size/
    collinearity makes an unregularized fit risky; the default
    (`regularized=False`) plain fit matches how this project's shipped
    2-feature blend (`submission.combine_predictions`) was fit.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    oof = np.empty(len(y), dtype=float)
    for train_idx, test_idx in folds:
        if regularized:
            # l1_ratios=(0.0,) is this sklearn build's non-deprecated way
            # to request pure L2 (it deprecated the `penalty=` kwarg on
            # LogisticRegressionCV in favor of l1_ratios/Cs); only
            # predict_proba is used below, never the legacy
            # .scores_/.coefs_paths_ attribute shapes, so the non-legacy
            # attribute layout is safe to opt into.
            clf = LogisticRegressionCV(Cs=cs, cv=inner_cv, l1_ratios=(0.0,),
                                        max_iter=2000, scoring="neg_log_loss",
                                        use_legacy_attributes=False)
        else:
            clf = LogisticRegression(C=np.inf, max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return oof


def paired_gate(label, candidate_oof, reference_oof, y, min_effect=0.003,
                 seed=config.RANDOM_STATE, verbose=True):
    """The corrected ensemble/recipe-comparison gate: a paired bootstrap
    over rows (`paired_bootstrap_ci`), not a paired delta compared against
    the *unpaired* across-fold sd of absolute scores (the bug found in
    `notebooks/22_calibration_refit_rowwise_cv.ipynb`'s `op06denoise` cell,
    2026-09-11 -- that denominator is driven by which rows land in which
    fold, identical for both arms, and can't resolve effects below its own
    ~0.011 SEM). Promoted from `notebooks/23_paired_gate_shipped_recipe_candidates.ipynb`.

    Adopt only if the whole 95% CI is negative (candidate reliably beats
    reference on every plausible row resample) AND the point delta exceeds
    `min_effect` in magnitude -- below `min_effect`, this project's own
    measured ~92% CV-to-leaderboard transfer ratio (submissions 1-2) would
    likely leave nothing visible on the real leaderboard anyway.

    Returns a dict: `label`, `score_candidate`, `score_reference`, `delta`,
    `ci_low`, `ci_high`, `clears`.
    """
    score_candidate = log_loss_score(y, candidate_oof)
    score_reference = log_loss_score(y, reference_oof)
    delta = score_candidate - score_reference
    ci_low, ci_high = paired_bootstrap_ci(y, candidate_oof, reference_oof, seed=seed)
    clears = (ci_high < 0) and (abs(delta) > min_effect)
    if verbose:
        print(f"{label}")
        print(f"  candidate row-CV log loss = {score_candidate:.4f}   reference = {score_reference:.4f}")
        print(f"  delta (candidate - reference) = {delta:+.4f}   "
              f"95% paired-bootstrap CI = [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"  -> {'CLEARS the gate (adopt)' if clears else 'does NOT clear the gate (keep reference)'} "
              f"(rule: whole CI < 0 AND |delta| > {min_effect})\n")
    return {"label": label, "score_candidate": score_candidate, "score_reference": score_reference,
            "delta": delta, "ci_low": ci_low, "ci_high": ci_high, "clears": clears}


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
