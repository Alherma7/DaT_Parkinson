"""Pure, unit-testable logic shared by submission_src/main.py. Kept here
(not inlined in main.py) so it's covered by pytest -- main.py itself
can't be run or verified against real data by Claude
(docs/superpowers/specs/2026-09-09-submission-packaging-design.md).
"""


def combine_predictions(uids, cnn_probs, baseline_probs, cnn_weight):
    """Blend per-uid CNN and classical-baseline probabilities into the
    submission's final prediction, in `uids`' exact order.

    `cnn_probs` and `baseline_probs` are dicts keyed by uid. Every uid in
    `uids` must have a CNN prediction (raises KeyError otherwise -- the
    CNN ensemble runs on every test volume unconditionally). A uid
    missing from `baseline_probs` (features.extract_baseline_features
    returned None for it -- a degenerate striatum mask) falls back to
    the CNN probability alone rather than crashing or dropping the row.

    Returns a list of floats, same length and order as `uids`.
    """
    predictions = []
    for uid in uids:
        cnn_p = cnn_probs[uid]
        if uid in baseline_probs:
            predictions.append(cnn_weight * cnn_p + (1 - cnn_weight) * baseline_probs[uid])
        else:
            predictions.append(cnn_p)
    return predictions
