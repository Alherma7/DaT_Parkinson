# 3D-CNN training: `train.py`, rung 2 (transfer + timing check), rung 3 (CV gate)

Status: approved for implementation planning (revised after Opus review +
external-research pass, 2026-09-09)
Date: 2026-09-09

## Purpose

Follow-up to `docs/superpowers/specs/2026-09-08-cnn-data-plumbing-design.md`,
which explicitly deferred this scope: a real training loop
(optimizer/scheduler, AMP, checkpointing, early stopping), rung 2 (single
fold + transfer check, full data), and rung 3 (full CV — the actual gate
decision against the classical baseline, `build_combat_baseline()` =
**0.5290 log loss**). Rung 0/1 (`notebooks/05_cnn_smoke_test.ipynb`) already
passed: right shapes/dtypes end to end, clean overfit on 24 volumes, sane
non-diverging loss curve on a 15%/1-fold subset (final val log loss 0.5638).

Augmentation (`augment.py`) is still out of scope, deferred until rung 2
shows whether overfitting on the full training set is actually a problem
worth solving. Left-right flip is already cleared as safe (`README.md`,
2026-09-08).

**Revision note**: an Opus review pass (2026-09-09) found that the first
draft of this spec had a real methodological flaw — see "Nested
cross-validation protocol" below — plus answers to the four open questions
it had flagged. The review's full findings aren't preserved as a separate
artifact; the resulting decisions are captured directly in the sections
below, and all citations are logged in `RESOURCES.md`'s new
"Training/validation methodology" entry.

## GPU capacity check (2026-09-09, synthetic data only)

RTX 4060 Laptop, 8 GiB VRAM, driver 610.88, 7.47 GiB free at rest.
`DatCNN` has 291,585 params. Benchmarked with random tensors matching
`config.TARGET_SHAPE` — no `.nii.gz`/label access.

| batch | AMP | peak VRAM | ms/step | vol/s |
|---|---|---|---|---|
| 8  | off | 0.163 GiB | 16.3  | 490  |
| 8  | on  | 0.095 GiB | 7.7   | 1037 |
| 16 | on  | 0.169 GiB | 10.1  | 1589 |
| 32 | on  | 0.315 GiB | 17.6  | 1823 |
| 64 | on  | 0.611 GiB | 35.3  | 1813 |
| 128| on  | 1.201 GiB | 74.0  | 1731 |
| 256| on  | 2.384 GiB | 144.7 | 1769 |
| 256| off | 4.606 GiB | 1150  | 223  |
| 512| off | (Windows shared-memory fallback, unusable) | 25243 | 20 |

- **AMP (`config.USE_AMP = True`, already set) roughly halves VRAM and
  doubles throughput** at every batch size tested. Use `torch.amp.GradScaler("cuda")`
  / `torch.autocast("cuda")` — the `torch.cuda.amp.*` spellings are
  deprecated on this project's torch 2.14.0+cu126.
- **GPU compute is not the bottleneck at this model size** (74-144 ms/step
  at batch 128-256). Wall-clock cost is dominated by CPU-side `nibabel`
  resampling per volume — which is exactly why the cache below matters far
  more than batch size for total run time.
- Batch-size decision is **not** made from this table alone — see
  "Batch size and learning rate" below; BatchNorm noise below batch~16 and
  Adam's non-linear LR scaling both matter more than raw throughput.

## Nested cross-validation protocol (applies to rung 2 and rung 3)

**The problem this fixes**: the first draft let each fold's own held-out
data pick both the early-stopping epoch and the best checkpoint, then
scored the gate metric on that same data. That's optimistic (the CNN gets
two looks at the fold `build_combat_baseline()` never gets one look at) and
makes the "apples to apples" comparison against 0.5290 false — the *split*
was shared, the *protocol* wasn't.

**Fix**: every outer evaluation fold (a rung-3 CV fold, or rung 2's held-out
family) is split again, inside itself, for model selection:

1. Take the *outer* fold's training portion (everything except the outer
   held-out rows).
2. Split *that* 90/10, stratified on target × in-plane-spacing family, via
   a second call to `evaluate.make_folds` (e.g. `n_splits=10`, use fold 0)
   — no new splitting code needed, this is the same tested primitive.
3. `train_one_fold(model, inner_train_loader, inner_val_loader, ...)` picks
   the best epoch using **only** the inner 10%. The outer held-out rows are
   never seen during training or stopping.
4. Reload the returned `best_state_dict` into a fresh model instance, then
   score the outer held-out rows exactly once, via `model.predict()` (not
   a re-implemented sigmoid) — this is the number that goes into the OOF
   vector / the family's LOFO score.

This applies identically to rung 3's 5 outer folds and to rung 2's
leave-one-family-out run (retained families → inner 90/10 split for
stopping; the held-out family is scored once, at the end, never touched
before that).

## Noise floor and gate decision rule (rung 3)

The first draft assumed repeating full CV to measure noise cost as much
GPU time as rung 3 itself — false once the cache (below) is in place: one
outer fold is ~1090 training volumes, batch 32, AMP → **~30s of GPU** per
fold, so a full 5-fold CV is roughly **3 minutes of GPU** once the
one-time ~1362-volume preprocessing pass is already cached. Five full CV
repeats cost roughly 15 minutes of GPU total. Do all of this, not a
weaker proxy:

1. **Persist OOF probabilities to disk** for every repeat (`outputs/` or
   `data/processed/`, per-uid, never printed row-by-row per the
   AI-assistant data rule — only aggregate numbers get reported back).
2. **5 full-CV repeats**, each varying both the model-init seed and
   `make_folds`' `random_state` together (Bouthillier et al. 2021,
   `RESOURCES.md`) — mirrors the 5-seed convention already used in
   notebooks 03/04, so the CNN's mean±sd is directly comparable to
   0.5290's own reported sd (0.0006-0.0011).
3. **Paired bootstrap on the pooled OOF vs. `build_combat_baseline()`'s
   own OOF** (already computable from `03`'s recorded predictions — persist
   those too if not already saved): resample the shared 1362 row indices
   with replacement (1000 replicates, fixed seed), compute the log-loss
   delta per replicate using the *same* resampled indices for both models,
   report the 2.5/97.5 percentile interval of the delta. Pairing cancels
   shared "which rows are hard" variance (Varoquaux 2018 warns the naive
   across-fold sd underestimates the true error bar — do not use it alone).
4. **Gate rule, decided now, not after seeing the number**: the CNN
   graduates past rung 3 only if **both** (a) the paired-bootstrap 95% CI
   on the delta vs. 0.5290 excludes zero, and (b) the 5-repeat mean beats
   0.5290 by more than 2x the larger of the two runs' standard deviations.
5. **Fallback if the gate isn't cleared**: ship `build_combat_baseline()`
   as the submission, and evaluate a CNN+ComBat probability blend on the
   already-persisted OOF vectors (near-zero additional cost, no GPU) before
   concluding the CNN adds nothing.

## Batch size and learning rate

Resolved empirically, not from theory alone (theory picks the candidates,
rung 2 measures which wins):

- **Candidates**: (batch=8, LR=1e-3) — the rung-0/1 config, kept as a
  continuity control; (batch=32, LR=2e-3) — Adam's sqrt-scaling rule
  (Malladi et al. 2022) applied to a 4x batch increase; (batch=32,
  LR=1e-3) — the unscaled control, to isolate whether the LR change or the
  batch change drives any difference.
- **Why batch 32 and not higher**: Masters & Luschi 2018 found batch 2-32
  is where stable convergence is easiest for small nets; `DatCNN`'s
  `BatchNorm3d` layers get statistically noisy under batch~16 (Wu & He
  2018) — batch 8 (current) is inside that noisy regime, batch 32 isn't.
  Cost is trivial either way (0.315 GiB of 7.47 GiB free).
- **Do not also switch on `config.LR_SCHEDULE` in the same experiment.**
  Chegodaev et al.'s candidate (cosine annealing, `T0=50, Tmult=1`) is one
  full cosine cycle across `config.EPOCHS=50` with no restart, and
  `config.PATIENCE=10` will almost always stop training well before epoch
  50 — early stopping would cut the schedule off before its low-LR
  fine-convergence tail does anything, while the decaying LR simultaneously
  flattens the val-loss curve early stopping needs to detect a plateau in.
  Keep `LR_SCHEDULE = None` for rung 2/3; treat the schedule as a separate
  experiment after the gate, with `T0` set to the actual observed
  early-stopping horizon, not 50.
- **Rung 2 runs all three candidates on the same fold-0 inner split**
  (~2 extra minutes of GPU with a warm cache) and rung 3 uses whichever
  wins on inner-validation log loss. Record the losing candidates' numbers
  too, not just the winner's — a negative result here is as valuable as
  the ones already logged for ImageNet transfer learning.

## Leave-one-family-out design (rung 2)

Hold out the **3.895mm family (n≈325)**, not the largest 2.46mm family
(n=528), as the primary probe:

- Wenzel et al. 2019 (`RESOURCES.md`) found the transfer failure is
  *asymmetric* — a model trained on high-resolution data collapses on
  low-resolution data (0.629 accuracy, effectively one-class), while the
  reverse direction works well (0.945). Holding out 2.46mm (the finest,
  most-identity-preserved-by-resampling family) trains on coarser data and
  tests on the *easy* direction — it would almost certainly pass and prove
  nothing.
- Holding out 3.895mm leaves 1037 training volumes, close to rung 3's
  per-fold ~1090 — the LOFO number is then roughly comparable to a rung-3
  fold's number without a large sample-size confound (2.46mm would leave
  only 834).
- 3.895mm is 100% oblique (`README.md`) — the cleanest available site
  signature — and `config.py`'s own comment already flags this family as
  "interpolated UP... must be revisited if it underperforms." This LOFO
  run directly answers that flagged question.
- **If time allows after the primary probe** (cheap at a warm cache, per
  the GPU numbers above), also run 2.46mm as a second, lower-priority
  sanity check — but 3.895mm is the one that must run.
- **Select the family programmatically**, not by the size quoted in this
  document: read `baseline_features.csv`'s `inplane_family` value counts
  in the notebook and pick by name, and sanitize the value before using it
  in a checkpoint filename (some family labels are `"other(3.123)"`-shaped
  per `notebooks/03`'s fallback bucket).
- **Report three numbers on the held-out family's rows, not one**: the
  family's own constant-base-rate log loss (its base rate differs from the
  global 0.548 — `README.md`'s pairwise family tests found real
  differences), `build_combat_baseline()` scored on those same rows
  (seconds, CPU), and the CNN's nested-CV score. A "bad" CNN number without
  these references is uninterpretable — it might just reflect a shifted
  base rate.

## Shared preprocessing cache: `src/cache.py`

The first draft proposed a notebook-local cache on the (mistaken) grounds
that a real-data cache can't be tested without real data. The caching
*mechanism* is dependency-injectable and pure — testable exactly like
`tests/test_dataset.py` already tests `DatParkinsonDataset` (monkeypatched
`load_volume`) — so it graduates to `src/`:

```python
class CachedVolumeStore:
    """uid -> data.load_volume(uid), computed at most once per uid,
    persisted to disk (one memmapped .npy + a uid->row-index json) so a
    kernel restart doesn't repeat the ~25 min preprocessing pass. Takes
    `load_fn` by injection (default data.load_volume) so it's testable
    against a fake loader returning synthetic arrays."""
    def __init__(self, uids, cache_path, load_fn=data.load_volume): ...
    def get(self, uid) -> np.ndarray: ...
```

- **Invalidation**: the cache file records the `config` values it was
  built with (`TARGET_SPACING`, `CROP_SIZE_MM`, `CROP_CENTER_MM`,
  `TARGET_SHAPE`, `BACKGROUND_PERCENTILE`, `BACKGROUND_MAX_FRACTION`);
  `CachedVolumeStore.__init__` rebuilds from scratch if any differ from
  what's on disk, rather than silently serving stale preprocessing.
- **Built eagerly, once, before any fold loop** in rung 2/3's notebooks —
  not lazily per-`__getitem__` — so its cost (measured once) is separated
  from per-epoch training cost, and so it's provably shared across every
  fold rather than accidentally rebuilt.
- **`DataLoader(..., num_workers=0)` whenever a `CachedVolumeStore`-backed
  dataset is used.** Multiple worker processes each get their own copy of
  anything the main process built (the same reason rung 0/1's notebook-
  local cache used `num_workers=0`) — with `config.NUM_WORKERS=4` this
  would silently rebuild the cache 4x and re-run every resample every
  epoch, turning a multi-minute run into hours with no error raised.
- Tests (`tests/test_cache.py`, synthetic `load_fn`, no real data): get()
  calls the injected loader once per uid regardless of repeat calls;
  cache persists across a fresh `CachedVolumeStore` instance pointed at
  the same path; a config-value mismatch triggers a rebuild rather than
  serving the old array.

## `src/features.py`: `inplane_family()` moves here

Currently defined inline in `notebooks/03_baseline_classical.ipynb`, but it
is now load-bearing for three consumers outside that notebook:
`evaluate.make_folds`'s stratification, `model.py`'s ComBat batch labels,
and this spec's rung-2 LOFO split. A pure function of one float (spacing)
belongs in `src/features.py` with a unit test, not duplicated or imported
awkwardly from a notebook. Notebook 03's own already-recorded results are
untouched — leave its local copy alone, just point new code (06, 07) at
the `src/` version.

## `src/train.py`

```python
def train_one_fold(model, train_loader, val_loader, optimizer, loss_fn,
                    epochs, patience, device, use_amp,
                    scheduler=None, seed=None) -> tuple[dict, dict]:
    """Runs up to `epochs` epochs with early stopping on val loss
    (`patience` epochs without improvement, config.PATIENCE). Returns
    (best_state_dict, history) where history =
    {"train_loss": [...], "val_loss": [...]}. Assumes `model` is already
    on `device`; moves each batch there. No file I/O -- the caller
    persists checkpoints and reloads best_state_dict before any inference
    pass (this function does not predict)."""
```

- `seed`, if given, seeds `torch.manual_seed` and each loader's
  `generator=` at the start of the call — makes a single fold reproducible
  in isolation, not dependent on how many folds ran before it in the same
  process (needed for rung 3's 5-seed noise-floor repeats).
- `scheduler`, if given, is `.step()`-ed once per epoch after the optimizer
  step — plumbing only; `LR_SCHEDULE` stays `None` for rung 2/3 per the
  batch-size section above, so this argument is exercised by tests but not
  used in the real runs yet.
- **Loss weighting**: `train_loss`/`val_loss` accumulate `loss.item() *
  x.shape[0]` and divide by the dataset size (not the batch count) —
  rung 0's code already did this correctly; a trailing partial batch would
  silently skew a naive batch-mean.
- **Validation is computed outside `autocast`** (or the loss is cast back
  to fp32 before accumulating) so AMP doesn't add noise to the early-
  stopping signal.
- **`val_loss` is a stopping signal, not the reported number.** `BCEWithLogitsLoss`
  clamps its internal log at -100 while sklearn's `log_loss` clips
  probabilities at machine epsilon — they can diverge on saturated
  predictions. Whatever gets written to `README.md`/compared against
  0.5290 is always recomputed via `evaluate.log_loss_score` on the actual
  reloaded-best-checkpoint predictions, never the raw training-loop `val_loss`.
- **No `pos_weight`.** `README.md` already logs `class_weight="balanced"`
  as an experiment to validate, not a default, and log loss is a proper
  scoring rule that `pos_weight` would distort. Plain `BCEWithLogitsLoss`
  at rungs 2 and 3.

**Testing (`tests/test_train.py`, synthetic only — tiny dummy model,
random tensors, no real volumes):**
1. Returns a state dict with the model's parameter keys; `history["val_loss"]`
   has length ≤ `epochs`.
2. A constructed case where val loss stops improving after epoch 1 stops
   training at exactly `epoch 1 + patience`.
3. A constructed case where the last epoch's val loss is deliberately worse
   than an earlier epoch's returns the earlier (best) epoch's weights, not
   the final ones.
4. Same `seed` given twice produces identical `history` on the same
   synthetic data; different seeds produce different histories.
5. A dummy `scheduler` (e.g. a `MagicMock`) has `.step()` called once per
   epoch.

## `notebooks/06_cnn_rung2.ipynb` — full-data transfer check, batch/LR pick, timing

Real `.nii.gz`/label access throughout — `[RUN ME]`, not run by Claude.

1. **Build the shared `CachedVolumeStore` over all 1362 uids first.**
   Report cache-build wall-clock time separately from every training
   number that follows (the first draft's mistake — conflating this
   one-time cost with per-fold training cost — is exactly what made rung
   3's budget look ~5x too expensive).
2. **Batch/LR mini-experiment**: outer fold 0 (from `evaluate.make_folds`,
   `n_splits=5`), nested per the protocol above, run three times —
   (8, 1e-3), (32, 2e-3), (32, 1e-3) — report each candidate's inner-
   validation log loss curve and per-epoch wall-clock time. This run's
   (32, winning-LR) result **is** the first of rung 3's 5 seed-repeats
   (reuse it, don't duplicate — the first draft ran an equivalent fold
   twice by accident); use a fresh seed for it distinct from rung 3's
   other 4 repeats.
3. **Leave-one-family-out on 3.895mm** (family selected from
   `baseline_features.csv`'s value counts, not hardcoded): nested split on
   the retained families for early stopping, score the held-out family
   once. Report the three reference numbers from the LOFO section above.
   Optionally repeat on 2.46mm if time allows.
4. Persist checkpoints (`checkpoints/rung2_fold0_batch{b}_lr{lr}.pt`,
   `checkpoints/rung2_lofo_<family>.pt`) and the held-out-row OOF
   probabilities used for reference-number computation.

## `notebooks/07_cnn_rung3.ipynb` — full CV, the gate decision

Real `.nii.gz`/label access throughout — `[RUN ME]`, not run by Claude.

- Reuses the `CachedVolumeStore` built in 06 if the notebook session is
  still warm, otherwise rebuilds once at the top (same eager-build,
  `num_workers=0` rule).
- 5-fold outer CV via `evaluate.make_folds`, nested per the protocol
  above, winning (batch, LR) from 06, `checkpoints/rung3_seed{s}_fold{i}.pt`.
- Repeated 5x total (seeds), the first repeat reused from 06's batch/LR
  winner per that section.
- Persist OOF probabilities per repeat; compute the 5-repeat mean±sd, the
  paired bootstrap vs. `build_combat_baseline()`'s OOF, and apply the gate
  rule from "Noise floor and gate decision rule" above — explicitly,
  in the notebook, not left for `README.md` to state after the fact.
- Per-family breakdown of the pooled OOF, same pattern as notebooks 02-04.

## Definition of done

- `pytest tests/` passes (46 existing tests + this plan's new
  `test_train.py` and `test_cache.py` tests + one new `test_features.py`
  test for the relocated `inplane_family`).
- `notebooks/06_cnn_rung2.ipynb` and `notebooks/07_cnn_rung3.ipynb`
  written, `[RUN ME]`-marked, not run by Claude.
- OOF probability vectors from both notebooks are persisted to disk
  (per-uid; Claude never reads their row-level contents, only the
  aggregate metrics reported back).
- The gate rule from "Noise floor and gate decision rule" is evaluated
  explicitly in `07`'s own findings cell — pass or fail, both are a valid
  outcome, but the rule must actually be applied, not just the raw numbers
  reported.
- `README.md` Progress/Next-steps updated only after the user runs both
  notebooks and reports back the numbers (AI-assistant data rule +
  notebook-graduation rule — this code stays notebook-referenced until
  rung 3 clears the gate).
- If the gate isn't cleared: the fallback (ship `build_combat_baseline()`,
  evaluate a CNN+ComBat OOF blend) is evaluated before declaring the CNN
  track closed, not skipped for lack of time.
