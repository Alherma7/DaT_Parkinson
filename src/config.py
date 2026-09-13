"""Paths, constants, and reproducibility settings for this project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_LABELS_PATH = DATA_RAW / "train_labels.csv"

SMOKE_TEST_DIR = DATA_RAW / "smoke_test"
SMOKE_TEST_NIFTI_DIR = SMOKE_TEST_DIR / "niftis"
SMOKE_TEST_SUBMISSION_FORMAT_PATH = SMOKE_TEST_DIR / "submission_format.csv"


def _resolve_runtime_paths(code_execution_root):
    """(running_in_code_execution, nifti_dir, submission_format_path).

    If `code_execution_root` exists, this process is running inside the
    DrivenData competition container (docs/superpowers/specs/
    2026-09-09-submission-packaging-design.md) -- test volumes and the
    submission format live under it, not under this repo's data/raw/.
    Otherwise, fall back to the local training layout (submission_format_path
    is None locally; nothing needs it outside the container).
    """
    if code_execution_root.exists():
        return (True, code_execution_root / "data" / "niftis",
                code_execution_root / "data" / "submission_format.csv")
    return False, DATA_RAW / "niftis", None


RUNNING_IN_CODE_EXECUTION, NIFTI_DIR, SUBMISSION_FORMAT_PATH = (
    _resolve_runtime_paths(Path("/code_execution"))
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
if not RUNNING_IN_CODE_EXECUTION:
    # In the competition container this directory is unused (checkpoints
    # ship pre-copied into submission_src/model_assets/) and the
    # filesystem outside /code_execution may not be writable -- skip the
    # mkdir there rather than risk a crash at import time.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
TARGET_COLUMN = "is_pathologic"
UID_COLUMN = "uid"

# --- Evaluation reference points (measured in notebooks/01_eda_volumes.ipynb) --
# Base rate and the log loss of a constant base-rate prediction. Every model
# number gets quoted against this; a model that does not beat it has found
# nothing. n=1362, 747 positive / 615 negative.
BASE_RATE = 0.548458
BASELINE_LOGLOSS = 0.688443

# --- Geometry normalization (EDA section 1 / 7) -------------------------------
# Resample to a common physical SPACING (physical size preserved), not to a
# common voxel shape -- deep-learning-imaging.md preprocessing step 2, and the
# organizers' explicit warning that shape/resolution both vary.
#
# 2.46 mm isotropic is both the median AND the modal voxel spacing across the
# 1362 training volumes (528 volumes are already exactly 2.46 mm in-plane, so
# resampling is the identity for 38.8% of the data). It also keeps the striatum
# well sampled: an ~11 mL striatum is ~739 voxels at 2.46 mm but only ~186
# voxels at 3.895 mm, so resampling everything down to the coarsest family
# (3.895 mm, 325 volumes) would throw away detail on the structure the task
# actually depends on. The trade-off accepted here is that the 3.895 mm family
# is interpolated UP -- that adds no information and must be revisited if the
# per-family metric breakdown shows that family underperforming.
TARGET_SPACING = (2.46, 2.46, 2.46)

# Smallest field of view observed across all 1362 training volumes, in mm
# (shape * spacing). Any fixed-mm crop must fit inside this or be padded:
# z is the binding axis -- 1 volume has FOV[z] = 108 mm, 4 are under 120 mm.
MIN_FOV_MM = (175.6, 175.6, 108.0)

# EDA section 6a, clean-subset run (2026-09-08): physical-volume mask
# (TARGET_ML=20.0) restricted to the central 70% of each axis, 2 largest
# connected components, 112/119 sampled volumes after excluding 7 uids
# whose mask still fragmented abnormally (see the notebook for the uid
# list and n_components -- one, ft07x5ic, was checked and has an
# unremarkable header, so it's a real per-file pixel-content anomaly, not
# a geometry/dtype pattern; data.py should sanity-check the detected
# region size defensively rather than assume every file behaves).
# CROP_SIZE_MM = 95th-percentile bbox x1.3 margin, rounded up to 5mm; it
# fits config.MIN_FOV_MM on all 3 axes, but only by 3mm on z (105 vs 108)
# -- data.py must pad/clamp explicitly for the handful of tight-FOV
# volumes (section 7: 1 volume <110mm, 4 <120mm), not assume headroom.
CROP_SIZE_MM = (135.0, 70.0, 105.0)
# Median offset of the striatum centroid from the volume's geometric
# centre, in mm (x=L-R, y=A-P, z=S-I; RAS convention). z is negative
# (inferior to centre), consistent with the independently-measured
# centroid_z ~ 0.37-0.39 (< 0.5) finding from the same section.
CROP_CENTER_MM = (1.5, 22.8, -15.5)
# CROP_SIZE_MM / TARGET_SPACING, rounded up to an even number per axis.
TARGET_SHAPE = (56, 30, 44)

# Fallback for data.load_volume(uid, center_mm="auto") when a volume's own
# striatum_center_mm is unmeasurable (degenerate mask, 0/1362 in training).
# The corrected population median (notebooks/25's op03offsets, all 1362
# volumes, RAS-resampled frame) -- a strictly better fallback than
# CROP_CENTER_MM above, which was a smaller stratified sample's median and
# is what "auto" mode replaces for every subject where the mask succeeds.
CROP_CENTER_FALLBACK_MM = (2.3, 22.4, -38.1)

# --- Intensity normalization (EDA section 5) ----------------------------------
# Per-volume, never a single global constant: max spans 8 -> 54157 and p99
# spans 2 -> 18326 across volumes, several orders of magnitude.
INTENSITY_NORM = "per_volume_zscore"

# Background removal. Boulkrinat et al. 2025 threshold at the volume's 30th
# percentile; that is safe only where the head occupies a small share of the
# field of view. FOV[x] ranges 175.6 -> 629.8 mm here, so in the tight-FOV
# volumes a flat 30% can cut into brain. The threshold is therefore bounded by
# a small fraction of each volume's own max -- see the notebook's section 5.
BACKGROUND_PERCENTILE = 30
BACKGROUND_MAX_FRACTION = 0.05

# Rung-4 experiment 5 (RESOURCES.md, Boulkrinat et al. 2025's
# preprocessing step 3: patch-wise Non-Local-Means denoising, applied
# after crop/before normalization -- data.py::denoise_volume). Off by
# default -- not yet gate-validated against the current CNN
# (notebooks/13_cnn_denoising.ipynb). Flipping this is the only change
# needed to promote or revert the experiment; data.load_volume has no
# separate denoising code path to drift from this flag.
USE_NLM_DENOISING = False

# --- Roadmap item 6: 2D thick-slab CNN, architecture diversity ---------------
# Wenzel et al. 2019 (RESOURCES.md): a 2D CNN over a single thick axial
# "slab" through the striatum -- 2x2x12mm voxels, i.e. high in-plane
# resolution but only a 12mm-thick slice along S-I, unlike the full 3D
# TARGET_SHAPE volume above. Same L-R/A-P field of view as CROP_SIZE_MM
# (136x70mm vs 135x70mm -- rounded to whole voxels at this coarser
# spacing) and the same striatum-centered CROP_CENTER_MM offset reused
# unchanged -- no new center estimate needed, model.py::DatSlab2DCNN.
#
# 12mm as a SINGLE resampled voxel (the literal reading of "2x2x12mm
# voxels") was tried first and rejected after the fold-0 sanity check
# scored barely above the base-rate baseline (2026-09-11): crop_or_pad's
# nearest-voxel rounding (+-0.5 voxel = +-6mm at this spacing) is
# negligible against the 3D track's 44-voxel/108mm-thick crop, but
# catastrophic against a 1-voxel/12mm-thick one. Resampling to a FINER
# 2mm z-spacing and averaging 6 voxels (6x2mm=12mm, the same total
# physical thickness Wenzel specifies) in data.load_slab fixed that
# +-6mm rounding error down to +-1mm -- but a second Opus review
# (2026-09-11) found this was NOT the dominant error term and the slab
# track should be considered unvalidated, not negatively validated:
# CROP_CENTER_MM is the *median* row of notebooks/01_eda_volumes.ipynb
# section 6a's per-volume striatum-centroid-offset table, whose z
# component has sd ~32.5mm (5-95th pct range roughly -97mm to +4mm) --
# a 12mm-thick window fixed at that one population median point contains
# real striatal tissue for only ~14-47% of subjects by a normal
# approximation to that spread (vs. ~89% for the 3D track's 108mm-thick
# crop, which tolerates the same offset error by sheer thickness).
# `load_slab`'s degenerate-crop rate (2.1%, up from <1% under the
# unfixed geometry) is fully explained by this: `normalize_intensity`
# only returns its input unchanged (triggering the warning) on a
# LITERALLY all-zero crop, so that 2.1% is "missed the head entirely",
# not "imprecisely centered" -- and the offset distribution is strongly
# left-skewed (mean -25.1mm vs median -15.5mm), so the unfixed 1-voxel
# geometry (which happened to sit nearer -18 to -24mm) was, by accident,
# closer to the population's center of mass than this "fixed" one is.
# Bottom line: this geometry presumes spatial normalization to a common
# frame, which this pipeline never performs -- the rounding-precision fix
# above is real but was not the reason notebooks/24's gate came back
# negative. See RESOURCES.md's Wenzel entry and
# notebooks/24_slab2d_architecture_diversity.ipynb's reflection cell for
# the full second-review findings and what remains untested (per-subject
# centering via features.striatum_mask, never attempted).
SLAB_TARGET_SPACING = (2.0, 2.0, 2.0)
SLAB_TARGET_SHAPE = (68, 35, 6)

# --- Deep-learning defaults (see deep-learning-imaging.md); tune after
# representation work, not before. ---------------------------------------------
SEED = RANDOM_STATE
DEVICE = "cuda"
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
# Chegodaev et al. 2026 (RESOURCES.md) report L2 = 1e-5 as part of the training
# protocol that mattered more than architecture on DaT-SPECT.
WEIGHT_DECAY = 1e-5
# None = constant LR. Géron Ch.11: with an adaptive optimizer (Adam) a schedule
# is an experiment to justify, not a default. The first candidate to try is
# Chegodaev et al.'s cosine annealing with warm restarts (T0=50, Tmult=1,
# eta_min=1e-6).
LR_SCHEDULE = None
PATIENCE = 10
USE_AMP = True
NUM_WORKERS = 4
