"""Paths, constants, and reproducibility settings for this project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
NIFTI_DIR = DATA_RAW / "niftis"
TRAIN_LABELS_PATH = DATA_RAW / "train_labels.csv"

SMOKE_TEST_DIR = DATA_RAW / "smoke_test"
SMOKE_TEST_NIFTI_DIR = SMOKE_TEST_DIR / "niftis"
SMOKE_TEST_SUBMISSION_FORMAT_PATH = SMOKE_TEST_DIR / "submission_format.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

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
