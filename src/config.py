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

# Deep-learning defaults (see deep-learning-imaging.md); tune after
# representation work, not before.
SEED = RANDOM_STATE
DEVICE = "cuda"
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
PATIENCE = 10
USE_AMP = True
NUM_WORKERS = 4
