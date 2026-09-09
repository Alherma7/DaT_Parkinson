"""Unit tests for src/config.py's runtime-environment path detection."""

import config


def test_resolve_runtime_paths_detects_existing_code_execution_root(tmp_path):
    fake_root = tmp_path / "code_execution"
    fake_root.mkdir()

    running, nifti_dir, submission_format_path = config._resolve_runtime_paths(fake_root)

    assert running is True
    assert nifti_dir == fake_root / "data" / "niftis"
    assert submission_format_path == fake_root / "data" / "submission_format.csv"


def test_resolve_runtime_paths_falls_back_when_root_is_missing(tmp_path):
    missing_root = tmp_path / "does_not_exist"

    running, nifti_dir, submission_format_path = config._resolve_runtime_paths(missing_root)

    assert running is False
    assert nifti_dir == config.DATA_RAW / "niftis"
    assert submission_format_path is None


def test_module_level_nifti_dir_defaults_to_local_layout():
    """Regression guard: on a normal dev machine (no /code_execution),
    the module-level NIFTI_DIR used by data.load_volume must still be the
    local training path, unchanged from before this task."""
    assert config.NIFTI_DIR == config.DATA_RAW / "niftis"
    assert config.RUNNING_IN_CODE_EXECUTION is False
