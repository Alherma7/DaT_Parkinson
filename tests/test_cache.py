"""Unit tests for src/cache.py. Synthetic load_fn only -- never real
patient data.
"""

import numpy as np

from cache import CachedVolumeStore

SHAPE = (2, 3, 4)


def _value_for(uid):
    return (abs(hash(uid)) % 100) / 10.0


def make_load_fn(calls):
    def load_fn(uid):
        calls.append(uid)
        return np.full(SHAPE, _value_for(uid), dtype=np.float32)
    return load_fn


def test_build_calls_load_fn_once_per_uid_then_never_again(tmp_path):
    uids = ["a", "b", "c"]
    calls = []
    store = CachedVolumeStore(uids, tmp_path / "cache", {"v": 1}, load_fn=make_load_fn(calls))

    assert sorted(calls) == sorted(uids)

    calls.clear()
    for uid in uids:
        store.get(uid)
        store.get(uid)
    assert calls == []


def test_returns_correct_volume_per_uid(tmp_path):
    uids = ["a", "b", "c"]
    store = CachedVolumeStore(uids, tmp_path / "cache", {"v": 1}, load_fn=make_load_fn([]))

    for uid in ["c", "a", "b", "a"]:
        expected = np.full(SHAPE, _value_for(uid), dtype=np.float32)
        np.testing.assert_array_equal(store.get(uid), expected)


def test_persists_across_instances_without_recalling_load_fn(tmp_path):
    uids = ["a", "b"]
    cache_dir = tmp_path / "cache"
    first = CachedVolumeStore(uids, cache_dir, {"v": 1}, load_fn=make_load_fn([]))

    def raising_load_fn(uid):
        raise AssertionError("load_fn should not be called when the disk cache is valid")

    second = CachedVolumeStore(uids, cache_dir, {"v": 1}, load_fn=raising_load_fn)
    for uid in uids:
        np.testing.assert_array_equal(second.get(uid), first.get(uid))


def test_config_fingerprint_mismatch_triggers_rebuild(tmp_path):
    uids = ["a", "b"]
    cache_dir = tmp_path / "cache"
    CachedVolumeStore(uids, cache_dir, {"shape": (56, 30, 44)}, load_fn=make_load_fn([]))

    calls_second = []
    CachedVolumeStore(uids, cache_dir, {"shape": (56, 30, 45)}, load_fn=make_load_fn(calls_second))
    assert sorted(calls_second) == sorted(uids)


def test_tuple_and_list_fingerprints_compare_equal(tmp_path):
    # JSON round-trips a tuple fingerprint value into a list -- the
    # comparison must normalize both sides, or the cache would silently
    # rebuild on every single instantiation.
    uids = ["a", "b"]
    cache_dir = tmp_path / "cache"
    CachedVolumeStore(uids, cache_dir, {"shape": (56, 30, 44)}, load_fn=make_load_fn([]))

    calls_second = []
    CachedVolumeStore(uids, cache_dir, {"shape": [56, 30, 44]}, load_fn=make_load_fn(calls_second))
    assert calls_second == []


def test_missing_fingerprint_file_triggers_rebuild_even_if_others_exist(tmp_path):
    uids = ["a", "b"]
    cache_dir = tmp_path / "cache"
    CachedVolumeStore(uids, cache_dir, {"v": 1}, load_fn=make_load_fn([]))
    (cache_dir / "fingerprint.json").unlink()  # simulate interrupted rebuild

    calls_second = []
    CachedVolumeStore(uids, cache_dir, {"v": 1}, load_fn=make_load_fn(calls_second))
    assert sorted(calls_second) == sorted(uids)


# --- was_reused ---------------------------------------------------------

def test_was_reused_is_false_on_first_build(tmp_path):
    store = CachedVolumeStore(["a", "b"], tmp_path / "cache", {"v": 1}, load_fn=make_load_fn([]))

    assert store.was_reused is False


def test_was_reused_is_true_when_disk_cache_matches(tmp_path):
    cache_dir = tmp_path / "cache"
    CachedVolumeStore(["a", "b"], cache_dir, {"v": 1}, load_fn=make_load_fn([]))

    second = CachedVolumeStore(["a", "b"], cache_dir, {"v": 1}, load_fn=make_load_fn([]))

    assert second.was_reused is True


def test_was_reused_is_false_when_fingerprint_mismatch_forces_rebuild(tmp_path):
    cache_dir = tmp_path / "cache"
    CachedVolumeStore(["a", "b"], cache_dir, {"v": 1}, load_fn=make_load_fn([]))

    second = CachedVolumeStore(["a", "b"], cache_dir, {"v": 2}, load_fn=make_load_fn([]))

    assert second.was_reused is False
