"""Tests for the BEHAVIOR-1K resolution variants.

The names here are load-bearing: conversion writes a directory, preprocessing
derives a cache key from that directory, and training looks the key up through
DATASET_MAP. If any pair disagrees the run reads the wrong data (or no data) and
nothing in the numbers says so, which is exactly what these tests pin down.
"""

import pytest

from robometer.data.b1k_variants import (
    BASE_RESOLUTION,
    PATCH_GRID,
    SUPPORTED_RESOLUTIONS,
    aligned_resolution,
    all_variants,
    dataset_map_entries,
    short_name_entries,
    variant,
)


# --------------------------------------------------------------------------
# 240 keeps the pre-existing names
# --------------------------------------------------------------------------


def test_base_resolution_keeps_unsuffixed_names():
    """Caches converted before this module existed must stay valid."""
    v = variant(240)
    assert v.suffix == ""
    assert v.dataset_dir == "datasets/b1k_rbm"
    assert v.dataset_map_key == "b1k"
    assert v.train_cache_key == "datasets_b1k_rbm_b1k_skill_train_b1k_skill_train"
    assert v.eval_cache_key == "datasets_b1k_rbm_b1k_skill_val_b1k_skill_val"


def test_default_is_the_pretraining_resolution():
    assert variant().resolution == BASE_RESOLUTION == 240


# --------------------------------------------------------------------------
# higher resolutions get their own namespace
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resolution", [480, 720])
def test_higher_resolutions_are_suffixed(resolution):
    v = variant(resolution)
    assert v.dataset_dir == f"datasets/b1k_rbm_{resolution}"
    assert v.dataset_map_key == f"b1k_{resolution}"
    assert v.train_cache_key == f"datasets_b1k_rbm_{resolution}_b1k_skill_train_b1k_skill_train"


def test_no_two_resolutions_share_a_directory():
    """Conversion skips clips whose mp4 exists, so a shared directory would keep
    the first resolution's clips and silently ignore the second."""
    dirs = [v.dataset_dir for v in all_variants()]
    assert len(dirs) == len(set(dirs))


def test_no_two_resolutions_share_a_cache_key():
    keys = [k for v in all_variants() for k in (v.train_cache_key, v.eval_cache_key)]
    assert len(keys) == len(set(keys))


def test_train_and_val_keys_differ():
    for v in all_variants():
        assert v.train_cache_key != v.eval_cache_key


# --------------------------------------------------------------------------
# the cache key must match what preprocessing actually writes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resolution", SUPPORTED_RESOLUTIONS)
def test_cache_key_is_the_path_with_slashes_replaced(resolution):
    """Mirrors DatasetPreprocessor.load_datasets: "<path>/<subset>" -> "_"."""
    v = variant(resolution)
    assert v.train_cache_key == f"{v.train_path}/{v.train_subset}".replace("/", "_")
    assert v.eval_cache_key == f"{v.eval_path}/{v.eval_subset}".replace("/", "_")


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resolution", SUPPORTED_RESOLUTIONS)
def test_registered_in_dataset_map(resolution):
    from robometer.data.dataset_category import DATASET_MAP

    v = variant(resolution)
    assert v.dataset_map_key in DATASET_MAP
    assert DATASET_MAP[v.dataset_map_key]["train"] == [v.train_cache_key]
    assert DATASET_MAP[v.dataset_map_key]["eval"] == [v.eval_cache_key]


def _short_name_mapping():
    """DS_SHORT_NAME_MAPPING, without needing the training dependencies.

    ``robometer.data.datasets.__init__`` imports RBMDataset and therefore torch,
    so the normal import only works in a full environment. The mapping itself is
    a plain dict, so fall back to loading the module file directly and keep this
    test runnable during data work.
    """
    try:
        from robometer.data.datasets.name_mapping import DS_SHORT_NAME_MAPPING

        return DS_SHORT_NAME_MAPPING
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_name_mapping_standalone", "robometer/data/datasets/name_mapping.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DS_SHORT_NAME_MAPPING


@pytest.mark.parametrize("resolution", SUPPORTED_RESOLUTIONS)
def test_registered_in_short_name_mapping(resolution):
    DS_SHORT_NAME_MAPPING = _short_name_mapping()

    v = variant(resolution)
    assert DS_SHORT_NAME_MAPPING[v.train_cache_key] == v.short_name
    assert DS_SHORT_NAME_MAPPING[v.eval_cache_key] == v.short_name_val


def test_short_names_are_resolution_independent():
    """Eval metric keys embed the short name, so keeping it fixed is what makes
    eval_rew_align/pearson_b1k_skill_val comparable across resolution runs."""
    names = {v.short_name for v in all_variants()}
    val_names = {v.short_name_val for v in all_variants()}
    assert names == {"b1k_skill"}
    assert val_names == {"b1k_skill_val"}


def test_data_source_has_a_success_cutoff():
    """data_source stays b1k_skill at every resolution, so one entry covers all."""
    from robometer.data.b1k_variants import DATA_SOURCE

    with open("robometer/data/dataset_success_cutoff.txt") as handle:
        sources = {line.split(",")[0].strip() for line in handle if "," in line}
    assert DATA_SOURCE in sources


def test_registration_covers_every_supported_resolution():
    assert set(dataset_map_entries()) == {v.dataset_map_key for v in all_variants()}
    assert len(short_name_entries()) == 2 * len(SUPPORTED_RESOLUTIONS)


# --------------------------------------------------------------------------
# patch alignment
# --------------------------------------------------------------------------


def test_480_is_patch_aligned_and_720_is_not():
    assert variant(480).is_patch_aligned
    assert not variant(720).is_patch_aligned
    assert variant(720).aligned_resolution == 704


def test_aligned_resolution_rounds_to_the_grid():
    assert aligned_resolution(240) == 256
    assert aligned_resolution(480) == 480
    assert aligned_resolution(704) == 704
    assert aligned_resolution(1) == PATCH_GRID


def test_rejects_nonsense_resolutions():
    with pytest.raises(ValueError):
        variant(0)
    with pytest.raises(ValueError):
        variant(-480)


# --------------------------------------------------------------------------
# the shell contract the pipeline script depends on
# --------------------------------------------------------------------------


def test_shell_output_is_assignments_the_driver_can_eval():
    from robometer.data.b1k_variants import _emit_shell

    lines = _emit_shell(480).splitlines()
    emitted = dict(line.split("=", 1) for line in lines)

    assert emitted["B1K_RES"] == "480"
    assert emitted["B1K_MAP_KEY"] == "b1k_480"
    assert emitted["B1K_PATCH_ALIGNED"] == "1"
    # Every name the driver reads must be present, or `set -u` aborts the run.
    for key in (
        "B1K_DATASET_DIR",
        "B1K_HUB_REPO_ID",
        "B1K_TRAIN_PATH",
        "B1K_EVAL_PATH",
        "B1K_TRAIN_SUBSET",
        "B1K_EVAL_SUBSET",
        "B1K_TRAIN_KEY",
        "B1K_VAL_KEY",
        "B1K_MAP_KEY",
        "B1K_SHORT_NAME",
        "B1K_SHORT_NAME_VAL",
        "B1K_ALIGNED_RES",
        "B1K_PATCH_ALIGNED",
    ):
        assert key in emitted, f"{key} missing from --shell output"


def test_shell_output_flags_unaligned_resolutions():
    from robometer.data.b1k_variants import _emit_shell

    emitted = dict(line.split("=", 1) for line in _emit_shell(720).splitlines())
    assert emitted["B1K_PATCH_ALIGNED"] == "0"
    assert emitted["B1K_ALIGNED_RES"] == "704"
