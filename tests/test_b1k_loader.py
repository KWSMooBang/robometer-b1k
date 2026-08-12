"""Tests for the BEHAVIOR-1K loader.

Pure-logic tests always run. Tests that touch the dataset are skipped unless
``B1K_ROOT`` points at a local copy:

    B1K_ROOT=/path/to/2026-challenge-demos pytest tests/test_b1k_loader.py
"""

import os
import pickle

import numpy as np
import pytest

from dataset_upload.dataset_loaders.b1k_loader import (
    ConversionStats,
    _frame_ranges,
    _normalize_whitelist,
    load_b1k_dataset,
    summarize_availability,
)
from dataset_upload.dataset_loaders.b1k_video import B1KSegmentFrameLoader, sample_indices

B1K_ROOT = os.environ.get("B1K_ROOT")
needs_data = pytest.mark.skipif(not B1K_ROOT, reason="set B1K_ROOT to run against the dataset")


# --------------------------------------------------------------------------
# pure logic
# --------------------------------------------------------------------------


def test_frame_ranges_plain():
    assert _frame_ranges([265, 1162]) == [(265, 1162)]


def test_frame_ranges_split():
    # 724 segments in the dataset carry an interrupted range
    assert _frame_ranges([[2795, 3007], [3209, 3638]]) == [(2795, 3007), (3209, 3638)]


def test_frame_ranges_malformed():
    assert _frame_ranges([]) == []
    assert _frame_ranges([1, 2, 3]) == []
    assert _frame_ranges(None) == []


def test_sample_indices_short_segment_kept_whole():
    assert sample_indices(10, 20, 64) == list(range(10, 20))


def test_sample_indices_downsamples_uniformly():
    idx = sample_indices(0, 900, 64)
    assert len(idx) == 64
    assert idx[0] == 0
    assert max(idx) < 900
    assert idx == sorted(idx)


def test_sample_indices_empty_range():
    assert sample_indices(5, 5, 64) == []
    assert sample_indices(9, 3, 64) == []


def test_normalize_whitelist_accepts_names_ids_and_indices():
    names = {0: "turning_on_radio", 7: "picking_up_trash"}
    assert _normalize_whitelist(["turning_on_radio"], names) == {0}
    assert _normalize_whitelist(["task-0007"], names) == {7}
    assert _normalize_whitelist([7], names) == {7}
    assert _normalize_whitelist(["7"], names) == {7}
    assert _normalize_whitelist(["turning on radio"], names) == {0}
    assert _normalize_whitelist(None, names) is None


def test_normalize_whitelist_rejects_unknown():
    with pytest.raises(ValueError):
        _normalize_whitelist(["no_such_task"], {0: "turning_on_radio"})


def test_frame_loader_is_picklable():
    loader = B1KSegmentFrameLoader("/tmp/x.mp4", 100, 200, max_frames=8)
    restored = pickle.loads(pickle.dumps(loader))
    assert restored.global_start == 100
    assert restored.indices() == loader.indices()


def test_conservation_check_catches_a_leak():
    stats = ConversionStats(segments_total=10, segments_kept=4)
    stats.segments_dropped["too_short"] = 3
    with pytest.raises(AssertionError):
        stats.check_conservation()
    stats.segments_dropped["oob"] = 3
    stats.check_conservation()


# --------------------------------------------------------------------------
# against the real dataset
# --------------------------------------------------------------------------


@needs_data
def test_availability_report():
    rows = summarize_availability(B1K_ROOT)
    assert rows
    for row in rows:
        assert row["ready"] <= row["episodes"]


@needs_data
def test_load_subset_shapes_and_labels():
    task_data = load_b1k_dataset(
        B1K_ROOT, max_episodes_per_task=3, split="all", verbose=False
    )
    trajs = [t for group in task_data.values() for t in group]
    assert trajs

    for traj in trajs:
        assert traj["is_robot"] is True
        assert traj["data_source"] == "b1k_skill"
        assert traj["quality_label"] == "successful"
        assert traj["partial_success"] == 1.0
        assert traj["task"] and traj["task"] == traj["task"].strip()
        assert traj["id"].startswith("b1k-")
        pickle.dumps(traj["frames"])  # spawn pool requirement


@needs_data
def test_decoded_clip_shape():
    task_data = load_b1k_dataset(
        B1K_ROOT, max_episodes_per_task=1, split="all", max_frames=16, verbose=False
    )
    traj = next(iter(task_data.values()))[0]
    frames = traj["frames"]()
    assert frames is not None
    assert frames.ndim == 4 and frames.shape[-1] == 3
    assert frames.dtype == np.uint8
    assert 8 <= frames.shape[0] <= 16


@needs_data
def test_ids_are_deterministic_across_runs():
    def ids():
        data = load_b1k_dataset(
            B1K_ROOT, max_episodes_per_task=3, split="all", verbose=False
        )
        return [t["id"] for group in data.values() for t in group]

    assert ids() == ids()


@needs_data
def test_train_and_val_splits_are_disjoint_by_episode():
    def episodes(split):
        data = load_b1k_dataset(B1K_ROOT, split=split, verbose=False)
        return {t["metadata"]["episode_index"] for group in data.values() for t in group}

    train, val = episodes("train"), episodes("val")
    assert train and val
    assert not (train & val)


@needs_data
def test_task_whitelist_restricts_output():
    rows = summarize_availability(B1K_ROOT)
    ready = [r for r in rows if r["ready"] > 0]
    name = ready[0]["task_name"]
    data = load_b1k_dataset(
        B1K_ROOT, task_whitelist=[name], max_episodes_per_task=2, split="all", verbose=False
    )
    names = {t["metadata"]["task_name"] for group in data.values() for t in group}
    assert names == {name}


@needs_data
def test_truncated_negatives_carry_partial_success():
    data = load_b1k_dataset(
        B1K_ROOT,
        max_episodes_per_task=5,
        split="all",
        truncated_negative_ratio=1.0,
        verbose=False,
    )
    trajs = [t for group in data.values() for t in group]
    negatives = [t for t in trajs if t["metadata"]["variant"] == "truncated"]
    assert negatives
    for neg in negatives:
        assert neg["quality_label"] == "failure"
        assert 0.30 <= neg["partial_success"] <= 0.70
        start, end = neg["metadata"]["segment_frames"]
        assert end > start
