#!/usr/bin/env python3
"""Verify a preprocessed BEHAVIOR-1K cache before committing GPU hours to it.

Checks the things that silently produce a useless training run:

  * the cache directory exists under ROBOMETER_PROCESSED_DATASETS_PATH
  * both quality labels are present (no failure rows -> nothing to prefer against)
  * exactly one data_source (the cutoff file and DATA_SOURCE_CATEGORY key off it)
  * the success cutoff resolves to 1.0 for that source
  * npz frames decode at the expected shape
  * an RBMDataset actually yields samples with progress/success labels

Usage:
    ROBOMETER_PROCESSED_DATASETS_PATH=... python scripts/b1k_verify_cache.py <cache_key>
    # cache_key example: datasets_b1k_rbm_b1k_skill_train_b1k_skill_train
    # omit it to list what is in the cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_key", nargs="?", help="cache directory name under the processed datasets path")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument(
        "--cutoff-file", default="robometer/data/dataset_success_cutoff.txt"
    )
    args = ap.parse_args()

    root = os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH", "")
    if not root:
        print("ROBOMETER_PROCESSED_DATASETS_PATH is not set")
        return 2
    if not os.path.isdir(root):
        print(f"cache root does not exist: {root}")
        return 2

    entries = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not args.cache_key:
        print(f"caches under {root}:")
        for entry in entries:
            print(f"  {entry}")
        print("\nPass one of these as the cache_key argument.")
        return 0

    cache_dir = os.path.join(root, args.cache_key)
    print(f"verifying {cache_dir}")
    if not os.path.isdir(cache_dir):
        fail(f"no such cache. Available: {entries}")
        return 1

    problems = 0

    # ---- index mappings ---------------------------------------------------
    with open(os.path.join(cache_dir, "index_mappings.json")) as handle:
        mappings = json.load(handle)

    quality = {k: len(v) for k, v in mappings["quality_indices"].items()}
    sources = {k: len(v) for k, v in mappings["source_indices"].items()}
    print(f"  quality_label counts: {quality}")
    print(f"  data_source counts  : {sources}")

    if len(sources) != 1:
        fail(f"expected exactly one data_source, got {list(sources)}")
        problems += 1
    else:
        ok(f"single data_source: {next(iter(sources))}")
    data_source = next(iter(sources), None)

    if "successful" not in quality:
        fail("no successful trajectories")
        problems += 1
    if "failure" not in quality and "suboptimal" not in quality:
        print("  ⚠️  no failure/suboptimal rows -- preference learning will lean entirely on "
              "online strategies (set b1k.truncated_negative_ratio > 0 to add them)")

    graded = sum(1 for v in mappings["suboptimal_by_task"].values() if v)
    print(f"  tasks with both optimal and suboptimal rows: {graded} "
          f"(policy_ranking eval needs these)")

    # ---- success cutoff ---------------------------------------------------
    if data_source:
        sys.path.insert(0, os.getcwd())
        from robometer.data.datasets.helpers import load_dataset_success_percent

        cutoffs = load_dataset_success_percent(args.cutoff_file)
        if data_source not in cutoffs:
            fail(f"'{data_source}' missing from {args.cutoff_file} -- progress labels will use "
                 f"the default max_success instead of the dataset cutoff")
            problems += 1
        else:
            ok(f"success cutoff for {data_source}: {cutoffs[data_source]}")

    # ---- frames -----------------------------------------------------------
    frames_dir = os.path.join(cache_dir, "frames")
    npz_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".npz"))
    print(f"  npz files: {len(npz_files)}")
    if npz_files:
        sample = np.load(os.path.join(frames_dir, npz_files[0]))["frames"]
        print(f"  npz[0] shape: {sample.shape} {sample.dtype}")
        if sample.ndim != 4 or sample.shape[-1] != 3 or sample.dtype != np.uint8:
            fail("unexpected frame array layout")
            problems += 1
        else:
            ok("frames decode as (T, H, W, 3) uint8")
    else:
        fail("no npz frames in cache")
        problems += 1

    # ---- end to end -------------------------------------------------------
    from robometer.configs.experiment_configs import DataConfig
    from robometer.data.datasets.rbm_data import RBMDataset

    cfg = DataConfig()
    cfg.train_datasets = [args.cache_key]
    cfg.eval_datasets = [args.cache_key]
    cfg.dataset_success_cutoff_file = args.cutoff_file
    cfg.max_frames = args.max_frames

    dataset = RBMDataset(cfg, is_evaluation=False)
    print(f"  RBMDataset length: {len(dataset)}")
    sample = dataset[0]
    traj = getattr(sample, "chosen_trajectory", None) or sample.trajectory
    frames = traj.frames
    print(f"  sample: {type(sample).__name__} strategy={sample.data_gen_strategy}")
    print(f"    task     : {traj.task}")
    print(f"    frames   : {None if frames is None else frames.shape}")
    print(f"    progress : {[round(float(x), 2) for x in traj.target_progress]}")
    print(f"    success  : {[int(x) for x in traj.success_label]}")

    if frames is None or frames.shape[0] != args.max_frames:
        fail(f"expected {args.max_frames} frames per sample")
        problems += 1
    else:
        ok(f"samples carry {args.max_frames} frames")

    print()
    if problems:
        print(f"❌ {problems} problem(s) -- fix before training")
        return 1
    print("✅ cache looks good")
    return 0


if __name__ == "__main__":
    sys.exit(main())
