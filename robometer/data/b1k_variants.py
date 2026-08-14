#!/usr/bin/env python3
"""Resolution variants of the BEHAVIOR-1K skill dataset.

The same segments can be converted at several frame resolutions so the effect of
resolution on subtask completion judgement can be measured. Every stage of the
pipeline needs a *different* name per resolution, and those names have to agree
with each other or training silently reads the wrong cache:

    conversion   datasets/b1k_rbm_480/b1k_skill_train      (mp4 + arrow)
    preprocess   <cache>/datasets_b1k_rbm_480_b1k_skill_train_b1k_skill_train
    registration DS_SHORT_NAME_MAPPING, DATASET_MAP["b1k_480"]
    training     data.train_datasets=[b1k_480]

This module is the single place those names are derived, so ``dataset_category``,
``name_mapping``, the pipeline script and the tests cannot drift apart. It has no
heavy imports on purpose -- ``scripts/b1k_pipeline.sh`` shells out to it.

240 keeps the original unsuffixed names (``datasets/b1k_rbm``, ``DATASET_MAP["b1k"]``)
so caches converted before this module existed stay valid.

``data_source`` stays ``b1k_skill`` at every resolution: it is a property of the
data, not of how many pixels it was stored at, and keeping it fixed means
``dataset_success_cutoff.txt`` needs one entry rather than one per resolution.
The short names are shared for the same reason -- eval metric keys such as
``eval_rew_align/pearson_b1k_skill_val`` stay comparable across resolution runs.
The cost is that two resolutions must not be trained on *in the same run*; their
metrics would collide. Compare them as separate runs.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "BASE_RESOLUTION",
    "SUPPORTED_RESOLUTIONS",
    "PATCH_GRID",
    "B1KVariant",
    "variant",
    "all_variants",
    "aligned_resolution",
]

# Resolution the repo's other 34 data_gen configs use, and the resolution
# Robometer-4B's RBM-1M pretraining ran at. Kept as the default for that reason.
BASE_RESOLUTION = 240

# Resolutions the pipeline knows how to build. Any value is accepted by
# ``variant()``; this list is what the driver script offers and what the tests
# check the registration for.
SUPPORTED_RESOLUTIONS: Tuple[int, ...] = (240, 480, 720)

# Qwen3-VL rounds each frame to a multiple of patch_size x merge_size (16 x 2)
# before patchifying. A resolution that is not a multiple of this is not an
# error -- qwen_vl_utils rounds it -- but the model then sees a size you did not
# choose, so the driver script warns.
PATCH_GRID = 32

_DATASET_ROOT = "datasets"
_DATASET_STEM = "b1k_rbm"
_TRAIN_SUBSET = "b1k_skill_train"
_VAL_SUBSET = "b1k_skill_val"

# Short names are resolution-independent; see the module docstring.
_SHORT_NAME = "b1k_skill"
_SHORT_NAME_VAL = "b1k_skill_val"

# Written into every converted row, and the key success cutoffs are looked up by.
DATA_SOURCE = "b1k_skill"


def _cache_key(dataset_path: str, subset: str) -> str:
    """Reproduce the cache directory name preprocess_datasets.py builds.

    See ``DatasetPreprocessor.load_datasets``: the key is "<path>/<subset>" with
    "/" and ":" replaced by "_". Mirrored rather than imported because that
    module pulls in torch and datasets.
    """
    return f"{dataset_path}/{subset}".replace("/", "_").replace(":", "_")


@dataclass(frozen=True)
class B1KVariant:
    """Every name one resolution of the dataset is known by."""

    resolution: int

    @property
    def suffix(self) -> str:
        """"" at 240, "_480" above it. Keeps pre-existing 240 paths untouched."""
        return "" if self.resolution == BASE_RESOLUTION else f"_{self.resolution}"

    # -- conversion -------------------------------------------------------
    @property
    def dataset_dir(self) -> str:
        """output.output_dir for generate_hf_dataset (mp4 tree + arrow dataset)."""
        return f"{_DATASET_ROOT}/{_DATASET_STEM}{self.suffix}"

    @property
    def hub_repo_id(self) -> str:
        return f"{_DATASET_STEM}{self.suffix}"

    # -- preprocess -------------------------------------------------------
    @property
    def train_path(self) -> str:
        return f"{self.dataset_dir}/{_TRAIN_SUBSET}"

    @property
    def eval_path(self) -> str:
        return f"{self.dataset_dir}/{_VAL_SUBSET}"

    @property
    def train_subset(self) -> str:
        return _TRAIN_SUBSET

    @property
    def eval_subset(self) -> str:
        return _VAL_SUBSET

    @property
    def train_cache_key(self) -> str:
        return _cache_key(self.train_path, _TRAIN_SUBSET)

    @property
    def eval_cache_key(self) -> str:
        return _cache_key(self.eval_path, _VAL_SUBSET)

    # -- registration / training -----------------------------------------
    @property
    def dataset_map_key(self) -> str:
        """The name used in data.train_datasets=[...] and custom_eval lists."""
        return f"b1k{self.suffix}"

    @property
    def short_name(self) -> str:
        return _SHORT_NAME

    @property
    def short_name_val(self) -> str:
        return _SHORT_NAME_VAL

    # -- sanity -----------------------------------------------------------
    @property
    def aligned_resolution(self) -> int:
        """What the processor will actually feed the vision tower."""
        return aligned_resolution(self.resolution)

    @property
    def is_patch_aligned(self) -> bool:
        return self.resolution % PATCH_GRID == 0


def aligned_resolution(resolution: int, grid: int = PATCH_GRID) -> int:
    """Round to the nearest multiple of the patch grid, the way smart_resize does."""
    return max(grid, int(round(resolution / grid)) * grid)


def variant(resolution: int = BASE_RESOLUTION) -> B1KVariant:
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return B1KVariant(resolution=int(resolution))


def all_variants() -> Tuple[B1KVariant, ...]:
    return tuple(variant(r) for r in SUPPORTED_RESOLUTIONS)


def short_name_entries() -> Dict[str, str]:
    """Cache key -> short name, for DS_SHORT_NAME_MAPPING."""
    entries: Dict[str, str] = {}
    for v in all_variants():
        entries[v.train_cache_key] = v.short_name
        entries[v.eval_cache_key] = v.short_name_val
    return entries


def dataset_map_entries() -> Dict[str, Dict[str, list]]:
    """DATASET_MAP fragment covering every supported resolution."""
    return {
        v.dataset_map_key: {"train": [v.train_cache_key], "eval": [v.eval_cache_key]}
        for v in all_variants()
    }


def _emit_shell(resolution: int) -> str:
    v = variant(resolution)
    pairs = [
        ("B1K_RES", str(v.resolution)),
        ("B1K_DATASET_DIR", v.dataset_dir),
        ("B1K_HUB_REPO_ID", v.hub_repo_id),
        ("B1K_TRAIN_PATH", v.train_path),
        ("B1K_EVAL_PATH", v.eval_path),
        ("B1K_TRAIN_SUBSET", v.train_subset),
        ("B1K_EVAL_SUBSET", v.eval_subset),
        ("B1K_TRAIN_KEY", v.train_cache_key),
        ("B1K_VAL_KEY", v.eval_cache_key),
        ("B1K_MAP_KEY", v.dataset_map_key),
        ("B1K_SHORT_NAME", v.short_name),
        ("B1K_SHORT_NAME_VAL", v.short_name_val),
        ("B1K_ALIGNED_RES", str(v.aligned_resolution)),
        ("B1K_PATCH_ALIGNED", "1" if v.is_patch_aligned else "0"),
    ]
    return "\n".join(f"{k}={shlex.quote(val)}" for k, val in pairs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("resolution", nargs="?", type=int, default=BASE_RESOLUTION)
    parser.add_argument(
        "--shell",
        action="store_true",
        help="emit `eval`-able shell assignments instead of a human-readable table",
    )
    args = parser.parse_args(argv)

    if args.shell:
        print(_emit_shell(args.resolution))
        return 0

    v = variant(args.resolution)
    rows = [
        ("resolution", str(v.resolution)),
        ("model sees", f"{v.aligned_resolution}x{v.aligned_resolution}"),
        ("dataset dir", v.dataset_dir),
        ("train cache key", v.train_cache_key),
        ("val cache key", v.eval_cache_key),
        ("DATASET_MAP key", v.dataset_map_key),
        ("data_source", DATA_SOURCE),
    ]
    width = max(len(k) for k, _ in rows)
    for key, val in rows:
        print(f"  {key.ljust(width)}  {val}")
    if not v.is_patch_aligned:
        print(
            f"\n  ! {v.resolution} is not a multiple of {PATCH_GRID}; the processor "
            f"will round frames to {v.aligned_resolution}x{v.aligned_resolution}."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
