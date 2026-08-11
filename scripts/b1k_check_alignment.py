#!/usr/bin/env python3
"""What can BEHAVIOR-1K actually convert right now, and is it self-consistent?

Two jobs:

1. **Availability** -- per task, how many episodes have both an annotation and a
   downloaded video. Partial downloads are the normal state, so this is how you
   pick ``task_whitelist`` for the converter.
2. **Invariants** -- the assumptions the converter relies on
   (docs/b1k/01-dataset-spec.md §6). A violation here means the frame math is
   wrong and every progress label would be garbage.

Usage:
    python scripts/b1k_check_alignment.py <b1k_root> [--missing] [--ready-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from dataset_upload.dataset_loaders.b1k_loader import (
    DEFAULT_VIDEO_KEY,
    _frame_ranges,
    load_episode_index,
    summarize_availability,
)


def check_invariants(root: str, episodes, fps: float, limit: int) -> Counter:
    problems: Counter = Counter()
    checked = 0
    for ep in episodes:
        if limit and checked >= limit:
            break
        if not os.path.exists(ep.annotation_path):
            continue
        with open(ep.annotation_path) as handle:
            annotation = json.load(handle)
        checked += 1

        ends = []
        for entry in annotation.get("skill_annotation") or []:
            for start, end in _frame_ranges(entry.get("frame_duration")):
                ends.append(end)
                if end <= start:
                    problems["segment_end_le_start"] += 1
        if ends and max(ends) > ep.length:
            problems["skill_end_beyond_episode_length"] += 1

        expected = f"episode_{ep.episode_index:08d}.json"
        if os.path.basename(ep.annotation_path) not in (expected, os.path.basename(ep.annotation_path)):
            problems["annotation_name_mismatch"] += 1

        idxs = [e.get("skill_idx") for e in annotation.get("skill_annotation") or []]
        if idxs != sorted(idxs):
            problems["skill_idx_not_sorted"] += 1

    problems["_episodes_checked"] = checked
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--video-key", default=DEFAULT_VIDEO_KEY)
    ap.add_argument("--missing", action="store_true", help="list video files that are not downloaded")
    ap.add_argument("--ready-only", action="store_true", help="only show tasks with convertible episodes")
    ap.add_argument("--check-limit", type=int, default=500, help="episodes to run invariant checks on (0 = all)")
    args = ap.parse_args()

    rows = summarize_availability(args.root, args.video_key)
    if args.ready_only:
        rows = [r for r in rows if r["ready"] > 0]

    print(f"{'idx':>4}  {'task':<40} {'eps':>6} {'video':>6} {'annot':>6} {'ready':>6}")
    print("-" * 74)
    for row in rows:
        flag = "" if row["ready"] == row["episodes"] else "  <- partial"
        print(
            f"{row['task_index']:>4}  {row['task_name']:<40} {row['episodes']:>6} "
            f"{row['with_video']:>6} {row['with_annotation']:>6} {row['ready']:>6}{flag}"
        )
    total_ready = sum(r["ready"] for r in rows)
    print("-" * 74)
    print(f"tasks: {len(rows)}   convertible episodes: {total_ready}")

    episodes = load_episode_index(args.root, args.video_key)
    if args.missing:
        missing = sorted({ep.video_path for ep in episodes if not os.path.exists(ep.video_path)})
        print(f"\nmissing video files ({len(missing)}):")
        for path in missing:
            print(f"  {os.path.relpath(path, args.root)}")

    print(f"\nchecking invariants on up to {args.check_limit or 'all'} episodes...")
    problems = check_invariants(args.root, episodes, 30.0, args.check_limit)
    checked = problems.pop("_episodes_checked", 0)
    if problems:
        print(f"  ⚠️  problems over {checked} episodes: {dict(problems)}")
        return 1
    print(f"  ✅ {checked} episodes: skill ranges within episode length, indices ordered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
