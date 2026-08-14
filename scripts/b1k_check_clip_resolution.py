#!/usr/bin/env python3
"""Check that converted BEHAVIOR-1K clips really carry the resolution asked for.

Worth its own stage because ``output.shortest_edge_size`` is not always honoured:
``create_trajectory_video`` (the legacy writer) leaves square input untouched, and
only ``create_trajectory_video_optimized`` scales it. B1K frames *are* square, so
a future switch between those two paths would change the resolution of every clip
without changing a single log line. Everything downstream -- cache key, dataset
name, metric name -- would still say 480 while the pixels said 720.

Usage:
    python scripts/b1k_check_clip_resolution.py datasets/b1k_rbm_480 --expect 480
    python scripts/b1k_check_clip_resolution.py datasets/b1k_rbm_480   # just report
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from typing import Optional, Tuple

import cv2


def probe(path: str) -> Optional[Tuple[int, int]]:
    """(width, height) of the first frame, or None if the clip will not open."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            ok, frame = capture.read()
            if not ok:
                return None
            height, width = frame.shape[:2]
        return width, height
    finally:
        capture.release()


def find_clips(root: str, limit: int, seed: int) -> list:
    clips = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".mp4"):
                clips.append(os.path.join(dirpath, name))
    clips.sort()
    if len(clips) > limit:
        # Sample rather than take the first N: clips are written in task order, so
        # the head of the list is one task and would hide a per-task problem.
        clips = random.Random(seed).sample(clips, limit)
    return clips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="converter output_dir, e.g. datasets/b1k_rbm_480")
    ap.add_argument("--expect", type=int, default=None, help="required shortest edge")
    ap.add_argument("--sample", type=int, default=64, help="how many clips to probe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dataset_dir):
        print(f"  ❌ no such directory: {args.dataset_dir}")
        return 1

    clips = find_clips(args.dataset_dir, args.sample, args.seed)
    if not clips:
        print(f"  ❌ no mp4 clips under {args.dataset_dir}")
        return 1

    sizes: Counter = Counter()
    unreadable = []
    for clip in clips:
        size = probe(clip)
        if size is None:
            unreadable.append(clip)
        else:
            sizes[size] += 1

    print(f"  probed {len(clips)} clip(s) under {args.dataset_dir}")
    for (width, height), count in sizes.most_common():
        print(f"    {width}x{height}: {count}")

    problems = 0
    if unreadable:
        print(f"  ❌ {len(unreadable)} clip(s) would not open, e.g. {unreadable[0]}")
        problems += 1

    if len(sizes) > 1:
        # Mixed sizes almost always mean two conversions landed in one directory.
        # The converter skips clips whose mp4 already exists, so the older
        # resolution survives and training silently reads a blend of both.
        print("  ❌ clips are not all the same size -- two conversions in one directory?")
        print(f"     remove {args.dataset_dir} and reconvert at a single resolution")
        problems += 1

    if args.expect is not None:
        for (width, height), _count in sizes.items():
            if min(width, height) != args.expect:
                print(f"  ❌ expected shortest edge {args.expect}, found {width}x{height}")
                problems += 1
                break
        else:
            if sizes:
                print(f"  ✅ shortest edge is {args.expect}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
