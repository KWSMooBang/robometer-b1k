#!/usr/bin/env python3
"""Find (and optionally delete) converted clips that will not decode.

Preprocessing opens every clip with decord and, when that raises, prints
"Error in _process_one: ..." and returns None -- the trajectory is then dropped
from the cache without being counted. A handful of broken clips therefore shrink
the training set silently, and the ratio of truncated negatives to positives
drifts from what the conversion report claims.

Re-running `convert` does not repair them: create_trajectory_video_optimized()
returns early when the output path already exists, so a half-written mp4 from an
interrupted or failed conversion survives every retry. The file has to be deleted
first, which is what --delete does.

Usage:
    # report only
    python scripts/b1k_check_clips.py datasets/b1k_rbm

    # delete the broken ones, then re-run: ./scripts/b1k_pipeline.sh convert
    python scripts/b1k_check_clips.py datasets/b1k_rbm --delete

    # check every clip rather than a sample (slow but exhaustive)
    python scripts/b1k_check_clips.py datasets/b1k_rbm --all
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

try:
    import decord

    _HAS_DECORD = True
except Exception:  # pragma: no cover - depends on the host wheel
    _HAS_DECORD = False


# Smallest plausible mp4. Anything at or below this is a stub the writer never
# finished, and reporting it as "0 frames" would hide the real cause.
_MIN_BYTES = 1024


def check_clip(path: str, min_frames: int) -> Optional[str]:
    """Return a reason string when the clip is unusable, or None when it is fine.

    Mirrors what preprocessing does -- decord first, OpenCV as the fallback --
    so a clip that passes here is one preprocessing can also read.
    """
    if not os.path.exists(path):
        return "missing"

    size = os.path.getsize(path)
    if size == 0:
        return "empty file"
    if size < _MIN_BYTES:
        return f"truncated ({size} bytes)"

    try:
        if _HAS_DECORD:
            reader = decord.VideoReader(path, num_threads=1)
            count = len(reader)
            if count < min_frames:
                return f"only {count} frame(s)"
            # Length alone is not proof: a file can report a frame count from a
            # header that no longer matches its payload. Force an actual decode
            # of the last frame, which is where truncation shows up.
            reader.get_batch([0, count - 1]).asnumpy()
            del reader
        else:
            import cv2

            cap = cv2.VideoCapture(path)
            try:
                if not cap.isOpened():
                    return "cannot open"
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if count < min_frames:
                    return f"only {count} frame(s)"
                cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
                ok, _frame = cap.read()
                if not ok:
                    return "last frame does not decode"
            finally:
                cap.release()
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"

    return None


def find_clips(root: str) -> list:
    clips = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".mp4"):
                clips.append(os.path.join(dirpath, name))
    clips.sort()
    return clips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="converter output_dir, e.g. datasets/b1k_rbm")
    ap.add_argument("--delete", action="store_true", help="remove broken clips so convert regenerates them")
    ap.add_argument("--all", action="store_true", help="check every clip (default: stop listing after --max-report)")
    ap.add_argument("--min-frames", type=int, default=1, help="treat clips shorter than this as broken")
    ap.add_argument("--max-report", type=int, default=20, help="how many broken clips to list")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dataset_dir):
        print(f"  ❌ no such directory: {args.dataset_dir}")
        return 2

    clips = find_clips(args.dataset_dir)
    if not clips:
        print(f"  ❌ no mp4 clips under {args.dataset_dir}")
        return 2

    reader = "decord" if _HAS_DECORD else "OpenCV"
    print(f"  checking {len(clips)} clip(s) under {args.dataset_dir} with {reader}")

    broken: list[tuple[str, str]] = []
    for index, clip in enumerate(clips, start=1):
        reason = check_clip(clip, args.min_frames)
        if reason is not None:
            broken.append((clip, reason))
        if index % 500 == 0:
            print(f"    {index}/{len(clips)} checked, {len(broken)} broken so far")

    if not broken:
        print(f"  ✅ all {len(clips)} clips decode")
        return 0

    print(f"\n  ❌ {len(broken)} of {len(clips)} clip(s) will not decode:")
    shown = broken if args.all else broken[: args.max_report]
    for clip, reason in shown:
        print(f"    {clip}  --  {reason}")
    if len(shown) < len(broken):
        print(f"    ... and {len(broken) - len(shown)} more (pass --all to list them)")

    if not args.delete:
        print(
            "\n  Re-running convert will NOT fix these: the converter skips clips whose\n"
            "  mp4 already exists. Delete them first:\n"
            f"    python {os.path.relpath(__file__)} {args.dataset_dir} --delete\n"
            "    ./scripts/b1k_pipeline.sh convert"
        )
        return 1

    removed = 0
    for clip, _reason in broken:
        try:
            os.remove(clip)
            removed += 1
        except OSError as exc:
            print(f"    could not remove {clip}: {exc}")
    print(f"\n  🗑️  removed {removed} clip(s)")
    print("  now re-run conversion to regenerate them:")
    print("    ./scripts/b1k_pipeline.sh convert")
    print("  the arrow dataset still references them, so conversion must rewrite the")
    print("  same paths before preprocessing is re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
