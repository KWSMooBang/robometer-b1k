#!/usr/bin/env python3
"""Contact sheets for BEHAVIOR-1K skill segments -- the G2 visual gate.

A frame-offset bug (mis-reading ``from_timestamp``, an off-by-one-episode slice)
passes every numeric check while making every progress label meaningless. The
only thing that catches it is looking at the clips: the last frame of a segment
must show the subtask *completed*.

The sampler deliberately favours episodes that sit deep inside an aggregated
mp4, since offset bugs are invisible on the first episode of a file (offset 0).

Usage:
    python scripts/b1k_preview_segments.py <b1k_root> -o out.png \
        --tasks turning_on_radio --rows 12
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

from dataset_upload.dataset_loaders.b1k_loader import load_b1k_dataset

THUMB = 150
PAD = 4
CAPTION_H = 22


def _draw_caption(width: int, text: str, subtext: str = ""):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, CAPTION_H), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), text, fill=(235, 235, 240))
    if subtext:
        draw.text((max(4, width - 8 * len(subtext) - 6), 2), subtext, fill=(140, 150, 170))
    return np.asarray(img)


def _thumbs(frames: np.ndarray, count: int) -> list[np.ndarray]:
    import cv2

    idx = [int(i * (len(frames) - 1) / max(1, count - 1)) for i in range(count)]
    return [cv2.resize(frames[i], (THUMB, THUMB), interpolation=cv2.INTER_AREA) for i in idx]


def build_sheet(trajs, cols: int) -> np.ndarray:
    rows = []
    for traj in trajs:
        frames = traj["frames"]()
        if frames is None:
            print(f"  !! no frames for {traj['id']}")
            continue
        thumbs = _thumbs(frames, cols)
        strip = np.full(
            (THUMB, cols * THUMB + (cols - 1) * PAD, 3), 20, dtype=np.uint8
        )
        for i, thumb in enumerate(thumbs):
            x = i * (THUMB + PAD)
            strip[:, x : x + THUMB] = thumb
        meta = traj.get("metadata", {})
        caption = _draw_caption(
            strip.shape[1],
            f"{traj['id']}  |  {traj['task']}",
            f"ep{meta.get('episode_index')} frames{meta.get('segment_frames')} "
            f"{meta.get('skill_type')} [{traj['quality_label']}]",
        )
        rows.append(np.vstack([caption, strip]))
        print(f"  {traj['id']:<34s} {traj['task']}")

    if not rows:
        raise SystemExit("no clips rendered")
    width = max(r.shape[1] for r in rows)
    padded = [
        np.pad(r, ((0, PAD), (0, width - r.shape[1]), (0, 0)), constant_values=20)
        for r in rows
    ]
    return np.vstack(padded)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", default="b1k_preview.png")
    ap.add_argument("--tasks", nargs="*", default=None, help="task names/ids to sample from")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--episodes-per-task", type=int, default=40)
    ap.add_argument("--truncated-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task_data = load_b1k_dataset(
        args.root,
        task_whitelist=args.tasks,
        max_episodes_per_task=args.episodes_per_task,
        split="all",
        truncated_negative_ratio=args.truncated_ratio,
        seed=args.seed,
    )
    trajs = [t for group in task_data.values() for t in group]

    # Prefer clips that start deep inside their aggregated mp4: an offset bug is
    # invisible at offset 0 (the first episode of a file).
    trajs.sort(key=lambda t: -t["frames"].global_start)
    deep = trajs[: max(args.rows * 4, 40)]
    rng = random.Random(args.seed)
    rng.shuffle(deep)

    # Spread the sample across distinct skills so one skill does not fill the sheet.
    by_skill: dict[str, list] = {}
    for traj in deep:
        by_skill.setdefault(traj["metadata"].get("skill_description", "?"), []).append(traj)
    picked, i = [], 0
    while len(picked) < args.rows and any(by_skill.values()):
        for skill in list(by_skill):
            if by_skill[skill]:
                picked.append(by_skill[skill].pop())
                if len(picked) >= args.rows:
                    break
        i += 1
        if i > 100:
            break

    print(f"rendering {len(picked)} clips (global_start range "
          f"{min(t['frames'].global_start for t in picked)}..{max(t['frames'].global_start for t in picked)})")
    sheet = build_sheet(picked, args.cols)

    import cv2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("\nCheck: the LAST frame of each row must show that subtask completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
