#!/usr/bin/env python3
"""Visualise what actually landed in a converted BEHAVIOR-1K dataset or npz cache.

``b1k_preview_segments.py`` renders clips straight from the source videos, so it
checks the *loader*. This one reads the **converted output** -- the HF dataset
written by generate_hf_dataset, or the npz cache written by preprocess_datasets --
and draws each clip next to the progress/success labels that training will
actually compute from it, via the real label functions
(``compute_progress_from_segment`` / ``compute_success_labels``).

That pairing is the point: a clip can look fine while its labels are wrong
(missing cutoff entry, partial_success not carried through, quality_label typo),
and a label table alone tells you nothing about whether the frames match.

Two inputs, auto-detected:

    datasets/b1k_rbm/b1k_skill_train                 # converted HF dataset + mp4s
    $ROBOMETER_PROCESSED_DATASETS_PATH/<cache_key>   # preprocessed npz cache

Two outputs:

    --out sheet.png      contact sheet, one row per clip  (default)
    --html gallery.html  self-contained page with playable clips

Usage:
    python scripts/b1k_visualize_dataset.py datasets/b1k_rbm/b1k_skill_train -n 12
    python scripts/b1k_visualize_dataset.py <cache_dir> --quality failure --html neg.html
    python scripts/b1k_visualize_dataset.py <dir> --task "pick up" --out picks.png
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import sys
from typing import Optional

import numpy as np

THUMB = 150
PAD = 4
CAPTION_H = 22
LABEL_H = 30

GREEN = (90, 200, 120)
RED = (215, 95, 95)
DIM = (110, 118, 132)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _is_cache_dir(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "frames")) and os.path.exists(
        os.path.join(path, "index_mappings.json")
    )


def load_rows(path: str, limit: Optional[int] = None) -> tuple[list[dict], str]:
    """Return (rows, kind) where kind is 'cache' or 'dataset'.

    Each row carries: id, task, quality_label, partial_success, data_source and
    either ``npz`` (cache) or ``video`` (converted dataset) as the frame source.
    """
    from datasets import load_from_disk

    if _is_cache_dir(path):
        dataset = load_from_disk(os.path.join(path, "processed_dataset"))
        kind = "cache"
    else:
        dataset = load_from_disk(path)
        kind = "dataset"

    rows = []
    # "frames" holds a path relative to the converter's output_dir, i.e. the
    # parent of the dataset directory.
    video_root = os.path.dirname(os.path.abspath(path))
    for row in dataset:
        entry = {
            "id": row["id"],
            "task": row["task"],
            "quality_label": row.get("quality_label", "successful"),
            "partial_success": row.get("partial_success"),
            "data_source": row.get("data_source", "unknown"),
        }
        frames_field = row["frames"]
        if kind == "cache":
            entry["npz"] = frames_field
        else:
            entry["video"] = (
                frames_field if os.path.isabs(frames_field) else os.path.join(video_root, frames_field)
            )
        rows.append(entry)
        if limit and len(rows) >= limit:
            break
    return rows, kind


def read_clip(row: dict, max_frames: int) -> Optional[np.ndarray]:
    if "npz" in row:
        path = row["npz"]
        if not os.path.exists(path):
            return None
        frames = np.load(path)["frames"]
    else:
        path = row["video"]
        if not os.path.exists(path):
            return None
        import cv2

        cap = cv2.VideoCapture(path)
        collected = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            collected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not collected:
            return None
        frames = np.stack(collected)

    if len(frames) > max_frames:
        idx = [int(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        frames = frames[idx]
    return frames


# --------------------------------------------------------------------------
# labels -- computed with the same functions training uses
# --------------------------------------------------------------------------


def compute_labels(row: dict, n_frames: int, cutoff_file: str, progress_pred_type: str):
    from robometer.data.datasets.helpers import (
        compute_progress_from_segment,
        compute_success_labels,
    )
    from robometer.data.datasets.helpers import load_dataset_success_percent

    cutoffs = load_dataset_success_percent(cutoff_file)
    source = row.get("data_source")
    cutoff = cutoffs.get(source, 1.0)

    progress = compute_progress_from_segment(
        num_frames_total=n_frames,
        frame_indices=list(range(n_frames)),
        progress_pred_type=progress_pred_type,
        success_cutoff=cutoff,
        partial_success=row.get("partial_success"),
    )
    success = compute_success_labels(
        progress, source, cutoffs, 1.0, row.get("quality_label")
    )
    return progress, success, cutoff


# --------------------------------------------------------------------------
# PNG contact sheet
# --------------------------------------------------------------------------


def _text_strip(width: int, height: int, left: str, right: str = "", bg=(24, 24, 28)):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), left, fill=(235, 235, 240))
    if right:
        draw.text((max(4, width - 7 * len(right) - 6), 2), right, fill=(140, 150, 170))
    return np.asarray(img)


def _label_strip(width: int, cols: int, progress, success):
    """Per-frame progress value plus a success bar under each thumbnail."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, LABEL_H), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    for i in range(cols):
        x = i * (THUMB + PAD)
        p = progress[i] if i < len(progress) else 0.0
        s = success[i] if i < len(success) else 0.0
        draw.text((x + 4, 1), f"p={p:.2f}", fill=(200, 205, 215))
        # progress bar
        draw.rectangle([x + 4, 15, x + THUMB - 6, 21], outline=DIM)
        filled = int((THUMB - 10) * max(0.0, min(1.0, p)))
        if filled > 0:
            draw.rectangle([x + 4, 15, x + 4 + filled, 21], fill=GREEN if s else DIM)
        draw.text((x + THUMB - 34, 1), "succ" if s else "", fill=GREEN)
    return np.asarray(img)


def build_sheet(rows, cols, max_frames, cutoff_file, progress_pred_type):
    import cv2

    strips = []
    for row in rows:
        frames = read_clip(row, max_frames)
        if frames is None:
            print(f"  !! missing frames for {row['id']}")
            continue
        n = len(frames)
        progress, success, cutoff = compute_labels(row, n, cutoff_file, progress_pred_type)

        shown = min(cols, n)
        idx = [int(i * (n - 1) / max(1, shown - 1)) for i in range(shown)]
        width = cols * THUMB + (cols - 1) * PAD

        strip = np.full((THUMB, width, 3), 20, dtype=np.uint8)
        for slot, frame_idx in enumerate(idx):
            thumb = cv2.resize(frames[frame_idx], (THUMB, THUMB), interpolation=cv2.INTER_AREA)
            border = GREEN if success[frame_idx] else RED
            thumb = cv2.copyMakeBorder(
                thumb[3:-3, 3:-3], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border
            )
            x = slot * (THUMB + PAD)
            strip[:, x : x + THUMB] = thumb

        ps = row.get("partial_success")
        caption = _text_strip(
            width,
            CAPTION_H,
            f"{row['id']}  |  {row['task']}",
            f"{row['quality_label']} ps={ps if ps is None else round(float(ps), 2)} "
            f"cutoff={cutoff} n={n}",
        )
        labels = _label_strip(width, shown, [progress[i] for i in idx], [success[i] for i in idx])
        strips.append(np.vstack([caption, strip, labels]))
        print(f"  {row['id']:<34s} {row['quality_label']:<11s} {row['task']}")

    if not strips:
        raise SystemExit("nothing to render")
    width = max(s.shape[1] for s in strips)
    padded = [
        np.pad(s, ((0, PAD), (0, width - s.shape[1]), (0, 0)), constant_values=20) for s in strips
    ]
    return np.vstack(padded)


# --------------------------------------------------------------------------
# HTML gallery
# --------------------------------------------------------------------------

_HTML_HEAD = """<!doctype html>
<meta charset="utf-8"><title>BEHAVIOR-1K clips</title>
<style>
 body{background:#14161a;color:#e7e9ee;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .meta{color:#93a;color:#8b93a7;margin-bottom:20px}
 .card{background:#1c1f26;border:1px solid #2a2f3a;border-radius:10px;padding:14px;margin-bottom:16px;
       display:grid;grid-template-columns:320px 1fr;gap:16px}
 video{width:320px;border-radius:6px;background:#000}
 .task{font-weight:600;font-size:15px;margin-bottom:2px}
 .id{color:#8b93a7;font-family:ui-monospace,Menlo,monospace;font-size:12px;margin-bottom:10px}
 .tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;margin-right:6px}
 .ok{background:#1d3a2a;color:#7fd6a0}.bad{background:#3a1f1f;color:#e08a8a}
 table{border-collapse:collapse;margin-top:10px;font-size:12px}
 td,th{padding:2px 8px;text-align:right;border-bottom:1px solid #262b35}
 th{color:#8b93a7;font-weight:500}
 .s1{color:#7fd6a0}.s0{color:#8b93a7}
 .bar{display:inline-block;height:7px;background:#3b4351;border-radius:3px;vertical-align:middle}
 .bar>i{display:block;height:100%;background:#7fd6a0;border-radius:3px}
</style>
"""


def build_html(rows, max_frames, cutoff_file, progress_pred_type, source_label: str) -> str:
    parts = [
        _HTML_HEAD,
        f"<h1>BEHAVIOR-1K converted clips</h1>",
        f'<div class="meta">{html.escape(source_label)} &middot; {len(rows)} clips &middot; '
        f"labels from compute_progress_from_segment / compute_success_labels</div>",
    ]

    for row in rows:
        frames = read_clip(row, max_frames)
        if frames is None:
            continue
        n = len(frames)
        progress, success, cutoff = compute_labels(row, n, cutoff_file, progress_pred_type)

        if "video" in row and os.path.exists(row["video"]):
            with open(row["video"], "rb") as handle:
                b64 = base64.b64encode(handle.read()).decode()
            media = f'<video src="data:video/mp4;base64,{b64}" controls loop muted></video>'
        else:
            media = _frames_to_gif_tag(frames)

        ps = row.get("partial_success")
        quality_class = "ok" if row["quality_label"] == "successful" else "bad"
        cells = "".join(
            f'<td class="{"s1" if success[i] else "s0"}">{progress[i]:.2f}</td>'
            for i in range(n)
        )
        succ_cells = "".join(
            f'<td class="{"s1" if success[i] else "s0"}">{int(success[i])}</td>' for i in range(n)
        )
        parts.append(
            f"""<div class="card">
 <div>{media}</div>
 <div>
  <div class="task">{html.escape(row['task'])}</div>
  <div class="id">{html.escape(row['id'])}</div>
  <span class="tag {quality_class}">{html.escape(row['quality_label'])}</span>
  <span class="tag">partial_success = {ps if ps is None else round(float(ps), 2)}</span>
  <span class="tag">cutoff = {cutoff}</span>
  <span class="tag">{n} frames</span>
  <table>
   <tr><th>frame</th>{''.join(f'<td>{i}</td>' for i in range(n))}</tr>
   <tr><th>progress</th>{cells}</tr>
   <tr><th>success</th>{succ_cells}</tr>
  </table>
 </div>
</div>"""
        )
        print(f"  {row['id']:<34s} {row['quality_label']:<11s} {row['task']}")

    return "\n".join(parts)


def _frames_to_gif_tag(frames: np.ndarray) -> str:
    """Embed npz frames as an animated GIF (no mp4 on disk in cache mode)."""
    import io

    from PIL import Image

    images = [Image.fromarray(f).resize((320, 320)) for f in frames]
    buffer = io.BytesIO()
    images[0].save(
        buffer, format="GIF", save_all=True, append_images=images[1:], duration=120, loop=0
    )
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/gif;base64,{b64}" width="320">'


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="converted HF dataset dir, or a preprocess cache dir")
    ap.add_argument("-n", "--num", type=int, default=12, help="clips to render")
    ap.add_argument("-o", "--out", default=None, help="PNG contact sheet path")
    ap.add_argument("--html", default=None, help="self-contained HTML gallery path")
    ap.add_argument("--cols", type=int, default=8, help="thumbnails per row (PNG mode)")
    ap.add_argument("--max-frames", type=int, default=16, help="frames read per clip")
    ap.add_argument("--quality", default=None, help="filter by quality_label, e.g. failure")
    ap.add_argument("--task", default=None, help="substring filter on the instruction")
    ap.add_argument("--ids", nargs="*", default=None, help="explicit trajectory ids")
    ap.add_argument(
        "--progress-pred-type",
        default="absolute_first_frame",
        help="must match data.progress_pred_type in the training config",
    )
    ap.add_argument("--cutoff-file", default="robometer/data/dataset_success_cutoff.txt")
    args = ap.parse_args()

    if not args.out and not args.html:
        args.out = "b1k_visualization.png"

    rows, kind = load_rows(args.path)
    print(f"loaded {len(rows)} rows from {args.path} ({kind})")

    if args.quality:
        rows = [r for r in rows if r["quality_label"] == args.quality]
    if args.task:
        needle = args.task.lower()
        rows = [r for r in rows if needle in r["task"].lower()]
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["id"] in wanted]
    if not rows:
        raise SystemExit("no rows matched the filters")

    # Spread the sample across instructions instead of taking the first N, which
    # would all be the same skill.
    by_task: dict[str, list] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    picked: list[dict] = []
    while len(picked) < args.num and any(by_task.values()):
        for task in list(by_task):
            if by_task[task]:
                picked.append(by_task[task].pop(0))
                if len(picked) >= args.num:
                    break
    print(f"rendering {len(picked)} clips")

    if args.out:
        import cv2

        sheet = build_sheet(picked, args.cols, args.max_frames, args.cutoff_file, args.progress_pred_type)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")
        print("green border / green bar = success label 1 for that frame")

    if args.html:
        page = build_html(
            picked, args.max_frames, args.cutoff_file, args.progress_pred_type, f"{args.path} ({kind})"
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.html)) or ".", exist_ok=True)
        with open(args.html, "w") as handle:
            handle.write(page)
        size_mb = os.path.getsize(args.html) / 1e6
        print(f"wrote {args.html}  ({size_mb:.1f} MB, self-contained)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
