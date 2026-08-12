#!/usr/bin/env python3
"""BEHAVIOR-1K 2026 challenge demos -> Robometer (RBM) trajectories.

One **skill segment** becomes one trajectory, not one episode. The annotations
split each episode into skills with exact frame boundaries, so a segment's last
frame is the moment that subtask completes -- which makes ``progress == 1.0``
mean "this subtask is done", the signal the subtask FSM transitions on.

    episode 0 "turning_on_radio", 1,956 frames
      skill 0  [   0,  265)  move to        -> "go to the radio"
      skill 1  [ 265, 1162)  pick up from   -> "pick up the radio from the coffee table"
      skill 2  [1162, 1434)  press          -> "press the radio"
      skill 3  [1434, 1776)  place on       -> "place the radio back on the coffee table"

Selecting a subset of the 100 tasks is first class: pass ``task_whitelist``
(names, ``task-0007`` ids, or indices) and/or ``max_episodes_per_task``. The
dataset is large enough that converting all of it is rarely what you want, and
partial downloads are the normal state -- episodes whose video or annotation is
missing are skipped and counted rather than raising.

See docs/b1k/02-converter-spec.md.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from dataset_upload.dataset_loaders.b1k_skill_templates import render_instruction
from dataset_upload.dataset_loaders.b1k_video import B1KSegmentFrameLoader

DEFAULT_VIDEO_KEY = "observation.rgb.zed_link_camera_0"
DEFAULT_DATA_SOURCE = "b1k_skill"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass
class EpisodeRef:
    """One row of meta/episodes, reduced to what the converter needs."""

    episode_index: int
    task_index: int
    task_name: str
    length: int
    demo_index_within_task: int
    annotation_path: str
    video_path: str
    video_offset: int  # first frame of this episode inside the aggregated mp4


@dataclass
class ConversionStats:
    episodes_seen: int = 0
    episodes_converted: int = 0
    episodes_skipped_no_annotation: int = 0
    episodes_skipped_no_video: int = 0
    segments_total: int = 0
    segments_kept: int = 0
    segments_dropped: Counter = field(default_factory=Counter)
    segments_flagged: Counter = field(default_factory=Counter)  # kept, but noteworthy
    trajectories_emitted: Counter = field(default_factory=Counter)
    tasks: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        out = asdict(self)
        for key in ("segments_dropped", "segments_flagged", "trajectories_emitted", "tasks"):
            out[key] = dict(sorted(self.__dict__[key].items()))
        out["segments_accounted_for"] = self.segments_kept + sum(self.segments_dropped.values())
        return out

    def check_conservation(self) -> None:
        """Every segment must be either kept or dropped for a named reason."""
        accounted = self.segments_kept + sum(self.segments_dropped.values())
        if accounted != self.segments_total:
            raise AssertionError(
                f"segment accounting leak: {self.segments_total} seen but "
                f"{self.segments_kept} kept + {sum(self.segments_dropped.values())} dropped "
                f"= {accounted}"
            )


# --------------------------------------------------------------------------
# meta reading
# --------------------------------------------------------------------------


def _read_parquet_rows(path: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _load_task_names(root: str) -> dict[int, str]:
    tasks_path = os.path.join(root, "meta", "tasks.parquet")
    if not os.path.exists(tasks_path):
        return {}
    return {row["task_index"]: row["task"] for row in _read_parquet_rows(tasks_path)}


def _episode_meta_files(root: str) -> list[str]:
    meta_dir = os.path.join(root, "meta", "episodes")
    found = []
    for dirpath, _dirnames, filenames in os.walk(meta_dir):
        for name in sorted(filenames):
            if name.endswith(".parquet"):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def _normalize_whitelist(
    whitelist: Optional[Iterable], task_names: dict[int, str]
) -> Optional[set[int]]:
    """Accept task names, ``task-0007`` folder ids, or plain indices."""
    if whitelist is None:
        return None
    by_name = {name: idx for idx, name in task_names.items()}
    selected: set[int] = set()
    for item in whitelist:
        if isinstance(item, int):
            selected.add(item)
            continue
        text = str(item).strip()
        if text in by_name:
            selected.add(by_name[text])
            continue
        match = re.fullmatch(r"task-(\d+)", text)
        if match:
            selected.add(int(match.group(1)))
            continue
        if text.isdigit():
            selected.add(int(text))
            continue
        # tolerate "turning on radio" for "turning_on_radio"
        underscored = text.replace(" ", "_")
        if underscored in by_name:
            selected.add(by_name[underscored])
            continue
        raise ValueError(f"unknown task in task_whitelist: {item!r}")
    return selected


def load_episode_index(
    root: str,
    video_key: str = DEFAULT_VIDEO_KEY,
    fps: float = 30.0,
) -> list[EpisodeRef]:
    """Read meta/episodes into EpisodeRef records (no annotation IO yet)."""
    task_names = _load_task_names(root)
    files = _episode_meta_files(root)
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root}/meta/episodes")

    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_col = f"videos/{video_key}/from_timestamp"

    episodes: list[EpisodeRef] = []
    for path in files:
        for row in _read_parquet_rows(path):
            if chunk_col not in row:
                raise KeyError(
                    f"video key {video_key!r} not in episode metadata; "
                    f"available: {[k for k in row if k.startswith('videos/')][:4]}"
                )
            task_index = int(row["task_index"])
            tasks = row.get("tasks")
            if isinstance(tasks, (list, tuple)) and tasks:
                task_name = str(tasks[0])
            else:
                task_name = task_names.get(task_index, f"task-{task_index:04d}")

            rel_video = os.path.join(
                "videos", video_key, f"chunk-{int(row[chunk_col]):03d}", f"file-{int(row[file_col]):03d}.mp4"
            )
            episodes.append(
                EpisodeRef(
                    episode_index=int(row["episode_index"]),
                    task_index=task_index,
                    task_name=task_name,
                    length=int(row["length"]),
                    demo_index_within_task=int(row.get("demo_index_within_task", 0)),
                    annotation_path=os.path.join(root, str(row["annotation_path"])),
                    video_path=os.path.join(root, rel_video),
                    video_offset=int(round(float(row[from_col]) * fps)),
                )
            )
    return episodes


def summarize_availability(
    root: str, video_key: str = DEFAULT_VIDEO_KEY
) -> list[dict]:
    """Per-task availability report -- what can actually be converted right now.

    Used to pick ``task_whitelist`` against a partial download.
    """
    episodes = load_episode_index(root, video_key)
    per_task: dict[int, dict] = {}
    video_exists: dict[str, bool] = {}
    for ep in episodes:
        entry = per_task.setdefault(
            ep.task_index,
            {"task_index": ep.task_index, "task_name": ep.task_name, "episodes": 0, "with_video": 0, "with_annotation": 0},
        )
        entry["episodes"] += 1
        if ep.video_path not in video_exists:
            video_exists[ep.video_path] = os.path.exists(ep.video_path)
        if video_exists[ep.video_path]:
            entry["with_video"] += 1
        if os.path.exists(ep.annotation_path):
            entry["with_annotation"] += 1
    for entry in per_task.values():
        entry["ready"] = min(entry["with_video"], entry["with_annotation"])
    return [per_task[k] for k in sorted(per_task)]


# --------------------------------------------------------------------------
# segment extraction
# --------------------------------------------------------------------------


def _frame_ranges(frame_duration) -> list[tuple[int, int]]:
    """Normalize frame_duration into a list of [start, end) pairs.

    724 segments carry a split range (``[[2795, 3007], [3209, 3638]]``) where the
    skill is interrupted and resumed.
    """
    if not isinstance(frame_duration, list) or not frame_duration:
        return []
    if isinstance(frame_duration[0], (int, float)):
        if len(frame_duration) != 2:
            return []
        return [(int(frame_duration[0]), int(frame_duration[1]))]
    ranges = []
    for item in frame_duration:
        if isinstance(item, list) and len(item) == 2:
            ranges.append((int(item[0]), int(item[1])))
    return ranges


def _iter_annotation_segments(annotation: dict, level: str) -> Iterable[tuple[int, dict]]:
    key = "skill_annotation" if level == "skill" else "primitive_annotation"
    idx_key = "skill_idx" if level == "skill" else "primitive_idx"
    desc_key = "skill_description" if level == "skill" else "primitive_description"
    for i, entry in enumerate(annotation.get(key) or []):
        yield int(entry.get(idx_key, i)), {
            "description": (entry.get(desc_key) or [None])[0],
            "object_id": entry.get("object_id") or [],
            "memory_prefix": entry.get("memory_prefix") or [],
            "spatial_prefix": entry.get("spatial_prefix") or [],
            "skill_type": (entry.get("skill_type") or [None])[0],
            "frame_duration": entry.get("frame_duration"),
        }


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def load_b1k_dataset(
    dataset_path: str,
    video_key: str = DEFAULT_VIDEO_KEY,
    annotation_level: str = "skill",
    max_frames: int = 64,
    min_segment_frames: int = 15,
    max_segment_frames: int = 3000,
    split: str = "train",
    val_demo_ratio: float = 0.1,
    task_whitelist: Optional[Iterable] = None,
    max_episodes_per_task: Optional[int] = None,
    max_segments_per_task: Optional[int] = None,
    truncated_negative_ratio: float = 0.0,
    truncated_alpha_range: tuple[float, float] = (0.30, 0.70),
    skip_split_range: bool = False,
    data_source: str = DEFAULT_DATA_SOURCE,
    include_task_context: bool = False,
    fps: float = 30.0,
    seed: int = 42,
    report_path: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Build RBM trajectory dicts from BEHAVIOR-1K skill annotations.

    Args:
        dataset_path: dataset root (the directory holding meta/, videos/, annotations/).
        video_key: which camera stream to cut clips from.
        annotation_level: ``skill`` (default) or ``primitive``.
        max_frames: frames kept per clip; the loader subsamples before decoding.
        min_segment_frames / max_segment_frames: segment length bounds, in frames.
        split: ``train``, ``val`` or ``all``. Episodes are split per task so both
            sides cover every task, and never split within an episode.
        val_demo_ratio: fraction of each task's demos routed to ``val``.
        task_whitelist: task names, ``task-0007`` ids or indices. None = all.
        max_episodes_per_task / max_segments_per_task: subsetting caps.
        truncated_negative_ratio: fraction of positives that also get a truncated
            failure clip (S4; 0.0 keeps this pass positives-only).
        data_source: written to every row; the key used by the cutoff file and
            DATA_SOURCE_CATEGORY, so keep it stable across train/val subsets.

    Returns:
        ``{task_key: [traj_dict, ...]}`` as ``flatten_task_data`` expects.
    """
    if annotation_level not in ("skill", "primitive"):
        raise ValueError(f"annotation_level must be 'skill' or 'primitive', got {annotation_level!r}")
    if split not in ("train", "val", "all"):
        raise ValueError(f"split must be 'train', 'val' or 'all', got {split!r}")

    rng = random.Random(seed)
    stats = ConversionStats()
    task_names = _load_task_names(dataset_path)
    whitelist = _normalize_whitelist(task_whitelist, task_names)

    episodes = load_episode_index(dataset_path, video_key, fps)
    stats.episodes_seen = len(episodes)

    # ---- task / split / cap selection ------------------------------------
    val_stride = max(2, int(round(1.0 / val_demo_ratio))) if val_demo_ratio > 0 else 0

    def in_split(ep: EpisodeRef) -> bool:
        if split == "all" or val_stride == 0:
            return True
        is_val = ep.demo_index_within_task % val_stride == 0
        return is_val if split == "val" else not is_val

    selected: list[EpisodeRef] = []
    per_task_count: Counter = Counter()
    for ep in sorted(episodes, key=lambda e: (e.task_index, e.demo_index_within_task)):
        if whitelist is not None and ep.task_index not in whitelist:
            continue
        if not in_split(ep):
            continue
        if max_episodes_per_task is not None and per_task_count[ep.task_index] >= max_episodes_per_task:
            continue
        per_task_count[ep.task_index] += 1
        selected.append(ep)

    if verbose:
        print(f"BEHAVIOR-1K: {len(episodes)} episodes in metadata, {len(selected)} selected "
              f"(split={split}, tasks={len(per_task_count)})")

    # Group by video file so the decoder keeps seeking forward within a file.
    selected.sort(key=lambda e: (e.video_path, e.video_offset))

    video_exists: dict[str, bool] = {}
    task_data: dict[str, list[dict]] = defaultdict(list)
    segments_per_task: Counter = Counter()

    for ep in selected:
        if ep.video_path not in video_exists:
            video_exists[ep.video_path] = os.path.exists(ep.video_path)
        if not video_exists[ep.video_path]:
            stats.episodes_skipped_no_video += 1
            continue
        if not os.path.exists(ep.annotation_path):
            stats.episodes_skipped_no_annotation += 1
            continue

        with open(ep.annotation_path) as handle:
            annotation = json.load(handle)

        emitted_for_episode = 0
        for skill_idx, seg in _iter_annotation_segments(annotation, annotation_level):
            ranges = _frame_ranges(seg["frame_duration"])
            stats.segments_total += 1

            if not ranges:
                stats.segments_dropped["bad_frame_duration"] += 1
                continue
            is_split_range = len(ranges) > 1
            if is_split_range:
                if skip_split_range:
                    stats.segments_dropped["split_range"] += 1
                    continue
                # Kept, using the first range only -- not a drop.
                stats.segments_flagged["split_range_truncated_to_first"] += 1
            start, end = ranges[0]

            if end - start <= 0:
                stats.segments_dropped["neg_len"] += 1
                continue
            if end > ep.length:
                stats.segments_dropped["oob"] += 1
                continue
            if end - start < min_segment_frames:
                stats.segments_dropped["too_short"] += 1
                continue
            if end - start > max_segment_frames:
                stats.segments_dropped["too_long"] += 1
                continue

            instruction = render_instruction(
                seg["description"],
                seg["object_id"],
                seg["memory_prefix"],
                seg["spatial_prefix"],
                task_name=ep.task_name,
                include_task_context=include_task_context,
            )
            if instruction is None:
                stats.segments_dropped["no_template"] += 1
                continue

            if max_segments_per_task is not None and segments_per_task[ep.task_index] >= max_segments_per_task:
                stats.segments_dropped["task_cap"] += 1
                continue

            task_key = f"{ep.task_name}/{seg['description']}"
            base_meta = {
                "episode_index": ep.episode_index,
                "task_index": ep.task_index,
                "task_name": ep.task_name,
                "skill_idx": skill_idx,
                "skill_description": seg["description"],
                "skill_type": seg["skill_type"],
                "segment_frames": [start, end],
                "episode_length": ep.length,
                "split_range": is_split_range,
            }

            task_data[task_key].append(
                _make_traj(
                    ep, skill_idx, start, end, instruction, "positive",
                    quality_label="successful", partial_success=1.0,
                    max_frames=max_frames, fps=fps, data_source=data_source, metadata=base_meta,
                )
            )
            stats.trajectories_emitted["positive"] += 1
            stats.segments_kept += 1
            segments_per_task[ep.task_index] += 1
            emitted_for_episode += 1

            # ---- truncated negative (S4) ---------------------------------
            if truncated_negative_ratio > 0 and rng.random() < truncated_negative_ratio:
                alpha = rng.uniform(*truncated_alpha_range)
                cut = start + int(round(alpha * (end - start)))
                if cut - start >= min_segment_frames:
                    meta = dict(base_meta, variant="truncated", alpha=alpha, segment_frames=[start, cut])
                    task_data[task_key].append(
                        _make_traj(
                            ep, skill_idx, start, cut, instruction, "truncated",
                            quality_label="failure", partial_success=alpha,
                            max_frames=max_frames, fps=fps, data_source=data_source, metadata=meta,
                        )
                    )
                    stats.trajectories_emitted["truncated"] += 1
                else:
                    stats.segments_flagged["truncated_too_short"] += 1

        if emitted_for_episode:
            stats.episodes_converted += 1
            stats.tasks[ep.task_name] += 1

    stats.check_conservation()
    total = sum(len(v) for v in task_data.values())
    if verbose:
        print(f"  emitted {total} trajectories over {len(task_data)} task/skill groups")
        print(f"  positives={stats.trajectories_emitted['positive']} "
              f"truncated={stats.trajectories_emitted['truncated']}")
        if stats.segments_dropped:
            print(f"  dropped: {dict(sorted(stats.segments_dropped.items()))}")
        if stats.episodes_skipped_no_video:
            print(f"  ⚠️  {stats.episodes_skipped_no_video} episodes skipped: video file not downloaded")
        if stats.episodes_skipped_no_annotation:
            print(f"  ⚠️  {stats.episodes_skipped_no_annotation} episodes skipped: annotation missing")

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        report = stats.as_dict()
        report.update(
            {
                "split": split,
                "video_key": video_key,
                "annotation_level": annotation_level,
                "data_source": data_source,
                "trajectories_total": total,
                "task_groups": len(task_data),
                "unique_instructions": len({t["task"] for v in task_data.values() for t in v}),
            }
        )
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=1)
        if verbose:
            print(f"  report -> {report_path}")

    if not task_data:
        raise RuntimeError(
            "no trajectories produced. Check that meta/episodes covers the requested tasks and "
            "that the videos for those episodes are downloaded (see summarize_availability)."
        )
    return dict(task_data)


def _make_traj(
    ep: EpisodeRef,
    skill_idx: int,
    start: int,
    end: int,
    instruction: str,
    variant: str,
    *,
    quality_label: str,
    partial_success: float,
    max_frames: int,
    fps: float,
    data_source: str,
    metadata: dict,
) -> dict[str, Any]:
    """Assemble one trajectory dict in the shape create_hf_trajectory expects."""
    return {
        "id": f"b1k-{ep.episode_index:06d}-{skill_idx:03d}-{variant}",
        "task": instruction,
        "frames": B1KSegmentFrameLoader(
            video_path=ep.video_path,
            global_start=ep.video_offset + start,
            global_end=ep.video_offset + end,
            max_frames=max_frames,
            fps=fps,
        ),
        "is_robot": True,
        "quality_label": quality_label,
        "partial_success": partial_success,
        "data_source": data_source,
        "preference_group_id": None,
        "preference_rank": None,
        "metadata": dict(metadata, variant=variant),
    }
