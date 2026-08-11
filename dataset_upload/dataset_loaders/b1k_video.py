#!/usr/bin/env python3
"""Frame extraction from BEHAVIOR-1K's aggregated LeRobot v3 videos.

One mp4 holds many episodes back to back (~46 min, 82k frames), so every read
is a seek into the middle of a long file. Three backends are tried in order:

    decord  -- what robometer/data/scripts/preprocess_datasets.py uses; fastest,
               but has no macOS arm64 wheel
    av      -- PyAV; seeks to the nearest keyframe and decodes forward
    cv2     -- OpenCV; last resort, per-frame POS_FRAMES seeking

The backend is resolved lazily inside the worker so that the loader objects stay
picklable across the ``spawn`` pool that generate_hf_dataset uses.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_BACKEND: Optional[str] = None


def resolve_backend() -> str:
    """Return the name of the first available decoder backend."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    for name, module in (("decord", "decord"), ("av", "av"), ("cv2", "cv2")):
        try:
            __import__(module)
        except ImportError:
            continue
        _BACKEND = name
        return name
    raise ImportError("no video backend available; install one of: decord, av, opencv-python")


def sample_indices(start: int, end: int, max_frames: int) -> list[int]:
    """Uniformly pick at most ``max_frames`` indices from ``[start, end)``."""
    span = end - start
    if span <= 0:
        return []
    if span <= max_frames:
        return list(range(start, end))
    return [start + (i * span) // max_frames for i in range(max_frames)]


def _read_decord(path: str, indices: Sequence[int]) -> np.ndarray:
    import decord  # type: ignore

    reader = decord.VideoReader(path, num_threads=1)
    total = len(reader)
    clipped = [min(i, total - 1) for i in indices]
    frames = reader.get_batch(clipped).asnumpy()
    del reader
    return frames


def _read_av(path: str, indices: Sequence[int], fps: float) -> np.ndarray:
    import av  # type: ignore

    wanted = sorted(set(indices))
    out: dict[int, np.ndarray] = {}

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)

        # Seek to the keyframe at or before the first wanted frame.
        target_pts = int(wanted[0] / fps / time_base)
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        remaining = set(wanted)
        last_wanted = wanted[-1]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            idx = int(round(frame.pts * time_base * fps))
            if idx in remaining:
                out[idx] = frame.to_ndarray(format="rgb24")
                remaining.discard(idx)
            if idx >= last_wanted and not remaining:
                break
            if idx > last_wanted:
                break

    if not out:
        raise RuntimeError(f"decoded no frames from {path} for indices {wanted[:4]}...")

    # Nearest-available substitution keeps clip length stable when a pts lands
    # a frame off; ordering follows the requested indices.
    available = sorted(out)
    frames = []
    for i in indices:
        key = i if i in out else min(available, key=lambda a: abs(a - i))
        frames.append(out[key])
    return np.stack(frames)


def _read_cv2(path: str, indices: Sequence[int]) -> np.ndarray:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return np.stack(frames)


def read_frames(path: str, indices: Sequence[int], fps: float = 30.0) -> np.ndarray:
    """Read the given absolute frame indices from ``path`` as (T, H, W, 3) uint8."""
    if not indices:
        raise ValueError("no indices requested")

    backend = resolve_backend()
    if backend == "decord":
        frames = _read_decord(path, indices)
    elif backend == "av":
        frames = _read_av(path, indices, fps)
    else:
        frames = _read_cv2(path, indices)

    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)
    return frames


class B1KSegmentFrameLoader:
    """Picklable on-demand loader for one skill segment.

    Holds only primitives so it survives the ``spawn`` pool in
    ``generate_hf_dataset.convert_dataset_to_hf_format``. Subsamples to
    ``max_frames`` *before* returning -- the longest segment in the dataset is
    8,946 frames, which at 720x720x3 would be ~13 GB decoded in full.
    """

    __slots__ = ("video_path", "global_start", "global_end", "max_frames", "fps")

    def __init__(
        self,
        video_path: str,
        global_start: int,
        global_end: int,
        max_frames: int = 64,
        fps: float = 30.0,
    ):
        self.video_path = str(video_path)
        self.global_start = int(global_start)
        self.global_end = int(global_end)
        self.max_frames = int(max_frames)
        self.fps = float(fps)

    def indices(self) -> list[int]:
        return sample_indices(self.global_start, self.global_end, self.max_frames)

    def __call__(self) -> Optional[np.ndarray]:
        idx = self.indices()
        if not idx:
            return None
        try:
            return read_frames(self.video_path, idx, self.fps)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
            print(f"⚠️  frame read failed for {self.video_path} [{self.global_start}:{self.global_end}]: {exc}")
            return None

    def __repr__(self) -> str:
        return (
            f"B1KSegmentFrameLoader({self.video_path!r}, "
            f"{self.global_start}, {self.global_end}, max_frames={self.max_frames})"
        )
