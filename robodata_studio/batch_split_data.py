#!/usr/bin/env python3
"""Batch split one or more LeRobot datasets into atomic-action datasets.

This script is meant for collection folders such as:

  solid/
    2026-06-01_lerobot/
    2026-06-02_lerobot/

Each child dataset is segmented with the same ordered subtask list. The output
is grouped by atomic action:

  solid/
    solid_pick_funnel_to_reactor/
    solid_press_tare_button/
    ...

Segmentation uses the collection convention that the robot returns to the
episode's initial home pose after every atomic action.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STUDIO_DIR = Path(__file__).resolve().parent
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

from split_lerobot_atomic_actions import _detect_video_keys  # noqa: E402
from split_lerobot_atomic_actions import _episode_path  # noqa: E402
from split_lerobot_atomic_actions import _episode_stats  # noqa: E402
from split_lerobot_atomic_actions import _import_pyarrow  # noqa: E402
from split_lerobot_atomic_actions import _load_json  # noqa: E402
from split_lerobot_atomic_actions import _load_jsonl  # noqa: E402
from split_lerobot_atomic_actions import _output_episode_path  # noqa: E402
from split_lerobot_atomic_actions import _output_video_path  # noqa: E402
from split_lerobot_atomic_actions import _set_column  # noqa: E402
from split_lerobot_atomic_actions import _slug  # noqa: E402
from split_lerobot_atomic_actions import _verify_preserved_data  # noqa: E402
from split_lerobot_atomic_actions import _video_path  # noqa: E402
from split_lerobot_atomic_actions import _write_csv  # noqa: E402
from split_lerobot_atomic_actions import _write_json  # noqa: E402
from split_lerobot_atomic_actions import _write_jsonl  # noqa: E402

def _resolve_ffmpeg(ffmpeg: str) -> str:
    env_ffmpeg = os.environ.get("ROBODATA_STUDIO_FFMPEG")
    if env_ffmpeg:
        return env_ffmpeg

    requested = str(ffmpeg or "ffmpeg")
    if requested != "ffmpeg":
        return requested

    try:
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled
    except Exception:
        pass

    return requested


TASK_PRESETS: dict[str, list[tuple[str, str]]] = {
    "liquid": [
        ("beaker_to_graduated_cylinder", "pour solution from the beaker into the graduated cylinder"),
        ("graduated_cylinder_to_reactor", "pour solution from the graduated cylinder into the reactor"),
    ],
    "solid": [
        ("pick_funnel_to_reactor", "pick up the funnel and place it on the reactor"),
        ("pick_weighing_boat_to_balance", "pick up the weighing boat and place it on the balance"),
        ("press_tare_button", "press the tare button on the balance"),
        ("scoop_solid_to_weighing_boat", "scoop solid into the weighing boat"),
        ("pour_solid_to_reactor", "pour the solid into the reactor"),
    ],
    "mix_distill": [
        ("return_funnel_to_rack", "pick up the funnel and put it back on the funnel rack"),
        ("place_distillation_rack", "place the distillation rack"),
        ("turn_reactor_knob", "turn the reactor knob"),
    ],
}
TASK_PRESETS["mixed_distillation"] = TASK_PRESETS["mix_distill"]
TASK_PRESETS["distill"] = TASK_PRESETS["mix_distill"]


@dataclass(frozen=True)
class SubtaskSpec:
    label: str
    prompt: str


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    source_id: str
    info: dict[str, Any]
    episodes: list[dict[str, Any]]


@dataclass(frozen=True)
class SegmentPlan:
    label: str
    task: str
    source_root: Path
    source_id: str
    source_episode: dict[str, Any]
    source_episode_index: int
    start: int
    end: int
    raw_start: int
    raw_end: int
    trimmed_start: int
    trimmed_end: int
    boundary_confidence: float


@dataclass(frozen=True)
class VideoJob:
    source_video: Path
    output_video: Path
    start_frame: int
    frame_count: int


def _parse_specs(text: str, preset: str) -> list[SubtaskSpec]:
    specs: list[SubtaskSpec] = []
    if text.strip():
        for index, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            if "|" in line:
                label, prompt = line.split("|", 1)
            elif "\t" in line:
                label, prompt = line.split("\t", 1)
            elif ":" in line:
                label, prompt = line.split(":", 1)
            else:
                prompt = line
                label = line
            label = _slug(label.strip().lower()) or f"atomic_{index + 1:02d}"
            prompt = prompt.strip() or label.replace("_", " ")
            specs.append(SubtaskSpec(label=label, prompt=prompt))
        return specs

    preset_key = preset.strip().lower()
    if preset_key not in TASK_PRESETS:
        raise ValueError(f"Unknown task preset {preset!r}. Available presets: {', '.join(sorted(TASK_PRESETS))}")
    return [SubtaskSpec(label=label, prompt=prompt) for label, prompt in TASK_PRESETS[preset_key]]


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "meta" / "episodes.jsonl").is_file()


def _discover_sources(source_roots: list[Path], planned_output_names: set[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for source in source_roots:
        root = source.expanduser().resolve()
        candidates = [root] if _is_lerobot_dataset(root) else sorted(child for child in root.iterdir() if child.is_dir())
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or candidate.name in planned_output_names:
                continue
            if _is_lerobot_dataset(resolved):
                discovered.append(resolved)
                seen.add(resolved)
    if not discovered:
        raise FileNotFoundError("No LeRobot datasets found. Pass a dataset root or a parent directory containing datasets.")
    return discovered


#
def _parse_dim_indices(text: str, width: int) -> set[int]:
    indices: set[int] = set()
    for raw_item in str(text or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            index = width + index
        if 0 <= index < width:
            indices.add(index)
    return indices


def _feature_dim_names(info: dict[str, Any], key: str) -> list[str]:
    names = ((info.get("features") or {}).get(key) or {}).get("names")
    if isinstance(names, list) and names and isinstance(names[0], list):
        return [str(item) for item in names[0]]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def _arm_dims(names: list[str], width: int, arm: str) -> set[int]:
    arm = arm.strip().lower()
    if arm not in {"left", "right"}:
        return set()
    if names:
        prefix = f"{arm}_"
        return {index for index, name in enumerate(names[:width]) if str(name).lower().startswith(prefix)}
    if width == 14:
        return set(range(0, 7)) if arm == "left" else set(range(7, 14))
    midpoint = width // 2
    return set(range(0, midpoint)) if arm == "left" else set(range(midpoint, width))


def _ignored_dims(info: dict[str, Any], key: str, args: argparse.Namespace, width: int) -> set[int]:
    ignore = _parse_dim_indices(args.home_ignore_dims, width)
    names = _feature_dim_names(info, key)
    inactive_arm = str(getattr(args, "inactive_arm", "") or "").strip().lower()
    active_arm = str(getattr(args, "active_arm", "") or "").strip().lower()
    if inactive_arm in {"left", "right"}:
        ignore.update(_arm_dims(names, width, inactive_arm))
    if active_arm in {"left", "right"}:
        other = "right" if active_arm == "left" else "left"
        ignore.update(_arm_dims(names, width, other))
    return {dim for dim in ignore if 0 <= dim < width}


def _active_diffs(left: list[float], right: list[float], ignore_dims: set[int] | None = None) -> list[float]:
    width = min(len(left), len(right))
    ignored = ignore_dims or set()
    return [left[dim] - right[dim] for dim in range(width) if dim not in ignored]


def _active_values(values: list[float], ignore_dims: set[int] | None = None) -> list[float]:
    ignored = ignore_dims or set()
    return [value for dim, value in enumerate(values) if dim not in ignored]

def _row_vector(row: dict[str, Any], key: str) -> list[float]:
    value = row.get(key)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    try:
        return [float(item) for item in value]
    except Exception:
        return []


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _mean_pose(vectors: list[list[float]], count: int) -> list[float]:
    count = min(max(count, 1), len(vectors))
    width = len(vectors[0])
    return [sum(vector[dim] for vector in vectors[:count]) / count for dim in range(width)]


# change
def _home_ratio(state: list[float], home_pose: list[float], threshold: float, ignore_dims: set[int] | None = None,) -> float:
    width = min(len(state), len(home_pose))
    if width <= 0:
        return 0.0

    ignore_dims = ignore_dims or set()
    dims = [dim for dim in range(width) if dim not in ignore_dims]
    if not dims:
        return 0.0

    close = sum(1 for dim in dims if abs(state[dim] - home_pose[dim]) <= threshold)
    return close / len(dims)


def _state_velocity(states: list[list[float]], index: int, ignore_dims: set[int] | None = None) -> float:
    if index <= 0:
        return 0.0
    return _norm(_active_diffs(states[index], states[index - 1], ignore_dims))


def _segment_edge_mask(
    states: list[list[float]],
    actions: list[list[float]],
    home_pose: list[float],
    *,
    joint_threshold: float,
    min_home_ratio: float,
    state_velocity_threshold: float,
    action_idle_threshold: float,
    home_ignore_dims: set[int] | None = None,
) -> list[bool]:
    mask: list[bool] = []
    for index, state in enumerate(states):
        near_home = _home_ratio(state, home_pose, joint_threshold, home_ignore_dims) >= min_home_ratio
        state_idle = state_velocity_threshold <= 0 or _state_velocity(states, index, home_ignore_dims) <= state_velocity_threshold
        action_idle = (
            action_idle_threshold <= 0
            or index >= len(actions)
            or not actions[index]
            or _norm(_active_values(actions[index], home_ignore_dims)) <= action_idle_threshold
        )
        mask.append(near_home and state_idle and action_idle)
    return mask


def _edge_count(mask: list[bool], *, leading: bool) -> int:
    values = mask if leading else list(reversed(mask))
    count = 0
    for item in values:
        if not item:
            break
        count += 1
    return count


def _trim_segment_bounds(
    raw_start: int,
    raw_end: int,
    states: list[list[float]],
    actions: list[list[float]],
    home_pose: list[float],
    ignore_dims: set[int],
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    segment_states = states[raw_start:raw_end]
    segment_actions = actions[raw_start:raw_end]
    if not segment_states:
        return raw_start, raw_end, 0, 0
    
    mask = _segment_edge_mask(
        segment_states,
        segment_actions,
        home_pose,
        joint_threshold=args.joint_threshold,
        min_home_ratio=args.edge_home_ratio,
        state_velocity_threshold=args.edge_state_velocity_threshold,
        action_idle_threshold=args.edge_action_idle_threshold,
        home_ignore_dims=ignore_dims,
    )
    leading = _edge_count(mask, leading=True)
    trailing = _edge_count(mask, leading=False)
    trim_start = raw_start
    trim_end = raw_end
    if leading >= args.min_edge_home_frames:
        trim_start = min(raw_end, raw_start + max(0, leading - args.keep_edge_home_frames))
    if trailing >= args.min_edge_home_frames:
        trim_end = max(trim_start, raw_end - max(0, trailing - args.keep_edge_home_frames))
    if trim_end - trim_start < args.min_segment_frames:
        return raw_start, raw_end, 0, 0
    return trim_start, trim_end, trim_start - raw_start, raw_end - trim_end


# def _score_boundaries(
#     states: list[list[float]],
#     actions: list[list[float]],
#     specs: list[SubtaskSpec],
#     args: argparse.Namespace,
# ) -> tuple[list[int], list[float]]:
#     if not states:
#         return [], []
#     needed = len(specs) - 1
#     if needed <= 0:
#         return [], []
#     home_pose = _mean_pose(states, args.home_window)
#     home_ignore_dims = _parse_dim_indices(args.home_ignore_dims, len(home_pose))
#     scored: list[dict[str, float]] = []
#     for index, state in enumerate(states):
#         action_norm = _norm(actions[index]) if index < len(actions) and actions[index] else 0.0
#         velocity = _state_velocity(states, index)
#         ratio = _home_ratio(state, home_pose, args.joint_threshold, home_ignore_dims)
#         score = ratio - min(velocity, 1.0) * 0.35 - min(action_norm, 1.0) * 0.15
#         scored.append({"frame": float(index), "score": score, "home_ratio": ratio, "velocity": velocity})

#     margin = max(args.margin, args.min_gap)
#     cluster_peaks: list[dict[str, float]] = []
#     current_cluster: list[dict[str, float]] = []
#     for item in scored[margin : max(margin, len(scored) - margin)]:
#         if item["home_ratio"] < args.min_home_ratio:
#             if current_cluster:
#                 cluster_peaks.append(max(current_cluster, key=lambda candidate: candidate["score"]))
#                 current_cluster = []
#             continue
#         if current_cluster and int(item["frame"]) - int(current_cluster[-1]["frame"]) > args.min_gap:
#             cluster_peaks.append(max(current_cluster, key=lambda candidate: candidate["score"]))
#             current_cluster = []
#         current_cluster.append(item)
#     if current_cluster:
#         cluster_peaks.append(max(current_cluster, key=lambda candidate: candidate["score"]))

#     if len(cluster_peaks) == needed:
#         cluster_peaks.sort(key=lambda item: item["frame"])
#         return [int(item["frame"]) for item in cluster_peaks], [float(item["home_ratio"]) for item in cluster_peaks]

#     boundaries: list[int] = []
#     confidences: list[float] = []
#     for transition_index in range(1, len(specs)):
#         target = round(len(states) * transition_index / len(specs))
#         search_start = max(margin, target - args.search_radius)
#         search_end = min(len(states) - margin, target + args.search_radius)
#         window = [
#             item
#             for item in cluster_peaks
#             if search_start <= int(item["frame"]) <= search_end and int(item["frame"]) not in boundaries
#         ]
#         if not window:
#             window = [
#                 item
#                 for item in scored[search_start:search_end]
#                 if item["home_ratio"] >= args.fallback_home_ratio
#             ]
#         if not window:
#             continue
#         chosen = max(window, key=lambda item: item["score"])
#         boundaries.append(int(chosen["frame"]))
#         confidences.append(float(chosen["home_ratio"]))
#     return boundaries, confidences


def _score_boundaries(
    states: list[list[float]],
    actions: list[list[float]],
    specs: list[SubtaskSpec],
    ignore_dims: set[int],
    args: argparse.Namespace,
) -> tuple[list[int], list[float]]:
    if not states:
        return [], []

    needed = len(specs) - 1
    if needed <= 0:
        return [], []

    home_pose = _mean_pose(states, args.home_window)

    stride = max(1, int(args.boundary_sample_stride))
    margin = max(args.margin, args.min_gap)
    sample_start = margin
    tail_ignore = max(0, int(getattr(args, "tail_ignore_frames", 0) or 0))
    sample_end = max(sample_start, len(states) - max(margin, tail_ignore))

    if sample_end <= sample_start:
        return [], []

    samples: list[dict[str, float]] = []
    for index in range(sample_start, sample_end, stride):
        state = states[index]
        action_norm = _norm(_active_values(actions[index], ignore_dims)) if index < len(actions) and actions[index] else 0.0
        velocity = _state_velocity(states, index, ignore_dims)
        ratio = _home_ratio(state, home_pose, args.joint_threshold, ignore_dims)
        score = ratio - min(velocity, 1.0) * 0.35 - min(action_norm, 1.0) * 0.15

        samples.append(
            {
                "frame": float(index),
                "score": float(score),
                "home_ratio": float(ratio),
                "velocity": float(velocity),
                "action_norm": float(action_norm),
            }
        )

    intervals: list[dict[str, float]] = []
    current: list[dict[str, float]] = []

    def close_interval() -> None:
        nonlocal current

        if not current:
            return

        start = int(current[0]["frame"])
        end = int(current[-1]["frame"])

        scores = [item["score"] for item in current]
        ratios = [item["home_ratio"] for item in current]

        center = (start + end) // 2
        avg_score = sum(scores) / len(scores)
        peak_score = max(scores)
        confidence = max(ratios)

        intervals.append(
            {
                "start": float(start),
                "end": float(end),
                "center": float(center),
                "score": float(avg_score),
                "peak_score": float(peak_score),
                "home_ratio": float(confidence),
                "sample_count": float(len(current)),
                "duration": float(end - start + 1),
            }
        )
        current = []

    seen_non_home = False

    for sample in samples:
        frame = int(sample["frame"])
        is_home = sample["home_ratio"] >= args.min_home_ratio

        if is_home:
            if not seen_non_home:
                # Ignore the initial home region before the first real action.
                continue

            if current:
                previous_frame = int(current[-1]["frame"])
                if frame - previous_frame > args.min_gap:
                    close_interval()

            current.append(sample)
        else:
            seen_non_home = True
            close_interval()

    # Do not close a trailing home interval.
    # The final return-to-home after the last task is not a boundary.

    intervals = [item for item in intervals if int(item["end"]) < sample_end]

    if len(intervals) < needed:
        raise RuntimeError(
            f"Found {len(intervals)} high-score boundary intervals, expected at least {needed}. "
            f"The initial home region and final return-to-home region are ignored. "
            f"Try lowering --min-home-ratio, increasing --joint-threshold, "
            f"or reducing --boundary-sample-stride."
        )

    selected: list[dict[str, float]] = []
    used: set[int] = set()
    selection = str(getattr(args, "boundary_selection", "ordered") or "ordered").strip().lower()
    if selection == "top":
        selected = sorted(
            intervals,
            key=lambda item: (
                item["score"],
                item["peak_score"],
                item["home_ratio"],
                item["duration"],
            ),
            reverse=True,
        )[:needed]
        selected.sort(key=lambda item: item["center"])
    else:
        for transition_index in range(1, len(specs)):
            target = round(len(states) * transition_index / len(specs))
            search_start = max(sample_start, target - args.search_radius)
            search_end = min(sample_end, target + args.search_radius)
            window = [
                (index, item)
                for index, item in enumerate(intervals)
                if index not in used and search_start <= int(item["center"]) <= search_end
            ]
            if window:
                chosen_index, chosen = max(
                    window,
                    key=lambda pair: (
                        pair[1]["score"],
                        pair[1]["peak_score"],
                        pair[1]["home_ratio"],
                        pair[1]["duration"],
                        -abs(int(pair[1]["center"]) - target),
                    ),
                )
            else:
                window = [
                    (index, item)
                    for index, item in enumerate(intervals)
                    if index not in used
                ]
                if not window:
                    continue
                chosen_index, chosen = max(
                    window,
                    key=lambda pair: (
                        -abs(int(pair[1]["center"]) - target),
                        pair[1]["score"],
                        pair[1]["peak_score"],
                        pair[1]["home_ratio"],
                        pair[1]["duration"],
                    ),
                )
            used.add(chosen_index)
            selected.append(chosen)
        selected.sort(key=lambda item: item["center"])

    if len(selected) != needed:
        raise RuntimeError(f"Selected {len(selected)} boundaries, expected {needed}. Try increasing --search-radius.")

    boundaries = [int(item["center"]) for item in selected]
    confidences = [float(item["home_ratio"]) for item in selected]
    return boundaries, confidences

def _episode_segments(
    source: SourceDataset,
    episode: dict[str, Any],
    table: Any,
    specs: list[SubtaskSpec],
    args: argparse.Namespace,
) -> list[SegmentPlan]:
    rows = table.to_pylist()
    if not rows:
        return []
    states = [_row_vector(row, args.state_key) for row in rows]
    if not states or not states[0]:
        raise KeyError(f"{source.root} episode {episode['episode_index']} missing numeric state key {args.state_key!r}")
    actions = [_row_vector(row, args.action_key) for row in rows]
    home_pose = _mean_pose(states, args.home_window)
    ignore_dims = _ignored_dims(source.info, args.state_key, args, len(home_pose))
    boundaries, confidences = _score_boundaries(states, actions, specs, ignore_dims, args)
    if len(boundaries) != len(specs) - 1:
        raise RuntimeError(
            f"{source.root.name} episode={episode['episode_index']} found {len(boundaries)} boundaries, "
            f"expected {len(specs) - 1}. Increase --search-radius or --joint-threshold."
            f"or reducing --boundary-sample-stride."
        )

    raw_bounds = [0, *boundaries, len(rows)]
    plans: list[SegmentPlan] = []
    for index, spec in enumerate(specs):
        raw_start = raw_bounds[index]
        raw_end = raw_bounds[index + 1]
        start, end, trimmed_start, trimmed_end = _trim_segment_bounds(raw_start, raw_end, states, actions, home_pose, ignore_dims, args)
        if end <= start:
            continue
        plans.append(
            SegmentPlan(
                label=spec.label,
                task=spec.prompt,
                source_root=source.root,
                source_id=source.source_id,
                source_episode=episode,
                source_episode_index=int(episode["episode_index"]),
                start=start,
                end=end,
                raw_start=raw_start,
                raw_end=raw_end,
                trimmed_start=trimmed_start,
                trimmed_end=trimmed_end,
                boundary_confidence=confidences[index - 1] if index > 0 and index - 1 < len(confidences) else 1.0,
            )
        )
    return plans


def _rewrite_table(table: Any, *, new_episode_index: int, global_start_index: int, task_index: int):
    length = table.num_rows
    table = _set_column(table, "episode_index", [new_episode_index] * length)
    table = _set_column(table, "frame_index", list(range(length)))
    table = _set_column(table, "index", list(range(global_start_index, global_start_index + length)))
    table = _set_column(table, "task_index", [task_index] * length)
    if "timestamp" in table.column_names and length > 0:
        values = table.column("timestamp").to_pylist()
        start = float(values[0])
        table = _set_column(table, "timestamp", [float(value) - start for value in values])
    return table


def _clip_video(
    source_video: Path,
    output_video: Path,
    *,
    start_frame: int,
    frame_count: int,
    args: argparse.Namespace,
) -> bool:
    if not source_video.exists():
        print(f"warning: missing video {source_video}", file=sys.stderr)
        return False
    output_video.parent.mkdir(parents=True, exist_ok=True)
    vf = f"trim=start_frame={start_frame}:end_frame={start_frame + frame_count},setpts=PTS-STARTPTS"
    command = [
        args.ffmpeg,
        "-y",
        "-hwaccel",
        "none",
        "-i",
        str(source_video),
        "-vf",
        vf,
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.video_preset,
        "-pix_fmt",
        "yuv420p",
    ]
    if args.ffmpeg_threads > 0:
        command.extend(["-threads", str(args.ffmpeg_threads)])
    if args.lossless_video:
        command.extend(["-qp", "0"])
    else:
        command.extend(["-crf", str(args.video_crf)])
    command.extend(["-movflags", "+faststart", str(output_video)])
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {source_video}\n{completed.stderr}")
    return True


def _clip_video_job(job: VideoJob, args: argparse.Namespace) -> bool:
    return _clip_video(
        job.source_video,
        job.output_video,
        start_frame=job.start_frame,
        frame_count=job.frame_count,
        args=args,
    )


def _clip_videos(video_jobs: list[VideoJob], args: argparse.Namespace) -> tuple[int, int]:
    if not video_jobs:
        return 0, 0

    workers = max(1, int(args.video_workers))
    if workers == 1 or len(video_jobs) == 1:
        written = 0
        missing = 0
        for job in video_jobs:
            if _clip_video_job(job, args):
                written += 1
            else:
                missing += 1
        return written, missing

    written = 0
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_clip_video_job, job, args) for job in video_jobs]
        for future in as_completed(futures):
            if future.result():
                written += 1
            else:
                missing += 1
    return written, missing


def _load_sources(source_paths: list[Path], planned_output_names: set[str]) -> list[SourceDataset]:
    sources: list[SourceDataset] = []
    for root in _discover_sources(source_paths, planned_output_names):
        sources.append(
            SourceDataset(
                root=root,
                source_id=_slug(root.name),
                info=_load_json(root / "meta" / "info.json"),
                episodes=_load_jsonl(root / "meta" / "episodes.jsonl"),
            )
        )
    return sources


def _video_keys_for_sources(sources: list[SourceDataset], args: argparse.Namespace) -> dict[Path, list[str]]:
    explicit = [item.strip() for item in str(args.video_keys or "").split(",") if item.strip()]
    if explicit:
        return {source.root: explicit for source in sources}
    return {source.root: _detect_video_keys(source.root, source.info, source.episodes) for source in sources}


def split_batch(args: argparse.Namespace) -> dict[str, Any]:
    specs = _parse_specs(args.subtasks, args.task_preset)  #得到基础的原子动作的列表，如果没有参数定义新的动作标签和prompt，那么使用默认的设置（默认solid及其标签）
    if not specs:
        raise ValueError("At least one subtask is required")

    source_roots = [Path(item) for item in args.source_root] #查找这个输入路径下的所有lerobot数据集，可以输入多条路径
    default_output_root = source_roots[0].expanduser().resolve() #只有一条输出路径
    if _is_lerobot_dataset(default_output_root):
        default_output_root = default_output_root.parent
    output_root = (Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root)
    repo_prefix = _slug(args.repo_prefix or output_root.name or "atomic")
    planned_output_names = {f"{repo_prefix}_{spec.label}" for spec in specs}
    sources = _load_sources(source_roots, planned_output_names)
    if not sources:
        raise ValueError("No source datasets selected")

    if args.split_videos:
        args.ffmpeg = _resolve_ffmpeg(args.ffmpeg)
        if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).exists():
            raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg}")

    _, pq = _import_pyarrow()
    video_keys_by_source = _video_keys_for_sources(sources, args)
    source_by_root = {source.root: source for source in sources}
    source_tables: dict[tuple[Path, int], Any] = {}
    segments_by_label: dict[str, list[SegmentPlan]] = {spec.label: [] for spec in specs}
    failed_episodes: list[dict[str, Any]] = []

    for source in sources:
        for episode in source.episodes:
            episode_index = int(episode["episode_index"])
            try:
                table = pq.read_table(_episode_path(source.root, source.info, episode_index))
                if args.cache_source_tables:
                    source_tables[(source.root, episode_index)] = table
                plans = _episode_segments(source, episode, table, specs, args)
            except Exception as exc:
                if not args.skip_failed_episodes:
                    raise
                failed_episodes.append(
                    {
                        "source_dataset": str(source.root),
                        "episode_index": episode_index,
                        "error": str(exc),
                    }
                )
                continue
            for plan in plans:
                segments_by_label[plan.label].append(plan)

    if not any(segments_by_label.values()):
        raise RuntimeError("No segments were produced")

    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "source_roots": [str(source.root) for source in sources],
        "output_root": str(output_root),
        "repo_prefix": repo_prefix,
        "task_preset": args.task_preset,
        "subtasks": [{"label": spec.label, "prompt": spec.prompt} for spec in specs],
        "datasets": [],
        "failed_episodes": failed_episodes,
        "split_videos": args.split_videos,
        "lossless_video": args.lossless_video,
    }

    base_info = dict(sources[0].info)
    chunk_size = max(int(base_info.get("chunks_size") or 1000), 1)
    for task_index, spec in enumerate(specs):
        label_segments = segments_by_label[spec.label]
        dataset_root = output_root / f"{repo_prefix}_{spec.label}"
        if dataset_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{dataset_root} exists; pass --overwrite")
            shutil.rmtree(dataset_root)
        (dataset_root / "meta").mkdir(parents=True, exist_ok=True)

        output_episodes: list[dict[str, Any]] = []
        output_stats: list[dict[str, Any]] = []
        report_rows: list[dict[str, Any]] = []
        global_index = 0
        videos_written = 0
        videos_missing = 0
        video_jobs: list[VideoJob] = []

        for new_episode_index, plan in enumerate(label_segments):
            source = source_by_root[plan.source_root]
            source_table = source_tables.get((source.root, plan.source_episode_index))
            if source_table is None:
                source_table = pq.read_table(_episode_path(source.root, source.info, plan.source_episode_index))
            source_slice = source_table.slice(plan.start, plan.end - plan.start)
            table = _rewrite_table(
                source_slice,
                new_episode_index=new_episode_index,
                global_start_index=global_index,
                task_index=0,
            )
            out_path = _output_episode_path(dataset_root, new_episode_index, chunk_size)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_path)
            _verify_preserved_data(source_slice, table, out_path)

            frame_count = table.num_rows
            episode = dict(plan.source_episode)
            episode.update(
                {
                    "episode_index": new_episode_index,
                    "source_dataset": plan.source_id,
                    "source_dataset_root": str(plan.source_root),
                    "source_episode_index": plan.source_episode_index,
                    "source_start_frame": plan.start,
                    "source_end_frame": plan.end,
                    "source_raw_start_frame": plan.raw_start,
                    "source_raw_end_frame": plan.raw_end,
                    "trimmed_start_frames": plan.trimmed_start,
                    "trimmed_end_frames": plan.trimmed_end,
                    "length": frame_count,
                    "tasks": [spec.prompt],
                }
            )
            output_episodes.append(episode)
            output_stats.append({"episode_index": new_episode_index, "stats": _episode_stats(table, base_info)})

            if args.split_videos:
                for video_key in video_keys_by_source.get(source.root, []):
                    video_jobs.append(
                        VideoJob(
                            source_video=_video_path(source.root, source.info, plan.source_episode_index, video_key),
                            output_video=_output_video_path(dataset_root, new_episode_index, video_key, chunk_size),
                            start_frame=plan.start,
                            frame_count=frame_count,
                        )
                    )

            report_rows.append(
                {
                    "label": spec.label,
                    "new_episode_index": new_episode_index,
                    "source_dataset": plan.source_id,
                    "source_episode_index": plan.source_episode_index,
                    "raw_start": plan.raw_start,
                    "raw_end": plan.raw_end,
                    "start": plan.start,
                    "end": plan.end,
                    "trimmed_start": plan.trimmed_start,
                    "trimmed_end": plan.trimmed_end,
                    "frames": frame_count,
                    "task": spec.prompt,
                    "boundary_confidence": f"{plan.boundary_confidence:.4f}",
                }
            )
            global_index += frame_count

        if args.split_videos:
            videos_written, videos_missing = _clip_videos(video_jobs, args)

        output_info = dict(base_info)
        output_info.update(
            {
                "total_episodes": len(output_episodes),
                "total_frames": global_index,
                "total_chunks": int(math.ceil(len(output_episodes) / chunk_size)),
                "total_tasks": 1,
                "total_videos": videos_written,
                "splits": {"train": f"0:{len(output_episodes)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
        _write_json(dataset_root / "meta" / "info.json", output_info)
        _write_jsonl(dataset_root / "meta" / "episodes.jsonl", output_episodes)
        _write_jsonl(dataset_root / "meta" / "episodes_stats.jsonl", output_stats)
        _write_jsonl(dataset_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": spec.prompt}])
        _write_csv(dataset_root / "meta" / "atomic_batch_split_report.csv", report_rows)

        summary["datasets"].append(
            {
                "label": spec.label,
                "prompt": spec.prompt,
                "dataset": dataset_root.name,
                "dataset_root": str(dataset_root),
                "episodes": len(output_episodes),
                "frames": global_index,
                "videos_written": videos_written,
                "videos_missing": videos_missing,
                "report": str(dataset_root / "meta" / "atomic_batch_split_report.csv"),
            }
        )

    _write_json(output_root / f"{repo_prefix}_atomic_batch_split_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True, help="LeRobot dataset root or parent folder.")
    parser.add_argument("--output-root", default="", help="Output parent. Defaults to the source parent folder.")
    parser.add_argument("--repo-prefix", default="", help="Output dataset prefix. Defaults to output parent folder name.")
    parser.add_argument("--task-preset", default="solid", choices=sorted(TASK_PRESETS))
    parser.add_argument(
        "--subtasks",
        default="",
        help="Ordered lines: label | prompt. Overrides --task-preset when non-empty.",
    )
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--joint-threshold", type=float, default=0.035)
    parser.add_argument("--min-home-ratio", type=float, default=0.65)
    parser.add_argument("--fallback-home-ratio", type=float, default=0.45)
    parser.add_argument("--edge-home-ratio", type=float, default=0.65)
    parser.add_argument("--home-window", type=int, default=10)
    parser.add_argument("--search-radius", type=int, default=80)
    parser.add_argument("--min-gap", type=int, default=20)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--min-edge-home-frames", type=int, default=5)
    parser.add_argument("--keep-edge-home-frames", type=int, default=0)
    parser.add_argument("--min-segment-frames", type=int, default=20)
    parser.add_argument("--edge-state-velocity-threshold", type=float, default=0.02)
    parser.add_argument("--edge-action-idle-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed-episodes", action="store_true")
    parser.add_argument("--split-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-keys", default="")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", default="fast")
    parser.add_argument(
        "--video-workers",
        type=int,
        default=1,
        help="Number of ffmpeg video clipping jobs to run concurrently.",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Set ffmpeg -threads for each video job. Use 0 to let ffmpeg decide.",
    )
    parser.add_argument("--lossless-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cache-source-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache source parquet tables after boundary detection to avoid rereading them during export.",
    )
    parser.add_argument(
        "--home-ignore-dims",
        default="",
        help="Comma-separated state/action dimensions ignored for home ratio, velocity, and action norm.",
    )
    parser.add_argument(
        "--inactive-arm",
        default="",
        choices=["", "left", "right"],
        help="Ignore one unused arm when detecting return-home boundaries.",
    )
    parser.add_argument(
        "--active-arm",
        default="",
        choices=["", "left", "right"],
        help="Only use one active arm when detecting return-home boundaries.",
    )
    parser.add_argument(
        "--tail-ignore-frames",
        type=int,
        default=120,
        help="Ignore this many frames at the end so the final return-home pose is not selected as a boundary.",
    )
    parser.add_argument(
        "--boundary-selection",
        default="ordered",
        choices=["ordered", "top"],
        help="ordered selects one boundary near each expected transition; top keeps the old global top-score behavior.",
    )
    parser.add_argument(
        "--boundary-sample-stride",
        type=int,
        default=5,
        help="Sample every N frames when detecting high-score boundary intervals.",
    )
    return parser.parse_args()


def main() -> None:
    summary = split_batch(_parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



## 现在存在的问题是，我需要从这里找出n-1峰值个平台，但是全部任务结束之后，机械臂回位，这时候也是一个相似度比较高的位置，那么现在实验的结果就是，末尾的相似度超过了第一第二个原子任务之间pause_pose的相似度
## 导致第一第二个任务没有办法切分，第五个任务是空白任务，中间的任务与subtask标签错位。请问可以怎么修改，让最后结束的位置不考虑进来。切掉也好，或者只是用先升后降的平台也好，只要能实现目标就行
