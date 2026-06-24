from __future__ import annotations

# Example:
#   python scripts/trim_idle_edges_dataset.py \
#     --dataset-root /data/vla/hdy/RoboTwin-main/policy/pi05/training_data/emchem_atomic_merged_video_526 \
#     --output-dataset /data/vla/hdy/RoboTwin-main/policy/pi05/training_data/emchem_atomic_merged_video_526_trimmed \
#     --action-idle-threshold 0.01 \
#     --min-edge-idle-frames 5 \
#     --overwrite
#
# Dry run first:
#   python scripts/trim_idle_edges_dataset.py \
#     --dataset-root /data/vla/hdy/RoboTwin-main/policy/pi05/training_data/emchem_atomic_merged_video_526 \
#     --output-dataset /data/vla/hdy/RoboTwin-main/policy/pi05/training_data/emchem_atomic_merged_video_526_trimmed \
#     --dry-run
#
# Stricter trimming that also requires the observed state to be nearly static:
#   python scripts/trim_idle_edges_dataset.py \
#     --dataset-root /path/to/source_dataset \
#     --output-dataset /path/to/source_dataset_trimmed \
#     --also-require-state-idle \
#     --state-idle-threshold 0.002 \
#     --overwrite
#
# Video handling:
#   The script trims parquet rows and matching mp4 files with the same frame range.
#   Use --no-trim-videos only for datasets without videos, or for metadata-only testing.
#   If video keys cannot be detected automatically, pass:
#     --video-keys observation.images.laptop,observation.images.phone

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def _ensure_numpy_fallback_path() -> None:
    pkgs_dir = Path(sys.prefix) / "pkgs"
    if not pkgs_dir.exists():
        return
    candidates = sorted(pkgs_dir.glob("numpy-base-*/Lib/site-packages"), reverse=True)
    for candidate in candidates:
        if (candidate / "numpy" / "lib").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


_ensure_numpy_fallback_path()

import numpy as np


@dataclasses.dataclass
class Args:
    dataset_root: str
    output_dataset: str
    action_key: str = "action"
    state_key: str = "observation.state"
    action_idle_threshold: float = 0.01
    state_idle_threshold: float = 0.002
    also_require_state_idle: bool = False
    min_edge_idle_frames: int = 5
    keep_edge_idle_frames: int = 0
    min_keep_frames: int = 20
    episode_start: int = 0
    episode_stop: int | None = None
    max_episodes: int | None = None
    overwrite: bool = False
    resume: bool = False
    dry_run: bool = False
    reset_timestamps: bool = True
    trim_videos: bool = True
    video_keys: str = ""
    ffmpeg: str = "ffmpeg"
    video_crf: int = 23
    video_preset: str = "fast"
    lossless_video: bool = False
    workers: int = 1


@dataclasses.dataclass
class EpisodePlan:
    old_episode_index: int
    new_episode_index: int
    output_episode: dict[str, Any]
    original_length: int
    kept_length: int
    trim_start: int
    trim_end: int
    trimmed_start: int
    trimmed_end: int
    raw_leading_idle: int
    raw_trailing_idle: int
    action_idle_ratio: float


@dataclasses.dataclass
class WriteResult:
    videos_written: int = 0
    videos_missing: int = 0
    episodes_reused: int = 0


def _import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        _ensure_numpy_fallback_path()
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    return pa, pq


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_episode_stats(source_meta_dir: Path) -> dict[int, dict[str, Any]]:
    path = source_meta_dir / "episodes_stats.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source {path}. LeRobot training requires meta/episodes_stats.jsonl in local datasets."
        )
    rows = _load_jsonl(path)
    return {int(row["episode_index"]): row for row in rows}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _stat_value(value: Any) -> list[Any] | Any:
    payload = np.asarray(value).tolist()
    return payload if isinstance(payload, list) else [payload]


def _table_column_stats(table: Any, column_name: str, info: dict[str, Any]) -> dict[str, Any] | None:
    feature = (info.get("features") or {}).get(column_name)
    if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}:
        return None

    values = table.column(column_name).to_pylist()
    if not values:
        return {"count": [0]}
    try:
        array = np.asarray(values)
    except Exception:
        return None
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        return None

    return {
        "min": _stat_value(np.min(array, axis=0)),
        "max": _stat_value(np.max(array, axis=0)),
        "mean": _stat_value(np.mean(array, axis=0)),
        "std": _stat_value(np.std(array, axis=0)),
        "count": [int(array.shape[0])],
    }


def _write_episode_stats(
    output_root: Path,
    info: dict[str, Any],
    output_episodes: list[dict[str, Any]],
    source_stats_by_episode: dict[int, dict[str, Any]],
) -> None:
    _, pq = _import_pyarrow()
    rows: list[dict[str, Any]] = []
    for episode in output_episodes:
        new_episode_index = int(episode["episode_index"])
        old_episode_index = int(episode.get("source_episode_index", new_episode_index))
        source_stats = source_stats_by_episode.get(old_episode_index)
        if source_stats is None:
            raise KeyError(f"Missing source stats for episode_index={old_episode_index}")

        table = pq.read_table(_output_episode_path(output_root, info, new_episode_index))
        stats = dict(source_stats.get("stats", {}))
        for column_name in table.column_names:
            column_stats = _table_column_stats(table, column_name, info)
            if column_stats is not None:
                stats[column_name] = column_stats

        rows.append({"episode_index": new_episode_index, "stats": stats})

    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    lines = [",".join(keys)]
    for row in rows:
        values = []
        for key in keys:
            text = "" if row.get(key) is None else str(row.get(key))
            if any(ch in text for ch in [",", "\n", '"']):
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _episode_name(index: int) -> str:
    return f"episode_{index:06d}.parquet"


def _episode_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = max(int(info.get("chunks_size") or 1000), 1)
    chunk_index = episode_index // chunk_size
    template = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    return dataset_root / template.format(episode_chunk=chunk_index, episode_index=episode_index)


def _output_episode_path(output_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = max(int(info.get("chunks_size") or 1000), 1)
    return output_root / f"data/chunk-{episode_index // chunk_size:03d}/{_episode_name(episode_index)}"


def _video_path(dataset_root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunk_size = max(int(info.get("chunks_size") or 1000), 1)
    chunk_index = episode_index // chunk_size
    template = info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return dataset_root / template.format(
        episode_chunk=chunk_index,
        video_key=video_key,
        episode_index=episode_index,
    )


def _parse_video_keys(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _detect_video_keys(source_root: Path, info: dict[str, Any], episodes: list[dict[str, Any]]) -> list[str]:
    keys = []
    for key, feature in (info.get("features") or {}).items():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            keys.append(key)
    if keys:
        return keys

    if not episodes:
        return []
    first_episode = int(episodes[0]["episode_index"])
    chunk_size = max(int(info.get("chunks_size") or 1000), 1)
    chunk_dir = source_root / "videos" / f"chunk-{first_episode // chunk_size:03d}"
    if not chunk_dir.exists():
        return []

    for child in sorted(chunk_dir.iterdir()):
        if child.is_dir() and (child / f"episode_{first_episode:06d}.mp4").exists():
            keys.append(child.name)
    return keys


def _select_episodes(episodes: list[dict[str, Any]], args: Args) -> list[dict[str, Any]]:
    selected = [entry for entry in episodes if int(entry["episode_index"]) >= args.episode_start]
    if args.episode_stop is not None:
        selected = [entry for entry in selected if int(entry["episode_index"]) < args.episode_stop]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]
    if not selected:
        raise ValueError("No episodes selected.")
    return selected


def _stack(rows: list[dict[str, Any]], key: str, episode_index: int) -> np.ndarray:
    if key not in rows[0]:
        raise KeyError(f"episode={episode_index} missing column {key!r}")
    return np.stack([np.asarray(row[key], dtype=np.float64) for row in rows], axis=0)


def _norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(values.reshape(values.shape[0], -1), axis=-1)


def _state_idle_mask(states: np.ndarray, threshold: float) -> np.ndarray:
    if states.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    velocity = _norm(np.diff(states, axis=0))
    frame_mask = np.zeros((states.shape[0],), dtype=bool)
    if velocity.size:
        frame_mask[0] = velocity[0] < threshold
        frame_mask[1:] = velocity < threshold
    return frame_mask


def _edge_count(mask: np.ndarray, *, leading: bool) -> int:
    values = mask if leading else mask[::-1]
    count = 0
    for item in values:
        if not bool(item):
            break
        count += 1
    return count


def _trim_bounds(idle_mask: np.ndarray, args: Args) -> tuple[int, int, int, int]:
    total = int(idle_mask.shape[0])
    leading = _edge_count(idle_mask, leading=True)
    trailing = _edge_count(idle_mask, leading=False)

    trim_start = 0
    trim_end = total
    if leading >= args.min_edge_idle_frames:
        trim_start = max(0, leading - args.keep_edge_idle_frames)
    if trailing >= args.min_edge_idle_frames:
        trim_end = min(total, total - trailing + args.keep_edge_idle_frames)

    if trim_end - trim_start < args.min_keep_frames:
        return 0, total, 0, 0
    return trim_start, trim_end, trim_start, total - trim_end


def _set_column(table: Any, name: str, values: list[Any] | np.ndarray):
    if name not in table.column_names:
        return table
    pa, _ = _import_pyarrow()
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    array = pa.array(values, type=field.type)
    return table.set_column(index, field, array)


def _rewrite_indices(table: Any, *, new_episode_index: int, global_start_index: int, reset_timestamps: bool):
    length = table.num_rows
    table = _set_column(table, "episode_index", [new_episode_index] * length)
    table = _set_column(table, "frame_index", list(range(length)))
    table = _set_column(table, "index", list(range(global_start_index, global_start_index + length)))

    if reset_timestamps and "timestamp" in table.column_names and length > 0:
        timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        timestamps = timestamps - float(timestamps[0])
        table = _set_column(table, "timestamp", timestamps.tolist())
    return table


def _copy_static_meta(source_meta_dir: Path, output_meta_dir: Path) -> None:
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tasks.jsonl"]:
        source = source_meta_dir / name
        if source.exists():
            shutil.copy2(source, output_meta_dir / name)


def _trim_video(
    source_video: Path,
    output_video: Path,
    *,
    trim_start: int,
    trim_end: int,
    original_length: int,
    args: Args,
) -> None:
    if not source_video.exists():
        raise FileNotFoundError(f"Missing source video: {source_video}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    if trim_start == 0 and trim_end == original_length:
        shutil.copy2(source_video, output_video)
        return

    vf = f"trim=start_frame={trim_start}:end_frame={trim_end},setpts=PTS-STARTPTS"
    cmd = [
        args.ffmpeg,
        "-y",
        "-i",
        str(source_video),
        "-vf",
        vf,
        "-frames:v",
        str(trim_end - trim_start),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.video_preset,
        "-pix_fmt",
        "yuv420p",
    ]
    if args.lossless_video:
        cmd.extend(["-qp", "0"])
    else:
        cmd.extend(["-crf", str(args.video_crf)])
    cmd.extend(["-movflags", "+faststart", str(output_video)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {source_video} -> {output_video}\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def _verify_preserved_data(source_slice: Any, written_table: Any, output_path: Path, reset_timestamps: bool) -> None:
    rewritten = {"episode_index", "frame_index", "index"}
    if reset_timestamps:
        rewritten.add("timestamp")
    preserved_columns = [name for name in source_slice.column_names if name not in rewritten]
    if not preserved_columns:
        return
    expected = source_slice.select(preserved_columns)
    actual = written_table.select(preserved_columns)
    if not expected.equals(actual, check_metadata=False):
        raise RuntimeError(f"Data changed before writing trimmed parquet: {output_path}")
    _, pq = _import_pyarrow()
    persisted = pq.read_table(output_path, columns=preserved_columns)
    if not expected.equals(persisted, check_metadata=False):
        raise RuntimeError(f"Data changed after writing trimmed parquet: {output_path}")


def _build_episode_plan(table: Any, episode_entry: dict[str, Any], new_episode_index: int, args: Args) -> EpisodePlan | None:
    old_episode_index = int(episode_entry["episode_index"])
    rows = table.to_pylist()
    if not rows:
        return None

    actions = _stack(rows, args.action_key, old_episode_index)
    states = _stack(rows, args.state_key, old_episode_index) if args.state_key in rows[0] else None
    if states is not None and actions.ndim == 2 and states.ndim == 2:
        dims = min(actions.shape[-1], states.shape[-1])
        action_idle = _norm(actions[:, :dims] - states[:, :dims]) < args.action_idle_threshold
    else:
        action_idle = _norm(actions) < args.action_idle_threshold

    idle_mask = action_idle
    if args.also_require_state_idle and states is not None:
        idle_mask = idle_mask & _state_idle_mask(states, args.state_idle_threshold)

    trim_start, trim_end, trimmed_start, trimmed_end = _trim_bounds(idle_mask, args)
    kept_length = trim_end - trim_start

    output_episode = dict(episode_entry)
    output_episode["episode_index"] = new_episode_index
    output_episode["source_episode_index"] = old_episode_index
    output_episode["length"] = kept_length

    return EpisodePlan(
        old_episode_index=old_episode_index,
        new_episode_index=new_episode_index,
        output_episode=output_episode,
        original_length=table.num_rows,
        kept_length=kept_length,
        trim_start=trim_start,
        trim_end=trim_end,
        trimmed_start=trimmed_start,
        trimmed_end=trimmed_end,
        raw_leading_idle=_edge_count(idle_mask, leading=True),
        raw_trailing_idle=_edge_count(idle_mask, leading=False),
        action_idle_ratio=float(np.mean(action_idle)),
    )


def _analyze_episode(
    source_root: Path,
    info: dict[str, Any],
    args: Args,
    new_episode_index: int,
    episode_entry: dict[str, Any],
) -> EpisodePlan | None:
    _, pq = _import_pyarrow()
    old_episode_index = int(episode_entry["episode_index"])
    table = pq.read_table(_episode_path(source_root, info, old_episode_index))
    return _build_episode_plan(table, episode_entry, new_episode_index, args)


def _write_episode_outputs(
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    args: Args,
    video_keys: list[str],
    plan: EpisodePlan,
    global_start_index: int,
    table: Any | None = None,
) -> WriteResult:
    _, pq = _import_pyarrow()
    if table is None:
        table = pq.read_table(_episode_path(source_root, info, plan.old_episode_index))

    source_slice = table.slice(plan.trim_start, plan.kept_length)
    trimmed_table = _rewrite_indices(
        source_slice,
        new_episode_index=plan.new_episode_index,
        global_start_index=global_start_index,
        reset_timestamps=args.reset_timestamps,
    )

    output_path = _output_episode_path(output_root, info, plan.new_episode_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(trimmed_table, output_path)
    _verify_preserved_data(source_slice, trimmed_table, output_path, args.reset_timestamps)

    result = WriteResult()
    if args.trim_videos:
        for video_key in video_keys:
            source_video = _video_path(source_root, info, plan.old_episode_index, video_key)
            output_video = _video_path(output_root, info, plan.new_episode_index, video_key)
            if not source_video.exists():
                result.videos_missing += 1
                print(f"warning: missing video {source_video}")
                continue
            _trim_video(
                source_video,
                output_video,
                trim_start=plan.trim_start,
                trim_end=plan.trim_end,
                original_length=plan.original_length,
                args=args,
            )
            result.videos_written += 1
    return result


def _existing_episode_outputs(
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    args: Args,
    video_keys: list[str],
    plan: EpisodePlan,
) -> WriteResult | None:
    if not _output_episode_path(output_root, info, plan.new_episode_index).exists():
        return None

    result = WriteResult(episodes_reused=1)
    if not args.trim_videos:
        return result

    for video_key in video_keys:
        source_video = _video_path(source_root, info, plan.old_episode_index, video_key)
        output_video = _video_path(output_root, info, plan.new_episode_index, video_key)
        if not source_video.exists():
            result.videos_missing += 1
            continue
        if not output_video.exists():
            return None
        result.videos_written += 1
    return result


def _plan_report_row(plan: EpisodePlan) -> dict[str, Any]:
    return {
        "old_episode_index": plan.old_episode_index,
        "new_episode_index": plan.new_episode_index,
        "original_length": plan.original_length,
        "kept_length": plan.kept_length,
        "trimmed_start": plan.trimmed_start,
        "trimmed_end": plan.trimmed_end,
        "trimmed_total": plan.trimmed_start + plan.trimmed_end,
        "raw_leading_idle": plan.raw_leading_idle,
        "raw_trailing_idle": plan.raw_trailing_idle,
        "action_idle_ratio": plan.action_idle_ratio,
    }


def _resolve_workers(workers: int) -> int:
    if workers == 0:
        return max(os.cpu_count() or 1, 1)
    return max(workers, 1)


def _write_report(output_dir: Path, report_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    worst = sorted(report_rows, key=lambda row: int(row["trimmed_total"]), reverse=True)[:20]
    video_keys = ", ".join(summary["video_keys"]) if summary["video_keys"] else "not detected"
    lines = [
        "# 静止帧裁剪报告",
        "",
        f"- 源数据集：`{summary['source_dataset']}`",
        f"- 输出数据集：`{summary['output_dataset']}`",
        f"- 处理 episode 数：{summary['episodes']}",
        f"- 原始帧数：{summary['original_frames']}",
        f"- 保留帧数：{summary['kept_frames']}",
        f"- 裁剪帧数：{summary['trimmed_frames']}",
        f"- 裁剪比例：{summary['trimmed_ratio']:.2%}",
        f"- 视频裁剪：{summary['trim_videos']}",
        f"- 视频 key：{video_keys}",
        f"- 成功写出视频数：{summary['videos_written']}",
        f"- 缺失视频数：{summary['videos_missing']}",
        f"- action idle 阈值：{summary['action_idle_threshold']}",
        f"- state idle 阈值：{summary['state_idle_threshold']}",
        f"- 是否要求 state 同时静止：{summary['also_require_state_idle']}",
        "",
        "## 结论",
        "",
    ]
    if summary["trimmed_ratio"] >= 0.2:
        lines.append("- 裁剪比例较高，说明数据首尾留白对训练分布的影响可能比较明显。")
    elif summary["trimmed_ratio"] >= 0.05:
        lines.append("- 存在一定首尾留白，清洗后建议重新训练，或至少重新计算 norm stats。")
    else:
        lines.append("- 首尾连续静止帧比例不高，慢动作问题可能不主要来自 episode 边缘留白。")
    if summary["trim_videos"] and summary["videos_missing"]:
        lines.append("- 有视频文件缺失，建议检查 `videos/` 目录和 `--video-keys` 是否匹配。")
    lines.extend(
        [
            "- `trim_report.csv` 记录了每条 episode 的裁剪帧数，建议优先回看裁剪最多的 episode。",
            "- `episodes_stats.jsonl` 会同步写出：图像/视频统计沿用源数据集，数值列从裁剪后的 parquet 重新计算。",
            "",
            "## 裁剪最多的 Episode",
            "",
            "| old_episode | new_episode | 原始帧数 | 保留帧数 | 开头裁剪 | 结尾裁剪 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst:
        lines.append(
            f"| {row['old_episode_index']} | {row['new_episode_index']} | {row['original_length']} | "
            f"{row['kept_length']} | {row['trimmed_start']} | {row['trimmed_end']} |"
        )
    (output_dir / "trim_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(args: Args) -> None:
    source_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_dataset).resolve()
    source_meta_dir = source_root / "meta"
    output_meta_dir = output_root / "meta"
    output_data_dir = output_root / "data"

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")
    if output_root.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(output_root)
    if output_root.exists() and not args.overwrite and not args.resume and not args.dry_run:
        raise FileExistsError(f"Output dataset already exists: {output_root}. Use --overwrite to replace it.")

    info = _load_json(source_meta_dir / "info.json")
    episodes = _select_episodes(_load_jsonl(source_meta_dir / "episodes.jsonl"), args)
    source_stats_by_episode = _load_episode_stats(source_meta_dir)
    video_keys = _parse_video_keys(args.video_keys) or _detect_video_keys(source_root, info, episodes)
    if args.trim_videos and not video_keys and ((source_root / "videos").exists() or int(info.get("total_videos") or 0) > 0):
        raise ValueError(
            "Video trimming is enabled, but no video keys were detected. "
            "Pass --video-keys, for example: --video-keys observation.images.laptop,observation.images.phone"
        )
    if args.trim_videos and video_keys and shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).exists():
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg}")
    report_rows: list[dict[str, Any]] = []
    output_episodes: list[dict[str, Any]] = []
    global_index = 0
    videos_written = 0
    videos_missing = 0
    episodes_reused = 0
    worker_count = _resolve_workers(args.workers)

    if not args.dry_run:
        output_meta_dir.mkdir(parents=True, exist_ok=True)
        output_data_dir.mkdir(parents=True, exist_ok=True)
        _copy_static_meta(source_meta_dir, output_meta_dir)

    if worker_count == 1:
        _, pq = _import_pyarrow()
        for new_episode_index, episode_entry in enumerate(episodes):
            old_episode_index = int(episode_entry["episode_index"])
            table = pq.read_table(_episode_path(source_root, info, old_episode_index))
            plan = _build_episode_plan(table, episode_entry, new_episode_index, args)
            if plan is None:
                continue

            if not args.dry_run:
                result = _existing_episode_outputs(source_root, output_root, info, args, video_keys, plan) if args.resume else None
                if result is None:
                    result = _write_episode_outputs(
                        source_root,
                        output_root,
                        info,
                        args,
                        video_keys,
                        plan,
                        global_index,
                        table=table,
                    )
                videos_written += result.videos_written
                videos_missing += result.videos_missing
                episodes_reused += result.episodes_reused

            output_episodes.append(plan.output_episode)
            report_rows.append(_plan_report_row(plan))
            global_index += plan.kept_length
    else:
        plans: list[EpisodePlan] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_analyze_episode, source_root, info, args, new_episode_index, episode_entry)
                for new_episode_index, episode_entry in enumerate(episodes)
            ]
            for future in concurrent.futures.as_completed(futures):
                plan = future.result()
                if plan is not None:
                    plans.append(plan)

        plans.sort(key=lambda plan: plan.new_episode_index)
        global_starts: dict[int, int] = {}
        for plan in plans:
            global_starts[plan.new_episode_index] = global_index
            output_episodes.append(plan.output_episode)
            report_rows.append(_plan_report_row(plan))
            global_index += plan.kept_length

        if not args.dry_run:
            write_plans = []
            for plan in plans:
                result = _existing_episode_outputs(source_root, output_root, info, args, video_keys, plan) if args.resume else None
                if result is None:
                    write_plans.append(plan)
                    continue
                videos_written += result.videos_written
                videos_missing += result.videos_missing
                episodes_reused += result.episodes_reused

            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _write_episode_outputs,
                        source_root,
                        output_root,
                        info,
                        args,
                        video_keys,
                        plan,
                        global_starts[plan.new_episode_index],
                    )
                    for plan in write_plans
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    videos_written += result.videos_written
                    videos_missing += result.videos_missing
                    episodes_reused += result.episodes_reused

    total_original = int(sum(int(row["original_length"]) for row in report_rows))
    total_kept = int(sum(int(row["kept_length"]) for row in report_rows))
    total_trimmed = total_original - total_kept
    summary = {
        "source_dataset": str(source_root),
        "output_dataset": str(output_root),
        "episodes": len(output_episodes),
        "original_frames": total_original,
        "kept_frames": total_kept,
        "trimmed_frames": total_trimmed,
        "trimmed_ratio": float(total_trimmed / total_original) if total_original else 0.0,
        "action_idle_threshold": args.action_idle_threshold,
        "state_idle_threshold": args.state_idle_threshold,
        "also_require_state_idle": args.also_require_state_idle,
        "min_edge_idle_frames": args.min_edge_idle_frames,
        "keep_edge_idle_frames": args.keep_edge_idle_frames,
        "min_keep_frames": args.min_keep_frames,
        "dry_run": args.dry_run,
        "trim_videos": args.trim_videos,
        "video_keys": video_keys,
        "videos_written": videos_written,
        "videos_missing": videos_missing,
        "resume": args.resume,
        "episodes_reused": episodes_reused,
        "workers": worker_count,
        "data_verified": not args.dry_run,
        "lossless_video": args.lossless_video,
    }

    if not args.dry_run:
        output_info = dict(info)
        chunk_size = max(int(info.get("chunks_size") or 1000), 1)
        output_info["total_episodes"] = len(output_episodes)
        output_info["total_frames"] = total_kept
        output_info["total_chunks"] = int(math.ceil(len(output_episodes) / chunk_size))
        output_info["total_videos"] = videos_written
        output_info["splits"] = {"train": f"0:{len(output_episodes)}"}
        _write_json(output_meta_dir / "info.json", output_info)
        _write_jsonl(output_meta_dir / "episodes.jsonl", output_episodes)
        _write_episode_stats(output_root, info, output_episodes, source_stats_by_episode)

    report_dir = output_root if not args.dry_run else source_root / "trim_idle_edges_dry_run"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "trim_summary.json", summary)
    _write_csv(report_dir / "trim_report.csv", report_rows)
    _write_report(report_dir, report_rows, summary)

    print(f"source: {source_root}")
    print(f"output: {output_root}")
    print(f"episodes={len(output_episodes)} original_frames={total_original} kept_frames={total_kept}")
    print(f"trimmed_frames={total_trimmed} trimmed_ratio={summary['trimmed_ratio']:.2%}")
    print(f"videos_written={videos_written} videos_missing={videos_missing} video_keys={video_keys}")
    print(f"episodes_reused={episodes_reused} resume={args.resume}")
    print(f"workers={worker_count}")
    print(f"report: {report_dir / 'trim_report.md'}")


def _parse_args() -> Args:
    defaults = Args(dataset_root="", output_dataset="")
    parser = argparse.ArgumentParser(description="Trim leading/trailing idle frames from a LeRobot parquet/video dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--action-key", default=defaults.action_key)
    parser.add_argument("--state-key", default=defaults.state_key)
    parser.add_argument("--action-idle-threshold", type=float, default=defaults.action_idle_threshold)
    parser.add_argument("--state-idle-threshold", type=float, default=defaults.state_idle_threshold)
    parser.add_argument("--also-require-state-idle", action="store_true", default=defaults.also_require_state_idle)
    parser.add_argument("--min-edge-idle-frames", type=int, default=defaults.min_edge_idle_frames)
    parser.add_argument("--keep-edge-idle-frames", type=int, default=defaults.keep_edge_idle_frames)
    parser.add_argument("--min-keep-frames", type=int, default=defaults.min_keep_frames)
    parser.add_argument("--episode-start", type=int, default=defaults.episode_start)
    parser.add_argument("--episode-stop", type=int, default=defaults.episode_stop)
    parser.add_argument("--max-episodes", type=int, default=defaults.max_episodes)
    parser.add_argument("--overwrite", action="store_true", default=defaults.overwrite)
    parser.add_argument("--resume", action="store_true", default=defaults.resume)
    parser.add_argument("--dry-run", action="store_true", default=defaults.dry_run)
    parser.add_argument("--reset-timestamps", action=argparse.BooleanOptionalAction, default=defaults.reset_timestamps)
    parser.add_argument("--trim-videos", action=argparse.BooleanOptionalAction, default=defaults.trim_videos)
    parser.add_argument("--video-keys", default=defaults.video_keys)
    parser.add_argument("--ffmpeg", default=defaults.ffmpeg)
    parser.add_argument("--video-crf", type=int, default=defaults.video_crf)
    parser.add_argument("--video-preset", default=defaults.video_preset)
    parser.add_argument(
        "--lossless-video",
        action=argparse.BooleanOptionalAction,
        default=defaults.lossless_video,
        help="Encode trimmed videos with lossless H.264 (QP 0).",
    )
    parser.add_argument("--workers", type=int, default=defaults.workers, help="Parallel episode workers. Use 0 for CPU count.")
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(_parse_args())
