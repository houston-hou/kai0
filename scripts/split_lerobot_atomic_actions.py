#!/usr/bin/env python3
"""Split a LeRobot dataset into atomic-action datasets from manual labels.

Label JSON examples:

{
  "pick_beaker": [
    {"episode_index": 0, "start": 12, "end": 86, "task": "pick up the beaker"}
  ],
  "pour_water": [
    {"episode_index": 0, "start": 86, "end": 151}
  ]
}

or:

{
  "segments": [
    {"label": "pick_beaker", "episode_index": 0, "start": 12, "end": 86}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segment:
    label: str
    episode_index: int
    start: int
    end: int
    task: str


def _import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pkgs_dir = Path(sys.prefix) / "pkgs"
        if pkgs_dir.exists():
            candidates = sorted(pkgs_dir.glob("numpy-base-*/Lib/site-packages"), reverse=True)
            for candidate in candidates:
                if (candidate / "numpy" / "lib").exists() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                    break
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

    return pa, pq


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


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


def _slug(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or "atomic_action"


def _episode_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = max(int(info.get("chunks_size") or 1000), 1)
    chunk_index = episode_index // chunk_size
    template = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    return dataset_root / template.format(episode_chunk=chunk_index, episode_index=episode_index)


def _output_episode_path(output_root: Path, episode_index: int, chunk_size: int = 1000) -> Path:
    return output_root / f"data/chunk-{episode_index // chunk_size:03d}/episode_{episode_index:06d}.parquet"


def _video_path(dataset_root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunk_size = max(int(info.get("chunks_size") or 1000), 1)
    chunk_index = episode_index // chunk_size
    template = info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return dataset_root / template.format(episode_chunk=chunk_index, video_key=video_key, episode_index=episode_index)


def _output_video_path(output_root: Path, episode_index: int, video_key: str, chunk_size: int = 1000) -> Path:
    return output_root / f"videos/chunk-{episode_index // chunk_size:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _detect_video_keys(source_root: Path, info: dict[str, Any], episodes: list[dict[str, Any]]) -> list[str]:
    keys = [
        key
        for key, feature in (info.get("features") or {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if keys or not episodes:
        return keys
    first_episode = int(episodes[0]["episode_index"])
    chunk_dir = source_root / "videos" / f"chunk-{first_episode // max(int(info.get('chunks_size') or 1000), 1):03d}"
    if not chunk_dir.exists():
        return []
    return [child.name for child in sorted(chunk_dir.iterdir()) if child.is_dir()]


def _set_column(table: Any, name: str, values: list[Any] | np.ndarray):
    if name not in table.column_names:
        return table
    pa, _ = _import_pyarrow()
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _rewrite_table(table: Any, *, new_episode_index: int, global_start_index: int, task_index: int = 0):
    length = table.num_rows
    table = _set_column(table, "episode_index", [new_episode_index] * length)
    table = _set_column(table, "frame_index", list(range(length)))
    table = _set_column(table, "index", list(range(global_start_index, global_start_index + length)))
    table = _set_column(table, "task_index", [task_index] * length)
    if "timestamp" in table.column_names and length > 0:
        timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        timestamps = timestamps - float(timestamps[0])
        table = _set_column(table, "timestamp", timestamps.tolist())
    return table


def _stat_value(value: Any) -> list[Any]:
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


def _episode_stats(table: Any, info: dict[str, Any]) -> dict[str, Any]:
    stats = {}
    for column_name in table.column_names:
        column_stats = _table_column_stats(table, column_name, info)
        if column_stats is not None:
            stats[column_name] = column_stats
    return stats


def _clip_video(
    source_video: Path,
    output_video: Path,
    *,
    start_frame: int,
    frame_count: int,
    fps: float,
    ffmpeg: str,
    crf: int,
    preset: str,
) -> bool:
    if not source_video.exists():
        print(f"warning: missing video {source_video}")
        return False
    output_video.parent.mkdir(parents=True, exist_ok=True)
    start_seconds = start_frame / fps
    duration = frame_count / fps
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_video),
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output_video),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {source_video}\n{completed.stderr}")
    return True


def _normalize_segments(payload: Any) -> list[Segment]:
    raw_segments: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        raw_segments = payload["segments"]
    elif isinstance(payload, list):
        raw_segments = payload
    elif isinstance(payload, dict):
        for label, entries in payload.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    raw_segments.append({"label": label, **entry})
    else:
        raise ValueError("Label JSON must be a list, a {segments: [...]} object, or a label-to-segments object")

    segments = []
    for entry in raw_segments:
        label = str(entry.get("label") or entry.get("name") or entry.get("action") or "").strip()
        if not label:
            raise ValueError(f"Segment missing label: {entry}")
        start = int(entry.get("start", entry.get("start_frame", entry.get("startFrame", 0))))
        end = int(entry.get("end", entry.get("end_frame", entry.get("endFrame", 0))))
        if end <= start:
            raise ValueError(f"Segment end must be greater than start: {entry}")
        episode_index = int(entry.get("episode_index", entry.get("episode", 0)))
        task = str(entry.get("task") or entry.get("prompt") or label.replace("_", " "))
        segments.append(Segment(label=label, episode_index=episode_index, start=start, end=end, task=task))
    if not segments:
        raise ValueError("No manual segments found in label JSON")
    return segments


def split_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    source_meta = source_root / "meta"
    info = _load_json(source_meta / "info.json")
    source_episodes = _load_jsonl(source_meta / "episodes.jsonl")
    source_episode_map = {int(item["episode_index"]): item for item in source_episodes}
    segments = _normalize_segments(_load_json(args.labels_json))
    groups: dict[str, list[Segment]] = defaultdict(list)
    for segment in segments:
        groups[segment.label].append(segment)

    _, pq = _import_pyarrow()
    fps = float(info.get("fps") or 30)
    chunk_size = max(int(info.get("chunks_size") or 1000), 1)
    video_keys = [item.strip() for item in args.video_keys.split(",") if item.strip()] or _detect_video_keys(source_root, info, source_episodes)
    if args.split_videos and video_keys and shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).exists():
        raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg}")

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "datasets": [],
        "total_segments": len(segments),
        "video_keys": video_keys,
    }

    for label, label_segments in groups.items():
        label_slug = _slug(label)
        dataset_root = output_root / f"{args.repo_prefix}_{label_slug}"
        if dataset_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{dataset_root} exists; pass --overwrite")
            shutil.rmtree(dataset_root)
        (dataset_root / "meta").mkdir(parents=True, exist_ok=True)

        output_episodes = []
        output_stats = []
        report_rows = []
        global_index = 0
        videos_written = 0
        videos_missing = 0
        task_text = label_segments[0].task

        for new_episode_index, segment in enumerate(label_segments):
            if segment.episode_index not in source_episode_map:
                raise KeyError(f"Unknown episode_index={segment.episode_index}")
            source_table = pq.read_table(_episode_path(source_root, info, segment.episode_index))
            end = min(segment.end, source_table.num_rows)
            start = max(0, min(segment.start, end))
            if end <= start:
                raise ValueError(f"Empty segment after clamping: {segment}")
            table = source_table.slice(start, end - start)
            table = _rewrite_table(table, new_episode_index=new_episode_index, global_start_index=global_index)
            out_path = _output_episode_path(dataset_root, new_episode_index, chunk_size)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_path)

            frame_count = table.num_rows
            episode = dict(source_episode_map[segment.episode_index])
            episode.update(
                {
                    "episode_index": new_episode_index,
                    "source_episode_index": segment.episode_index,
                    "source_start_frame": start,
                    "source_end_frame": end,
                    "length": frame_count,
                    "tasks": [task_text],
                }
            )
            output_episodes.append(episode)
            output_stats.append({"episode_index": new_episode_index, "stats": _episode_stats(table, info)})

            if args.split_videos:
                for video_key in video_keys:
                    wrote = _clip_video(
                        _video_path(source_root, info, segment.episode_index, video_key),
                        _output_video_path(dataset_root, new_episode_index, video_key, chunk_size),
                        start_frame=start,
                        frame_count=frame_count,
                        fps=fps,
                        ffmpeg=args.ffmpeg,
                        crf=args.video_crf,
                        preset=args.video_preset,
                    )
                    if wrote:
                        videos_written += 1
                    else:
                        videos_missing += 1

            report_rows.append(
                {
                    "label": label,
                    "new_episode_index": new_episode_index,
                    "source_episode_index": segment.episode_index,
                    "start": start,
                    "end": end,
                    "frames": frame_count,
                    "task": task_text,
                }
            )
            global_index += frame_count

        output_info = dict(info)
        output_info.update(
            {
                "total_episodes": len(output_episodes),
                "total_frames": global_index,
                "total_chunks": int(math.ceil(len(output_episodes) / chunk_size)),
                "total_videos": videos_written if args.split_videos else int(info.get("total_videos") or 0),
                "splits": {"train": f"0:{len(output_episodes)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
        _write_json(dataset_root / "meta" / "info.json", output_info)
        _write_jsonl(dataset_root / "meta" / "episodes.jsonl", output_episodes)
        _write_jsonl(dataset_root / "meta" / "episodes_stats.jsonl", output_stats)
        _write_jsonl(dataset_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_text}])
        _write_csv(dataset_root / "meta" / "atomic_split_report.csv", report_rows)
        _write_json(dataset_root / "meta" / "atomic_split_summary.json", {"label": label, "segments": report_rows})
        summary["datasets"].append(
            {
                "label": label,
                "dataset": dataset_root.name,
                "episodes": len(output_episodes),
                "frames": global_index,
                "videos_written": videos_written,
                "videos_missing": videos_missing,
            }
        )

    _write_json(output_root / f"{args.repo_prefix}_atomic_split_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--labels-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-prefix", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-keys", default="")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", default="fast")
    return parser.parse_args()


def main() -> None:
    summary = split_dataset(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
