from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


def _ensure_numpy_fallback_path() -> None:
    pkgs_dir = Path(sys.prefix) / "pkgs"
    for candidate in sorted(pkgs_dir.glob("numpy-base-*/Lib/site-packages"), reverse=True):
        if (candidate / "numpy" / "lib").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


_ensure_numpy_fallback_path()

import numpy as np


def _import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        _ensure_numpy_fallback_path()
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    return pa, pq


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "min": np.min(values, axis=0).reshape(-1).tolist(),
        "max": np.max(values, axis=0).reshape(-1).tolist(),
        "mean": np.mean(values, axis=0).reshape(-1).tolist(),
        "std": np.std(values, axis=0).reshape(-1).tolist(),
        "count": [int(values.shape[0])],
    }


class InferenceParquetRecorder:
    """Collect synchronous inference requests as one local LeRobot episode."""

    IMAGE_KEYS = ("top_head", "hand_right", "hand_left")

    def __init__(
        self,
        output_root: str | Path,
        *,
        fps: float,
        prompt: str,
        inference_mode: str = "sync",
        flush_every_chunks: int = 1,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.fps = float(fps)
        self.prompt = prompt
        self.inference_mode = inference_mode
        self.flush_every_chunks = max(int(flush_every_chunks), 1)
        self.records: list[dict[str, Any]] = []
        self.output_parquet = self.output_root / "data/chunk-000/episode_000000.parquet"

    def append(
        self,
        *,
        request_step: int,
        timestamp: float,
        state: np.ndarray,
        image_bytes: dict[str, bytes],
        action_sequence: np.ndarray,
        executed_actions: np.ndarray,
        inference_ms: float,
    ) -> None:
        actions = np.asarray(action_sequence, dtype=np.float32)
        executed = np.asarray(executed_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected action sequence [N, 14], got {actions.shape}")
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.size != 14:
            raise ValueError(f"Expected state [14], got {state_array.shape}")
        missing = [key for key in self.IMAGE_KEYS if key not in image_bytes]
        if missing:
            raise KeyError(f"Missing inference images: {missing}")

        row_index = len(self.records)
        self.records.append(
            {
                "timestamp": float(timestamp),
                "frame_index": row_index,
                "episode_index": 0,
                "index": row_index,
                "task_index": 0,
                "request_step": int(request_step),
                "chunk_index": row_index,
                "action_count": int(actions.shape[0]),
                "inference_ms": float(inference_ms),
                "observation.state": state_array.tolist(),
                "action": actions[0].tolist(),
                "observation.images.top_head": image_bytes["top_head"],
                "observation.images.hand_right": image_bytes["hand_right"],
                "observation.images.hand_left": image_bytes["hand_left"],
                "prompt": self.prompt,
                "inference_mode": self.inference_mode,
                "action_sequence": json.dumps(actions.tolist(), separators=(",", ":")),
                "executed_action_sequence": json.dumps(executed.tolist(), separators=(",", ":")),
            }
        )
        if len(self.records) % self.flush_every_chunks == 0:
            self.flush()

    def _table(self):
        pa, _ = _import_pyarrow()
        list14 = pa.list_(pa.float32(), 14)
        schema = pa.schema(
            [
                pa.field("timestamp", pa.float64()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
                pa.field("request_step", pa.int64()),
                pa.field("chunk_index", pa.int64()),
                pa.field("action_count", pa.int64()),
                pa.field("inference_ms", pa.float32()),
                pa.field("observation.state", list14),
                pa.field("action", list14),
                pa.field("observation.images.top_head", pa.binary()),
                pa.field("observation.images.hand_right", pa.binary()),
                pa.field("observation.images.hand_left", pa.binary()),
                pa.field("prompt", pa.string()),
                pa.field("inference_mode", pa.string()),
                pa.field("action_sequence", pa.string()),
                pa.field("executed_action_sequence", pa.string()),
            ]
        )
        return pa.Table.from_pylist(self.records, schema=schema)

    def flush(self) -> Path | None:
        if not self.records:
            return None
        _, pq = _import_pyarrow()
        self.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_parquet.with_suffix(".parquet.tmp")
        table = self._table()
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, self.output_parquet)
        self._write_metadata()
        return self.output_parquet

    def close(self) -> Path | None:
        return self.flush()

    def _write_metadata(self) -> None:
        row_count = len(self.records)
        features = {
            "timestamp": {"dtype": "float64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "request_step": {"dtype": "int64", "shape": [1]},
            "chunk_index": {"dtype": "int64", "shape": [1]},
            "action_count": {"dtype": "int64", "shape": [1]},
            "inference_ms": {"dtype": "float32", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "observation.images.top_head": {"dtype": "image", "shape": [3, 224, 224]},
            "observation.images.hand_right": {"dtype": "image", "shape": [3, 224, 224]},
            "observation.images.hand_left": {"dtype": "image", "shape": [3, 224, 224]},
            "prompt": {"dtype": "string", "shape": [1]},
            "inference_mode": {"dtype": "string", "shape": [1]},
            "action_sequence": {"dtype": "string", "shape": [1]},
            "executed_action_sequence": {"dtype": "string", "shape": [1]},
        }
        info = {
            "codebase_version": "inference-trace-v1",
            "robot_type": "agilex",
            "total_episodes": 1,
            "total_frames": row_count,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1 if row_count else 0,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": features,
        }
        _write_json(self.output_root / "meta/info.json", info)
        _write_jsonl(
            self.output_root / "meta/episodes.jsonl",
            [{"episode_index": 0, "tasks": [self.prompt], "length": row_count}],
        )
        _write_jsonl(self.output_root / "meta/tasks.jsonl", [{"task_index": 0, "task": self.prompt}])

        numeric_columns = {
            "timestamp": np.asarray([row["timestamp"] for row in self.records], dtype=np.float64)[:, None],
            "frame_index": np.arange(row_count, dtype=np.int64)[:, None],
            "episode_index": np.zeros((row_count, 1), dtype=np.int64),
            "index": np.arange(row_count, dtype=np.int64)[:, None],
            "task_index": np.zeros((row_count, 1), dtype=np.int64),
            "request_step": np.asarray([row["request_step"] for row in self.records], dtype=np.int64)[:, None],
            "chunk_index": np.arange(row_count, dtype=np.int64)[:, None],
            "action_count": np.asarray([row["action_count"] for row in self.records], dtype=np.int64)[:, None],
            "inference_ms": np.asarray([row["inference_ms"] for row in self.records], dtype=np.float32)[:, None],
            "observation.state": np.asarray([row["observation.state"] for row in self.records], dtype=np.float32),
            "action": np.asarray([row["action"] for row in self.records], dtype=np.float32),
        }
        stats = {name: _stats(values) for name, values in numeric_columns.items()}
        _write_jsonl(
            self.output_root / "meta/episodes_stats.jsonl",
            [{"episode_index": 0, "stats": stats}],
        )
