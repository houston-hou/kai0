#!/usr/bin/env python3
"""Run a served policy on LeRobot observations and compare predicted actions.

This script is intentionally client-side: start the normal policy server first,
then use this script to feed real LeRobot frames/states into that server and
plot predicted action chunks against the dataset's recorded action trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np


DEFAULT_IMAGE_MAP = {
    "top_head": "observation.images.cam_high",
    "hand_left": "observation.images.cam_left_wrist",
    "hand_right": "observation.images.cam_right_wrist",
}

ACTION_NAMES = [
    "left_0",
    "left_1",
    "left_2",
    "left_3",
    "left_4",
    "left_5",
    "left_gripper",
    "right_0",
    "right_1",
    "right_2",
    "right_3",
    "right_4",
    "right_5",
    "right_gripper",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_dataset_root(dataset_root: str, repo_id: str | None) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if (root / "meta" / "info.json").exists():
        return root
    if repo_id:
        candidate = (root / repo_id).resolve()
        if (candidate / "meta" / "info.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not find meta/info.json under {root} or {root / (repo_id or '<repo_id>')}")


def _episode_chunk(episode_index: int, info: dict[str, Any]) -> int:
    chunk_size = int(info.get("chunks_size") or 1000)
    return episode_index // chunk_size


def _data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    template = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    return root / template.format(episode_chunk=_episode_chunk(episode_index, info), episode_index=episode_index)


def _video_path(root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    template = info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return root / template.format(
        episode_chunk=_episode_chunk(episode_index, info),
        video_key=video_key,
        episode_index=episode_index,
    )


def _read_table(path: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read LeRobot parquet files. Run this script in the same environment "
            "used by RoboData Studio / LeRobot, or install pyarrow there."
        ) from exc

    table = pq.read_table(path)

    def read_column(name: str) -> np.ndarray:
        column = table[name].combine_chunks()
        if isinstance(column, pa.FixedSizeListArray):
            return column.flatten().to_numpy().reshape(len(column), column.type.list_size)
        if isinstance(column, pa.ListArray):
            value_type = column.type
            while isinstance(value_type, pa.ListType):
                value_type = value_type.value_type
            return np.asarray(column.to_pylist(), dtype=value_type.to_pandas_dtype())
        return column.to_numpy(zero_copy_only=False)

    return {name: read_column(name) for name in table.column_names}


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to read videos/images for evaluation.") from exc
    return cv2


def _import_av():
    try:
        import av
    except ImportError:
        return None
    return av


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write evaluation plots.") from exc
    return plt


def _parse_image_map(text: str) -> dict[str, str]:
    if not text:
        return dict(DEFAULT_IMAGE_MAP)
    output = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid image map item: {item!r}. Expected target=source.")
        output[key.strip()] = value.strip()
    return output


def _decode_image_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _decode_image_value(bytes(value["bytes"]))
        if value.get("path"):
            return _read_image_path(Path(value["path"]))
    if isinstance(value, (bytes, bytearray)):
        cv2 = _import_cv2()
        encoded = np.frombuffer(bytes(value), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image bytes")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if isinstance(value, str):
        return _read_image_path(Path(value))
    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return array.astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image value with shape {array.shape}")


def _read_image_path(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._av = _import_av()
        self._container = None
        self._stream = None
        self._frames = None
        self._next_frame_index = 0
        self._cv2 = None
        self._cap = None
        if self._av is not None:
            self._open_av()
        else:
            self._open_cv2()

    def _open_av(self) -> None:
        self._container = self._av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._frames = self._container.decode(self._stream)
        self._next_frame_index = 0

    def _open_cv2(self) -> None:
        self._cv2 = _import_cv2()
        self._cap = self._cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")

    def _reset_av(self) -> None:
        if self._container is not None:
            self._container.close()
        self._open_av()

    def read_rgb(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        if self._av is not None:
            if frame_index < self._next_frame_index:
                self._reset_av()
            assert self._frames is not None
            for frame in self._frames:
                current = self._next_frame_index
                self._next_frame_index += 1
                if current == frame_index:
                    return frame.to_rgb().to_ndarray()
            raise RuntimeError(f"Could not read frame {frame_index} from {self.path}")

        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = self._cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {self.path}")
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        if self._cap is not None:
            self._cap.release()


def _build_video_readers(root: Path, info: dict[str, Any], episode_index: int) -> dict[str, VideoReader]:
    readers = {}
    for key, feature in (info.get("features") or {}).items():
        if (feature or {}).get("dtype") == "video":
            readers[key] = VideoReader(_video_path(root, info, episode_index, key))
    return readers


def _image_for_key(
    source_key: str,
    frame_index: int,
    table: dict[str, np.ndarray],
    video_readers: dict[str, VideoReader],
    camera_color_order: str,
    model_color_order: str,
) -> np.ndarray:
    if source_key in video_readers:
        image = video_readers[source_key].read_rgb(frame_index)
    elif source_key in table:
        image = _decode_image_value(table[source_key][frame_index])
        if camera_color_order == "bgr":
            cv2 = _import_cv2()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise KeyError(f"Image key {source_key!r} not found in videos or parquet table")
    if model_color_order == "bgr":
        cv2 = _import_cv2()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _prompt_for_row(args: argparse.Namespace, table: dict[str, np.ndarray], tasks: list[dict[str, Any]], frame_index: int) -> str:
    if args.prompt:
        return args.prompt
    if "prompt" in table:
        value = table["prompt"][frame_index]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
    if "task_index" in table and tasks:
        task_index = int(np.asarray(table["task_index"][frame_index]).reshape(-1)[0])
        for item in tasks:
            if int(item.get("task_index", -1)) == task_index:
                return str(item.get("task", ""))
    return ""


def _plot_overlay(starts: np.ndarray, true_actions: np.ndarray, pred_actions: np.ndarray, output: Path) -> None:
    plt = _import_pyplot()
    dims = min(true_actions.shape[-1], pred_actions.shape[-1], len(ACTION_NAMES))
    fig, axes = plt.subplots(7, 2, figsize=(18, 22), sharex=False)
    axes = axes.ravel()
    for dim in range(dims):
        ax = axes[dim]
        for sample_idx, start in enumerate(starts):
            horizon = min(true_actions.shape[1], pred_actions.shape[1])
            xs = np.arange(start, start + horizon)
            ax.plot(xs, true_actions[sample_idx, :horizon, dim], color="#1f77b4", linewidth=1.0, alpha=0.65)
            ax.plot(xs, pred_actions[sample_idx, :horizon, dim], color="#d62728", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_title(ACTION_NAMES[dim])
        ax.grid(True, alpha=0.25)
    axes[0].plot([], [], color="#1f77b4", label="dataset action")
    axes[0].plot([], [], color="#d62728", linestyle="--", label="policy action")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_error(mae: np.ndarray, rmse: np.ndarray, output: Path) -> None:
    plt = _import_pyplot()
    dims = min(mae.size, len(ACTION_NAMES))
    xs = np.arange(dims)
    width = 0.38
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(xs - width / 2, mae[:dims], width, label="MAE")
    ax.bar(xs + width / 2, rmse[:dims], width, label="RMSE")
    ax.set_xticks(xs, ACTION_NAMES[:dims], rotation=45, ha="right")
    ax.set_ylabel("action error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_metrics(path: Path, mae: np.ndarray, rmse: np.ndarray) -> None:
    dims = min(mae.size, len(ACTION_NAMES))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dim", "name", "mae", "rmse"])
        writer.writeheader()
        for dim in range(dims):
            writer.writerow({"dim": dim, "name": ACTION_NAMES[dim], "mae": mae[dim], "rmse": rmse[dim]})


def _create_policy_client(host: str, port: int):
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "openpi_client is required to call the policy server. Run this script in the same environment "
            "used by the robot inference client / policy server."
        ) from exc
    return websocket_client_policy.WebsocketClientPolicy(host, port)


def _default_output_dir() -> Path:
    return ROOT_DIR / "robodata_studio" / "work" / f"lerobot_policy_eval_{time.strftime('%Y%m%d_%H%M%S')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True, help="LeRobot dataset root, or parent root when --repo_id is set.")
    parser.add_argument("--repo_id", default=None, help="Repo id under dataset_root, optional.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=0, help="Compare this many actions; 0 uses policy output length.")
    parser.add_argument("--prompt", default="", help="Override prompt. Empty means use prompt/task from dataset.")
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--action_key", default="action")
    parser.add_argument(
        "--image_map",
        default=",".join(f"{dst}={src}" for dst, src in DEFAULT_IMAGE_MAP.items()),
        help="Comma list target=source image keys.",
    )
    parser.add_argument("--camera_color_order", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--model_color_order", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _resolve_dataset_root(args.dataset_root, args.repo_id)
    info = _load_json(root / "meta" / "info.json")
    tasks = _load_jsonl(root / "meta" / "tasks.jsonl")
    table = _read_table(_data_path(root, info, args.episode))
    if args.state_key not in table:
        raise KeyError(f"Missing state key: {args.state_key}")
    if args.action_key not in table:
        raise KeyError(f"Missing action key: {args.action_key}")

    image_map = _parse_image_map(args.image_map)
    video_readers = _build_video_readers(root, info, args.episode)
    policy = _create_policy_client(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(table[args.action_key])
    starts = []
    true_chunks = []
    pred_chunks = []
    prompts = []
    try:
        for item in range(args.count):
            frame_index = args.start + item * args.stride
            if frame_index >= total_frames - 1:
                break
            images = {
                target_key: _image_for_key(
                    source_key,
                    frame_index,
                    table,
                    video_readers,
                    args.camera_color_order,
                    args.model_color_order,
                )
                for target_key, source_key in image_map.items()
            }
            prompt = _prompt_for_row(args, table, tasks, frame_index)
            payload = {
                "state": np.asarray(table[args.state_key][frame_index], dtype=np.float32).reshape(-1),
                "images": images,
                "prompt": prompt,
            }
            result = policy.infer(payload)
            pred = np.asarray(result["actions"], dtype=np.float32)
            if pred.ndim != 2:
                raise ValueError(f"Expected policy actions [horizon, dim], got {pred.shape}")
            horizon = int(args.horizon) if args.horizon > 0 else pred.shape[0]
            horizon = min(horizon, pred.shape[0], total_frames - frame_index)
            true = np.asarray(table[args.action_key][frame_index : frame_index + horizon], dtype=np.float32)
            pred = pred[:horizon, : true.shape[-1]]
            true = true[:, : pred.shape[-1]]
            starts.append(frame_index)
            true_chunks.append(true)
            pred_chunks.append(pred)
            prompts.append(prompt)
            print(f"[{len(starts):03d}] frame={frame_index} horizon={horizon} prompt={prompt!r}")
    finally:
        for reader in video_readers.values():
            reader.close()

    if not pred_chunks:
        raise RuntimeError("No samples were evaluated")

    min_horizon = min(chunk.shape[0] for chunk in pred_chunks)
    min_dim = min(chunk.shape[1] for chunk in pred_chunks + true_chunks)
    pred_arr = np.stack([chunk[:min_horizon, :min_dim] for chunk in pred_chunks])
    true_arr = np.stack([chunk[:min_horizon, :min_dim] for chunk in true_chunks])
    starts_arr = np.asarray(starts, dtype=np.int64)
    error = pred_arr - true_arr
    mae = np.mean(np.abs(error), axis=(0, 1))
    rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))

    _plot_overlay(starts_arr, true_arr, pred_arr, output_dir / "trajectory_overlay.png")
    _plot_error(mae, rmse, output_dir / "error_summary.png")
    _write_metrics(output_dir / "metrics.csv", mae, rmse)
    np.savez_compressed(
        output_dir / "predictions.npz",
        starts=starts_arr,
        true_actions=true_arr,
        predicted_actions=pred_arr,
        prompts=np.asarray(prompts),
        mae=mae,
        rmse=rmse,
    )

    print(f"Saved evaluation to: {output_dir}")
    print(f"Mean MAE: {float(np.mean(mae)):.6f}")
    print(f"Mean RMSE: {float(np.mean(rmse)):.6f}")


if __name__ == "__main__":
    main()
