from __future__ import annotations

import json
import math
import os
import base64
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
from urllib.parse import unquote

from flask import Flask, Response, abort, jsonify, request, send_from_directory


STUDIO_DIR = Path(__file__).resolve().parent
ROOT_DIR = STUDIO_DIR.parent
TRAINING_DATA_ROOT = ROOT_DIR / "training_data"
CHECKPOINTS_ROOT = ROOT_DIR / "checkpoints"
TRIM_SCRIPT = ROOT_DIR / "scripts" / "trim_idle_edges_dataset.py"
ATOMIC_SPLIT_SCRIPT = ROOT_DIR / "scripts" / "split_lerobot_atomic_actions.py"
SINGLE_CONVERTER = ROOT_DIR / "examples" / "aloha_real" / "lzc_mod_convert_aloha_data_to_lerobot_robotwin.py"
MERGED_CONVERTER = ROOT_DIR / "examples" / "aloha_real" / "convert_aloha_data_to_lerobot_robotwin_merged.py"

app = Flask(__name__, static_folder=None)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_pyarrow_parquet():
    ensure_numpy_fallback_path()
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        ensure_numpy_fallback_path()
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            abort(500, description=f"pyarrow unavailable: {exc}")
    return pq


def ensure_numpy_fallback_path() -> None:
    candidate = numpy_fallback_site_packages()
    if candidate and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def numpy_fallback_site_packages() -> Path | None:
    pkgs_dir = Path(sys.prefix) / "pkgs"
    if not pkgs_dir.exists():
        return None
    candidates = sorted(pkgs_dir.glob("numpy-base-*/Lib/site-packages"), reverse=True)
    for candidate in candidates:
        numpy_lib = candidate / "numpy" / "lib"
        if numpy_lib.exists():
            return candidate
    return None


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    candidate = numpy_fallback_site_packages()
    if candidate:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(candidate) if not existing else str(candidate) + os.pathsep + existing
    return env


def feature_keys(info: dict[str, Any], dtype: str | None = None, numeric: bool = False) -> list[str]:
    output: list[str] = []
    for key, feature in (info.get("features") or {}).items():
        feature_dtype = str((feature or {}).get("dtype") or "").lower()
        if dtype is not None and feature_dtype == dtype:
            output.append(key)
        elif numeric and key != "timestamp" and feature_dtype not in {"image", "video"}:
            output.append(key)
    return sorted(output)


def dataset_root(dataset_id: str) -> Path:
    if "/" in dataset_id or "\\" in dataset_id or dataset_id in {"", ".", ".."}:
        abort(400, description="Invalid dataset id")
    root = (TRAINING_DATA_ROOT / dataset_id).resolve()
    allowed_root = TRAINING_DATA_ROOT.resolve()
    if root != allowed_root and allowed_root not in root.parents:
        abort(400, description="Invalid dataset path")
    if not (root / "meta" / "info.json").exists():
        abort(404, description=f"Dataset not found: {dataset_id}")
    return root


def make_episode_name(index: int) -> str:
    return f"episode_{index:06d}.parquet"


def episode_index_from_name(episode_name: str) -> int:
    if not episode_name.startswith("episode_") or not episode_name.endswith(".parquet"):
        abort(400, description="Invalid episode name")
    return int(episode_name.removeprefix("episode_").removesuffix(".parquet"))


def episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    return episode_index // max(int(info.get("chunks_size") or 1000), 1)


def episode_parquet_path(root: Path, info: dict[str, Any], episode_name: str) -> Path:
    episode_index = episode_index_from_name(episode_name)
    chunk = episode_chunk(info, episode_index)
    template = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    return root / template.format(episode_chunk=chunk, episode_index=episode_index)


def video_path(root: Path, info: dict[str, Any], episode_name: str, video_key: str) -> Path:
    episode_index = episode_index_from_name(episode_name)
    chunk = episode_chunk(info, episode_index)
    template = info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return root / template.format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)


def safe_dataset_file(root: Path, relative_path: str) -> Path:
    normalized = (root / relative_path).resolve()
    if root != normalized and root not in normalized.parents:
        abort(400, description="Invalid dataset file path")
    return normalized


def row_value_to_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if hasattr(value, "tolist"):
        payload = value.tolist()
        return [float(item) for item in payload] if isinstance(payload, list) else [float(payload)]
    if value is None:
        return []
    return [float(value)]


def detect_image_mime(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"GIF87a") or payload.startswith(b"GIF89a"):
        return "image/gif"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def image_value_to_payload(root: Path, dataset_id: str, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes:
            raw = bytes(image_bytes)
            mime = detect_image_mime(raw)
            encoded = base64.b64encode(raw).decode("ascii")
            return {"src": f"data:{mime};base64,{encoded}"}
        if image_path:
            return {"src": f"/api/datasets/{dataset_id}/file?path={image_path}"}
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        mime = detect_image_mime(raw)
        encoded = base64.b64encode(raw).decode("ascii")
        return {"src": f"data:{mime};base64,{encoded}"}
    if isinstance(value, str):
        path = safe_dataset_file(root, value)
        if path.exists():
            return {"src": f"/api/datasets/{dataset_id}/file?path={value}"}
    return None


def trim_report_dir(job: dict[str, Any]) -> Path:
    payload = job.get("payload") or {}
    if payload.get("dryRun", True):
        return Path(str(payload["sourceRoot"])) / "trim_idle_edges_dry_run"
    return Path(str(payload["outputRoot"]))


def load_trim_report(job: dict[str, Any]) -> dict[str, Any]:
    report_dir = trim_report_dir(job)
    summary = load_json(report_dir / "trim_summary.json", None)
    markdown_path = report_dir / "trim_report.md"
    csv_path = report_dir / "trim_report.csv"
    return {
        "reportDir": str(report_dir),
        "summary": summary,
        "markdown": markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else "",
        "csv": csv_path.read_text(encoding="utf-8") if csv_path.exists() else "",
    }


def dataset_summary(dataset_id: str) -> dict[str, Any]:
    root = dataset_root(dataset_id)
    info = load_json(root / "meta" / "info.json", {})
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    parquet_count = len(list((root / "data").rglob("episode_*.parquet"))) if (root / "data").exists() else 0
    video_count = len(list((root / "videos").rglob("*.mp4"))) if (root / "videos").exists() else 0
    return {
        "id": dataset_id,
        "label": dataset_id,
        "root": str(root.relative_to(ROOT_DIR)).replace("\\", "/"),
        "absoluteRoot": str(root),
        "fps": info.get("fps"),
        "totalEpisodes": info.get("total_episodes", len(episodes)),
        "totalFrames": info.get("total_frames"),
        "totalTasks": info.get("total_tasks", len(tasks)),
        "totalVideos": info.get("total_videos", video_count),
        "parquetCount": parquet_count,
        "videoCount": video_count,
        "numericKeys": feature_keys(info, numeric=True),
        "videoKeys": feature_keys(info, dtype="video"),
        "imageKeys": feature_keys(info, dtype="image"),
        "updatedAt": root.stat().st_mtime,
    }


def list_datasets() -> list[dict[str, Any]]:
    if not TRAINING_DATA_ROOT.exists():
        return []
    output = []
    for path in sorted(item for item in TRAINING_DATA_ROOT.iterdir() if item.is_dir()):
        if (path / "meta" / "info.json").exists():
            output.append(dataset_summary(path.name))
    return output


def create_job(kind: str, payload: dict[str, Any]) -> str:
    job_id = f"{kind}-{int(time.time() * 1000)}"
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "createdAt": time.time(),
            "startedAt": None,
            "finishedAt": None,
            "payload": payload,
            "command": [],
            "stdout": "",
            "stderr": "",
            "returnCode": None,
        }
    return job_id


def update_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


def run_trim_job(job_id: str, command: list[str]) -> None:
    update_job(job_id, status="running", startedAt=time.time(), command=command)
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            env=subprocess_env(),
            text=True,
            capture_output=True,
            check=False,
        )
        update_job(
            job_id,
            status="succeeded" if completed.returncode == 0 else "failed",
            finishedAt=time.time(),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returnCode=completed.returncode,
        )
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job:
            update_job(job_id, report=load_trim_report(job))
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            finishedAt=time.time(),
            stderr=f"{exc}\n{traceback.format_exc()}",
            returnCode=-1,
        )


def append_job_log(job_id: str, text: str) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["stdout"] = str(_jobs[job_id].get("stdout") or "") + text


def append_job_error(job_id: str, text: str) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["stderr"] = str(_jobs[job_id].get("stderr") or "") + text


def run_command_for_job(job_id: str, command: list[str], *, env: dict[str, str] | None = None) -> int:
    append_job_log(job_id, "\n$ " + " ".join(command) + "\n")
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=env or subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        append_job_log(job_id, completed.stdout)
    if completed.stderr:
        append_job_error(job_id, completed.stderr)
    return int(completed.returncode)


def safe_external_dir(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path.resolve()


def repo_component(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or "task"


def discover_hdf5_tasks(raw_root: Path) -> list[dict[str, Any]]:
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    candidates = [raw_root] if list(raw_root.glob("*.hdf5")) else [item for item in raw_root.iterdir() if item.is_dir()]
    tasks = []
    for task_dir in sorted(candidates):
        hdf5_files = sorted(task_dir.rglob("*.hdf5"))
        if not hdf5_files:
            continue
        instruction_files = [
            name
            for name in ["instruction.json", "instructions.json", "instructions_lzc_mod.json"]
            if (task_dir / name).exists()
        ]
        tasks.append(
            {
                "name": task_dir.name,
                "path": str(task_dir),
                "relativePath": str(task_dir.relative_to(raw_root)) if task_dir != raw_root else ".",
                "hdf5Count": len(hdf5_files),
                "hasInfo": (task_dir / "info.json").exists(),
                "instructionFiles": instruction_files,
            }
        )
    return tasks


def run_hdf5_conversion_job(job_id: str) -> None:
    update_job(job_id, status="running", startedAt=time.time())
    with _jobs_lock:
        payload = dict(_jobs[job_id]["payload"])
    try:
        raw_root = safe_external_dir(str(payload["rawRoot"]))
        repo_prefix = str(payload.get("repoPrefix") or "emchem_atomic").strip()
        selected_tasks = [str(item).strip() for item in payload.get("taskNames") or [] if str(item).strip()]
        conversion_mode = str(payload.get("conversionMode") or payload.get("splitMode") or "separate")
        output_mode = str(payload.get("outputMode") or "video")

        task_infos = discover_hdf5_tasks(raw_root)
        if selected_tasks:
            task_infos = [item for item in task_infos if item["name"] in selected_tasks]
        if not task_infos:
            raise ValueError("No HDF5 task directories selected")

        work_root = STUDIO_DIR / "work" / job_id
        if work_root.exists():
            shutil.rmtree(work_root)
        work_root.mkdir(parents=True, exist_ok=True)

        hf_root = TRAINING_DATA_ROOT
        env = subprocess_env()
        env["HF_LEROBOT_HOME"] = str(hf_root)

        source_dirs = [Path(task["path"]) for task in task_infos]
        output_repos: list[str] = []
        converted_repos: list[str] = []
        if conversion_mode == "merged":
            repo_id = str(payload.get("mergedRepoId") or f"{repo_prefix}_merged").strip()
            cmd = ["uv", "run", str(MERGED_CONVERTER), "--repo_id", repo_id, "--raw_dirs"]
            cmd.extend(str(path) for path in source_dirs)
            cmd.extend(["--mode", output_mode])
            code = run_command_for_job(job_id, cmd, env=env)
            if code != 0:
                raise RuntimeError(f"Merged conversion failed with code {code}")
            converted_repos.append(repo_id)
        else:
            for task, source_dir in zip(task_infos, source_dirs):
                repo_id = f"{repo_prefix}_{repo_component(task['name'])}"
                cmd = [
                    "uv",
                    "run",
                    str(SINGLE_CONVERTER),
                    "--raw_dir",
                    str(source_dir),
                    "--repo_id",
                    repo_id,
                    "--mode",
                    output_mode,
                ]
                code = run_command_for_job(job_id, cmd, env=env)
                if code != 0:
                    raise RuntimeError(f"Conversion failed for {task['name']} with code {code}")
                converted_repos.append(repo_id)

        output_repos.extend(converted_repos)

        update_job(
            job_id,
            status="succeeded",
            finishedAt=time.time(),
            returnCode=0,
            outputRepos=output_repos,
            workRoot=str(work_root),
        )
    except Exception as exc:
        append_job_error(job_id, f"\n{exc}\n{traceback.format_exc()}")
        update_job(job_id, status="failed", finishedAt=time.time(), returnCode=-1)


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def row_vector(row: dict[str, Any], key: str) -> list[float]:
    try:
        return [float(value) for value in row_value_to_list(row.get(key))]
    except Exception:
        return []


def slug_label(value: str, index: int) -> str:
    label = repo_component(value).lower()
    return label if label != "task" else f"atomic_{index + 1:02d}"


def parse_atomic_specs(text: str) -> list[dict[str, str]]:
    specs = []
    for index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            name, prompt = line.split("|", 1)
        elif "\t" in line:
            name, prompt = line.split("\t", 1)
        elif ":" in line:
            name, prompt = line.split(":", 1)
        else:
            prompt = line
            name = slug_label(prompt, index)
        name = slug_label(name.strip(), index)
        prompt = prompt.strip()
        if not prompt:
            prompt = name.replace("_", " ")
        specs.append({"name": name, "prompt": prompt})
    return specs


def suggest_atomic_keyframes(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(payload.get("datasetId") or "").strip()
    if not dataset_id:
        raise ValueError("datasetId is required")
    root = dataset_root(dataset_id)
    info = load_json(root / "meta" / "info.json", {})
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    if not episodes:
        raise ValueError(f"No episodes found in dataset: {dataset_id}")

    state_key = str(payload.get("stateKey") or "observation.state")
    action_key = str(payload.get("actionKey") or "action")
    joint_threshold = float(payload.get("jointThreshold") or 0.035)
    min_gap = max(int(payload.get("minGap") or 20), 1)
    margin = max(int(payload.get("margin") or min_gap), 1)
    home_window = max(int(payload.get("homeWindow") or 10), 1)
    specs = parse_atomic_specs(str(payload.get("subtasks") or payload.get("prompts") or ""))
    if not specs:
        raise ValueError("At least one subtask line is required")
    needed_keyframes = max(len(specs) - 1, 0)

    pq = import_pyarrow_parquet()
    columns = ["frame_index", state_key]
    features = info.get("features") or {}
    if action_key in features:
        columns.append(action_key)

    def score_episode(episode_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
        table = pq.read_table(
            episode_parquet_path(root, info, make_episode_name(episode_index)),
            columns=[column for column in columns if column in features or column == "frame_index"],
        )
        rows = table.to_pylist()
        if not rows:
            return [], [], 0, 0

        state_vectors = [row_vector(row, state_key) for row in rows]
        if not state_vectors or not state_vectors[0]:
            raise ValueError(f"Missing numeric state key: {state_key}")
        width = len(state_vectors[0])
        home_count = min(home_window, len(state_vectors))
        home_pose = [
            sum(vector[dim] for vector in state_vectors[:home_count]) / home_count
            for dim in range(width)
        ]
        action_vectors = [row_vector(row, action_key) for row in rows]

        scored = []
        previous_state = state_vectors[0]
        for index, state in enumerate(state_vectors):
            close_count = sum(1 for dim, value in enumerate(state[:width]) if abs(value - home_pose[dim]) <= joint_threshold)
            home_ratio = close_count / max(width, 1)
            velocity = vector_norm([value - previous_state[dim] for dim, value in enumerate(state[:width])]) if index else 0.0
            action_norm = vector_norm(action_vectors[index]) if index < len(action_vectors) and action_vectors[index] else 0.0
            score = home_ratio - min(velocity, 1.0) * 0.35 - min(action_norm, 1.0) * 0.15
            scored.append(
                {
                    "episodeIndex": episode_index,
                    "frame": index,
                    "score": score,
                    "homeRatio": home_ratio,
                    "stateVelocity": velocity,
                    "actionNorm": action_norm,
                }
            )
            previous_state = state

        all_candidates = [
            item for item in scored[margin : max(margin, len(scored) - margin)]
            if item["homeRatio"] >= float(payload.get("minHomeRatio") or 0.65)
        ]
        all_candidates.sort(key=lambda item: item["score"], reverse=True)

        selected: list[dict[str, Any]] = []
        if needed_keyframes:
            default_radius = max(min_gap, int(len(rows) / max(len(specs) * 4, 1)))
            search_radius = max(int(payload.get("searchRadius") or default_radius), 1)
            for transition_index in range(1, len(specs)):
                target = round(len(rows) * transition_index / len(specs))
                search_start = max(margin, target - search_radius)
                search_end = min(len(rows) - margin, target + search_radius)
                window = [
                    item for item in all_candidates
                    if search_start <= int(item["frame"]) <= search_end
                ]
                if not window:
                    window = [
                        item for item in scored[search_start:search_end]
                        if item["homeRatio"] >= float(payload.get("fallbackHomeRatio") or 0.45)
                    ]
                    window.sort(key=lambda item: item["score"], reverse=True)
                if not window:
                    continue
                chosen = dict(window[0])
                chosen.update(
                    {
                        "transitionIndex": transition_index,
                        "fromTask": specs[transition_index - 1]["name"],
                        "toTask": specs[transition_index]["name"],
                        "fromPrompt": specs[transition_index - 1]["prompt"],
                        "toPrompt": specs[transition_index]["prompt"],
                        "targetFrame": target,
                        "searchStart": search_start,
                        "searchEnd": search_end,
                        "role": "between_actions_home_boundary",
                    }
                )
                selected.append(chosen)
            selected.sort(key=lambda item: item["transitionIndex"])
        return selected, all_candidates[:10], len(rows), home_count

    segments = []
    preview_candidates = []
    processed = 0
    home_count_preview = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        selected, candidates, row_count, home_count = score_episode(episode_index)
        if row_count <= 0:
            continue
        processed += 1
        if not preview_candidates:
            preview_candidates = selected or candidates
            home_count_preview = home_count
        boundaries = [0] + [int(item["frame"]) for item in selected[:needed_keyframes]] + [row_count]
        if len(boundaries) != len(specs) + 1:
            continue
        for index, spec in enumerate(specs):
            segments.append(
                {
                    "label": spec["name"],
                    "episode_index": episode_index,
                    "start": boundaries[index],
                    "end": boundaries[index + 1],
                    "task": spec["prompt"],
                }
            )

    return {
        "datasetId": dataset_id,
        "episodeCount": processed,
        "subtasks": specs,
        "homePoseFrameWindow": home_count_preview,
        "candidates": preview_candidates[:20],
        "segments": segments,
        "labelsJson": json.dumps({"segments": segments}, indent=2, ensure_ascii=False),
    }


def run_atomic_split_job(job_id: str) -> None:
    update_job(job_id, status="running", startedAt=time.time())
    with _jobs_lock:
        payload = dict(_jobs[job_id]["payload"])
    try:
        dataset_id = str(payload.get("datasetId") or "").strip()
        labels_text = str(payload.get("labelsJson") or "").strip()
        if not dataset_id:
            raise ValueError("datasetId is required")
        if not labels_text:
            raise ValueError("labelsJson is required")
        source_root = dataset_root(dataset_id)
        repo_prefix = str(payload.get("repoPrefix") or f"{dataset_id}_atomic").strip()

        work_root = STUDIO_DIR / "work" / job_id
        if work_root.exists():
            shutil.rmtree(work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        labels_path = work_root / "manual_atomic_labels.json"
        labels_path.write_text(labels_text, encoding="utf-8")

        command = [
            sys.executable,
            str(ATOMIC_SPLIT_SCRIPT),
            "--dataset-root",
            str(source_root),
            "--labels-json",
            str(labels_path),
            "--output-root",
            str(TRAINING_DATA_ROOT),
            "--repo-prefix",
            repo_prefix,
            "--overwrite",
        ]
        if not payload.get("splitVideos", True):
            command.append("--no-split-videos")
        if payload.get("videoKeys"):
            command.extend(["--video-keys", str(payload["videoKeys"])])
        code = run_command_for_job(job_id, command)
        summary = load_json(TRAINING_DATA_ROOT / f"{repo_prefix}_atomic_split_summary.json", {})
        update_job(
            job_id,
            status="succeeded" if code == 0 else "failed",
            finishedAt=time.time(),
            returnCode=code,
            report=summary,
            outputRepos=[str(item["dataset"]) for item in summary.get("datasets", []) if item.get("dataset")],
        )
    except Exception as exc:
        append_job_error(job_id, f"\n{exc}\n{traceback.format_exc()}")
        update_job(job_id, status="failed", finishedAt=time.time(), returnCode=-1)


@app.get("/")
@app.get("/robodata_studio/")
def index() -> Response:
    return send_from_directory(STUDIO_DIR, "index.html")


@app.get("/robodata_studio/<path:asset_path>")
def assets(asset_path: str) -> Response:
    return send_from_directory(STUDIO_DIR, asset_path)


@app.get("/styles.css")
@app.get("/app.js")
def root_assets() -> Response:
    return send_from_directory(STUDIO_DIR, request.path.lstrip("/"))


@app.get("/api/datasets")
def api_datasets() -> Response:
    return jsonify({"datasets": list_datasets()})


@app.get("/api/datasets/<dataset_id>/summary")
def api_dataset_summary(dataset_id: str) -> Response:
    return jsonify(dataset_summary(unquote(dataset_id)))


@app.get("/api/datasets/<dataset_id>/episodes")
def api_dataset_episodes(dataset_id: str) -> Response:
    root = dataset_root(unquote(dataset_id))
    info = load_json(root / "meta" / "info.json", {})
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    existing = set(path.name for path in (root / "data").rglob("episode_*.parquet")) if (root / "data").exists() else set()
    for item in episodes:
        item["episode_name"] = make_episode_name(int(item["episode_index"]))
        item["exists"] = item["episode_name"] in existing
    return jsonify({"info": info, "episodes": episodes})


@app.get("/api/datasets/<dataset_id>/tasks")
def api_dataset_tasks(dataset_id: str) -> Response:
    root = dataset_root(unquote(dataset_id))
    return jsonify({"tasks": load_jsonl(root / "meta" / "tasks.jsonl")})


@app.get("/api/datasets/<dataset_id>/episode/<path:episode_name>/series")
def api_episode_series(dataset_id: str, episode_name: str) -> Response:
    root = dataset_root(unquote(dataset_id))
    normalized_episode = unquote(episode_name)
    info = load_json(root / "meta" / "info.json", {})
    numeric_keys = feature_keys(info, numeric=True)
    base_columns = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    columns = [column for column in [*base_columns, *numeric_keys] if column in (info.get("features") or {}) or column in base_columns]
    pq = import_pyarrow_parquet()
    table = pq.read_table(episode_parquet_path(root, info, normalized_episode), columns=columns)
    rows = table.to_pylist()
    feature_data = {key: [row_value_to_list(row.get(key)) for row in rows] for key in numeric_keys if key in table.column_names}
    row_meta = [
        {
            "timestamp": row.get("timestamp"),
            "frame_index": row.get("frame_index", idx),
            "episode_index": row.get("episode_index"),
            "index": row.get("index"),
            "task_index": row.get("task_index"),
        }
        for idx, row in enumerate(rows)
    ]
    return jsonify(
        {
            "rowCount": len(rows),
            "rowMeta": row_meta,
            "featureData": feature_data,
            "numericKeys": list(feature_data),
            "videoKeys": feature_keys(info, dtype="video"),
            "imageKeys": feature_keys(info, dtype="image"),
        }
    )


@app.get("/api/datasets/<dataset_id>/episode/<path:episode_name>/media")
def api_episode_media(dataset_id: str, episode_name: str) -> Response:
    root = dataset_root(unquote(dataset_id))
    normalized_episode = unquote(episode_name)
    info = load_json(root / "meta" / "info.json", {})
    videos = []
    for key in feature_keys(info, dtype="video"):
        path = video_path(root, info, normalized_episode, key)
        videos.append(
            {
                "key": key,
                "exists": path.exists(),
                "path": str(path.relative_to(ROOT_DIR)).replace("\\", "/") if path.exists() else "",
                "url": f"/api/datasets/{dataset_id}/episode/{normalized_episode}/video/{key}",
            }
        )
    return jsonify({"videos": videos})


@app.get("/api/datasets/<dataset_id>/episode/<path:episode_name>/frame/<int:row_index>/images")
def api_episode_frame_images(dataset_id: str, episode_name: str, row_index: int) -> Response:
    normalized_dataset_id = unquote(dataset_id)
    root = dataset_root(normalized_dataset_id)
    normalized_episode = unquote(episode_name)
    info = load_json(root / "meta" / "info.json", {})
    image_keys = feature_keys(info, dtype="image")
    if not image_keys:
        return jsonify({"images": {}})
    pq = import_pyarrow_parquet()
    parquet_path = episode_parquet_path(root, info, normalized_episode)
    pf = pq.ParquetFile(parquet_path)
    batch_size = 64
    skipped = 0
    selected_row: dict[str, Any] | None = None
    for batch in pf.iter_batches(columns=image_keys, batch_size=batch_size):
        records = batch.to_pylist()
        if row_index < skipped + len(records):
            selected_row = records[row_index - skipped]
            break
        skipped += len(records)
    if selected_row is None:
        abort(404, description=f"Row index out of range: {row_index}")
    images = {
        key: image_value_to_payload(root, normalized_dataset_id, selected_row.get(key))
        for key in image_keys
    }
    return jsonify({"images": images})


@app.get("/api/datasets/<dataset_id>/episode/<path:episode_name>/video/<path:video_key>")
def api_episode_video(dataset_id: str, episode_name: str, video_key: str) -> Response:
    root = dataset_root(unquote(dataset_id))
    normalized_episode = unquote(episode_name)
    key = unquote(video_key)
    info = load_json(root / "meta" / "info.json", {})
    path = video_path(root, info, normalized_episode, key)
    if not path.exists():
        abort(404, description=f"Video not found: {key}")
    return send_from_directory(path.parent, path.name, mimetype="video/mp4")


@app.get("/api/datasets/<dataset_id>/file")
def api_dataset_file(dataset_id: str) -> Response:
    root = dataset_root(unquote(dataset_id))
    relative_path = str(request.args.get("path") or "").strip()
    if not relative_path:
        abort(400, description="path is required")
    path = safe_dataset_file(root, relative_path)
    if not path.exists():
        abort(404, description=f"File not found: {relative_path}")
    return send_from_directory(path.parent, path.name)


@app.post("/api/editor/trim-idle")
def api_trim_idle() -> Response:
    if not TRIM_SCRIPT.exists():
        abort(500, description=f"Missing trim script: {TRIM_SCRIPT}")
    payload = request.get_json(force=True, silent=False) or {}
    dataset_id = str(payload.get("datasetId") or "").strip()
    source_root = dataset_root(dataset_id)
    output_name = str(payload.get("outputDataset") or f"{dataset_id}_trimmed").strip()
    if "/" in output_name or "\\" in output_name or output_name in {"", ".", ".."}:
        abort(400, description="Invalid output dataset name")
    output_root = (TRAINING_DATA_ROOT / output_name).resolve()
    if TRAINING_DATA_ROOT.resolve() not in output_root.parents:
        abort(400, description="Invalid output dataset path")

    command = [
        sys.executable,
        str(TRIM_SCRIPT),
        "--dataset-root",
        str(source_root),
        "--output-dataset",
        str(output_root),
        "--action-key",
        str(payload.get("actionKey") or "action"),
        "--state-key",
        str(payload.get("stateKey") or "observation.state"),
        "--action-idle-threshold",
        str(float(payload.get("actionIdleThreshold") or 0.01)),
        "--state-idle-threshold",
        str(float(payload.get("stateIdleThreshold") or 0.002)),
        "--min-edge-idle-frames",
        str(int(payload.get("minEdgeIdleFrames") or 5)),
        "--keep-edge-idle-frames",
        str(int(payload.get("keepEdgeIdleFrames") or 0)),
        "--min-keep-frames",
        str(int(payload.get("minKeepFrames") or 20)),
        "--workers",
        str(int(payload.get("workers") or 1)),
    ]
    if payload.get("alsoRequireStateIdle"):
        command.append("--also-require-state-idle")
    if payload.get("dryRun", True):
        command.append("--dry-run")
    if payload.get("overwrite"):
        command.append("--overwrite")
    if payload.get("resume"):
        command.append("--resume")
    if not payload.get("trimVideos", True):
        command.append("--no-trim-videos")
    if payload.get("videoKeys"):
        command.extend(["--video-keys", str(payload["videoKeys"])])
    if payload.get("maxEpisodes"):
        command.extend(["--max-episodes", str(int(payload["maxEpisodes"]))])

    job_payload = {**payload, "sourceRoot": str(source_root), "outputRoot": str(output_root)}
    job_id = create_job("trim-idle", job_payload)
    thread = threading.Thread(target=run_trim_job, args=(job_id, command), daemon=True)
    thread.start()
    return jsonify({"jobId": job_id, "job": _jobs[job_id]})


@app.post("/api/editor/suggest-keyframes")
def api_suggest_keyframes() -> Response:
    payload = request.get_json(force=True, silent=False) or {}
    try:
        return jsonify(suggest_atomic_keyframes(payload))
    except Exception as exc:
        abort(400, description=str(exc))


@app.post("/api/editor/split-atomic")
def api_split_atomic() -> Response:
    payload = request.get_json(force=True, silent=False) or {}
    if not str(payload.get("datasetId") or "").strip():
        abort(400, description="datasetId is required")
    if not str(payload.get("labelsJson") or "").strip():
        abort(400, description="labelsJson is required")
    job_id = create_job("split-atomic", payload)
    thread = threading.Thread(target=run_atomic_split_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"jobId": job_id, "job": _jobs[job_id]})


@app.post("/api/conversion/discover")
def api_conversion_discover() -> Response:
    payload = request.get_json(force=True, silent=False) or {}
    raw_root_text = str(payload.get("rawRoot") or "").strip()
    if not raw_root_text:
        abort(400, description="rawRoot is required")
    try:
        raw_root = safe_external_dir(raw_root_text)
        tasks = discover_hdf5_tasks(raw_root)
    except Exception as exc:
        abort(400, description=str(exc))
    return jsonify({"rawRoot": str(raw_root), "tasks": tasks})


@app.post("/api/conversion/hdf5-to-lerobot")
def api_hdf5_to_lerobot() -> Response:
    payload = request.get_json(force=True, silent=False) or {}
    if not str(payload.get("rawRoot") or "").strip():
        abort(400, description="rawRoot is required")
    job_id = create_job("hdf5-convert", payload)
    thread = threading.Thread(target=run_hdf5_conversion_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"jobId": job_id, "job": _jobs[job_id]})


@app.get("/api/jobs")
def api_jobs() -> Response:
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda item: item["createdAt"], reverse=True)
    return jsonify({"jobs": jobs})


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        abort(404, description=f"Job not found: {job_id}")
    return jsonify(job)


@app.get("/api/jobs/<job_id>/report")
def api_job_report(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        abort(404, description=f"Job not found: {job_id}")
    if job.get("kind") != "trim-idle":
        abort(400, description="Only trim-idle jobs have reports")
    return jsonify(load_trim_report(job))


@app.get("/api/training/preview-command")
def api_training_preview() -> Response:
    config = request.args.get("config", "pi05_base_aloha_measure_liquid_full")
    dataset = request.args.get("dataset", "<dataset>")
    steps = request.args.get("steps", "30000")
    command = f"uv run scripts/train.py {config} --dataset={dataset} --num_train_steps={steps}"
    return jsonify({"command": command, "checkpointsRoot": str(CHECKPOINTS_ROOT)})


@app.get("/api/inference/preview-command")
def api_inference_preview() -> Response:
    config = request.args.get("config", "pi05_base_aloha_measure_liquid_full")
    checkpoint = request.args.get("checkpoint", "<checkpoint_dir>")
    port = request.args.get("port", "8001")
    command = f"uv run scripts/serve_policy.py --port={port} policy:checkpoint --policy.config={config} --policy.dir={checkpoint}"
    return jsonify({"command": command})


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
def api_error(error: Any) -> Response:
    status = getattr(error, "code", 500)
    return jsonify({"error": getattr(error, "description", str(error)), "status": status}), status


@app.errorhandler(Exception)
def api_unhandled_error(error: Exception) -> Response:
    return jsonify({"error": str(error), "trace": traceback.format_exc(), "status": 500}), 500


if __name__ == "__main__":
    port = int(os.environ.get("ROBODATA_STUDIO_PORT", "8091"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
