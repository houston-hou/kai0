# RoboData Studio scripts

This directory contains the Studio server, UI, and the data-processing scripts it invokes.

## Data tools

- `server.py`: local Studio API and static-file server.
- `trim_idle_edges_dataset.py`: trim leading/trailing idle frames and matching video frames.
- `split_lerobot_atomic_actions.py`: export manually labeled `[start, end)` ranges as atomic LeRobot datasets.
- `inference_parquet.py`: reusable writer for inference-request traces.
- `agilex_inference_record_parquet.py`: independent AgileX client-side inference entrypoint with trace recording.
- `evaluate_lerobot_policy_actions.py`: feed LeRobot observations to a running policy server and plot predicted-vs-recorded actions.

## Recorded inference

Install the IPC requirements, including PyArrow, then run from the repository root:

```bash
python robodata_studio/agilex_inference_record_parquet.py \
  --host <policy-server-ip> \
  --port 8000 \
  --prompt "measure liquid" \
  --chunk_size 50 \
  --record_max_steps 1000 \
  --camera_color_order rgb \
  --model_color_order rgb \
  --record_image_format png \
  --record_output /path/to/inference_trace
```

The command stops after `record_max_steps` raw action steps and writes:

```text
inference_trace/
  data/chunk-000/episode_000000.parquet
  meta/info.json
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/tasks.jsonl
```

Register `inference_trace/` in the Studio path input. Each row is one model request and contains the three model-input frames, state, prompt, inference latency, first action, full returned action sequence, and actual published action sequence. PNG is the lossless default; JPEG can reduce file size. The recorder assumes ROS `passthrough` camera images are RGB by default; use `--camera_color_order bgr` only for cameras that publish OpenCV-style BGR frames. The policy input defaults to `--model_color_order rgb`, matching browser-displayed LeRobot training videos; the saved images are encoded from the same model input arrays. Use `--model_color_order bgr` only if the deployed policy was trained with BGR channel order. The `inference_mode` field is only a metadata label, so the recorder is independent of whether the policy server internally uses sync, RTC, or another implementation.

## Offline policy-vs-data evaluation

Start the normal policy server first, then run:

```bash
python robodata_studio/evaluate_lerobot_policy_actions.py \
  --dataset_root /mnt/hdy/emchem_pi05/training_data \
  --repo_id measure_liquid_full_atomic_beaker2cylinder_trimmed \
  --episode 0 \
  --host 127.0.0.1 \
  --port 8000 \
  --start 0 \
  --count 20 \
  --stride 10 \
  --model_color_order rgb
```

The evaluator uses real LeRobot observations as policy input, compares the returned action chunk with the recorded dataset actions starting at the same frame, and writes `trajectory_overlay.png`, `error_summary.png`, `metrics.csv`, and `predictions.npz` under `robodata_studio/work/`.
