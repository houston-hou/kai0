# RoboData Studio scripts

This directory contains the Studio server, UI, and the data-processing scripts it invokes.

## Data tools

- `server.py`: local Studio API and static-file server.
- `trim_idle_edges_dataset.py`: trim leading/trailing idle frames and matching video frames.
- `split_lerobot_atomic_actions.py`: export manually labeled `[start, end)` ranges as atomic LeRobot datasets.
- `inference_parquet.py`: reusable writer for inference-request traces.
- `agilex_inference_record_parquet.py`: independent AgileX client-side inference entrypoint with trace recording.

## Recorded inference

Install the IPC requirements, including PyArrow, then run from the repository root:

```bash
python robodata_studio/agilex_inference_record_parquet.py \
  --host <policy-server-ip> \
  --port 8000 \
  --prompt "measure liquid" \
  --chunk_size 50 \
  --record_max_steps 1000 \
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

Register `inference_trace/` in the Studio path input. Each row is one model request and contains the three model-input frames, state, prompt, inference latency, first action, full returned action sequence, and actual published action sequence. PNG is the lossless default; JPEG can reduce file size. The `inference_mode` field is only a metadata label, so the recorder is independent of whether the policy server internally uses sync, RTC, or another implementation.
