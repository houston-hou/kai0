#!/usr/bin/env python3
"""AgileX client-side inference with Parquet request/action recording.

This is a separate entrypoint and does not modify the existing robot inference scripts.
It records the actual client-side observations and returned/published actions regardless
of the policy server implementation behind the websocket endpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from robodata_studio.inference_parquet import InferenceParquetRecorder

import cv2
import numpy as np
import torch

from train_deploy_alignment.inference.agilex.inference import agilex_inference_openpi_sync as base


B2C_LEFT_INIT = [-0.021979, -0.008286, 0.0, -0.108728, 0.341955, 0.155252, -0.0007]
B2C_RIGHT_INIT = [0.090622, -0.011461, 0.000837, -0.111415, 0.306823, -0.011269, -0.00154]


def _recording_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--record_output", default="")
    parser.add_argument("--record_max_steps", type=int, default=1000)
    parser.add_argument("--record_flush_chunks", type=int, default=1)
    parser.add_argument("--record_image_format", choices=["png", "jpeg"], default="png")
    parser.add_argument("--record_jpeg_quality", type=int, default=90)
    parser.add_argument(
        "--camera_color_order",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color channel order returned by ROS cv_bridge passthrough images.",
    )
    parser.add_argument(
        "--model_color_order",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color channel order sent to the policy and recorded to parquet.",
    )
    parser.add_argument(
        "--swap_red_blue_for_model",
        dest="model_color_order",
        action="store_const",
        const="bgr",
        help="Compatibility alias for --model_color_order bgr.",
    )
    parser.add_argument(
        "--no_swap_red_blue_for_model",
        dest="model_color_order",
        action="store_const",
        const="rgb",
        help="Compatibility alias for --model_color_order rgb.",
    )
    parser.add_argument("--prompt", default=base.lang_embeddings)
    parser.add_argument("--inference_mode", default="sync", help="Metadata label for the server/policy mode.")
    return parser.parse_known_args()


def _all_args() -> argparse.Namespace:
    record_args, remaining = _recording_args()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = base.get_arguments()
    finally:
        sys.argv = original_argv
    for key, value in vars(record_args).items():
        setattr(args, key, value)
    if args.record_max_steps <= 0:
        raise ValueError("--record_max_steps must be positive")
    if not 1 <= args.record_jpeg_quality <= 100:
        raise ValueError("--record_jpeg_quality must be in [1, 100]")
    return args


def _build_payload(config: dict, camera_color_order: str, model_color_order: str) -> dict:
    latest = base.observation_window[-1]
    images = [latest["images"][name] for name in config["camera_names"]]
    if camera_color_order == "bgr":
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    else:
        images = [np.asarray(image) for image in images]
    images = base.image_tools.resize_with_pad(np.asarray(images), 224, 224)
    if model_color_order == "bgr":
        images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
    return {
        "state": np.asarray(latest["qpos"], dtype=np.float32),
        "images": {
            "top_head": images[0].transpose(2, 0, 1),
            "hand_right": images[1].transpose(2, 0, 1),
            "hand_left": images[2].transpose(2, 0, 1),
        },
        "prompt": base.lang_embeddings,
    }


def _image_payload(
    images: dict[str, np.ndarray],
    image_format: str,
    jpeg_quality: int,
    model_color_order: str,
) -> dict[str, bytes]:
    output = {}
    for key, chw_image in images.items():
        image = np.asarray(chw_image).transpose(1, 2, 0)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if model_color_order == "rgb" else image
        extension = ".png" if image_format == "png" else ".jpg"
        options = [] if image_format == "png" else [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        ok, encoded = cv2.imencode(extension, bgr, options)
        if not ok:
            raise RuntimeError(f"Failed to encode inference image: {key}")
        output[key] = encoded.tobytes()
    return output


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "inference_logs" / f"agilex_trace_{timestamp}"


def _warmup(args, config, ros_operator, policy) -> None:
    try:
        base.update_observation_window(args, config, ros_operator)
        payload = _build_payload(config, args.camera_color_order, args.model_color_order)
        policy.infer(payload)
        print("Warmup done.")
    except Exception as exc:
        base.rospy.logwarn(f"[startup_warmup] {exc}")


def model_inference_with_recording(args, config, ros_operator) -> Path | None:
    policy = base.websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    left0 = B2C_LEFT_INIT
    right0 = B2C_RIGHT_INIT
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    _warmup(args, config, ros_operator, policy)

    output_root = Path(args.record_output).expanduser() if args.record_output else _default_output()
    recorder = InferenceParquetRecorder(
        output_root,
        fps=args.publish_rate,
        prompt=args.prompt,
        inference_mode=args.inference_mode,
        flush_every_chunks=args.record_flush_chunks,
    )
    max_steps = min(int(args.max_publish_step), int(args.record_max_steps))
    rate = base.rospy.Rate(args.publish_rate)
    pre_action = np.asarray(base.observation_window[-1]["qpos"], dtype=np.float32).copy()
    started_at = time.time()
    step = 0
    parquet_path = None

    try:
        with torch.inference_mode():
            while step < max_steps and not base.rospy.is_shutdown() and not base.shutdown_event.is_set():
                base.update_observation_window(args, config, ros_operator)
                payload = _build_payload(config, args.camera_color_order, args.model_color_order)
                infer_started = time.perf_counter()
                result = policy.infer(payload)
                inference_ms = (time.perf_counter() - infer_started) * 1000.0
                returned_action_chunk = np.asarray(result.get("actions", []), dtype=np.float32)
                if returned_action_chunk.ndim != 2 or returned_action_chunk.shape[1] != config["state_dim"]:
                    raise ValueError(f"Invalid model action chunk shape: {returned_action_chunk.shape}")
                execution_chunk = returned_action_chunk[: config["chunk_size"]]
                if not len(execution_chunk):
                    raise RuntimeError("Model returned an empty action chunk")

                executed_actions = []
                request_step = step
                for chunk_offset, raw_action in enumerate(execution_chunk):
                    if step >= max_steps or base.rospy.is_shutdown() or base.shutdown_event.is_set():
                        break
                    if chunk_offset:
                        base.update_observation_window(args, config, ros_operator)
                    if args.use_actions_interpolation:
                        publish_actions = base.interpolate_action(args, pre_action, raw_action)
                    else:
                        publish_actions = raw_action[np.newaxis, :]
                    for action in publish_actions:
                        if args.ctrl_type != "joint":
                            raise ValueError("Recording entrypoint currently supports --ctrl_type joint only")
                        left_action = action[:7].copy()
                        right_action = action[7:14].copy()
                        right_action[6] = max(0.0, right_action[6] - base.RIGHT_OFFSET)
                        ros_operator.puppet_arm_publish(left_action, right_action)
                        executed_actions.append(np.concatenate([left_action, right_action]).astype(np.float32))
                        rate.sleep()
                    pre_action = raw_action.copy()
                    step += 1
                    print(f"Published Step {step}/{max_steps}")

                recorder.append(
                    request_step=request_step,
                    timestamp=time.time() - started_at,
                    state=payload["state"],
                    image_bytes=_image_payload(
                        payload["images"],
                        args.record_image_format,
                        args.record_jpeg_quality,
                        args.model_color_order,
                    ),
                    action_sequence=returned_action_chunk,
                    executed_actions=np.asarray(executed_actions, dtype=np.float32).reshape(-1, 14),
                    inference_ms=inference_ms,
                )
    finally:
        parquet_path = recorder.close()
        if parquet_path:
            print(f"Inference trace saved: {parquet_path}")
            print(f"Register this dataset in RoboData Studio: {recorder.output_root}")
    return parquet_path


def main() -> None:
    args = _all_args()
    base.lang_embeddings = args.prompt
    ros_operator = base.RosOperator(args)
    if args.seed is not None:
        base.set_seed(args.seed)
    config = base.get_config(args)
    signal.signal(signal.SIGINT, base._on_sigint)
    try:
        model_inference_with_recording(args, config, ros_operator)
    except KeyboardInterrupt:
        base.shutdown_event.set()


if __name__ == "__main__":
    main()
