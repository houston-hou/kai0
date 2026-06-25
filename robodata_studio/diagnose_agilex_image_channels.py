#!/usr/bin/env python3
"""Compare AgileX record/sync image preprocessing on live ROS frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def _wait_image(topic: str, timeout: float) -> tuple[np.ndarray, str]:
    msg = rospy.wait_for_message(topic, Image, timeout=timeout)
    image = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    return np.asarray(image), msg.encoding


def _resize_stack(images: list[np.ndarray]) -> np.ndarray:
    resized = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA) for img in images]
    return np.asarray(resized)


def _save_grid(path: Path, title: str, images: list[np.ndarray], labels: list[str], assume_rgb: bool) -> None:
    panels = []
    for image, label in zip(images, labels, strict=True):
        show = image if assume_rgb else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        show_bgr = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        canvas = np.full((show_bgr.shape[0] + 28, show_bgr.shape[1], 3), 255, dtype=np.uint8)
        canvas[28:, :, :] = show_bgr
        cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(canvas)
    grid = np.concatenate(panels, axis=1)
    cv2.putText(grid, title, (6, grid.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_front_topic", default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", default="/camera_r/color/image_raw")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out_dir", default="/tmp/agilex_channel_diag")
    args = parser.parse_args()

    rospy.init_node("agilex_channel_diag", anonymous=True, disable_signals=True)
    out_dir = Path(args.out_dir)

    topics = [
        ("front", args.img_front_topic),
        ("right", args.img_right_topic),
        ("left", args.img_left_topic),
    ]
    raw_images = []
    encodings = []
    for name, topic in topics:
        image, encoding = _wait_image(topic, args.timeout)
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"{name} image has unsupported shape {image.shape}, encoding={encoding}")
        raw_images.append(image)
        encodings.append(encoding)

    # record default path: camera_color_order=rgb, model_color_order=rgb
    record_rgb = _resize_stack([np.asarray(img) for img in raw_images])
    # sync/smoothing/RTC path: cv2.COLOR_BGR2RGB before resize.
    sync_rgb = _resize_stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in raw_images])

    labels = [f"{name}:{enc}" for (name, _), enc in zip(topics, encodings, strict=True)]
    _save_grid(out_dir / "raw_passthrough_as_rgb.png", "raw passthrough displayed as RGB", raw_images, labels, True)
    _save_grid(out_dir / "record_default_rgb.png", "record default payload", list(record_rgb), labels, True)
    _save_grid(out_dir / "sync_bgr2rgb.png", "sync/smoothing/RTC payload", list(sync_rgb), labels, True)

    same = _mae(record_rgb, sync_rgb)
    swapped = _mae(record_rgb[..., ::-1], sync_rgb)
    print(f"topics={topics}")
    print(f"encodings={encodings}")
    print(f"record_vs_sync_mae={same:.6f}")
    print(f"record_rb_swapped_vs_sync_mae={swapped:.6f}")
    if swapped + 1e-6 < same * 0.1:
        print("JUDGMENT=CHANNEL_MISMATCH record_default and sync are red/blue swapped")
    elif same <= swapped:
        print("JUDGMENT=CHANNEL_MATCH record_default and sync are closer in the same channel order")
    else:
        print("JUDGMENT=AMBIGUOUS check saved images manually")
    print(f"saved_dir={out_dir}")
    time.sleep(0.2)


if __name__ == "__main__":
    main()
