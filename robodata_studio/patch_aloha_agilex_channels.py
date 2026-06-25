#!/usr/bin/env python3
"""Patch robot-side AgileX inference scripts for RGB camera input.

This is intentionally narrow: it only fixes the known aloha runtime issue where
ROS publishes rgb8 images but the inference clients still force BGR->RGB.
"""

from __future__ import annotations

import argparse
from pathlib import Path


RELATIVE_FILES = [
    Path("train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_sync.py"),
    Path("train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing.py"),
    Path("train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_rtc.py"),
]

CANONICAL_CAMERA_NAMES = 'CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]'


def normalize_camera_names(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    camera_def_indices = [
        index for index, line in enumerate(lines) if "CAMERA_NAMES =" in line
    ]
    if not camera_def_indices:
        return text, 0

    first = camera_def_indices[0]
    new_lines = []
    changed = 0
    for index, line in enumerate(lines):
        if index == first:
            if line != CANONICAL_CAMERA_NAMES:
                changed += 1
            new_lines.append(CANONICAL_CAMERA_NAMES)
        elif index in camera_def_indices:
            changed += 1
        else:
            new_lines.append(line)

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + suffix, changed


def patch_text(text: str) -> tuple[str, int]:
    text, count = normalize_camera_names(text)
    replacements = [
        (
            "image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]",
            "image_arrs = [np.asarray(img) for img in image_arrs]",
        ),
        (
            "imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]",
            "imgs = [np.asarray(im) for im in imgs]",
        ),
    ]
    for old, new in replacements:
        occurrences = text.count(old)
        if occurrences:
            text = text.replace(old, new)
            count += occurrences
    return text, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/agilex/kai0-main",
        help="Robot-side kai0-main directory to patch.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    total = 0
    for relpath in RELATIVE_FILES:
        path = root / relpath
        text = path.read_text()
        patched, count = patch_text(text)
        if count:
            path.write_text(patched)
        print(f"{relpath}: replacements={count}")
        total += count
    print(f"total_replacements={total}")


if __name__ == "__main__":
    main()
