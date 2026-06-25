#!/usr/bin/env python3
"""Inspect, export, and merge pi05 action-expert checkpoint parameters.

Typical workflow:

  # On the training server, inspect or export only the fine-tuned action expert.
  uv run python scripts/pi05_action_expert_checkpoint.py inspect \
    --params checkpoints/<config>/<exp>/<step>/params

  uv run python scripts/pi05_action_expert_checkpoint.py export \
    --params checkpoints/<config>/<exp>/<step>/params \
    --output /tmp/pi05_action_expert_params

  # On the target machine, merge base pi05 VLM weights with exported action expert.
  uv run python scripts/pi05_action_expert_checkpoint.py merge \
    --base-params weights_cache/openpi-assets/checkpoints/pi05_base/params \
    --expert-params /tmp/pi05_action_expert_params \
    --output checkpoints/pi05_base_plus_action_expert/params
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
from collections.abc import Iterable
from typing import Any

from flax import traverse_util
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as _model


# The pi05 model has two Gemma experts inside PaliGemma.llm:
#   expert 0: VLM / language-image prefix, no suffix in module names
#   expert 1: action expert, module names ending in _1
_ACTION_EXPERT_PATTERNS = (
    re.compile(r"^PaliGemma/llm/(.*_1)(/|$)"),
    re.compile(r"^action_in_proj/"),
    re.compile(r"^action_out_proj/"),
    re.compile(r"^time_mlp_in/"),
    re.compile(r"^time_mlp_out/"),
)
_IMAGE_ENCODER_PATTERNS = (re.compile(r"^PaliGemma/img/"),)


def _selected_patterns(*, include_img: bool) -> tuple[re.Pattern[str], ...]:
    if include_img:
        return _ACTION_EXPERT_PATTERNS + _IMAGE_ENCODER_PATTERNS
    return _ACTION_EXPERT_PATTERNS


def _flatten(params: dict[str, Any]) -> dict[str, Any]:
    return traverse_util.flatten_dict(params, sep="/")


def _unflatten(params: dict[str, Any]) -> dict[str, Any]:
    return traverse_util.unflatten_dict(params, sep="/")


def _is_selected_key(key: str, *, include_img: bool) -> bool:
    return any(pattern.search(key) for pattern in _selected_patterns(include_img=include_img))


def _num_bytes(value: Any) -> int:
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    array = np.asarray(value)
    return int(array.nbytes)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _load_params(path: pathlib.Path | str) -> dict[str, Any]:
    try:
        return _model.restore_params(path, restore_type=np.ndarray)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to restore params from {path}. If this is a local Orbax checkpoint, "
            "the checkpoint may be incomplete or corrupted; try a different step or re-copy/re-save it."
        ) from exc


def _save_params(path: pathlib.Path, params: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": params})


def _summarize(flat_params: dict[str, Any], keys: Iterable[str]) -> tuple[int, int]:
    selected = list(keys)
    return len(selected), sum(_num_bytes(flat_params[key]) for key in selected)


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def inspect_params(args: argparse.Namespace) -> None:
    flat_params = _flatten(_load_params(args.params))
    expert_keys = [key for key in flat_params if _is_selected_key(key, include_img=args.include_img)]
    vlm_keys = [key for key in flat_params if not _is_selected_key(key, include_img=args.include_img)]

    total_count, total_size = _summarize(flat_params, flat_params)
    expert_count, expert_size = _summarize(flat_params, expert_keys)
    vlm_count, vlm_size = _summarize(flat_params, vlm_keys)

    print(f"params: {args.params}")
    print(f"total:         {total_count:6d} leaves, {_format_bytes(total_size)}")
    selected_label = "action+img" if args.include_img else "action expert"
    print(f"{selected_label}: {expert_count:6d} leaves, {_format_bytes(expert_size)}")
    print(f"other params:  {vlm_count:6d} leaves, {_format_bytes(vlm_size)}")
    print()
    print(f"sample {selected_label} keys:")
    for key in sorted(expert_keys)[: args.max_keys]:
        value = flat_params[key]
        shape = getattr(value, "shape", "?")
        dtype = getattr(value, "dtype", "?")
        print(f"  {key} shape={shape} dtype={dtype}")


def export_expert(args: argparse.Namespace) -> None:
    flat_params = _flatten(_load_params(args.params))
    expert_flat = {
        key: value for key, value in flat_params.items() if _is_selected_key(key, include_img=args.include_img)
    }
    if not expert_flat:
        raise ValueError(f"No pi05 selected keys matched in {args.params}")

    _save_params(args.output, _unflatten(expert_flat), overwrite=args.overwrite)
    count, size = _summarize(expert_flat, expert_flat)
    label = "action+img" if args.include_img else "action expert"
    print(f"exported {count} {label} leaves ({_format_bytes(size)}) to {args.output}")


def merge_expert(args: argparse.Namespace) -> None:
    base_flat = _flatten(_load_params(args.base_params))
    expert_flat = _flatten(_load_params(args.expert_params))
    expert_flat = {
        key: value for key, value in expert_flat.items() if _is_selected_key(key, include_img=args.include_img)
    }
    if not expert_flat:
        raise ValueError(f"No pi05 selected keys matched in {args.expert_params}")

    missing = sorted(key for key in expert_flat if key not in base_flat)
    if missing:
        preview = "\n".join(f"  {key}" for key in missing[:20])
        raise KeyError(
            "Expert checkpoint contains keys that do not exist in the base params. "
            "The model configs probably differ.\n"
            f"{preview}"
        )

    shape_mismatches = sorted(
        (key, _shape(base_flat[key]), _shape(value))
        for key, value in expert_flat.items()
        if _shape(base_flat[key]) != _shape(value)
    )
    if shape_mismatches:
        preview = "\n".join(
            f"  {key}: base={base_shape}, expert={expert_shape}"
            for key, base_shape, expert_shape in shape_mismatches[:20]
        )
        raise ValueError(
            "Expert checkpoint contains params with shapes that do not match the base params. "
            "The model configs probably differ.\n"
            f"{preview}"
        )

    merged_flat = dict(base_flat)
    merged_flat.update(expert_flat)
    _save_params(args.output, _unflatten(merged_flat), overwrite=args.overwrite)
    count, size = _summarize(expert_flat, expert_flat)
    label = "action+img" if args.include_img else "action expert"
    print(f"merged {count} {label} leaves ({_format_bytes(size)}) into {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--params", required=True)
    inspect_parser.add_argument("--max-keys", type=int, default=30)
    inspect_parser.add_argument("--include-img", action="store_true")
    inspect_parser.set_defaults(func=inspect_params)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--params", required=True)
    export_parser.add_argument("--output", required=True, type=pathlib.Path)
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.add_argument("--include-img", action="store_true")
    export_parser.set_defaults(func=export_expert)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--base-params", required=True)
    merge_parser.add_argument("--expert-params", required=True)
    merge_parser.add_argument("--output", required=True, type=pathlib.Path)
    merge_parser.add_argument("--overwrite", action="store_true")
    merge_parser.add_argument("--include-img", action="store_true")
    merge_parser.set_defaults(func=merge_expert)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
