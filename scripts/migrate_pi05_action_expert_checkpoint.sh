#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Migrate a pi05 action-expert checkpoint to a merged pi05-base checkpoint.

Run this on the target machine that has the trained checkpoint and pi05 base
params. If a reverse SSH tunnel is open, the script can first refresh the
Python merge helper from the laptop-side repository.

Example:
  bash scripts/migrate_pi05_action_expert_checkpoint.sh \
    --trained-checkpoint ~/checkpoints/boil_agilex_action_expert \
    --output-name boil_agilex_action_expert_merged \
    --asset-id boil_agilex

Common options:
  --trained-checkpoint PATH   Trained checkpoint step dir, params dir parent,
                              or experiment dir containing numeric step dirs.
  --output-name NAME          Output directory name under --output-root.
  --asset-id NAME             Required asset id for policy norm stats.
  --include-img               Also migrate PaliGemma/img params.
  --overwrite                 Replace existing export/output dirs.
  --skip-sync                 Do not rsync helper from the reverse tunnel.

Defaults:
  --base-params ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
  --output-root ~/checkpoints
  --repo-dir /mnt/hdy/kai0
  --source-repo /home/hdy/VLA/emchem_pi05
  --source-ssh hdy@127.0.0.1
  --source-port 2222
EOF
}

expand_path() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

require_arg() {
  local name="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "Missing required argument: $name" >&2
    usage >&2
    exit 2
  fi
}

find_checkpoint_step_dir() {
  local input="$1"

  if [[ -d "$input/params" ]]; then
    printf '%s\n' "$input"
    return
  fi
  if [[ "$(basename "$input")" == "params" && -d "$input" ]]; then
    dirname "$input"
    return
  fi

  local latest
  latest="$(
    find "$input" -mindepth 2 -maxdepth 2 -type d -name params -printf '%h\n' 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "$latest" ]]; then
    echo "Could not find a params directory under: $input" >&2
    exit 1
  fi
  printf '%s\n' "$latest"
}

run_python_helper() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    python "$@"
  fi
}

trained_checkpoint=""
output_name=""
asset_id=""
base_params="~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
output_root="~/checkpoints"
repo_dir="/mnt/hdy/kai0"
source_repo="/home/hdy/VLA/emchem_pi05"
source_ssh="hdy@127.0.0.1"
source_port="2222"
assets_source=""
export_root="/tmp"
include_img=0
overwrite=0
skip_sync=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trained-checkpoint|--train-checkpoint)
      trained_checkpoint="$2"
      shift 2
      ;;
    --output-name)
      output_name="$2"
      shift 2
      ;;
    --asset-id)
      asset_id="$2"
      shift 2
      ;;
    --base-params)
      base_params="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --repo-dir)
      repo_dir="$2"
      shift 2
      ;;
    --source-repo)
      source_repo="$2"
      shift 2
      ;;
    --source-ssh)
      source_ssh="$2"
      shift 2
      ;;
    --source-port)
      source_port="$2"
      shift 2
      ;;
    --assets-source)
      assets_source="$2"
      shift 2
      ;;
    --export-root)
      export_root="$2"
      shift 2
      ;;
    --include-img)
      include_img=1
      shift
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    --skip-sync)
      skip_sync=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_arg "--trained-checkpoint" "$trained_checkpoint"
require_arg "--output-name" "$output_name"
require_arg "--asset-id" "$asset_id"

trained_checkpoint="$(expand_path "$trained_checkpoint")"
base_params="$(expand_path "$base_params")"
output_root="$(expand_path "$output_root")"
repo_dir="$(expand_path "$repo_dir")"
export_root="$(expand_path "$export_root")"
if [[ -n "$assets_source" ]]; then
  assets_source="$(expand_path "$assets_source")"
fi

helper="$repo_dir/scripts/pi05_action_expert_checkpoint.py"
step_dir="$(find_checkpoint_step_dir "$trained_checkpoint")"
trained_params="$step_dir/params"
if [[ -z "$assets_source" ]]; then
  assets_source="$step_dir/assets"
fi
output_dir="$output_root/$output_name"
export_dir="$export_root/${output_name}_selected_params"

if [[ ! -d "$base_params" ]]; then
  echo "Base params not found: $base_params" >&2
  exit 1
fi
if [[ ! -d "$trained_params" ]]; then
  echo "Trained params not found: $trained_params" >&2
  exit 1
fi
if [[ ! -d "$assets_source/$asset_id" ]]; then
  echo "Assets not found: $assets_source/$asset_id" >&2
  echo "Pass --assets-source if assets live outside the selected checkpoint step." >&2
  exit 1
fi
if [[ -e "$output_dir" && "$overwrite" -ne 1 ]]; then
  echo "Output already exists: $output_dir. Pass --overwrite to replace it." >&2
  exit 1
fi
if [[ -e "$export_dir" && "$overwrite" -ne 1 ]]; then
  echo "Export dir already exists: $export_dir. Pass --overwrite to replace it." >&2
  exit 1
fi

mkdir -p "$repo_dir/scripts" "$output_root" "$export_root"

if [[ "$skip_sync" -ne 1 ]]; then
  echo "Syncing helper from ${source_ssh}:${source_repo}/scripts/pi05_action_expert_checkpoint.py"
  rsync -avP -e "ssh -p ${source_port}" \
    "${source_ssh}:${source_repo}/scripts/pi05_action_expert_checkpoint.py" \
    "$helper"
fi

include_args=()
if [[ "$include_img" -eq 1 ]]; then
  include_args+=(--include-img)
fi
overwrite_args=()
if [[ "$overwrite" -eq 1 ]]; then
  overwrite_args+=(--overwrite)
  rm -rf "$export_dir" "$output_dir"
fi

echo "Selected checkpoint step: $step_dir"
echo "Exporting selected params to: $export_dir"
run_python_helper "$helper" export \
  --params "$trained_params" \
  --output "$export_dir" \
  "${include_args[@]}" \
  "${overwrite_args[@]}"

echo "Merging into pi05 base params: $output_dir/params"
run_python_helper "$helper" merge \
  --base-params "$base_params" \
  --expert-params "$export_dir" \
  --output "$output_dir/params" \
  "${include_args[@]}" \
  "${overwrite_args[@]}"

echo "Copying assets: $assets_source/$asset_id -> $output_dir/assets/$asset_id"
mkdir -p "$output_dir/assets"
rsync -a --delete "$assets_source/$asset_id/" "$output_dir/assets/$asset_id/"

cat <<EOF
Done.
Merged checkpoint: $output_dir
Use with serve_policy:
  uv run scripts/serve_policy.py policy:checkpoint --policy.config=<matching_config> --policy.dir=$output_dir
EOF
