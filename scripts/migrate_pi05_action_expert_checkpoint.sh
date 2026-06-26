#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Export and merge pi05 action-expert checkpoints.

Use --mode export on the machine that holds the trained checkpoint, then copy the
small export package to the machine that holds pi05_base, and use --mode merge
there. Use --mode full only when trained checkpoint and pi05_base are on the same
machine.

Export on lin:
  bash scripts/migrate_pi05_action_expert_checkpoint.sh \
    --mode export \
    --trained-checkpoint /mnt/hdy/kai0/checkpoints/beaker2cylinder_agilex \
    --output-name beaker2cylinder_agilex_export \
    --asset-id beaker2cylinder_agilex \
    --include-img \
    --overwrite

Copy export package from lin to gxl, for example:
  rsync -avP -e "ssh -p 2222" \
    /mnt/hdy/kai0/checkpoints/beaker2cylinder_agilex_export \
    hdy@127.0.0.1:~/checkpoints/

Merge on gxl:
  bash scripts/migrate_pi05_action_expert_checkpoint.sh \
    --mode merge \
    --export-package ~/checkpoints/beaker2cylinder_agilex_export \
    --output-name beaker2cylinder_agilex_merged \
    --asset-id beaker2cylinder_agilex \
    --include-img \
    --overwrite \
    --skip-sync

Options:
  --mode MODE                 export, merge, or full. Default: full.
  --trained-checkpoint PATH   Trained checkpoint step dir, params dir parent,
                              or experiment dir containing numeric step dirs.
  --export-package PATH       Export package dir. In export mode this is the
                              output package; in merge mode this is the input.
  --output-name NAME          Output directory name under --output-root.
  --asset-id NAME             Asset id for policy norm stats.
  --include-img               Also migrate PaliGemma/img params.
  --overwrite                 Replace existing export/output dirs.
  --skip-sync                 Do not rsync helper from the reverse tunnel.

Defaults:
  --base-params /home/hdy/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
  --output-root /mnt/hdy/kai0/checkpoints in export mode, ~/checkpoints otherwise
  --repo-dir /mnt/hdy/kai0
  --source-repo /home/hdy/VLA/emchem_pi05
  --source-ssh hdy@127.0.0.1
  --source-port 2222
EOF
}

expand_path() {
  local path="$1"
  local home_tilde="$HOME/~/"
  if [[ "$path" == "~" ]]; then
    printf '%s\n' "$HOME"
  elif [[ "$path" == "~/"* ]]; then
    printf '%s/%s\n' "$HOME" "${path#~/}"
  elif [[ "$path" == "$HOME/~" ]]; then
    printf '%s\n' "$HOME"
  elif [[ "$path" == "$HOME/~/"* ]]; then
    printf '%s/%s\n' "$HOME" "${path#$home_tilde}"
  else
    printf '%s\n' "$path"
  fi
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

sync_helper() {
  if [[ "$skip_sync" -eq 1 ]]; then
    return
  fi
  mkdir -p "$repo_dir/scripts"
  echo "Syncing helper from ${source_ssh}:${source_repo}/scripts/pi05_action_expert_checkpoint.py"
  rsync -avP -e "ssh -p ${source_port}" \
    "${source_ssh}:${source_repo}/scripts/pi05_action_expert_checkpoint.py" \
    "$helper"
}

remove_if_overwrite() {
  local path="$1"
  if [[ -e "$path" ]]; then
    if [[ "$overwrite" -ne 1 ]]; then
      echo "Path already exists: $path. Pass --overwrite to replace it." >&2
      exit 1
    fi
    rm -rf "$path"
  fi
}

copy_assets() {
  local src_assets="$1"
  local dst_root="$2"
  if [[ ! -d "$src_assets/$asset_id" ]]; then
    echo "Assets not found: $src_assets/$asset_id" >&2
    echo "Pass --assets-source if assets live outside the selected checkpoint step." >&2
    exit 1
  fi
  mkdir -p "$dst_root/assets"
  rsync -a --delete "$src_assets/$asset_id/" "$dst_root/assets/$asset_id/"
}

mode="full"
trained_checkpoint=""
export_package=""
output_name=""
asset_id=""
base_params="/home/hdy/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
output_root=""
repo_dir="/mnt/hdy/kai0"
source_repo="/home/hdy/VLA/emchem_pi05"
source_ssh="hdy@127.0.0.1"
source_port="2222"
assets_source=""
include_img=0
overwrite=0
skip_sync=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --trained-checkpoint|--train-checkpoint)
      trained_checkpoint="$2"
      shift 2
      ;;
    --export-package)
      export_package="$2"
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

case "$mode" in
  export|merge|full) ;;
  *)
    echo "Invalid --mode: $mode. Expected export, merge, or full." >&2
    exit 2
    ;;
esac

base_params="$(expand_path "$base_params")"
if [[ -z "$output_root" ]]; then
  if [[ "$mode" == "export" ]]; then
    output_root="/mnt/hdy/kai0/checkpoints"
  else
    output_root="~/checkpoints"
  fi
fi
output_root="$(expand_path "$output_root")"
repo_dir="$(expand_path "$repo_dir")"
helper="$repo_dir/scripts/pi05_action_expert_checkpoint.py"
if [[ -n "$trained_checkpoint" ]]; then
  trained_checkpoint="$(expand_path "$trained_checkpoint")"
fi
if [[ -n "$export_package" ]]; then
  export_package="$(expand_path "$export_package")"
fi
if [[ -n "$assets_source" ]]; then
  assets_source="$(expand_path "$assets_source")"
fi

include_args=()
if [[ "$include_img" -eq 1 ]]; then
  include_args+=(--include-img)
fi
overwrite_args=()
if [[ "$overwrite" -eq 1 ]]; then
  overwrite_args+=(--overwrite)
fi

if [[ "$mode" == "export" || "$mode" == "full" ]]; then
  require_arg "--trained-checkpoint" "$trained_checkpoint"
  require_arg "--asset-id" "$asset_id"
  if [[ -z "$export_package" ]]; then
    if [[ "$mode" == "full" ]]; then
      require_arg "--output-name" "$output_name"
      export_package="${TMPDIR:-/tmp}/${output_name}_export_package"
    else
      require_arg "--output-name" "$output_name"
      export_package="$output_root/$output_name"
    fi
  fi

  step_dir="$(find_checkpoint_step_dir "$trained_checkpoint")"
  trained_params="$step_dir/params"
  if [[ -z "$assets_source" ]]; then
    assets_source="$step_dir/assets"
  fi
  if [[ ! -d "$trained_params" ]]; then
    echo "Trained params not found: $trained_params" >&2
    exit 1
  fi

  sync_helper
  remove_if_overwrite "$export_package"
  mkdir -p "$export_package"

  echo "Selected checkpoint step: $step_dir"
  echo "Exporting selected params to: $export_package/params"
  run_python_helper "$helper" export \
    --params "$trained_params" \
    --output "$export_package/params" \
    "${include_args[@]}" \
    "${overwrite_args[@]}"

  echo "Copying assets: $assets_source/$asset_id -> $export_package/assets/$asset_id"
  copy_assets "$assets_source" "$export_package"
  printf '%s\n' "$step_dir" > "$export_package/source_step.txt"

  if [[ "$mode" == "export" ]]; then
    cat <<EOF
Done.
Export package: $export_package
Copy this directory to the pi05-base machine, then run --mode merge.
EOF
    exit 0
  fi
fi

if [[ "$mode" == "merge" || "$mode" == "full" ]]; then
  require_arg "--asset-id" "$asset_id"
  require_arg "--output-name" "$output_name"
  if [[ -z "$export_package" && "$mode" == "merge" ]]; then
    echo "Missing required argument: --export-package" >&2
    usage >&2
    exit 2
  fi
  output_dir="$output_root/$output_name"

  if [[ ! -d "$base_params" ]]; then
    echo "Base params not found: $base_params" >&2
    exit 1
  fi
  if [[ ! -d "$export_package/params" ]]; then
    echo "Exported params not found: $export_package/params" >&2
    exit 1
  fi
  if [[ ! -d "$export_package/assets/$asset_id" ]]; then
    echo "Exported assets not found: $export_package/assets/$asset_id" >&2
    exit 1
  fi

  sync_helper
  remove_if_overwrite "$output_dir"

  echo "Merging into pi05 base params: $output_dir/params"
  run_python_helper "$helper" merge \
    --base-params "$base_params" \
    --expert-params "$export_package/params" \
    --output "$output_dir/params" \
    "${include_args[@]}" \
    "${overwrite_args[@]}"

  echo "Copying assets: $export_package/assets/$asset_id -> $output_dir/assets/$asset_id"
  copy_assets "$export_package/assets" "$output_dir"

  cat <<EOF
Done.
Merged checkpoint: $output_dir
Use with serve_policy:
  uv run scripts/serve_policy.py policy:checkpoint --policy.config=<matching_config> --policy.dir=$output_dir
EOF
fi
