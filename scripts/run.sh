# #!/bin/bash
# set -e

# echo "==================== Cloud Training Script ===================="

# # ===================== 0. 项目路径 =====================
# PROJECT_DIR="/mnt/hdy/kai0"
# cd "$PROJECT_DIR"

# echo "PWD=$(pwd)"
# echo "PROJECT_DIR=$PROJECT_DIR"

# # ===================== 1. 缓存和代理 =====================
# rm -rf ~/.cache
# ln -sfn /mnt/.cache ~/.cache || true

# export http_proxy=http://192.168.32.28:18000
# export https_proxy=http://192.168.32.28:18000

# # ===================== 2. 离线 HuggingFace =====================
# # 如果权重和数据都已经在本地，保持离线
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# # ===================== 3. uv / venv 路径 =====================
# # 已经改好了，不是必须，但建议保留，防止后续 uv 又用 /root/.local
# export UV_PYTHON_INSTALL_DIR=/mnt/hdy/uv_python
# export UV_CACHE_DIR=/mnt/hdy/uv_cache

# # ===================== 4. 激活 venv =====================
# source .venv/bin/activate

# PYTHON_BIN="$(which python)"

# echo "Using python:"
# echo "$PYTHON_BIN"

# echo "Resolved python:"
# readlink -f "$PYTHON_BIN"

# "$PYTHON_BIN" --version

# # 强制确认 python 在 /mnt 下，避免再次指向 /root/.local
# RESOLVED_PYTHON="$(readlink -f "$PYTHON_BIN")"
# case "$RESOLVED_PYTHON" in
#     /mnt/*)
#         echo "Python path OK: $RESOLVED_PYTHON"
#         ;;
#     *)
#         echo "ERROR: Python is not under /mnt: $RESOLVED_PYTHON"
#         exit 1
#         ;;
# esac

# # ===================== 5. Python 搜索路径 =====================
# export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/packages/openpi-client/src:$PYTHONPATH"
# echo "PYTHONPATH=$PYTHONPATH"

# # ===================== 6. CUDA / XLA =====================
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# unset XLA_PYTHON_CLIENT_ALLOCATOR
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# # ===================== 7. wandb =====================
# # 如果云端能联网，并且已经配置 WANDB_API_KEY，用 online
# # 如果云端不能联网，改成 offline
# export WANDB_MODE=online
# export WANDB_PROJECT=kai0

# # 不建议把 key 写死在脚本里，最好在平台环境变量里配置
# # 如果必须写在脚本里，再取消注释下一行：
# # export WANDB_API_KEY="你的_wandb_key"

# echo "WANDB_MODE=$WANDB_MODE"
# echo "WANDB_PROJECT=$WANDB_PROJECT"

# # ===================== 8. 调试 import =====================
# echo "==================== Debug Imports ===================="

# "$PYTHON_BIN" -c "import sys; print('python:', sys.executable); print(sys.version)"
# "$PYTHON_BIN" -c "import torch; print('torch:', torch.__version__)"
# "$PYTHON_BIN" -c "import numpy; print('numpy:', numpy.__version__)"
# "$PYTHON_BIN" -c "import openpi; print('openpi ok')"
# "$PYTHON_BIN" -c "import wandb; print('wandb:', wandb.__version__)"


echo "==================== Start Training ===================="


# funnel2reactor_agilex
# boat2balance_agilex
# solid2reactor_agilex
# press_button_agilex
# solid2boat_agilex


NAME="solid2boat_agilex"
ASSET_ID="solid2boat_agilex"
CKPT_STEP="10000"
SSH_PORT="2222"

LOCAL_PROJECT="/mnt/hdy/kai0"
REMOTE_PROJECT="/home/hdy/kai0"

LOCAL_CKPT_DIR="${LOCAL_PROJECT}/checkpoints"
REMOTE_CKPT_DIR="/home/hdy/checkpoints"

cd "$LOCAL_PROJECT"
source .venv/bin/activate

echo "==================== Task: ${NAME} ===================="

# # 配置 config 文件
# python scripts/compute_norm_states_fast.py --config-name "$NAME"

# # 训练
# LD_LIBRARY_PATH="${LOCAL_PROJECT}/ffmpeg_libs:${LD_LIBRARY_PATH:-}" \
# python scripts/train.py "$NAME" \
#   --exp_name="$NAME" \
#   --batch-size=256 \
#   --num_train_steps=11000 \
#   --num_workers=64 \
#   --resume

# # 导出 expert 参数
# python scripts/pi05_action_expert_checkpoint.py export \
#   --params "${LOCAL_CKPT_DIR}/${ASSET_ID}/${NAME}/${CKPT_STEP}/params" \
#   --output "${LOCAL_CKPT_DIR}/${ASSET_ID}_expert/params" \
#   --overwrite

# # 复制 assets
# mkdir -p "${LOCAL_CKPT_DIR}/${ASSET_ID}_expert/assets"

# if [ -d "assets/${NAME}" ]; then
#   cp -a "assets/${NAME}/." "${LOCAL_CKPT_DIR}/${ASSET_ID}_expert/assets/"
# else
#   cp -a "assets/${NAME}" "${LOCAL_CKPT_DIR}/${ASSET_ID}_expert/assets/"
# fi

# 传 expert 参数
# 前提：ssh -N -T -R 2222:10.132.91.10:22 lin 已经建立
ssh -p "$SSH_PORT" hdy@127.0.0.1 "mkdir -p ${REMOTE_CKPT_DIR}"

rsync -avzP \
  --exclude="train_state" \
  -e "ssh -p ${SSH_PORT}" \
  "${LOCAL_CKPT_DIR}/${ASSET_ID}_expert" \
  hdy@127.0.0.1:"${REMOTE_CKPT_DIR}/"

# 在远端 merge
ssh -p "$SSH_PORT" hdy@127.0.0.1 "cd ${REMOTE_PROJECT} && source active_py.sh && source .venv/bin/activate &&  python scripts/pi05_action_expert_checkpoint.py merge \
  --base-params /home/hdy/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --expert-params ${REMOTE_CKPT_DIR}/${ASSET_ID}_expert/params \
  --output ${REMOTE_CKPT_DIR}/${ASSET_ID}_expert/params_merged \
  --overwrite"

echo "==================== Finished: ${NAME} ===================="





