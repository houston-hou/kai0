NAME="funnel2reactor_agilex"
ASSET_ID="funnel2reactor_agilex"

BS="16"
STEP="50"
WORKERS="2"


source .venv/bin/activate
# 配置config 文件 
uv run python scripts/compute_norm_states_fast.py --config-name "$NAME"

#训练
python scripts/train.py "$NAME" --exp_name="$NAME" --batch_size="$BS" --num_train_steps="$STEP" num_workers="$WORKERS"

#导出expert参数
cd /mnt/hdy/kai0

uv run python scripts/pi05_action_expert_checkpoint.py export \
  --params /mnt/hdy/kai0/checkpoints/“$ASSET_ID”/"$NAME"/"$STEP"/params \
  --output /mnt/hdy/kai0/checkpoints/"$ASSET_ID"_expert/params \
  --overwrite

cp assets/"$NAME" /mnt/hdy/kai0/checkpoints/"$ASSET_ID"_expert/assets/

#传expert参数
ssh -N -T -R 2222:10.132.91.10:22 lin
# rsync -avzP --exclude -e "ssh -p 2222" hdy@127.0.0.1:/path/to/src /mnt/hdy/organ_data/
rsync -avzP --exclude="train_state" -e "ssh -p 2222" /mnt/hdy/kai0/checkpoints/"$ASSET_ID"_expert  hdy@127.0.0.1:/home/hdy/checkpoints/

ssh hdy@127.0.0.1 "cd /home/hdy/kai0 && uv run python scripts/pi05_action_expert_checkpoint.py merge \
  --base-params /home/hdy/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --expert-params /home/hdy/checkpoints/"$ASSET_ID"_expert/params \
  --output /home/hdy/checkpoints/"$ASSET_ID"_expert/params_merged \
  --overwrite"








