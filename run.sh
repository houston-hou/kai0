# uv run python scripts/pi05_action_expert_checkpoint.py merge \
#   --base-params /home/hdy/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
#   --expert-params /home/hdy/checkpoints/boil_distill_place_distillation_rack_agilex_0626_11000_expert/params \
#   --output /home/hdy/checkpoints/boil_distill_place_distillation_rack_agilex_0626_11000_merged/params \
#   --overwrite

  
# uv run python scripts/pi05_action_expert_checkpoint.py merge \
#   --base-params /home/hdy/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
#   --expert-params /home/hdy/checkpoints/boil_distill_return_funnel_to_rack_agilex_0626_11000_expert/params \
#   --output /home/hdy/checkpoints/boil_distill_return_funnel_to_rack_agilex_0626_11000_expert_merged/params \
#   --overwrite

  
uv run python scripts/pi05_action_expert_checkpoint.py merge \
  --base-params /home/hdy/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --expert-params /home/hdy/checkpoints/boil_distill_turn_reactor_knob_agilex_0626_11000_expert/params \
  --output /home/hdy/checkpoints/boil_distill_turn_reactor_knob_agilex_0626_11000_expert_merged/params \
  --overwrite
  