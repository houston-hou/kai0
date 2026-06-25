#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
cd /home/agilex/kai0-main
python3 /tmp/diagnose_agilex_image_channels.py "$@"
