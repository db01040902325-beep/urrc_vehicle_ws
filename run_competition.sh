#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source "$HOME/urrc_2/install/setup.bash"
source "$HOME/urrc_vehicle_ws/install/setup.bash"

exec ros2 launch \
  urrc_lidar_racer \
  competition.launch.py
