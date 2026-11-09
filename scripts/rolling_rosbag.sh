#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${URRC_ROSBAG_DIR:-$HOME/urrc_rosbags}"
MAX_BAGS="${URRC_ROSBAG_MAX:-10}"

mkdir -p "$ROOT_DIR"

if ! [[ "$MAX_BAGS" =~ ^[0-9]+$ ]] || [ "$MAX_BAGS" -lt 1 ]; then
  MAX_BAGS=10
fi

# Oldest first. Delete before starting the new recording so the total count
# never exceeds MAX_BAGS.
mapfile -t EXISTING < <(
  find "$ROOT_DIR" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name 'run_*' \
    -printf '%T@ %p\n' \
  | sort -n \
  | cut -d' ' -f2-
)

while [ "${#EXISTING[@]}" -ge "$MAX_BAGS" ]; do
  OLDEST="${EXISTING[0]}"
  echo "[URRC ROSBAG] Removing oldest recording: $OLDEST"
  rm -rf -- "$OLDEST"
  EXISTING=("${EXISTING[@]:1}")
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/run_$STAMP"

echo "[URRC ROSBAG] Recording: $OUT_DIR"
echo "[URRC ROSBAG] Keeping latest $MAX_BAGS runs"

exec ros2 bag record \
  -o "$OUT_DIR" \
  /scan \
  /rear_scan \
  /imu \
  /camera/start_allowed \
  /camera/start_state \
  /camera/road_center_offset \
  /camera/road_confidence \
  /race/lidar_cmd_vel \
  /race/controller_mode \
  /race/control_authority \
  /race/stop_reason \
  /race/front_clearance_m \
  /race/steering_deg \
  /race/speed_mps \
  /race/preview_angle_deg \
  /race/curve_prediction_deg \
  /race/curve_speed_limit_mps \
  /race/lateral_speed_limit_mps \
  /race/hairpin_direction \
  /race/left_wall_distance_m \
  /race/right_wall_distance_m \
  /race/center_align_active \
  /race/front_vehicle_distance_m \
  /race/front_vehicle_detected \
  /race/sensor_health \
  /model/urrc_car/cmd_vel \
  /model/urrc_car/odometry
