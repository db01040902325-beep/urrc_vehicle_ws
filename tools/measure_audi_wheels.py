import bpy
import math
import os
import xml.etree.ElementTree as ET

from mathutils import Vector, Matrix


SDF = os.path.expanduser(
    "~/urrc_vehicle_ws/models/urrc_car/model_f1_lidar_audi_clean.sdf"
)

MAX_X = 0.52
MAX_Y = 0.23
MAX_Z = 0.16


def mesh_world_points(obj, depsgraph):
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()

    pts = [
        ev.matrix_world @ v.co
        for v in mesh.vertices
    ]

    ev.to_mesh_clear()
    return pts


def bounds(points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]

    return (
        min(xs), max(xs),
        min(ys), max(ys),
        min(zs), max(zs)
    )


def dims(points):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(points)

    return (
        xmax - xmin,
        ymax - ymin,
        zmax - zmin
    )


# =========================================================
# 선택된 Audi 객체
# =========================================================

objects = [
    obj
    for obj in bpy.context.selected_objects
    if obj.type == 'MESH'
]

if len(objects) < 5:
    raise RuntimeError(
        "Audi Body + Wheels를 선택한 뒤 저장하십시오."
    )

wheel_objects = [
    obj
    for obj in objects
    if obj.name.startswith("Audi- Wheel.")
]

if len(wheel_objects) != 4:
    print("Detected wheel objects:")
    for obj in wheel_objects:
        print(obj.name)

    raise RuntimeError(
        f"Wheel 객체가 4개여야 합니다. 현재 {len(wheel_objects)}개"
    )

depsgraph = bpy.context.evaluated_depsgraph_get()

object_points = {}

for obj in objects:
    object_points[obj.name] = mesh_world_points(
        obj,
        depsgraph
    )

all_points = [
    p
    for pts in object_points.values()
    for p in pts
]


# =========================================================
# 기존 clean exporter와 동일한 축 보정
# =========================================================

dx, dy, dz = dims(all_points)

axis_matrix = Matrix.Identity(4)

if dy > dx and dy > dz:
    axis_matrix = Matrix.Rotation(
        math.radians(-90.0),
        4,
        'Z'
    )

elif dz > dx and dz > dy:
    axis_matrix = Matrix.Rotation(
        math.radians(90.0),
        4,
        'Y'
    )

rotated = [
    axis_matrix @ p
    for p in all_points
]

dx, dy, dz = dims(rotated)

second_matrix = Matrix.Identity(4)

if dz > dy:
    second_matrix = Matrix.Rotation(
        math.radians(90.0),
        4,
        'X'
    )

rotation = second_matrix @ axis_matrix

rotated = [
    rotation @ p
    for p in all_points
]

dx, dy, dz = dims(rotated)

scale_value = min(
    MAX_X / dx,
    MAX_Y / dy,
    MAX_Z / dz
)

scale_matrix = Matrix.Scale(
    scale_value,
    4
)

scaled = [
    scale_matrix @ p
    for p in rotated
]

xmin, xmax, ymin, ymax, zmin, zmax = bounds(scaled)

cx = (xmin + xmax) * 0.5
cy = (ymin + ymax) * 0.5

center_matrix = Matrix.Translation(
    (-cx, -cy, -zmin)
)

normalization = (
    center_matrix
    @ scale_matrix
    @ rotation
)


# =========================================================
# 현재 Gazebo Audi visual pose 읽기
# =========================================================

tree = ET.parse(SDF)
root = tree.getroot()
model = root.find("model")

chassis = next(
    link
    for link in model.findall("link")
    if link.get("name") == "chassis"
)

audi_visual = next(
    visual
    for visual in chassis.findall("visual")
    if visual.get("name") == "green_audi_clean_visual"
)

pose_text = audi_visual.find("pose").text.strip()

values = [
    float(v)
    for v in pose_text.split()
]

tx, ty, tz, roll, pitch, yaw = values

visual_pose = (
    Matrix.Translation((tx, ty, tz))
    @ Matrix.Rotation(yaw, 4, 'Z')
    @ Matrix.Rotation(pitch, 4, 'Y')
    @ Matrix.Rotation(roll, 4, 'X')
)

final_transform = (
    visual_pose
    @ normalization
)


# =========================================================
# 각 Audi Wheel의 실제 chassis 좌표
# =========================================================

wheels = []

for obj in wheel_objects:

    pts = [
        final_transform @ p
        for p in object_points[obj.name]
    ]

    xmin, xmax, ymin, ymax, zmin, zmax = bounds(pts)

    cx_w = (xmin + xmax) * 0.5
    cy_w = (ymin + ymax) * 0.5
    cz_w = (zmin + zmax) * 0.5

    sx = xmax - xmin
    sy = ymax - ymin
    sz = zmax - zmin

    wheels.append({
        "name": obj.name,
        "x": cx_w,
        "y": cy_w,
        "z": cz_w,
        "sx": sx,
        "sy": sy,
        "sz": sz,
    })


# +X = LiDAR 방향 = FRONT
wheels.sort(
    key=lambda w: w["x"],
    reverse=True
)

front = wheels[:2]
rear = wheels[2:]

front_left = max(
    front,
    key=lambda w: w["y"]
)

front_right = min(
    front,
    key=lambda w: w["y"]
)

rear_left = max(
    rear,
    key=lambda w: w["y"]
)

rear_right = min(
    rear,
    key=lambda w: w["y"]
)


def print_wheel(label, w):
    print(
        f"{label:12s}: "
        f"x={w['x']:+.4f}  "
        f"y={w['y']:+.4f}  "
        f"z={w['z']:+.4f}  "
        f"size=({w['sx']:.4f}, "
        f"{w['sy']:.4f}, "
        f"{w['sz']:.4f})"
    )


print("")
print("==============================================")
print("AUDI WHEEL GEOMETRY IN GAZEBO CHASSIS FRAME")
print("==============================================")

print_wheel("FRONT LEFT", front_left)
print_wheel("FRONT RIGHT", front_right)
print_wheel("REAR LEFT", rear_left)
print_wheel("REAR RIGHT", rear_right)

front_x = (
    front_left["x"]
    + front_right["x"]
) / 2.0

rear_x = (
    rear_left["x"]
    + rear_right["x"]
) / 2.0

wheelbase = abs(
    front_x - rear_x
)

front_track = abs(
    front_left["y"]
    - front_right["y"]
)

rear_track = abs(
    rear_left["y"]
    - rear_right["y"]
)

# 차량이 정상적으로 서 있다면
# wheel axis = Y,
# X/Z가 타이어 직경에 해당
diameters = []

widths = []

for w in wheels:
    diameters.append(
        (w["sx"] + w["sz"]) / 2.0
    )
    widths.append(
        w["sy"]
    )

wheel_radius = (
    sum(diameters)
    / len(diameters)
    / 2.0
)

wheel_width = (
    sum(widths)
    / len(widths)
)

print("")
print("----------------------------------------------")
print(f"Wheelbase       : {wheelbase:.4f} m")
print(f"Front track     : {front_track:.4f} m")
print(f"Rear track      : {rear_track:.4f} m")
print(f"Wheel radius    : {wheel_radius:.4f} m")
print(f"Wheel width     : {wheel_width:.4f} m")
print("----------------------------------------------")

print("")
print("현재 SDF visual pose:")
print(pose_text)
print("")
