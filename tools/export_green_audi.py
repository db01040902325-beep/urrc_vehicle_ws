import bpy
import math
import os

from mathutils import Vector, Matrix


OUT = os.path.expanduser(
    "~/urrc_vehicle_ws/models/urrc_car/meshes/green_Audi_clean.glb"
)

MAX_X = 0.52
MAX_Y = 0.23
MAX_Z = 0.16


# =========================================================
# 여러 원본 객체의 현재 World Bounds
# =========================================================
def world_bounds(objects):
    points = []

    for obj in objects:
        for corner in obj.bound_box:
            points.append(
                obj.matrix_world @ Vector(corner)
            )

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]

    return (
        min(xs), max(xs),
        min(ys), max(ys),
        min(zs), max(zs)
    )


# =========================================================
# Bake 이후 단일 Mesh의 실제 vertex bounds
# =========================================================
def mesh_bounds(obj):
    verts = [
        v.co
        for v in obj.data.vertices
    ]

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]

    return (
        min(xs), max(xs),
        min(ys), max(ys),
        min(zs), max(zs)
    )


def mesh_dimensions(obj):
    xmin, xmax, ymin, ymax, zmin, zmax = mesh_bounds(obj)

    return (
        xmax - xmin,
        ymax - ymin,
        zmax - zmin
    )


def apply_mesh_transform(obj, matrix):
    obj.data.transform(matrix)
    obj.data.update()
    bpy.context.view_layer.update()


# =========================================================
# 선택한 Audi Mesh
# =========================================================
source_objects = [
    obj
    for obj in bpy.context.selected_objects
    if obj.type == 'MESH'
]

print("")
print("====================================")
print("SOURCE AUDI OBJECTS")
print("====================================")

for obj in source_objects:
    print(obj.name)

print("COUNT =", len(source_objects))

if len(source_objects) < 5:
    raise RuntimeError(
        "Audi Body + Wheels가 모두 선택되어 있어야 합니다."
    )


# =========================================================
# 리그 적용 결과를 실제 Mesh에 Bake
# =========================================================
depsgraph = bpy.context.evaluated_depsgraph_get()

baked_objects = []

collection = bpy.data.collections.new(
    "URRC_EXPORT_TEMP"
)

bpy.context.scene.collection.children.link(
    collection
)

for src in source_objects:

    evaluated = src.evaluated_get(
        depsgraph
    )

    mesh = bpy.data.meshes.new_from_object(
        evaluated,
        depsgraph=depsgraph
    )

    # 현재 Blender에서 보이는 World 위치를 vertex에 Bake
    mesh.transform(
        src.matrix_world
    )

    new_obj = bpy.data.objects.new(
        "BAKED_" + src.name,
        mesh
    )

    collection.objects.link(
        new_obj
    )

    new_obj.parent = None
    new_obj.matrix_world = Matrix.Identity(4)

    baked_objects.append(
        new_obj
    )


# =========================================================
# 하나의 Visual Mesh로 Join
# =========================================================
bpy.ops.object.select_all(
    action='DESELECT'
)

for obj in baked_objects:
    obj.select_set(True)

bpy.context.view_layer.objects.active = baked_objects[0]

bpy.ops.object.join()

car = bpy.context.active_object
car.name = "URRC_GREEN_AUDI"

car.parent = None
car.matrix_world = Matrix.Identity(4)


# =========================================================
# 원본 크기
# =========================================================
dx, dy, dz = mesh_dimensions(car)

print("")
print("BEFORE AXIS CORRECTION")
print(f"X = {dx:.4f}")
print(f"Y = {dy:.4f}")
print(f"Z = {dz:.4f}")


# =========================================================
# 가장 긴 축 = 차량 길이 = Gazebo X
# =========================================================
if dy > dx and dy > dz:

    apply_mesh_transform(
        car,
        Matrix.Rotation(
            math.radians(-90.0),
            4,
            'Z'
        )
    )

elif dz > dx and dz > dy:

    apply_mesh_transform(
        car,
        Matrix.Rotation(
            math.radians(90.0),
            4,
            'Y'
        )
    )


# =========================================================
# 자동차 폭(Y)이 높이(Z)보다 커야 함
# =========================================================
dx, dy, dz = mesh_dimensions(car)

if dz > dy:

    apply_mesh_transform(
        car,
        Matrix.Rotation(
            math.radians(90.0),
            4,
            'X'
        )
    )


# =========================================================
# 차량 규격에 맞춰 균일 축소
# =========================================================
dx, dy, dz = mesh_dimensions(car)

scale = min(
    MAX_X / dx,
    MAX_Y / dy,
    MAX_Z / dz
)

apply_mesh_transform(
    car,
    Matrix.Scale(scale, 4)
)


# =========================================================
# X/Y 중앙 정렬
# 바닥 Z = 0
# =========================================================
xmin, xmax, ymin, ymax, zmin, zmax = mesh_bounds(car)

cx = (xmin + xmax) * 0.5
cy = (ymin + ymax) * 0.5

apply_mesh_transform(
    car,
    Matrix.Translation(
        (-cx, -cy, -zmin)
    )
)


# =========================================================
# PBR 단순화
# Base Color 유지
# =========================================================
materials = set()

for slot in car.material_slots:
    if slot.material:
        materials.add(slot.material)

for mat in materials:

    if not mat.use_nodes:
        continue

    for node in mat.node_tree.nodes:

        if node.type != 'BSDF_PRINCIPLED':
            continue

        metallic = node.inputs.get('Metallic')
        roughness = node.inputs.get('Roughness')

        if metallic is not None:
            for link in list(metallic.links):
                mat.node_tree.links.remove(link)

            metallic.default_value = 0.0

        if roughness is not None:
            for link in list(roughness.links):
                mat.node_tree.links.remove(link)

            roughness.default_value = 0.45


# =========================================================
# 최종 크기
# =========================================================
dx, dy, dz = mesh_dimensions(car)

print("")
print("====================================")
print("FINAL BAKED URRC AUDI")
print("====================================")
print(f"Length X : {dx:.4f} m")
print(f"Width  Y : {dy:.4f} m")
print(f"Height Z : {dz:.4f} m")
print("====================================")

if dx > 0.55:
    raise RuntimeError("차량 길이 제한 초과")

if dy > 0.25:
    raise RuntimeError("차량 폭 제한 초과")

if dz > 0.18:
    raise RuntimeError("차량 높이 제한 초과")

if dx < 0.40:
    raise RuntimeError("차량 길이가 비정상적으로 짧습니다.")


# =========================================================
# Export
# =========================================================
bpy.ops.object.select_all(
    action='DESELECT'
)

car.select_set(True)
bpy.context.view_layer.objects.active = car

os.makedirs(
    os.path.dirname(OUT),
    exist_ok=True
)

bpy.ops.export_scene.gltf(
    filepath=OUT,
    export_format='GLB',
    use_selection=True
)

print("")
print("[PASS] Rig baked")
print("[PASS] Parent removed")
print("[PASS] Armature removed")
print("[PASS] Joined visual mesh")
print("[PASS]", OUT)
