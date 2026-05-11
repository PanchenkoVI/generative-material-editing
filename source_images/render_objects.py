import bpy
import sys
import math
from pathlib import Path
from mathutils import Vector

argv = sys.argv
argv = argv[argv.index("--") + 1:]

material_arg = argv[0]
model_path = Path(argv[1])
output_dir = Path(argv[2])
num_views = int(argv[3])
skip_original = len(argv) > 4 and argv[4] == "--skip_original"

material_path = None if material_arg == "SKIP" else Path(material_arg)
 
def fibonacci_sphere_views(n, phi_min, phi_max):
    views = []
    gr = (1 + math.sqrt(5)) / 2
    span = phi_max - phi_min
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        phi_full = math.acos(y)
        phi = phi_min + (phi_full / math.pi) * span
        theta = (2 * math.pi * i / gr) % (2 * math.pi)
        views.append((theta, phi))
    return views
 
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.wm.obj_import(filepath=str(model_path))

remove_keywords = ["plane", "background", "ground", "floor", "light", "lamp", "env", "sky"]

car_parts = []
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    if any(k in obj.name.lower() for k in remove_keywords):
        bpy.data.objects.remove(obj, do_unlink=True)
    else:
        car_parts.append(obj)

if not car_parts:
    raise RuntimeError("No mesh objects left")

bpy.ops.object.select_all(action='DESELECT')
for o in car_parts:
    o.select_set(True)
bpy.context.view_layer.objects.active = car_parts[0]
bpy.ops.object.join()
obj = bpy.context.active_object

# BOUNDS + CENTER 
bpy.context.view_layer.update()
verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
min_v = Vector((min(v.x for v in verts),
                min(v.y for v in verts),
                min(v.z for v in verts)))
max_v = Vector((max(v.x for v in verts),
                max(v.y for v in verts),
                max(v.z for v in verts)))
size = max(max_v - min_v)

obj.location -= Vector(((min_v.x + max_v.x)/2,
                        (min_v.y + max_v.y)/2,
                        min_v.z))
target_center = Vector((0, 0, (max_v.z - min_v.z) / 2))

# RENDER SETTINGS 
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = True

scene.view_settings.view_transform = "AgX"
scene.view_settings.exposure = -0.5
scene.cycles.sample_clamp_indirect = 1
scene.cycles.sample_clamp_direct = 10
 
# WORLD OFF (тёмный, чтобы не мешал) 
world = scene.world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[1].default_value = 0.05
 
# CAMERA 
bpy.ops.object.camera_add()
camera = bpy.context.active_object
scene.camera = camera 
 
# LIGHT RIG 
power = size * size * 150
def add_area(loc, energy_mult, size_mult):
    bpy.ops.object.light_add(type="AREA")
    l = bpy.context.active_object
    l.location = loc
    l.data.energy = power * energy_mult
    l.data.size = size * size_mult
    return l

# Освещение с уменьшенными тенями
top  = add_area((0, 0, size*3),       0.25, 4)
key  = add_area((size*1.5, -size*1.5, size*1.6), 0.4, 2.5)
fill = add_area((-size*1.5, size*1.5, size*1.2), 0.6, 3)
# Дополнительный свет снизу для подсветки теней (глубокие объекты)
bottom = add_area((0, 0, -size*1.2),   0.25, 3)

# VIEWS
# Ограничиваем углы: от ~60° (сверху) до 85° (чуть выше горизонта),
# чтобы исключить вид снизу и сильные тени.
phi_min = 0.33 * math.pi      # ~60°
phi_max = 0.47 * math.pi      # ~85° (чуть выше горизонта)
views = fibonacci_sphere_views(num_views, phi_min, phi_max)
 
# 1. ОРИГИНАЛ (без материала) 
if not skip_original:
    original_root = Path(__file__).parent.parent / "training_assets" / "original_renders" / model_path.stem
    original_root.mkdir(parents=True, exist_ok=True)

    for i, (theta, phi) in enumerate(views):
        dist = size * 2
        tgt = target_center - Vector((0, 0, size * 0.1))
        camera.location = tgt + Vector((
            dist * math.sin(phi) * math.cos(theta),
            dist * math.sin(phi) * math.sin(theta),
            dist * math.cos(phi)
        ))
        d = tgt - camera.location
        camera.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

        scene.render.filepath = str(original_root / f"view_{i:03d}.png")
        bpy.ops.render.render(write_still=True)

# 2. МАТЕРИАЛ  
if material_path is not None:
    mat = bpy.data.materials.new("SynthMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (200, 0)
    links.new(bsdf.outputs[0], out.inputs[0])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-400, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-200, 0)
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    def tex_node(name, colorspace, y_offset=0):
        p = material_path / name
        if not p.exists():
            return None
        img = bpy.data.images.load(str(p))
        img.colorspace_settings.name = colorspace
        n = nodes.new("ShaderNodeTexImage")
        n.image = img
        n.location = (0, y_offset)
        n.projection = 'BOX'
        n.projection_blend = 0.2
        links.new(mapping.outputs[0], n.inputs["Vector"])
        return n

    bc = tex_node("basecolor.png", 'sRGB', 0)
    nr = tex_node("normal.png", 'Non-Color', -200)
    rg = tex_node("roughness.png", 'Non-Color', -400)
    mt = tex_node("metallic.png", 'Non-Color', -600)

    if bc: links.new(bc.outputs[0], bsdf.inputs["Base Color"])
    if rg: links.new(rg.outputs[0], bsdf.inputs["Roughness"])
    if mt: links.new(mt.outputs[0], bsdf.inputs["Metallic"])
    if nr:
        nm = nodes.new("ShaderNodeNormalMap")
        nm.location = (200, -200)
        links.new(nr.outputs[0], nm.inputs[0])
        links.new(nm.outputs[0], bsdf.inputs["Normal"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (theta, phi) in enumerate(views):
        dist = size * 2
        tgt = target_center - Vector((0, 0, size * 0.1))
        camera.location = tgt + Vector((
            dist * math.sin(phi) * math.cos(theta),
            dist * math.sin(phi) * math.sin(theta),
            dist * math.cos(phi)
        ))
        d = tgt - camera.location
        camera.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

        scene.render.filepath = str(output_dir / f"view_{i:03d}.png")
        bpy.ops.render.render(write_still=True)

print("DONE")