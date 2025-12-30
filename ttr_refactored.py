import bpy, sys, math, addon_utils
from mathutils import Euler

def main(argv):
    """Main function for the Texture Transfer Rigging pipeline."""
    print("=== TTR PIPE STARTED ===")

    # --------------------------------------------------
    # Args
    # blender -b --python ttr.py -- in.glb image.png out.glb baked.png
    # --------------------------------------------------
    MODEL_PATH, IMAGE_PATH, OUT_GLB, OUT_TEX = argv

    # --------------------------------------------------
    # Clean startup
    # --------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    view = bpy.context.view_layer

    # --------------------------------------------------
    # Enable Rigify addon (MUST be before using metarig)
    # --------------------------------------------------
    if not addon_utils.check("rigify")[1]:
        addon_utils.enable("rigify", default_set=True)
        bpy.ops.wm.save_userpref()

    # --------------------------------------------------
    # Import GLB
    # --------------------------------------------------
    bpy.ops.import_scene.gltf(filepath=MODEL_PATH)
    mesh = next(o for o in scene.objects if o.type == 'MESH')

    view.objects.active = mesh
    mesh.select_set(True)

    # Apply transforms to avoid rig issues
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # --------------------------------------------------
    # Camera (front view)
    # --------------------------------------------------
    cam_data = bpy.data.cameras.new("BakeCam")
    cam = bpy.data.objects.new("BakeCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    cam.location = (0, -3, 0)
    cam.rotation_euler = Euler((math.radians(90), 0, 0))

    # --------------------------------------------------
    # Load source image
    # --------------------------------------------------
    src_img = bpy.data.images.load(IMAGE_PATH)

    # --------------------------------------------------
    # Projection material (Camera coords)
    # --------------------------------------------------
    proj_mat = bpy.data.materials.new("ProjectionMat")
    proj_mat.use_nodes = True
    nodes = proj_mat.node_tree.nodes
    links = proj_mat.node_tree.links
    nodes.clear()

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = src_img

    coord = nodes.new("ShaderNodeTexCoord")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")

    links.new(coord.outputs["Camera"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    mesh.data.materials.clear()
    mesh.data.materials.append(proj_mat)

    # --------------------------------------------------
    # UV unwrap (real UVs)
    # --------------------------------------------------
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66)
    bpy.ops.object.mode_set(mode='OBJECT')

    # --------------------------------------------------
    # Bake target image
    # --------------------------------------------------
    baked_img = bpy.data.images.new("BakedDiffuse", 2048, 2048)

    bake_node = nodes.new("ShaderNodeTexImage")
    bake_node.image = baked_img
    nodes.active = bake_node   # CRITICAL

    # --------------------------------------------------
    # Cycles bake setup
    # --------------------------------------------------
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1
    scene.cycles.device = 'CPU'

    # Force camera coords evaluation
    scene.frame_set(1)

    bpy.ops.object.bake(
        type='DIFFUSE',
        pass_filter={'COLOR'},
        margin=4
    )

    baked_img.filepath_raw = OUT_TEX
    baked_img.file_format = 'PNG'
    baked_img.save()

    # --------------------------------------------------
    # Final baked material
    # --------------------------------------------------
    final_mat = bpy.data.materials.new("FinalMat")
    final_mat.use_nodes = True
    fn = final_mat.node_tree.nodes
    fl = final_mat.node_tree.links
    fn.clear()

    ftex = fn.new("ShaderNodeTexImage")
    ftex.image = baked_img
    fbsdf = fn.new("ShaderNodeBsdfPrincipled")
    fout = fn.new("ShaderNodeOutputMaterial")

    fl.new(ftex.outputs["Color"], fbsdf.inputs["Base Color"])
    fl.new(fbsdf.outputs["BSDF"], fout.inputs["Surface"])

    mesh.data.materials.clear()
    mesh.data.materials.append(final_mat)

    # --------------------------------------------------
    # Rigify auto-rig (human)
    # --------------------------------------------------
    bpy.ops.object.armature_human_metarig_add()
    metarig = view.objects.active
    metarig.location = mesh.location

    bpy.ops.object.select_all(action='DESELECT')
    metarig.select_set(True)
    view.objects.active = metarig

    bpy.ops.pose.rigify_generate()

    # Get the generated rig (it's named "rig" by default)
    rig = scene.objects.get("rig")
    if not rig:
        # Fallback to active object
        rig = view.objects.active

    print(f"DEBUG: Rig object = {rig.name if rig else 'NONE'}")
    print(f"DEBUG: Mesh object = {mesh.name}")

    # --------------------------------------------------
    # Bind mesh to rig
    # --------------------------------------------------
    bpy.ops.object.mode_set(mode='OBJECT')  # Ensure object mode
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    rig.select_set(True)
    view.objects.active = rig

    print("DEBUG: Attempting ARMATURE_AUTO parent (this may take time)...")
    print(f"DEBUG: Mesh vertices = {len(mesh.data.vertices)}")

    # ARMATURE_AUTO can be slow but should work
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    print("DEBUG: Parent set complete")

    # --------------------------------------------------
    # Export GLB (force mesh + rig inclusion)
    # --------------------------------------------------
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    rig.select_set(True)
    view.objects.active = rig

    print(f"DEBUG: Exporting to {OUT_GLB}")
    print(f"DEBUG: Selected objects = {[o.name for o in bpy.context.selected_objects]}")

    bpy.ops.export_scene.gltf(
        filepath=OUT_GLB,
        export_format='GLB',
        use_selection=True,
        export_apply=True
    )

    print("🔥 TTR PIPE FINISHED 🔥")

if __name__ == "__main__":
    # To run from command line:
    # blender -b --python ttr_refactored.py -- in.glb image.png out.glb baked.png
    if "--" in sys.argv:
        try:
            argv = sys.argv[sys.argv.index("--") + 1:]
            if len(argv) == 4:
                main(argv)
            else:
                print("ERROR: Incorrect number of arguments.")
                print("Usage: blender -b --python ttr_refactored.py -- <in.glb> <image.png> <out.glb> <baked.png>")
        except ValueError:
            print("ERROR: Could not find '--' separator in arguments.")
    else:
        print("ERROR: Script requires arguments passed after '--'")
        print("Usage: blender -b --python ttr_refactored.py -- <in.glb> <image.png> <out.glb> <baked.png>")