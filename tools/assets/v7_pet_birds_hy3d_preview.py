"""Four cardinal CPU Blender views of a generated bird; record exact camera matrices."""
import argparse,json,sys,math
from pathlib import Path
import bpy
from mathutils import Vector
p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--size",type=int,default=512);p.add_argument("--front-sign",type=int,choices=[-1,1],default=1)
a=p.parse_args(sys.argv[sys.argv.index("--")+1:]);a.output_dir.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
if a.input.suffix.lower()==".glb":bpy.ops.import_scene.gltf(filepath=str(a.input))
else:bpy.ops.wm.obj_import(filepath=str(a.input),forward_axis="Z",up_axis="Y")
meshes=[o for o in bpy.data.objects if o.type=="MESH"]
points=[o.matrix_world@Vector(v) for o in meshes for v in o.bound_box]
lo=Vector([min(v[i] for v in points) for i in range(3)]);hi=Vector([max(v[i] for v in points) for i in range(3)]);center=(lo+hi)/2;extent=max(hi-lo)
for o in meshes:
    if not o.data.materials:
        mat=bpy.data.materials.new("NeutralShape");mat.diffuse_color=(.55,.55,.55,1);o.data.materials.append(mat)
    for mat in o.data.materials:
        if mat:
            mat.use_nodes=True;bs=mat.node_tree.nodes.get("Principled BSDF")
            if bs:
                bs.inputs["Metallic"].default_value=0;bs.inputs["Roughness"].default_value=.85;bs.inputs["Specular IOR Level"].default_value=.15
scene=bpy.context.scene;scene.render.engine="CYCLES";scene.cycles.device="CPU";scene.cycles.samples=16;scene.render.threads_mode="FIXED";scene.render.threads=8
scene.render.resolution_x=scene.render.resolution_y=a.size;scene.render.resolution_percentage=100;scene.render.image_settings.file_format="JPEG";scene.render.image_settings.quality=90
world=bpy.data.worlds.new("PreviewWorld");scene.world=world;world.use_nodes=True;world.node_tree.nodes["Background"].inputs["Color"].default_value=(.65,.65,.65,1);world.node_tree.nodes["Background"].inputs["Strength"].default_value=.5;scene.view_settings.view_transform="Standard";scene.view_settings.look="None"
scene.view_settings.exposure=0;scene.view_settings.gamma=1
for name,loc,power in [("Key",center+Vector((2,-3,4))*extent,600*extent**2),("Fill",center+Vector((-3,2,2))*extent,350*extent**2)]:
    light=bpy.data.lights.new(name,"AREA");light.energy=power;light.shape="DISK";light.size=2*extent;obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);obj.location=loc;obj.rotation_euler=(center-loc).to_track_quat("-Z","Y").to_euler()
camdata=bpy.data.cameras.new("ReviewCamera");cam=bpy.data.objects.new("ReviewCamera",camdata);scene.collection.objects.link(cam);scene.camera=cam;camdata.type="ORTHO";camdata.ortho_scale=extent*1.15;camdata.clip_start=.001;camdata.clip_end=1000
rows=[]
for name,offset in [("front",(0,-3*a.front_sign,0)),("right",(3,0,0)),("rear",(0,3*a.front_sign,0)),("left",(-3,0,0))]:
    cam.location=center+Vector(offset)*extent;cam.rotation_euler=(center-cam.location).to_track_quat("-Z","Y").to_euler()
    path=a.output_dir/(name+".jpg");scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
    rows.append({"name":name,"image":str(path),"camera_world_matrix":list(map(list,cam.matrix_world)),"projection_matrix":list(map(list,cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),x=a.size,y=a.size))),"ortho_scale":camdata.ortho_scale,"size":a.size})
(a.output_dir/"cameras.json").write_text(json.dumps({"input":str(a.input),"blender_bounds_min":list(lo),"blender_bounds_max":list(hi),"renderer":"Blender Cycles CPU; isolated asset preview only","views":rows},indent=2)+"\n")
