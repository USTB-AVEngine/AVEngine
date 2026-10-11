"""Resolve reviewed 2D landmark pixels onto the generated mesh with Blender rays."""
import argparse,json,sys
from pathlib import Path
import bpy
from mathutils import Vector,Matrix
p=argparse.ArgumentParser();p.add_argument("--cameras",type=Path,required=True);p.add_argument("--annotations",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index("--")+1:]);meta=json.loads(a.cameras.read_text());marks=json.loads(a.annotations.read_text())
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=meta["input"]);deps=bpy.context.evaluated_depsgraph_get()
rows={}
for name,annotation in marks.items():
    camera=next(r for r in meta["views"] if r["name"]==annotation["view"]);x,y=annotation["pixel"];size=camera["size"]
    inverse=(Matrix(camera["projection_matrix"])@Matrix(camera["camera_world_matrix"]).inverted()).inverted()
    n=inverse@Vector((2*x/size-1,1-2*y/size,-1,1));f=inverse@Vector((2*x/size-1,1-2*y/size,1,1));n=Vector(n[:3])/n.w;f=Vector(f[:3])/f.w
    hit,point,normal,index,obj,matrix=bpy.context.scene.ray_cast(deps,n,(f-n).normalized())
    if not hit:raise RuntimeError(f"Annotation ray missed mesh: {name} {annotation}")
    rows[name]={"annotation":annotation,"point_local_glb":[point.x,point.z,-point.y],"hit_object":obj.name,"hit_polygon":index,"method":"reviewed pixel to native Blender mesh ray; approximate semantic landmark"}
a.output.write_text(json.dumps({"input":meta["input"],"cameras":str(a.cameras),"annotations":str(a.annotations),"landmarks":rows},indent=2)+"\n");print(json.dumps(rows))
