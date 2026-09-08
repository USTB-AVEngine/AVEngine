#!/usr/bin/env python3
"""Refine only v6 living plants and pet beds, retaining full room exports."""
from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).parent))
import refine_room_details as d
import refine_family_interiors as f
import polish_furnished_room as h


def plant_leaf(name, start, direction, length, width, rise, mat, stem_mat, coll):
    """Curved lanceolate lamina, folded across its midrib and slightly rippled."""
    tangent=Vector((math.cos(direction),math.sin(direction),0))
    side=Vector((-tangent.y,tangent.x,0))
    verts=[]; faces=[]; center=[]
    rows,cols=26,10
    for j in range(rows+1):
        t=j/rows
        mid=start+tangent*(length*t)+Vector((0,0,rise*math.sin(math.pi*t*.90)-.095*t*t))
        center.append(mid+Vector((0,0,.002)))
        half=width*math.sin(math.pi*t)**.78
        half=max(half,.0004)
        for k in range(cols+1):
            u=-1+2*k/cols
            # Raised midrib, restrained edge curl, and asymmetric natural waviness.
            z=-.025*(abs(u)**1.4)*math.sin(math.pi*t)+.006*math.sin(t*math.tau*2.5+direction)*u*u
            verts.append(tuple(mid+side*(u*half)+Vector((0,0,z))))
    for j in range(rows):
        for k in range(cols):
            a=j*(cols+1)+k;faces.append((a,a+1,a+cols+2,a+cols+1))
    leaf=d.create_mesh(name,verts,faces,mat,coll)
    f.recalc(leaf);f.finish(leaf)
    solid=leaf.modifiers.new("LeafThickness","SOLIDIFY");solid.thickness=.0012
    rib=d.create_curve_mesh(name+"_Midrib",[tuple(v) for v in center],stem_mat,coll,bevel_depth=.0008)
    f.finish(rib)
    return leaf


def plants():
    coll=d.ensure_collection("Props")
    greens=[d.create_material("LivingProps_Leaf_"+str(i),color,roughness=.43)
            for i,color in enumerate([(.042,.12,.026,1),(.06,.17,.035,1),(.095,.22,.049,1)])]
    stem_mat=d.create_material("LivingProps_Stems",(.10,.18,.045,1),roughness=.57)
    soil_mat=d.create_material("LivingProps_PottingSoil",(.025,.014,.007,1),roughness=.98)
    pots=[o for o in bpy.data.objects if o.type=="MESH" and o.name.startswith("LivingPlantPot")]
    # Delete only the identified source foliage, never unrelated decorative objects.
    for o in list(bpy.data.objects):
        if o.name.startswith(("LivingPlantLeaf_","PolishPlantLeaf_","PolishPlantStem_")):
            bpy.data.objects.remove(o,do_unlink=True)
    result=[]
    for index,pot in enumerate(pots):
        size=d.local_dimensions(pot); radius=size[0]/2;height=size[2]
        center=pot.location.copy();bottom=center.z-height/2;top=center.z+height/2
        mat=pot.data.materials[0]
        # Preserve stable pot ID/location and outer footprint, adding an actual rim/cavity.
        f.lathe(pot,[(.001,-height/2),(.17,-height/2),(.205,-height/2+.025),
                    (radius,height/2-.015),(radius-.004,height/2),
                    (radius-.018,height/2),(radius-.020,height/2-.020),
                    (.175,-height/2+.035),(.001,-height/2+.035)],mat)
        pot.location=center
        soil_z=top-.04
        soil=d.create_cylinder(pot.name+"_Soil",(center.x,center.y,soil_z-.013),radius-.025,.026,soil_mat,coll,vertices=96)
        f.finish(soil)
        # Granular soil surface consists of actual small irregular clods, not a shader.
        for i in range(22):
            angle=i*2.39996;rad=(radius-.045)*math.sqrt((i+.5)/22)
            xyz=(center.x+rad*math.cos(angle),center.y+rad*math.sin(angle),soil_z+.003)
            clod=d.create_cylinder(pot.name+"_SoilClod_"+str(i),xyz,.005+(i%3)*.002,.006,soil_mat,coll,vertices=7)
            f.finish(clod)
        for i in range(13):
            angle=i*2.39996+index*.7
            tall=i%4
            stem_h=.18+.07*tall+.015*math.sin(i*2)
            outward=.045+.012*(i%3)
            stem_start=Vector((center.x+.026*math.cos(angle),center.y+.026*math.sin(angle),soil_z))
            leaf_start=Vector((center.x+outward*math.cos(angle),center.y+outward*math.sin(angle),soil_z+stem_h))
            points=[]
            for j in range(20):
                t=j/19
                p=stem_start.lerp(leaf_start,t)
                p+=Vector((.015*math.cos(angle)*math.sin(math.pi*t),.015*math.sin(angle)*math.sin(math.pi*t),0))
                points.append(tuple(p))
            d.create_curve_mesh(pot.name+"_Petiole_"+str(i),points,stem_mat,coll,bevel_depth=.0025)
            leaf=plant_leaf(pot.name+"_Leaf_"+str(i),leaf_start,angle,
                            .30+.035*(i%4),.055+.011*(i%3),.18+.035*tall,
                            greens[i%3],stem_mat,coll)
            result.append(leaf.name)
    bpy.context.view_layer.update()
    return result


def pet_beds():
    result=[]
    mat=bpy.data.materials.get("Family_sage") or bpy.data.materials.get("Family_sage.001")
    if mat is None:
        mat=bpy.data.materials.get("Family_fabric")
    mat=mat.copy();mat.name="LivingProps_BedFabric"
    normal=mat.node_tree.nodes.get("ScannedNormalMap")
    if normal:normal.inputs["Strength"].default_value=.80
    seam=d.create_material("LivingProps_BedSeam",(.18,.24,.16,1),roughness=.85)
    for obj in [o for o in bpy.data.objects if o.type=="MESH" and o.name.startswith("DogBed")]:
        w,dep,height=d.local_dimensions(obj);n=144
        # Radial closed squircle: low center, soft raised perimeter, rolled sidewall.
        profile=[(.001,.73),(.20,.735),(.40,.75),(.57,.78),(.70,.82),
                 (.78,.87),(.84,.94),(.90,.985),(.95,1.0),
                 (.99,.96),(1.,.80),(1.,.57),(.99,.30),(.965,.10),
                 (.90,.0),(.72,.0),(.4,.0),(.001,.0)]
        verts=[];faces=[]
        for radius,z in profile:
            for i in range(n):
                a=math.tau*i/n
                x=math.copysign(abs(math.cos(a))**.46,math.cos(a))*w/2*radius
                y=math.copysign(abs(math.sin(a))**.46,math.sin(a))*dep/2*radius
                wrinkle=.0035*math.sin(a*17+.3)*math.exp(-((radius-.81)/.13)**2)*math.sin(z*math.pi)
                verts.append((x,y,height*(z-.5)+wrinkle))
        for j in range(len(profile)-1):
            for i in range(n):
                k=(i+1)%n
                faces.append((j*n+i,j*n+k,(j+1)*n+k,(j+1)*n+i))
        faces.extend([tuple(reversed(range(n))),tuple((len(profile)-1)*n+i for i in range(n))])
        new=d.create_mesh(obj.name+"_SoftBolster",verts,faces,mat,obj.users_collection[0])
        f.replace(obj,new);f.recalc(obj);f.finish(obj,.3)
        bpy.context.view_layer.update()
        points=[]
        for i in range(n):
            a=math.tau*i/n
            points.append((math.copysign(abs(math.cos(a))**.46,math.cos(a))*w/2*.995,
                           math.copysign(abs(math.sin(a))**.46,math.sin(a))*dep/2*.995,
                           height*(.44-.5)))
        d.create_pipe_loop(obj.name+"_SideSeam",obj,points,seam,obj.users_collection[0],bevel_depth=.0018)
        result.append(obj.name)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--room",choices=["room_a","room_c"],required=True)
    a=p.parse_args(sys.argv[sys.argv.index("--")+1:])
    source=a.input_root.resolve(strict=True);out=a.output_root.resolve()
    assert not out.exists(),f"refusing existing output {out}"
    blend=next(source.glob("*.blend"))
    sem=json.loads((source/"object_semantics.json").read_text())
    anchors=json.loads((source/"functional_anchors.json").read_text())
    out.mkdir(parents=True)
    for name in ["textures","visual","usd","renders"]:(out/name).mkdir()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    leaves=plants();beds=pet_beds()
    # Keep all existing cameras, lighting and unchanged production geometry.
    for im in bpy.data.images:
        if im.source!="FILE" or not im.filepath:continue
        src=Path(bpy.path.abspath(im.filepath))
        if src.is_file():
            dst=out/"textures"/src.name
            if not dst.exists():shutil.copy2(src,dst)
            im.filepath=str(dst)
    shutil.copy2(source/"material_provenance.json",out/"material_provenance.json")
    cameras=json.loads((source/"review_camera_specs.json").read_text())["cameras"]
    # Only temporary asset-review closeups; no production camera is added or changed.
    close_specs=([( "PLANT",(-1.40,1.25,1.20),(-.20,2.30,.85)),
                  ("BED",(-1.60,2.75,.85),(-.35,4.05,.12))] if a.room=="room_a" else
                 [("PLANT",(-.60,.60,1.20),(.55,1.65,.85)),
                  ("BED_CLEAR",(1.65,2.65,.88),(.65,3.65,.13))])
    scene=bpy.context.scene;scene.render.engine="CYCLES";scene.cycles.device="CPU";scene.cycles.samples=24
    scene.render.threads_mode="FIXED";scene.render.threads=4
    scene.render.resolution_x=960;scene.render.resolution_y=540
    bpy.context.view_layer.update()
    dest=out/(a.room+"_living_props_v7.blend");bpy.ops.wm.save_as_mainfile(filepath=str(dest))
    glb=h.export_selection(out);usd=out/"usd"/(a.room+"_living_props_v7.usda")
    usd_record=h.load_usd_exporter().export_static_usd(scene,usd)
    objects=h.scene_static_mesh_records(scene,"visual/"+glb.name,"usd/"+usd.name)
    for o in objects:
        if o["object_id"].startswith("DiningGlass"):o["category"]="prop"
    by={o["object_id"]:o for o in objects}
    # Ordinary direct comparison proves scope: all untouched object bounds remain.
    changed_prefixes=("LivingPlant","PolishPlant","DogBed")
    for previous in sem["objects"]:
        name=previous["object_id"]
        if name.startswith(changed_prefixes):continue
        assert name in by and by[name]["world_bounds_m"]==previous["world_bounds_m"],name
    sem.update(objects=objects,furniture_objects=[o for o in objects if o["category"] in ["furniture","prop","cabinet","appliance"]],
               detail_profile="living_props_v7_candidate",source_blend=str(blend))
    anchors.update(scene_object_ids=[o["object_id"] for o in objects])
    d.write_json(out/"object_semantics.json",sem);d.write_json(out/"functional_anchors.json",anchors)
    shutil.copy2(source/"lighting.json",out/"lighting.json")
    d.write_json(out/"review_camera_specs.json",dict(status="review_only",cameras=cameras,temporary_inspection_views=[dict(label=label,position_m=pos,target_m=target,production_camera=False) for label,pos,target in close_specs]))
    report=dict(status="research_candidate",source_root=str(source),scope=["living_plants","pet_beds"],
                changed_leaves=leaves,changed_beds=beds,unchanged_bounds_check="pass",seat_points_preserved=True,
                native_execution="not_run_pending_root_choice",artifacts=dict(blend=str(dest),visual_glb=str(glb),usd=usd_record))
    d.write_json(out/"detail_report.json",report);d.write_json(out/"polish_report.json",report)
    # Exported Blend and camera specs retain the exact source camera objects.
    # Reuse one existing review camera transiently after exports for close inspection.
    camera=bpy.data.objects[cameras[0]["camera_id"]]
    scene.camera=camera
    scene.render.filepath=str(out/"renders"/(camera.name+".png"))
    bpy.ops.render.render(write_still=True)
    old_matrix=camera.matrix_world.copy();old_lens=camera.data.lens
    for label,pos,target in close_specs:
        camera.location=pos;camera.data.lens=48;d.look_at(camera,Vector(target))
        scene.render.filepath=str(out/"renders"/("REVIEW_PROPS_"+a.room.upper()+"_"+label+".png"))
        bpy.ops.render.render(write_still=True)
    camera.matrix_world=old_matrix;camera.data.lens=old_lens
    print("LIVING_PROPS_COMPLETE",str(out),flush=True)

if __name__=="__main__":main()
