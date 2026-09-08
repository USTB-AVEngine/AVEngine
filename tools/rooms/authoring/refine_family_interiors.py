#!/usr/bin/env python3
"""Refine editable A/C family furniture; reuse the shared room exporters.

Review cameras are asset inspection only. Source seat centers, tops and fronts
are retained; production cameras, people, sound and QA remain AVEngine-owned.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
from pathlib import Path
import bpy
import bmesh
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import refine_room_details as d
import polish_furnished_room as h


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def finish(obj, tile=None):
    for p in obj.data.polygons:
        p.use_smooth = True
    if tile:
        d.projected_uvs(obj, tile)
    return obj


def replace(obj, new):
    old = obj.data
    obj.data = new.data
    obj.modifiers.clear()
    bpy.data.objects.remove(new, do_unlink=True)
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj


def recalc(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()


def pillow(obj, size, mat, exponent=.42):
    """Closed superellipsoid with continuously rounded, crowned surfaces."""
    verts, faces = [], []
    def power(x):
        return math.copysign(abs(x) ** exponent, x)
    nu, nv = 96, 48
    for j in range(1, nv):
        v = -math.pi/2 + math.pi*j/nv
        for i in range(nu):
            u = 2*math.pi*i/nu
            verts.append((size[0]/2*power(math.cos(v))*power(math.cos(u)),
                          size[1]/2*power(math.cos(v))*power(math.sin(u)),
                          size[2]/2*power(math.sin(v))))
    for j in range(nv-2):
        for i in range(nu):
            ni = (i+1)%nu
            faces.append((j*nu+i,j*nu+ni,(j+1)*nu+ni,(j+1)*nu+i))
    lo, hi = len(verts), len(verts)+1
    verts.extend([(0,0,-size[2]/2),(0,0,size[2]/2)])
    for i in range(nu):
        ni=(i+1)%nu
        faces.extend([(lo,ni,i),(hi,(nv-2)*nu+i,(nv-2)*nu+ni)])
    new=d.create_mesh(obj.name+"_Soft",verts,faces,mat,obj.users_collection[0])
    replace(obj,new)
    recalc(obj)
    finish(obj,.30)
    return obj


def piping(obj, size, mat, vertical=False):
    # Seam follows the same superellipse at its equator, below the seat top.
    bpy.context.view_layer.update()
    pts=[]
    for i in range(128):
        t=math.tau*i/128
        x=math.copysign(abs(math.cos(t))**.42,math.cos(t))
        y=math.copysign(abs(math.sin(t))**.42,math.sin(t))
        pts.append((x*size[0]*.497,0,y*size[2]*.497) if vertical else
                   (x*size[0]*.497,y*size[1]*.497,0))
    seam=d.create_pipe_loop(obj.name+"_SewnSeam",obj,pts,mat,obj.users_collection[0],bevel_depth=.0018)
    # Existing helper handles yaw; include any deliberate pillow lean.
    seam.rotation_euler=obj.rotation_euler.copy()
    finish(seam,.30)


def materials(root,out):
    mats={}
    for key,folder,tile in [("wood","walnut_veneer_02_2k",1.),("floor","oak_wood_planks_2k",1.2),("fabric","terlenka_2k",.3),("wall","white_plaster_02_2k",1.)]:
        maps=[]
        for name,ext,nc in [("basecolor","jpg",False),("roughness","exr",True),("normal_gl","exr",True)]:
            maps.append(d.load_source_image(root/folder/(name+"."+ext),out/"textures"/(key+"_"+name+(".png" if nc else ".jpg")),noncolor=nc))
        mat=d.create_material("Family_"+key,(.6,.6,.6,1),roughness=.65)
        d.connect_pbr_texture(mat,*maps,roughness=.7)
        mat["physical_uv_tile_m"]=tile
        mats[key]=mat
    # Tint is baked into pixels, never a non-exportable multiply node.
    import numpy as np
    for key,tint in [("sage",(.52,.65,.58)),("ochre",(.68,.41,.22)),("wood",(.47,.34,.23))]:
        source_mat=mats["wood"] if key=="wood" else mats["fabric"]
        original=source_mat.node_tree.nodes.get("ScannedBaseColor").image
        pixels=np.empty(len(original.pixels),dtype=np.float32)
        original.pixels.foreach_get(pixels)
        pixels=pixels.reshape((-1,4))
        pixels[:,:3]*=np.array(tint,dtype=np.float32)
        im=bpy.data.images.new("Family_"+key+"_baked",width=original.size[0],height=original.size[1])
        im.pixels.foreach_set(pixels.reshape(-1))
        im.filepath_raw=str(out/"textures"/(key+"_fabric.png")); im.file_format="PNG"; im.save()
        mat=source_mat.copy(); mat.name="Family_"+key
        mat.node_tree.nodes.get("ScannedBaseColor").image=im
        mats[key]=mat
    wall=mats["wall"]; shader=wall.node_tree.nodes.get("Principled BSDF")
    for link in list(shader.inputs["Base Color"].links):wall.node_tree.links.remove(link)
    shader.inputs["Base Color"].default_value=(.66,.63,.56,1)
    wall.node_tree.nodes.get("ScannedNormalMap").inputs["Strength"].default_value=.12
    mats["ceramic"]=d.create_material("Family_CreamStoneware",(.72,.68,.56,1),roughness=.29)
    mats["seam"]=d.create_material("Family_LinenSeam",(.36,.33,.27,1),roughness=.85)
    mats["paper"]=d.create_material("Family_BookPaper",(.75,.71,.60,1),roughness=.85)
    mats["cover"]=d.create_material("Family_BookCloth",(.14,.23,.20,1),roughness=.77)
    mats["metal"]=d.create_material("Family_AntiqueBrass",(.26,.19,.10,1),roughness=.36,metallic=.7)
    for obj in list(bpy.data.objects):
        if obj.type!="MESH" or not obj.data.materials: continue
        names=" ".join(m.name.lower() for m in obj.data.materials if m)
        key=None
        if any(t in names for t in ["oak","walnut","wood"]): key="floor" if obj.name.lower().startswith("floor") else "wood"
        if "plaster" in names: key="wall"
        if any(t in names for t in ["fabric","sofa","pillow"]): key="fabric"
        if key:
            assign(obj,mats[key]); d.projected_uvs(obj,float(mats[key].get("physical_uv_tile_m",.3)))
    return mats


def soft_furniture(m):
    for obj in list(bpy.data.objects):
        n=obj.name
        if obj.type!="MESH": continue
        size=d.local_dimensions(obj)
        if n.startswith("SofaCushion"):
            pillow(obj,size,m["fabric"]); piping(obj,size,m["seam"])
        elif n.startswith("PolishSofaBackCushion"):
            # Match each back to the actual seat, including the wide C sofa.
            group,index=map(int,n.rsplit("_",2)[1:])
            suffix="" if group==0 else ".001"
            seat=bpy.data.objects.get("SofaCushion_"+str(index)+suffix)
            if seat:
                sz=d.local_dimensions(seat)
                obj.location.x=seat.location.x if abs(math.sin(obj.rotation_euler.z))<.5 else obj.location.x
                obj.location.y=seat.location.y if abs(math.sin(obj.rotation_euler.z))>.5 else obj.location.y
                size=(sz[0]*.97,.25,.57)
            obj.location.z=.57+size[2]/2-.015
            pillow(obj,size,m["fabric"],.46); piping(obj,size,m["seam"],True)
        elif n.startswith("PolishSofaPillow"):
            obj.location.z=.79
            # Move toward the seat front so the throw pillow rests against the back.
            obj.location += obj.rotation_euler.to_matrix() @ Vector((0,-.14,0))
            obj.rotation_euler.x=math.radians(-9)
            obj.rotation_euler.y=math.radians(5 if n.endswith("0") else -6)
            pillow(obj,(.42,.21,.43),m["sage"] if n.endswith("0") else m["ochre"],.55)
            piping(obj,(.42,.21,.43),m["seam"],True)
        elif n.startswith("SofaBase"):
            new=d.create_box(n+"_Frame",(0,0,0),(size[0],size[1],.21),m["wood"],obj.users_collection[0],bevel=.022)
            replace(obj,new); obj.location.z=.285
            bevel=obj.modifiers.new("FrameRoundover","BEVEL"); bevel.width=.018; bevel.segments=5
        elif n.startswith("SofaBack"):
            obj.location.z=.69
            pillow(obj,(size[0],.21,.83),m["fabric"],.27)
        elif n.startswith("SofaArm"):
            pillow(obj,size,m["sage"],.42)
        elif n.startswith("PolishSofaLeg"):
            obj.location.z=.095
            for v in obj.data.vertices: v.co.z*=.19/size[2]
            assign(obj,m["wood"]); d.projected_uvs(obj,1.)
        elif n.startswith("LivingThrow"):
            # Replace the solid slab by a folded cloth with subtle drape.
            w,dep=.52,.44
            verts=[]; faces=[]
            for j in range(25):
                y=-dep/2+dep*j/24
                for i in range(25):
                    x=-w/2+w*i/24
                    z=.005*math.sin(x*55)+.002*math.cos(y*41)
                    verts.append((x,y,z))
            for j in range(24):
                for i in range(24):
                    a=j*25+i; faces.append((a,a+1,a+26,a+25))
            new=d.create_mesh(n+"_Cloth",verts,faces,m["sage"],obj.users_collection[0])
            replace(obj,new);obj.location.z=.585
            sol=obj.modifiers.new("FoldedClothThickness","SOLIDIFY");sol.thickness=.004
            finish(obj,.3)


def chairs(m):
    seats=[o for o in bpy.data.objects if o.type=="MESH" and "DiningChair" in o.name and "_Seat" in o.name]
    for seat in seats:
        size=d.local_dimensions(seat)
        pillow(seat,size,m["fabric"],.30)
        back=bpy.data.objects.get(seat.name.replace("_Seat","_Back"))
        assert back is not None
        # Short bowed wood back with an open lumbar gap; stable ID and yaw.
        w=.50; thick=.025; height=.22; n=32
        verts=[];faces=[]
        for side in [-1,1]:
            for i in range(n+1):
                x=-w/2+w*i/n; y=.038*(1-(2*x/w)**2)+side*thick/2
                verts.extend([(x,y,-height/2),(x,y,height/2)])
        stride=(n+1)*2
        for i in range(n):
            a=i*2;b=a+2;c=stride+b;e=stride+a
            faces.extend([(a,b,b+1,a+1),(e,e+1,c+1,c),(a,e,c,b),(a+1,b+1,c+1,e+1)])
        faces.extend([(0,1,stride+1,stride),(2*n,stride+2*n,stride+2*n+1,2*n+1)])
        new=d.create_mesh(back.name+"_BentWood",verts,faces,m["wood"],back.users_collection[0]);replace(back,new)
        back.location.z=.93; recalc(back);finish(back,1.)
        be=back.modifiers.new("BentPlyRoundover","BEVEL");be.width=.008;be.segments=4
        no=back.modifiers.new("BackWeightedNormals","WEIGHTED_NORMAL");no.keep_sharp=True
        for side in [-1,1]:
            d.create_box_local(back.name+"_Post_"+str(side),seat,(side*.205,.20,.245),(.03,.035,.54),m["wood"],seat.users_collection[0],bevel=.009)
            d.create_box_local(seat.name+"_SideRail_"+str(side),seat,(side*.205,0,-.078),(.032,.41,.054),m["wood"],seat.users_collection[0],bevel=.006)
            d.create_box_local(seat.name+"_CrossRail_"+str(side),seat,(0,side*.19,-.078),(.43,.032,.054),m["wood"],seat.users_collection[0],bevel=.006)
    for obj in bpy.data.objects:
        if obj.type=="MESH" and "DiningChair" in obj.name and "Leg" in obj.name:
            assign(obj,m["wood"]);d.projected_uvs(obj,1.)
            for mod in obj.modifiers:
                if mod.type=="BEVEL":mod.width=.009;mod.segments=5
    return [s.name for s in seats]


def lathe(obj,profile,mat):
    n=96;verts=[];faces=[]
    for r,z in profile:
        for i in range(n):
            t=math.tau*i/n;verts.append((r*math.cos(t),r*math.sin(t),z))
    for j in range(len(profile)-1):
        for i in range(n):
            k=(i+1)%n;faces.append((j*n+i,j*n+k,(j+1)*n+k,(j+1)*n+i))
    # Ends have tiny nonzero radii to keep a regular closed ring topology.
    faces.extend([tuple(reversed(range(n))),tuple((len(profile)-1)*n+i for i in range(n))])
    new=d.create_mesh(obj.name+"_Turned",verts,faces,mat,obj.users_collection[0]);replace(obj,new);recalc(obj);finish(obj)


def tabletop(m):
    tables=[o for o in bpy.data.objects if o.type=="MESH" and (o.name.startswith("DiningTop") or o.name.startswith("CoffeeTable") and "Leg" not in o.name)]
    props=d.ensure_collection("Props")
    for obj in list(bpy.data.objects):
        n=obj.name
        if obj.type!="MESH" or not any(t in n for t in ["DiningPlate","DiningGlass","DiningSharedBowl","LivingMug","DiningNapkin"]):continue
        choices=[t for t in tables if ("Dining" in t.name)==("Dining" in n)]
        table=min(choices,key=lambda t:(t.location-obj.location).length)
        size=d.local_dimensions(table);p=table.matrix_world.inverted()@obj.location
        r=.12 if "Plate" in n else .15 if "Bowl" in n else .05
        p.x=max(-size[0]/2+r+.018,min(size[0]/2-r-.018,p.x))
        p.y=max(-size[1]/2+r+.018,min(size[1]/2-r-.018,p.y))
        if "Glass" in n:
            # Separate cups from shallow plates rather than intersecting them.
            if size[0]>size[1]:
                p.y*=.38
                if abs(p.x)<.28:p.x=math.copysign(.28,p.x)
            else:
                p.x*=.38
                if abs(p.y)<.28:p.y=math.copysign(.28,p.y)
        p.z=size[2]/2;obj.location=table.matrix_world@p;obj.rotation_euler=(0,0,0)
        if "Plate" in n:
            lathe(obj,[(.001,0),(.080,0),(.118,.010),(.123,.018),(.119,.022),(.090,.009),(.001,.006)],m["ceramic"])
        elif "Bowl" in n:
            lathe(obj,[(.001,0),(.072,0),(.10,.025),(.147,.075),(.15,.09),(.145,.094),(.137,.080),(.093,.03),(.068,.008),(.001,.008)],m["ceramic"])
        elif "Napkin" in n:
            obj.location.z+=d.local_dimensions(obj)[2]/2
        else:
            lathe(obj,[(.001,0),(.039,0),(.046,.006),(.049,.095),(.048,.102),(.044,.102),(.044,.095),(.041,.011),(.001,.009)],m["ceramic"])
            if "Mug" in n:
                pts=[(.044+.030*math.sin(math.pi*i/48),0,.052+.033*math.cos(math.pi*i/48)) for i in range(49)]
                d.create_curve_mesh(n+"_Handle",pts,m["ceramic"],props,bevel_depth=.005,center=obj.location)
    for table in tables:
        size=d.local_dimensions(table);assign(table,m["wood"]);d.projected_uvs(table,1.)
        for mod in table.modifiers:
            if mod.type=="BEVEL":mod.width=.03;mod.segments=8
        if "Coffee" not in table.name:continue
        for i in range(2):
            z=size[2]/2+.012+i*.027
            center=(-.26+i*.025,-.09,z)
            d.create_box_local(table.name+"_BookPages_"+str(i),table,center,(.25,.18,.021),m["paper"],props,bevel=.0015)
            for sign in [-1,1]:
                d.create_box_local(table.name+"_BookCover_"+str(i)+"_"+str(sign),table,(center[0],center[1],z+sign*.0115),(.26,.188,.002),m["cover"],props,bevel=.001)


def wall_decor(room):
    # V3 picture rows overlapped and were placed from sofa depth rather than walls.
    for obj in list(bpy.data.objects):
        if not obj.name.startswith("PolishWallArt"):continue
        index=int(obj.name.rsplit("_",1)[1])
        if index >= (3 if room=="room_a" else 2):
            bpy.data.objects.remove(obj,do_unlink=True);continue
        frame="Frame" in obj.name
        obj.rotation_euler=(0,0,0)
        obj.location=((-5.25+index*.80) if room=="room_a" else (-4.05 if index==0 else 1.10),
                      (4.845 if room=="room_a" else 5.045)-(0 if frame else .038),1.92)
        if room=="room_a":obj.scale.x=.57
    for obj in bpy.data.objects:
        if obj.name.startswith("PolishWallShelf"):
            obj.rotation_euler=(0,0,0)
            obj.location=(-5.2,4.775,1.40) if room=="room_a" else (-4.05,4.975,1.26)


def review_specs(room):
    return ([("OVERALL",(-5.45,-.20,1.62),(-3.55,3.05,.88),24),
             ("SOFA",(-3.25,1.20,1.37),(-4.60,3.64,.68),43),
             ("DINING",(.8,-4.30,1.48),(2.65,-2.08,.79),39),
             ("TABLEWARE",(1.75,-3.25,1.28),(2.75,-2.15,.83),53)] if room=="room_a" else
            [("OVERALL",(-4.75,-3.80,1.65),(-1.4,.25,.87),24),
             ("SOFA",(-.7,.35,1.36),(-2.65,2.50,.70),43),
             ("DINING",(-2.15,-3.80,1.50),(.05,-1.8,.80),41),
             ("TABLEWARE",(-1.25,-3.02,1.34),(-.1,-1.8,.84),52)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--texture-root",type=Path,required=True);p.add_argument("--room",choices=["room_a","room_c"],required=True)
    p.add_argument("--samples",type=int,default=32)
    a=p.parse_args(sys.argv[sys.argv.index("--")+1:]);source=a.input_root.resolve(strict=True);out=a.output_root.resolve()
    assert not out.exists(),f"refusing existing output {out}"
    blend,sem,anchors,_=d.load_source(source,None)
    out.mkdir(parents=True);[(out/x).mkdir() for x in ["textures","renders","visual","usd"]]
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    d.ensure_collection("ArchitecturalDetails")
    shutil.copy2(a.texture_root/"manifest.json",out/"material_provenance.json")
    m=materials(a.texture_root,out);soft_furniture(m);chair_ids=chairs(m);tabletop(m);wall_decor(a.room)
    for o in bpy.data.objects:
        if o.type=="MESH" and o.data.materials and o.data.materials[0]==m["wood"]:d.projected_uvs(o,1.)
    bpy.context.view_layer.update()
    cameras=[]
    for label,pos,target,lens in review_specs(a.room):
        name="REVIEW_"+a.room.upper()+"_"+label
        data=bpy.data.cameras.new(name);data.lens=lens;data.sensor_width=36
        obj=bpy.data.objects.new(name,data);d.ensure_collection("Cameras").objects.link(obj);obj.location=pos;d.look_at(obj,Vector(target))
        cameras.append(dict(camera_id=name,position_m=pos,target_m=target,forward_world=list((Vector(target)-Vector(pos)).normalized()),focal_length_mm=lens,fov_horizontal_deg=math.degrees(2*math.atan(18/lens)),production_camera=False,purpose="review_only_asset_inspection",visibility_note="all_production_geometry_visible"))
    # Copy all surviving source texture dependencies into this package.
    for im in bpy.data.images:
        if im.source!="FILE" or not im.filepath:continue
        src=Path(bpy.path.abspath(im.filepath))
        if src.is_file() and not src.is_relative_to(out):
            dst=out/"textures"/("source_"+src.name)
            if not dst.exists():shutil.copy2(src,dst)
            im.filepath=str(dst)
    scene=bpy.context.scene
    scene.view_settings.exposure=-.4
    scene.render.engine="CYCLES";scene.cycles.device="CPU";scene.cycles.samples=a.samples;scene.cycles.use_denoising=True
    scene.render.threads_mode="FIXED";scene.render.threads=6
    scene.render.resolution_x=960;scene.render.resolution_y=540;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    out_blend=out/(a.room+"_detailed_v6.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    glb=h.export_selection(out);usd=out/"usd"/(a.room+"_detailed_v6.usda");usd_record=h.load_usd_exporter().export_static_usd(scene,usd)
    objects=h.scene_static_mesh_records(scene,"visual/"+glb.name,"usd/"+usd.name)
    for record in objects:
        if record["object_id"].startswith("DiningGlass"):record["category"]="prop"
    seats=sem["seat_points"]
    obj_by={o["object_id"]:o for o in objects}
    checks=[]
    for seat in seats:
        obj_id=seat.get("surface_object_id") or seat.get("scene_object_id")
        if obj_id in obj_by:
            actual=obj_by[obj_id]["world_bounds_m"]["max_m"][2]
            expected=seat.get("support_height_m",seat.get("surface_height_m"))
            checks.append(dict(object_id=obj_id,actual_top_m=actual,expected_top_m=expected))
            assert abs(actual-expected)<.002,(obj_id,actual,expected)
    sem.update(objects=objects,seat_points=seats,source_blend=str(blend),source_reference_status="reference_informed_not_exact_reconstruction" if a.room=="room_a" else "authored_open_family_room",detail_profile="family_interiors_v6",furniture_objects=[o for o in objects if o["category"] in ["furniture","prop","cabinet","appliance"]])
    anchors.update(seat_points=seats,scene_object_ids=[o["object_id"] for o in objects])
    d.write_json(out/"object_semantics.json",sem);d.write_json(out/"functional_anchors.json",anchors)
    lighting=h.scene_lighting_records(scene);d.write_json(out/"lighting.json",dict(kind="avengine_scene_lighting_manifest",room_id=sem.get("room_id"),lights=lighting))
    d.write_json(out/"review_camera_specs.json",dict(status="review_only",cameras=cameras))
    report=dict(status="research_candidate",qualification_claim=False,source_root=str(source),changes=["smooth_crowned_upholstery_and_seams","lowered_back_cushions_to_seat_contact","exposed_wood_frame_and_grounded_legs","bent_plywood_chair_backs_and_open_joinery","hollow_stoneware_on_measured_tabletops","editable_books_and_folded_cloth","CC0_scanned_PBR_real_scale_UV"],chair_ids=chair_ids,seat_checks=checks,seat_reference_policy="source_centers_tops_fronts_preserved",source_texture_root=str(a.texture_root),native_execution="pending_root_spear_ue",artifacts=dict(blend=str(out_blend),visual_glb=str(glb),usd=usd_record),review_cameras=cameras)
    d.write_json(out/"detail_report.json",report);d.write_json(out/"polish_report.json",report)
    print("ASSET_EXPORT_READY",str(out),flush=True)
    for camera in cameras:
        scene.camera=bpy.data.objects[camera["camera_id"]];scene.render.filepath=str(out/"renders"/(camera["camera_id"]+".png"));bpy.ops.render.render(write_still=True)
    print("FAMILY_DETAIL_COMPLETE",str(out),flush=True)

if __name__=="__main__":main()
