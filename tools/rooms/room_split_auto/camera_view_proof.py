"""Native full-house CPU RGB evidence for offline HM3D split room scopes.

All outputs use exclusive creation. Room polygons are used after native rendering
for annotations and evidence, never as renderer geometry or an RGB black mask.
"""
from __future__ import annotations
import argparse,ctypes,datetime,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import shape,Polygon,Point
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
from tools.rooms.room_split_auto.pipeline import load_native
from tools.rooms.room_split_auto.software_render import verify_software_devices

def dump(path,data):
    with Path(path).open("x",encoding="utf8") as f:
        json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()

def save_png(path,array):
    with Path(path).open("xb") as f:Image.fromarray(array).save(f,format="PNG")

def filled_footprint(g):
    if g.geom_type=="Polygon":return Polygon(g.exterior)
    return shapely.union_all([filled_footprint(x) for x in g.geoms])

def camera_basis(forward):
    f=np.array(forward,dtype=float);f/=np.linalg.norm(f)
    right=np.cross(f,np.array([0.,1.,0.]));right/=np.linalg.norm(right)
    up=np.cross(right,f)
    return np.column_stack([right,up,-f])

def project(points,projection,view,width,height):
    a=np.asarray(points,dtype=float).reshape(-1,3)
    clip=np.c_[a,np.ones(len(a))]@(projection@view).T
    ndc=clip[:,:3]/clip[:,3,None]
    return np.c_[(ndc[:,0]+1)*width/2-.5,(1-ndc[:,1])*height/2-.5],clip[:,3]

def unproject(depth,projection,view):
    height,width=depth.shape
    yy,xx=np.mgrid[:height,:width]
    # Native Habitat PINHOLE depth is camera-plane forward depth.
    eye=np.stack([(2*(xx+.5)/width-1)*depth/projection[0,0],
        (1-2*(yy+.5)/height)*depth/projection[1,1],-depth,
        np.ones_like(depth)],axis=-1)
    return (eye@np.linalg.inv(view).T)[...,:3]

def font(size=19):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)

def banner(draw,text,origin=(10,10),color=(255,255,255),size=19):
    f=font(size);bounds=draw.textbbox(origin,text,font=f)
    draw.rectangle((bounds[0]-5,bounds[1]-5,bounds[2]+5,bounds[3]+5),fill=(18,25,32))
    draw.text(origin,text,font=f,fill=color)

def annotate_view(rgb,depth,projection,view,block,house_floors,region_blocks,forward,direction):
    height,width=depth.shape
    world=unproject(depth,projection,view)
    fy=block["floor_y_m"];g=shape(block["floor_polygon_xz_m"])
    # Fill semantic holes and ignore a 20 cm boundary band: furniture holes,
    # small disconnected scan fragments and cut-edge raster pixels are not proof.
    footprint=filled_footprint(g).buffer(.20)
    ground=shapely.union_all([shape(x["floor_polygon"]) for _,x in house_floors])
    valid=np.isfinite(depth)&(depth>.05)&(depth<99.9)&np.isfinite(world).all(axis=2)
    outside=valid&~shapely.contains_xy(footprint,world[...,0],world[...,2])
    ground_hits=shapely.contains_xy(ground.buffer(.025),world[...,0],world[...,2])
    floor_height=np.abs(world[...,1]-fy)<=.18
    bright=np.max(rgb,axis=2)>=8
    evidence=outside&ground_hits&floor_height&bright
    labels,count=ndimage.label(evidence)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    regions=sorted(range(1,len(sizes)),key=lambda i:(-sizes[i],i))
    annotation=rgb.copy()
    annotation[evidence]=(rgb[evidence].astype(float)*.70+np.array([10,235,235])*.30).astype("uint8")
    image=Image.fromarray(annotation);draw=ImageDraw.Draw(image)
    banner(draw,f"{direction.upper()} | cyan = depth-confirmed FLOOR OUTSIDE selected room")
    records=[]
    for rank,label in enumerate(regions[:3]):
        if sizes[label]<30:continue
        patch=labels==label
        dist=ndimage.distance_transform_edt(patch)
        py,px=np.unravel_index(np.argmax(dist),dist.shape)
        xyz=world[py,px];owner=[]
        for row,floor in house_floors:
            if shape(floor["floor_polygon"]).covers(Point(xyz[[0,2]])):
                owner.append(row["room_label"])
        adjacent=[]
        for b in region_blocks:
            if b["id"]!=block["id"] and shape(b["floor_polygon_xz_m"]).covers(Point(xyz[[0,2]])):
                adjacent.append(b["id"].split("__")[-1])
        record={"pixel_xy":[int(px),int(py)],"world_xyz_m":xyz.tolist(),
            "native_rgb":rgb[py,px].tolist(),"depth_m":float(depth[py,px]),
            "semantic_ground_regions":owner,"other_split_blocks":adjacent,
            "conservative_distance_from_selected_footprint_m":float(filled_footprint(g).distance(Point(xyz[[0,2]]))),
            "connected_visible_floor_pixels":int(sizes[label])}
        records.append(record)
        tag="OUT "+str(rank+1)+" "+("/".join(adjacent or owner) or "other floor")
        draw.ellipse((px-8,py-8,px+8,py+8),outline=(0,255,255),width=3)
        tx=max(14,min(width-250,px+20));ty=max(50,min(height-45,py-34-rank*25))
        draw.line((px,py,tx,ty+12),fill=(0,255,255),width=2)
        banner(draw,tag,(tx,ty),(0,255,255),17)
    if not records:banner(draw,"No conservative outside-floor patch in this direction",(10,45),(255,210,130),17)
    witness=block["placement_witness"]
    speaker_points=np.array([witness["source_1_m"],witness["source_2_m"]])
    speaker_px,speaker_depth=project(speaker_points,projection,view,width,height)
    marker_records=[]
    for i,(point,screen,z) in enumerate(zip(speaker_points,speaker_px,speaker_depth),1):
        inside=bool(z>0 and 0<=screen[0]<width and 0<=screen[1]<height)
        marker_records.append({"label":"S"+str(i),"pixel_xy":screen.tolist(),"forward_depth_m":float(z),"in_frame":inside})
        if inside:
            x,y=screen;color=(255,105,100) if i==1 else (255,195,65)
            draw.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
            banner(draw,"S"+str(i)+" (position only)",(max(10,min(width-225,x+12)),max(70,min(height-35,y))),color,16)
    yy,xx=np.nonzero(evidence)
    if len(yy):
        chosen=np.linspace(0,len(yy)-1,min(16,len(yy)),dtype=int)
        samples=[{"pixel_xy":[int(xx[k]),int(yy[k])],"world_xyz_m":world[yy[k],xx[k]].tolist(),"rgb":rgb[yy[k],xx[k]].tolist()} for k in chosen]
        pp,_=project(np.array([a["world_xyz_m"] for a in samples]),projection,view,width,height)
        maxerror=float(np.max(np.linalg.norm(pp-np.array([a["pixel_xy"] for a in samples]),axis=1)))
    else:samples=[];maxerror=None
    return image,np.asarray(evidence,dtype=np.uint8)*255,{
        "outside_geometry_pixel_count":int(np.count_nonzero(outside&bright)),
        "outside_ground_nonblack_pixel_count":int(np.count_nonzero(evidence)),
        "annotation_rule":"native depth surface within 0.18m of semantic floor; inside same-level semantic ground; outside filled selected footprint buffered 0.20m; native RGB max channel>=8",
        "outline_points":records,"independent_ray_check_samples":samples,
        "projection_roundtrip_max_error_px":maxerror,"speaker_markers":marker_records,
        "rendered_human_meshes":False,"raw_rgb_unchanged":True}

def overhead_overlay(rgb,projection,view,block,all_blocks,cut_lines,camera,source_points,forward,hfov):
    h,w=rgb.shape[:2];im=Image.fromarray(rgb).convert("RGBA")
    layer=Image.new("RGBA",im.size);draw=ImageDraw.Draw(layer)
    def pts(xz,y=None):
        xz=np.asarray(xz,dtype=float)
        world=np.c_[xz[:,0],np.full(len(xz),block["floor_y_m"] if y is None else y),xz[:,1]]
        screen,_=project(world,projection,view,w,h)
        return [tuple(a) for a in screen]
    g=shape(block["floor_polygon_xz_m"]);parts=[g] if g.geom_type=="Polygon" else list(g.geoms)
    for p in parts:
        pxy=pts(p.exterior.coords)
        draw.polygon(pxy,fill=(20,235,220,34),outline=(0,255,220,255),width=4)
        for ring in p.interiors:draw.line(pts(ring.coords),fill=(0,255,220,110),width=1)
    for c in cut_lines:
        if c.get("floor_id")!=block["floor_id"] or not c.get("active_in_final_partition"):continue
        line=c.get("line_xz_m")
        if line:draw.line(pts(line),fill=(255,105,220,210) if c["type"]=="door" else (245,230,115,135),width=3)
    f=np.asarray(forward)[[0,2]];f/=np.linalg.norm(f);cam=np.asarray(camera)[[0,2]]
    for sign,color in [(1,(255,255,70,48)),(-1,(190,130,255,40))]:
        poly=[cam.tolist()]
        for a in np.linspace(-np.radians(hfov)/2,np.radians(hfov)/2,35):
            rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            poly.append((cam+rot@(f*sign)*6).tolist())
        draw.polygon(pts(poly),fill=color,outline=color[:3]+(150,),width=2)
    for b in all_blocks:
        if b["floor_id"]!=block["floor_id"]:continue
        p=shape(b["floor_polygon_xz_m"]).representative_point();xy=pts([[p.x,p.y]])[0]
        if w*.01<xy[0]<w*.99 and h*.01<xy[1]<h*.99:
            draw.text(xy,b["id"].split("__")[-1],font=font(16),fill=(255,255,255,240),stroke_width=2,stroke_fill=(15,15,15,220))
    marked=Image.alpha_composite(im,layer).convert("RGB");draw=ImageDraw.Draw(marked)
    banner(draw,"WHOLE HOUSE | green outline = selected placement scope",(12,10),size=20)
    banner(draw,"yellow cone: forward  |  purple cone: reverse  |  magenta: semantic door",(12,42),size=17)
    for label,pos,color in [("CAM",camera,(90,255,100)),("S1",source_points[0],(255,105,100)),("S2",source_points[1],(255,195,65))]:
        xy=pts([[pos[0],pos[2]]])[0];x,y=xy
        draw.ellipse((x-6,y-6,x+6,y+6),fill=color,outline=(15,15,15),width=2)
        banner(draw,label,(x+10,y+2),color,16)
    return marked

def run(args):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    root=Path(args.root);out=Path(args.output)
    if not (out.parent/"code_audit_v1.json").exists():raise RuntimeError("Code audit must precede native rendering")
    source=root/"delivery_all_v2/regions"/(args.house+"__"+args.region+".json")
    d=json.loads(source.read_text());block=next(b for b in d["blocks"] if b["id"].endswith(args.block))
    if block["decision"]!="retain" or not block["placement_witness"]["found"]:raise ValueError("Existing retained placement witness required")
    target=out/(args.house+"__"+args.region+"__"+args.block);target.mkdir(parents=True,exist_ok=False)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.time()
    dump(target/"start.json",{"pid":os.getpid(),"machine":os.uname().nodename,"started_at_utc":started,"source_json":str(source),"script":str(Path(__file__).resolve()),"cpu_only":True})
    inventory=root/"inventory_v1"/(args.house+".json");inv=json.loads(inventory.read_text())
    house_floors=[(row,floor) for row in inv["rows"] for floor in row["floors"] if abs(floor["floor_y_m"]-block["floor_y_m"])<=.3]
    all_ground=shapely.union_all([shape(f["floor_polygon"]) for row in inv["rows"] for f in row["floors"]])
    x0,z0,x1,z1=all_ground.bounds;cx=(x0+x1)/2;cz=(z0+z1)/2
    span=max(x1-x0,z1-z0)*1.12+1.0
    scene=Path(d["source_geometry"]["scene_directory"]);sid=scene.name.split("-",1)[1];glb=scene/(sid+".glb")
    checks={str(p):sha(p) for p in [source,inventory,glb,scene/(sid+".semantic.glb"),scene/(sid+".semantic.txt"),scene/(sid+".basis.navmesh")]}
    hs=load_native();import magnum as mn
    import quaternion
    extensions=verify_software_devices()
    cfg=hs.SimulatorConfiguration();cfg.scene_id=str(glb);cfg.scene_dataset_config_file="default";cfg.gpu_device_id=-1;cfg.enable_physics=False;cfg.load_semantic_mesh=False
    sensors=[]
    for uuid,kind in [("proof_rgb",hs.SensorType.COLOR),("proof_depth",hs.SensorType.DEPTH)]:
        s=hs.CameraSensorSpec();s.uuid=uuid;s.sensor_type=kind;s.sensor_subtype=hs.SensorSubType.PINHOLE
        s.resolution=[720,1280];s.position=mn.Vector3(0,0,0);s.orientation=mn.Vector3(0,0,0);s.hfov=85.;s.near=.05;s.far=100.;s.gpu2gpu_transfer=False;s.noise_model="None"
        if kind==hs.SensorType.DEPTH:s.channels=1
        sensors.append(s)
    s=hs.CameraSensorSpec();s.uuid="proof_overhead";s.sensor_type=hs.SensorType.COLOR;s.sensor_subtype=hs.SensorSubType.ORTHOGRAPHIC
    s.resolution=[1200,1200];s.position=mn.Vector3(0,0,0);s.orientation=mn.Vector3(-np.pi/2,np.pi,0);s.ortho_scale=10.;s.near=28.2;s.far=30.3;s.gpu2gpu_transfer=False
    sensors.append(s)
    agent=hs.agent.AgentConfiguration();agent.sensor_specifications=sensors;agent.action_space={}
    sim=None
    try:
        sim=hs.Simulator(hs.Configuration(cfg,[agent]))
        gl=ctypes.CDLL("libGL.so.1");gl.glGetString.argtypes=[ctypes.c_uint];gl.glGetString.restype=ctypes.c_char_p
        renderer=gl.glGetString(0x1F01).decode();vendor=gl.glGetString(0x1F00).decode()
        if "llvmpipe" not in renderer.lower():raise RuntimeError("Refuse observation on a non-CPU GL renderer")
        print(json.dumps({"pid":os.getpid(),"renderer":renderer,"EGL":extensions}),flush=True)
        overhead=sim._sensors["proof_overhead"]._sensor_object.render_camera
        overhead.projection_matrix=mn.Matrix4.orthographic_projection(mn.Vector2(span,span),28.2,30.3)
        state=hs.AgentState();state.position=mn.Vector3(cx,block["floor_y_m"]+30,cz);sim.initialize_agent(0,state)
        obs=sim.get_sensor_observations()
        overhead_rgb=np.asarray(obs["proof_overhead"],dtype=np.uint8)[...,:3].copy()
        op=np.asarray(overhead.projection_matrix,dtype=float);ov=np.asarray(overhead.camera_matrix,dtype=float)
        save_png(target/"whole_house_overhead_raw.png",overhead_rgb)
        witness=block["placement_witness"];camera=np.asarray(witness["camera_m"]);speakers=np.array([witness["source_1_m"],witness["source_2_m"]])
        forward=speakers.mean(axis=0)-camera;forward/=np.linalg.norm(forward)
        overlay=overhead_overlay(overhead_rgb,op,ov,block,d["blocks"],d["cut_lines"],camera,speakers,forward,85.)
        with (target/"whole_house_overhead_annotated.png").open("xb") as f:overlay.save(f,format="PNG")
        pf=hs.PathFinder()
        if not pf.load_nav_mesh(str(scene/(sid+".basis.navmesh"))):raise RuntimeError("Native ground support cannot be loaded")
        supports=[]
        for role,point,height in [("camera",camera,1.5),("source_1",speakers[0],1.2),("source_2",speakers[1],1.2)]:
            ground=point.copy();ground[1]-=height
            snapped=np.asarray(pf.snap_point(ground),dtype=float)
            if np.linalg.norm(snapped-ground)>.03:raise RuntimeError("Frozen witness ground support drift")
            supports.append({"role":role,"absolute_point_m":point.tolist(),"native_ground_support_m":snapped.tolist(),"height_above_native_ground_m":float(point[1]-snapped[1]),"inside_frozen_floor_polygon":bool(shape(block["floor_polygon_xz_m"]).covers(Point(point[[0,2]])))})
        views=[]
        for direction,fwd in [("forward",forward),("reverse",forward*np.array([-1.,1.,-1.]))]:
            state=hs.AgentState();state.position=mn.Vector3(camera.tolist());state.rotation=quaternion.from_rotation_matrix(camera_basis(fwd))
            sim.initialize_agent(0,state)
            rgb_camera=sim._sensors["proof_rgb"]._sensor_object.render_camera
            dep_camera=sim._sensors["proof_depth"]._sensor_object.render_camera
            projection=np.asarray(rgb_camera.projection_matrix,dtype=float);view=np.asarray(rgb_camera.camera_matrix,dtype=float)
            if not np.allclose(view,np.asarray(dep_camera.camera_matrix),atol=1e-7):raise RuntimeError("RGB/depth camera poses differ")
            obs=sim.get_sensor_observations();rgb=np.asarray(obs["proof_rgb"],dtype=np.uint8)[...,:3].copy();depth=np.asarray(obs["proof_depth"],dtype=np.float32).copy()
            save_png(target/(direction+"_raw.png"),rgb)
            with (target/(direction+"_depth.npy")).open("xb") as f:np.save(f,depth,allow_pickle=False)
            annotated,mask,evidence=annotate_view(rgb,depth,projection,view,block,house_floors,d["blocks"],fwd,direction)
            with (target/(direction+"_annotated.png")).open("xb") as f:annotated.save(f,format="PNG")
            save_png(target/(direction+"_outside_floor_mask.png"),mask)
            actual_hfov=float(np.degrees(2*np.arctan(1/projection[0,0])))
            if abs(actual_hfov-85)>.001:raise RuntimeError("Native FOV differs")
            actual_camera=np.linalg.inv(view)[:3,3]
            if np.linalg.norm(actual_camera-camera)>1e-5:raise RuntimeError("Native camera changed existing witness")
            views.append({"direction":direction,"forward_world":fwd.tolist(),"camera_position_m":actual_camera.tolist(),"projection_matrix":projection.tolist(),"view_matrix":view.tolist(),"hfov_native_deg":actual_hfov,"resolution_hw":[720,1280],"evidence":evidence})
            print(direction,"outside floor pixels",evidence["outside_ground_nonblack_pixel_count"],flush=True)
        final_checks={p:sha(p) for p in checks}
        if final_checks!=checks:raise RuntimeError("Read-only source identity changed during native render")
        record={"schema":"hm3d_camera_room_scope_proof_v1","status":"rendered","created_at_utc":started,"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"machine":os.uname().nodename,"pid":os.getpid(),"code_commit_audited":"f59e40bb3fba4b10c701a8ae39caf1e93c594ada","source_json":str(source),"block":block,"source_floor_area_m2":d["source_floor_area_m2"],"scene_glb":str(glb),"cpu_gl_renderer":renderer,"cpu_gl_vendor":vendor,"software_device_extensions":extensions,"gpu_device_id":-1,"scene_crop_or_black_mask":False,"acoustics":"not_run","rendered_human_meshes":False,"witness_supports":supports,"production_calibration":{"hfov_deg":85,"resolution_hw":[720,1280],"near_m":.05,"far_m":100.,"camera_default_height_m":1.55,"task_frozen_camera_height_m":1.5,"task_frozen_source_height_m":1.2},"opposite_view_semantics":"yaw changed by 180 degrees; same pitch; identical camera point","whole_house_overhead":{"bounds_all_house_semantic_floors_xz_m":[x0,z0,x1,z1],"frame_center_xz_m":[cx,cz],"frame_span_m":span,"projection":op.tolist(),"view":ov.tolist(),"floor_y_m":block["floor_y_m"],"vertical_section_limits_m":[block["floor_y_m"]-.3,block["floor_y_m"]+1.8],"vertical_section_note":"Standard roof-cut orthographic floor view; full-house XZ extent, no room crop or edited geometry."},"views":views,"input_sha256_before":checks,"input_sha256_after":final_checks,"elapsed_s":time.time()-clock,"peak_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        dump(target/"proof.json",record)
    finally:
        if sim is not None:sim.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--house",required=True);p.add_argument("--region",required=True);p.add_argument("--block",required=True)
    run(p.parse_args())
