"""CPU whole-house gallery for admitted native rooms and v2 split candidates."""
from pathlib import Path
import argparse,base64,ctypes,datetime,io,json,math,os,resource,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np
import shapely
from shapely.geometry import shape,mapping,box
from PIL import Image,ImageDraw,ImageFont
from tools.rooms.room_split_auto.pipeline import load_native
from tools.rooms.room_split_auto.software_render import verify_software_devices
from tools.rooms.room_split_auto.camera_view_proof import project
from tools.rooms.room_selection.geometry import short_side

def dump(p,d):
    with Path(p).open("x") as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2)

def render(job,out):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    started=time.time();hs=load_native();import magnum as mn
    ext=verify_software_devices();scene=Path(job["scene_directory"]);sid=scene.name.split("-",1)[1]
    cfg=hs.SimulatorConfiguration();cfg.scene_id=str(scene/(sid+".glb"));cfg.gpu_device_id=-1;cfg.enable_physics=False
    sensor=hs.CameraSensorSpec();sensor.uuid="gallery";sensor.sensor_type=hs.SensorType.COLOR;sensor.sensor_subtype=hs.SensorSubType.ORTHOGRAPHIC;sensor.resolution=[1200,1200];sensor.ortho_scale=10
    sensor.position=mn.Vector3(0,0,0);sensor.orientation=mn.Vector3(-np.pi/2,np.pi,0);sensor.near=28.2;sensor.far=30.3;sensor.gpu2gpu_transfer=False
    agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[sensor];sim=None;results=[]
    try:
        sim=hs.Simulator(hs.Configuration(cfg,[agent]))
        gl=ctypes.CDLL("libGL.so.1");gl.glGetString.argtypes=[ctypes.c_uint];gl.glGetString.restype=ctypes.c_char_p
        renderer=gl.glGetString(0x1F01).decode()
        if "llvmpipe" not in renderer.lower():raise RuntimeError("Nonsoftware GL device")
        camera=sim._sensors["gallery"]._sensor_object.render_camera
        camera.projection_matrix=mn.Matrix4.orthographic_projection(mn.Vector2(job["span"],job["span"]),28.2,30.3)
        for floor in job["floors"]:
            y=floor["y"];state=hs.AgentState();state.position=mn.Vector3(job["cx"],y+30,job["cz"]);sim.initialize_agent(0,state)
            rgb=np.asarray(sim.get_sensor_observations()["gallery"],dtype=np.uint8)[...,:3].copy()
            stem=job["house"]+"__"+floor["key"];p=Path(out)/(stem+".png")
            with p.open("xb") as f:Image.fromarray(rgb).save(f,format="PNG")
            record={"path":str(p),"projection":np.asarray(camera.projection_matrix).tolist(),"view":np.asarray(camera.camera_matrix).tolist(),"floor_y_m":y,"span_m":job["span"],"size_px":[1200,1200],"whole_house_xz_bounds":job["bounds"],"renderer":renderer,"EGL":ext,"pid":os.getpid(),"nice":os.getpriority(os.PRIO_PROCESS,0),"scene_glb":str(scene/(sid+".glb"))}
            dump(p.with_suffix(".json"),record);results.append(dict(house=job["house"],floor_key=floor["key"],metadata=str(p.with_suffix(".json"))))
    finally:
        if sim is not None:sim.close()
    return dict(house=job["house"],seconds=time.time()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,frames=results)

def overlay(item,meta):
    base=Image.open(meta["path"]).convert("RGBA");w,h=base.size;g=shape(item["geometry"])
    layer=Image.new("RGBA",base.size);d=ImageDraw.Draw(layer)
    def pts(coords):
        arr=np.asarray(coords);world=np.c_[arr[:,0],np.full(len(arr),item["floor_y_m"]),arr[:,1]]
        pixels,_=project(world,np.asarray(meta["projection"]),np.asarray(meta["view"]),w,h)
        return [tuple(x) for x in pixels]
    parts=[g] if g.geom_type=="Polygon" else list(g.geoms)
    for p in parts:
        if p.geom_type!="Polygon":continue
        d.polygon(pts(p.exterior.coords),fill=(35,157,237,110),outline=(12,75,230,255),width=4)
        for ring in p.interiors:d.polygon(pts(ring.coords),fill=(0,0,0,0))
    rect=box(*g.bounds);corners=pts(rect.exterior.coords)
    for a,b in zip(corners,corners[1:]):
        a=np.asarray(a);b=np.asarray(b);length=np.linalg.norm(b-a)
        for t in np.arange(0,length,18):
            aa=a+(b-a)*t/max(length,1);bb=a+(b-a)*min(t+11,length)/max(length,1)
            d.line([tuple(aa),tuple(bb)],fill=(255,245,15,255),width=4)
    image=Image.alpha_composite(base,layer).convert("RGB");d=ImageDraw.Draw(image)
    font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",32,index=2)
    small=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",23,index=2)
    label=f'{item["area_m2"]:.2f} m²  短边 {item["short_side_m"]:.2f} m'
    d.rectangle((10,10,1000,105),fill=(20,28,38));d.text((20,12),item["label"],font=small,fill="white");d.text((20,48),label,font=font,fill="white")
    q=g.representative_point();x,y=pts([(q.x,q.y)])[0]
    d.text((x,y),f'{item["area_m2"]:.1f}',font=font,fill="white",stroke_width=3,stroke_fill=(10,55,150),anchor="mm")
    scale=5*w/meta["span_m"];sx=30;sy=h-46
    d.rectangle((sx-15,sy-37,sx+scale+15,sy+20),fill=(20,28,38))
    d.line((sx,sy,sx+scale,sy),fill="white",width=6);d.line((sx,sy-9,sx,sy+9),fill="white",width=3);d.line((sx+scale,sy-9,sx+scale,sy+9),fill="white",width=3)
    d.text((sx+scale/2,sy-13),"5 m",font=small,fill="white",anchor="mb")
    return image

def run(root,workers=12):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    root=Path(root);out=root/"size_gallery_30_50_v1";raw=out/"raw";raw.mkdir(exist_ok=False);media=out/"media";media.mkdir(exist_ok=False)
    selection=json.loads((out/"selection_rows_v1.json").read_text());unique={}
    for r in selection:unique[(r["house"],r["room_label"])]=r
    inventory={}
    for p in (root/"inventory_v1").glob("*.json"):
        d=json.loads(p.read_text());inventory[d["house"]]=d
    items=[];missing=[];measured=[]
    for key,r in sorted(unique.items()):
        row=next((x for x in inventory.get(r["house"],{}).get("rows",[]) if x["room_label"]==r["room_label"]),None)
        if row is None:missing.append(key);continue
        area=row["floor_area_sum_m2"];measured.append(dict(house=key[0],region=key[1],measured_area_m2=area,listed_area_m2=float(r["floor_area_m2"])))
        if not 30<=area<=50:continue
        floor=max(row["floors"],key=lambda f:f["floor_area_m2"]);g=shapely.union_all([shape(f["floor_polygon"]) for f in row["floors"]])
        items.append(dict(group="native",id=key[0]+"__"+key[1],house=key[0],region=key[1],label=key[0]+"/"+key[1],area_m2=area,short_side_m=short_side(g),floor_y_m=floor["floor_y_m"],geometry=mapping(g),listed_area_m2=float(r["floor_area_m2"]),selection_file=r["selection_file"],floor_layers=[{k:f[k] for k in ["floor_id","floor_y_m","floor_area_m2"]} for f in row["floors"]]))
    for p in (root/"delivery_all_v2/regions").glob("*.json"):
        d=json.loads(p.read_text())
        for b in d["blocks"]:
            if b["decision"]=="retain" and 30<=b["floor_area_m2"]<=50:
                items.append(dict(group="split_v2",id=b["id"],house=b["house"],region=b["source_region"],label=b["house"]+"/"+b["source_region"]+"/"+b["id"].split("__")[-1],area_m2=b["floor_area_m2"],short_side_m=b["short_side_m"],floor_y_m=b["floor_y_m"],geometry=b["floor_polygon_xz_m"],source_json=str(p),v2_component_count=len(list(shape(b["floor_polygon_xz_m"]).geoms)) if shape(b["floor_polygon_xz_m"]).geom_type=="MultiPolygon" else 1))
    jobs={}
    for item in items:
        inv=inventory[item["house"]]
        if item["house"] not in jobs:
            allg=shapely.union_all([shape(f["floor_polygon"]) for r in inv["rows"] for f in r["floors"]]);x0,z0,x1,z1=allg.bounds
            jobs[item["house"]]=dict(house=item["house"],scene_directory=inv["source"],bounds=[x0,z0,x1,z1],cx=(x0+x1)/2,cz=(z0+z1)/2,span=max(x1-x0,z1-z0)*1.12+1.,floors=[])
        job=jobs[item["house"]];match=next((f for f in job["floors"] if abs(f["y"]-item["floor_y_m"])<=.05),None)
        if match is None:match=dict(key=f'Y{len(job["floors"]):02d}',y=item["floor_y_m"]);job["floors"].append(match)
        item["floor_key"]=match["key"]
    dump(out/"plan.json",dict(selection_rows=len(selection),unique_rooms=len(unique),native_items=sum(x["group"]=="native" for x in items),new_v2_items=sum(x["group"]=="split_v2" for x in items),missing=missing,measured_rows=measured,items=items,jobs=list(jobs.values()),workers=workers,cpu_only=True,nice=os.getpriority(os.PRIO_PROCESS,0),max_processes=workers+1,max_address_space_gib=(workers+1)*8))
    print("GALLERY_PLAN",len(items),len(jobs),"houses",flush=True)
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(render,j,raw):j["house"] for j in jobs.values()}
        for f in as_completed(futures):
            result=f.result();receipts.append(result);print("CPU_RENDERED",result["house"],flush=True)
    dump(out/"render_receipt.json",receipts)
    from html import escape
    html=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>30–50 m² 真实房间图集</title><style>body{background:#eef2f6;font:16px/1.6 system-ui;color:#213248;margin:0}main{max-width:1150px;margin:auto;padding:24px}article{background:white;padding:16px;margin:22px 0;border-radius:8px}img{width:100%;height:auto;aspect-ratio:1;background:#dce3ea}code{overflow-wrap:anywhere}small{color:#53697e}</style><main><p>结论：此页展示当前名单的原生30–50 m²房间，以及v2新切房间的真实整屋CPU俯视图，供判断面积上限；未改任何名单或冻结门槛。</p>']
    html.append(f'<h1>30–50 m² 房间长什么样</h1><p>实测原生 {sum(x["group"]=="native" for x in items)} 间；v2新切 {sum(x["group"]=="split_v2" for x in items)} 间。蓝色=语义地面；黄色虚线=该地面X/Z轴对齐外接矩形；每张有5米尺。完整原GLB，按本层常规去顶切片，XZ视域覆盖整套房子；CPU llvmpipe，nice=10。</p><p>新切组来自v2，包含owner已指出的飞地；连通性会在B修正。本页不将它们伪称为v3合格房间。矩形展示的是地面形状的外接矩形，生产旧rooms.json按成员面的框另在C量化。</p>')
    for group,title in [("native","现有名单：原生房间"),("split_v2","本次新切：v2保留房间")]:
        selected=sorted([x for x in items if x["group"]==group],key=lambda x:(-x["area_m2"],x["id"]))
        html.append("<h2>"+title+f'（{len(selected)}间）</h2>')
        for i,item in enumerate(selected):
            meta=json.loads((raw/(item["house"]+"__"+item["floor_key"]+".json")).read_text());im=overlay(item,meta)
            p=media/(item["id"]+".jpg")
            with p.open("xb") as f:im.save(f,format="JPEG",quality=88)
            item["image_path"]=str(p);item["render_metadata"]=str(raw/(item["house"]+"__"+item["floor_key"]+".json"))
            data="data:image/jpeg;base64,"+base64.b64encode(p.read_bytes()).decode()
            note=f'v2原多边形连通片={item["v2_component_count"]}' if group=="split_v2" else "名单出处："+item["selection_file"]
            html.append(f'<article><h3>{i+1}. {escape(item["label"])} — {item["area_m2"]:.3f} m²，短边 {item["short_side_m"]:.3f} m</h3><small>{escape(note)}；楼层语义地面高度 {item["floor_y_m"]:.3f} m</small><img data-src="{data}" loading="lazy" decoding="async" width="1200" height="1200" alt="{escape(item["label"],quote=True)}"></article>')
    html.append("</main><script>const o=new IntersectionObserver(es=>{for(const e of es)if(e.isIntersecting){const im=e.target;im.src=im.dataset.src;delete im.dataset.src;o.unobserve(im);}},{rootMargin:'0px',threshold:0.01});document.querySelectorAll('img[data-src]').forEach(im=>o.observe(im));</script></html>")
    with (out/"review.html").open("x") as f:f.write("\n".join(html))
    counts={group:{band:sum(x["group"]==group and lo<=x["area_m2"]<(hi if hi<50 else 50+1e-9) for x in items) for band,lo,hi in [("30_40",30,40),("40_50",40,50)]} for group in ["native","split_v2"]}
    dump(out/"summary.json",dict(counts=counts,items=items,missing=missing,html=str(out/"review.html"),nice=10,cpu_only=True,whole_house_frames=sum(len(j["floors"]) for j in jobs.values())))
    with (root/"PROGRESS_zh.md").open("a") as f:f.write("\nGALLERY_READY "+str(out/"review.html")+"\n")
    with (out/"COMPLETED.txt").open("x") as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    print("GALLERY_READY",str(out/"review.html"),counts,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--workers",type=int,default=12);a=p.parse_args();run(a.root,a.workers)
