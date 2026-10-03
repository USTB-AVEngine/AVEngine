"""Native textured Habitat-Sim views with saved camera calibration.

Orthographic images use a floor-relative cutaway, not a geometry-only proxy.
MP3D quality is computed by the merged black_metric on the unmodified PNG.
HM3D numeric measurements and judgments are never recomputed here.
"""
from __future__ import annotations
import json, math, os, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import shape, box
from .run import write_json, now
from .runtime import habitat_runtime_options
from .media import black_metric, overlay
from .protocol import decide
from .workflow import prediction
from .demo_textures import prepare_texture_scene


def native_matrices(sensor):
    camera=sensor._sensor_object.render_camera
    return np.asarray(camera.camera_matrix,dtype=float),np.asarray(camera.projection_matrix,dtype=float)


class Renderer:
    def __init__(self, source, rt):
        self.rt=rt; self.hs=hs=rt.habitat_sim
        cfg=hs.SimulatorConfiguration();cfg.scene_id=str(source)
        cfg.gpu_device_id=0;cfg.enable_physics=False;cfg.load_semantic_mesh=False
        cfg.physics_config_file=str(rt.physics_config_path);cfg.requires_textures=True
        from habitat_sim.gfx import NO_LIGHT_KEY
        cfg.scene_light_setup=NO_LIGHT_KEY
        top=hs.CameraSensorSpec();top.uuid="top";top.sensor_type=hs.SensorType.COLOR
        top.sensor_subtype=hs.SensorSubType.ORTHOGRAPHIC;top.resolution=rt.magnum.Vector2i([1536,1536])
        top.position=rt.magnum.Vector3(0,0,0);top.orientation=rt.magnum.Vector3(-math.pi/2,0,0);top.near=0.05;top.far=2.3
        rgb=hs.CameraSensorSpec();rgb.uuid="rgb";rgb.sensor_type=hs.SensorType.COLOR
        rgb.sensor_subtype=hs.SensorSubType.PINHOLE;rgb.resolution=rt.magnum.Vector2i([540,960])
        rgb.position=rt.magnum.Vector3(0,0,0);rgb.hfov=85.0;rgb.near=0.05;rgb.far=1000
        agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[top,rgb]
        self.sim=hs.Simulator(hs.Configuration(cfg,[agent]));self.agent=self.sim.get_agent(0)
        self.source=str(source)

    def pose(self,eye,yaw=0,elevation=0):
        state=self.hs.AgentState();state.position=np.asarray(eye,dtype=np.float32)
        state.rotation=self.rt.quaternion.from_rotation_vector([0,yaw,0])*self.rt.quaternion.from_rotation_vector([elevation,0,0])
        self.agent.set_state(state,reset_sensors=True)

    def grab(self,key):
        obs=self.sim.get_sensor_observations()
        view,projection=native_matrices(self.sim._sensors[key])
        image=Image.fromarray(np.asarray(obs[key])[:,:,:3].astype(np.uint8))
        return image,view,projection

    def topdown(self,geometry,floor_y,path):
        x0,z0,x1,z1=geometry.bounds
        span=max(2.0,max(x1-x0,z1-z0)*1.13+0.25)
        eye=[(x0+x1)/2,floor_y+1.65,(z0+z1)/2]
        self.pose(eye)
        obj=self.sim._sensors["top"]._sensor_object
        _,initial=native_matrices(self.sim._sensors["top"])
        obj.zoom((2/abs(initial[0,0]))/span)
        image,view,projection=self.grab("top")
        if not np.allclose(view@np.r_[eye,1],[0,0,0,1],atol=2e-4): raise ValueError("native view matrix does not map the camera origin")
        if not np.isclose(2/abs(projection[0,0]),span,rtol=2e-5): raise ValueError("orthographic span calibration failed")
        path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        entry=dict(floor_y_m=float(floor_y),view=view.tolist(),orientation="native Habitat calibration; +X right, +Z down",camera_m=eye,
            slice_height_above_floor_m=1.60,slice_bottom_below_floor_m=0.65,source_glb=self.source,
            calibration="native sensor projection/view matrices; orthographic span and camera origin numerically verified")
        record=dict(projection=projection.tolist(),span_m=float(span),path=str(path),width=image.width,height=image.height)
        write_json(path.with_suffix(".camera.json"),dict(**entry,images=[record]))
        return entry,record,image,dict(metadata_cache=str(path.with_suffix(".camera.json")),image_cache=str(path),orientation=entry["orientation"])

    def interior(self,camera,targets,path,yaw_offset=0):
        eye=np.asarray(camera,float)
        if targets:
            vectors=np.asarray(targets,float)-eye
            vectors/=np.linalg.norm(vectors,axis=1)[:,None]
            direction=vectors.sum(axis=0)
        else: direction=np.array([0,0,-1.0])
        yaw=math.atan2(-direction[0],-direction[2])+yaw_offset
        elevation=math.atan2(direction[1],np.linalg.norm(direction[[0,2]]))
        self.pose(eye,yaw,elevation);image,view,projection=self.grab("rgb")
        hfov=math.degrees(2*math.atan(1/abs(projection[0,0])))
        if abs(hfov-85)>0.01:raise ValueError("perspective does not have 85 degree horizontal FoV")
        path.parent.mkdir(parents=True,exist_ok=True);image.save(path,quality=89)
        return dict(path=str(path),camera_m=eye.tolist(),target_m=np.asarray(targets).tolist(),horizontal_fov_deg=hfov,width=image.width,height=image.height,view=view.tolist(),projection=projection.tolist(),source_glb=self.source)

    def close(self):self.sim.close()


def dominant_metrics(row):
    floors=row.get("floors",[])
    if not floors:return None
    selected=(row.get("floor_selection") or {}).get("selected_floor_id")
    return next((f for f in floors if f["floor_id"]==selected),max(floors,key=lambda f:f["metrics"]["floor_area_m2"]))


def visual_bbox(row):
    """Read a source registration box for display only, never measurement."""
    native = row.get("native_region") or {}
    if native.get("bbox_xz_m"):
        return native["bbox_xz_m"], "native"
    legacy = row.get("legacy_room") or {}
    if legacy.get("bbox_xz_m"):
        return legacy["bbox_xz_m"], "legacy_reference"
    return None, None


def house_layers(data):
    rows=data["rows"];family=data["family"]
    if family=="mp3d":
        ids=sorted({r["native_region"]["level_index"] for r in rows})
        groups=[(i,[r for r in rows if r["native_region"]["level_index"]==i]) for i in ids]
    else:
        # Descriptive building-level windows. This never changes the frozen
        # per-region <=0.3m windows or dominant-area qualification.
        heights=sorted(dominant_metrics(r)["metrics"]["floor_y_m"] for r in rows if dominant_metrics(r) and dominant_metrics(r)["metrics"]["floor_area_m2"]>=4)
        centres=[]
        for y in heights:
            if not centres or y-centres[-1]>1.5:centres.append(y)
        if not centres:centres=[0.0]
        groups=[]
        for i,c in enumerate(centres):
            groups.append((i,[r for r in rows if (not dominant_metrics(r) and i==0) or (dominant_metrics(r) and int(np.argmin(np.abs(np.asarray(centres)-dominant_metrics(r)["metrics"]["floor_y_m"])))==i)]))
    result=[]
    for i,rs in groups:
        measured=[dominant_metrics(r) for r in rs if dominant_metrics(r)]
        if measured:
            ordered=sorted([(f["metrics"]["floor_y_m"],f["metrics"]["floor_area_m2"]) for f in measured])
            weight=sum(w for _,w in ordered);running=0;fy=ordered[0][0]
            for y,w in ordered:
                running+=w
                if running>=weight/2:fy=y;break
        else:
            registered_heights = [(r.get("native_region") or {}).get("floor_y_m", (r.get("legacy_room") or {}).get("floor_y_m")) for r in rs]
            registered_heights = [float(y) for y in registered_heights if y is not None]
            fy = float(np.median(registered_heights)) if registered_heights else 0.0
        scopes=[]
        for r in rs:
            f=dominant_metrics(r)
            if f: scopes.append(shape(f["metrics"]["floor_polygon"]))
            else:
                bounds, _ = visual_bbox(r)
                if bounds:
                    scopes.append(box(*np.asarray(bounds).ravel()))
        from tools.rooms.room_screening.geometry import union_projected_polygons
        union=union_projected_polygons(scopes)
        if not union.is_empty:result.append(dict(level_id=i,floor_y_m=float(fy),room_labels=[r["room_label"] for r in rs],geometry=union))
    return result


def reference_target(camera, scope, floor_y, p):
    """Aim a display-only reference into the room, avoiding a vertical view."""
    eye = np.asarray(camera, float)
    point = scope.representative_point()
    x, z = point.x, point.y
    if np.linalg.norm(eye[[0, 2]] - [x, z]) < .5:
        from .media import polygons
        boundary = [v for polygon in polygons(scope) for v in polygon.exterior.coords]
        if boundary:
            x, z = max(boundary, key=lambda v: (v[0]-eye[0])**2 + (v[1]-eye[2])**2)
    return [float(x), float(floor_y + p["source_height_m"]), float(z)]


def reference_pose(renderer,row,p):
    """A explicitly marked native-supported preview if no placement exists."""
    views=(row.get("native_region") or {}).get("native_viewpoints",[])
    if views:
        camera = views[0]["camera_m"]
        floor = dominant_metrics(row)
        native = row.get("native_region") or {}
        scope = shape(floor["metrics"]["floor_polygon"] if floor else native.get("native_floor_polygon", {"type": "GeometryCollection", "geometries": []}))
        if scope.is_empty and native.get("bbox_xz_m"):
            scope = box(*np.asarray(native["bbox_xz_m"]).ravel())
        floor_y = floor["metrics"]["floor_y_m"] if floor else camera[1]-p["camera_height_m"]
        targets = [reference_target(camera, scope, floor_y, p)] if not scope.is_empty else []
        return dict(camera_m=camera, targets=targets, kind="native_acquisition_reference", source=views[0]["source"])
    floor=dominant_metrics(row)
    if not floor:
        bounds, origin = visual_bbox(row)
        if not bounds:
            return None
        scope = box(*np.asarray(bounds).ravel())
        low, high = renderer.sim.pathfinder.get_bounds()
        legacy_y = (row.get("legacy_room") or {}).get("floor_y_m")
        heights = ([float(legacy_y)] if legacy_y is not None else [])
        heights += np.arange(float(low[1]), float(high[1]) + 0.3, 0.3).tolist()
        for y in heights:
            for fx, fz in [(0.5, 0.5)] + [(a, b) for a in [.1, .3, .7, .9] for b in [.1, .3, .7, .9]]:
                pt = [scope.bounds[0] + fx * (scope.bounds[2] - scope.bounds[0]), y,
                      scope.bounds[1] + fz * (scope.bounds[3] - scope.bounds[1])]
                q = np.asarray(renderer.sim.pathfinder.snap_point(pt), float)
                if not np.isfinite(q).all() or abs(q[1] - y) > p["floor_height_separation_m"]:
                    continue
                from shapely.geometry import Point
                if not scope.covers(Point(q[0], q[2])):
                    continue
                camera = q + [0, p["camera_height_m"], 0]
                target = reference_target(camera, scope, q[1], p)
                return dict(camera_m=camera.tolist(), targets=[target],
                            kind="legacy_box_navmesh_reference" if origin == "legacy_reference" else "native_box_navmesh_reference",
                            source=row.get("legacy_rooms_source") or row.get("navmesh_source"),
                            navigation_source=row.get("navmesh_source"),
                            registration_bbox_xz_m=bounds,
                            note="display reference inside source box and existing navmesh; semantic floor and placement NOT established")
        return None
    m=floor["metrics"];scope=shape(m["floor_polygon"])
    support=shape(m.get("nav_grid_main_polygon",m["floor_polygon"]))
    candidates=[support.representative_point(),scope.representative_point(),support.centroid]
    from shapely.geometry import Point
    x0, z0, x1, z1 = scope.bounds
    candidates += [Point(x0 + a*(x1-x0), z0 + b*(z1-z0))
                   for a in np.linspace(.05, .95, 15) for b in np.linspace(.05, .95, 15)
                   if scope.covers(Point(x0 + a*(x1-x0), z0 + b*(z1-z0)))]
    for pt in candidates:
        if pt.is_empty:continue
        q=np.asarray(renderer.sim.pathfinder.snap_point([pt.x,m["floor_y_m"],pt.y]),float)
        if np.isfinite(q).all() and abs(q[1]-m["floor_y_m"])<=p["floor_height_separation_m"] and scope.covers(__import__("shapely").geometry.Point(q[0],q[2])):
            camera=q+[0,p["camera_height_m"],0]
            target=reference_target(camera, scope, m["floor_y_m"], p)
            return dict(camera_m=camera.tolist(),targets=[target],kind="native_navmesh_reference",source="native navigation support only; three-point placement NOT established")
    # Some real semantic floors have no support in the existing navmesh.
    # Show the scan from a furniture-free semantic point, explicitly without
    # navigation or placement evidence. Never feed this pose to qualification.
    furniture = shape(m.get("furniture_polygon", {"type": "GeometryCollection", "geometries": []}))
    free = scope.difference(furniture.buffer(p["placement_camera_clearance_m"]))
    if free.is_empty:
        free = scope
    from .media import polygons
    pieces = list(polygons(free))
    if not pieces:
        return None
    point = max(pieces, key=lambda g: g.area).representative_point()
    camera = [point.x, m["floor_y_m"] + p["camera_height_m"], point.y]
    corners = [(x, z) for x in [scope.bounds[0], scope.bounds[2]]
               for z in [scope.bounds[1], scope.bounds[3]]]
    tx, tz = max(corners, key=lambda v: (v[0]-point.x)**2 + (v[1]-point.y)**2)
    return dict(camera_m=camera, targets=[[tx, m["floor_y_m"]+1.2, tz]],
                kind="semantic_floor_reference_no_navmesh", floor_id=floor["floor_id"],
                source=row.get("semantic_source"), navigation_source=row.get("navmesh_source"),
                note="semantic-floor display point only; existing navmesh did not support a reference within this floor; placement NOT established")


def render_house(args,house,family,rt,inputs):
    src=args.output/("hm3d_houses" if family=="hm3d" else "houses")/(house+".json")
    if not src.exists():raise FileNotFoundError("CPU/source result not yet ready: "+str(src))
    data=json.loads(src.read_text());rows=data["rows"];p=inputs["parameters"]
    scene=Path(rows[0]["scene_directory"])
    source=Path(rows[0].get("collision_source") or (scene/(scene.name.split("-",1)[-1]+".glb")))
    started=time.time();out=args.output
    runtime_scene,texture_evidence=prepare_texture_scene(source,out)
    renderer=Renderer(runtime_scene,rt)
    if not renderer.sim.pathfinder.load_nav_mesh(rows[0]["navmesh_source"]):
        renderer.close()
        raise ValueError("existing source navmesh failed to load for native preview")
    try:
        levels=house_layers(data);data["demo_floor_images"]=[]
        for level in levels:
            path=out/"demo"/"assets"/house/("level_"+str(level["level_id"])+"_texture.png")
            entry,record,image,_=renderer.topdown(level.pop("geometry"),level["floor_y_m"],path)
            data["demo_floor_images"].append(dict(**level,path=str(path),entry=entry,image_record=record))
        for row in rows:
            label=row["room_label"];visual=dict(interiors=[],quality_images=[])
            if family=="mp3d":
                for floor in row.get("floors",[]):
                    m=floor["metrics"];scope=shape(m["floor_polygon"])
                    path=out/"demo"/"assets"/house/(label+"_"+floor["floor_id"]+"_quality.png")
                    overhead=renderer.topdown(scope,m["floor_y_m"],path)
                    m.update(black_metric(scope,m["floor_y_m"],*overhead[:3],p))
                    m["black_metric_source"]=str(path);floor["stage1"]=decide(m,row["room_type"],p)
                    visual["quality_images"].append(dict(path=str(path),floor_id=floor["floor_id"],kind="new native calibrated orthographic texture",black_fraction=m.get("black_fraction")))
                    parts=floor.get("split_parts",[])
                    for part in parts:
                        pm=part["metrics"];pm.update(black_metric(shape(pm["floor_polygon"]),m["floor_y_m"],*overhead[:3],p));part["stage1"]=decide(pm,row["room_type"],p)
                    if parts:
                        path2=out/"demo"/"assets"/house/(label+"_"+floor["floor_id"]+"_split.png")
                        overlay(path2,scope,m["floor_y_m"],[shape(part["metrics"]["floor_polygon"]) for part in parts],overhead)
                        floor["stage2"].update(overlay_path=str(path2),overlay_status="native calibrated textured overhead")
                        floor["stage2"]["reason_codes"]=[c for c in floor["stage2"].get("reason_codes",[]) if c not in {"SPLIT_RGB_OVERHEAD_MISSING","SPLIT_OVERHEAD_FLOOR_MISMATCH"}]
                selected=(row.get("floor_selection") or {}).get("selected_floor_id")
                if selected is not None:
                    floor=next(f for f in row["floors"] if f["floor_id"]==selected)
                    row["metrics"]=floor["metrics"];row["stage1"]=dict(floor["stage1"])
                if row["stage1"]["status"]!=prediction(row,p):raise ValueError("demo qualification diverged from frozen merged prediction")
                if row.get("stage2",{}).get("status")=="proposed":
                    row["stage2"]["reason_codes"]=[c for c in row["stage2"].get("reason_codes",[]) if c not in {"SPLIT_RGB_OVERHEAD_MISSING","SPLIT_OVERHEAD_FLOOR_MISMATCH"}]
            else:
                reference=row.get("overhead") or {}
                img=reference.get("image_cache")
                if img and Path(img).is_file():
                    path=out/"demo"/"assets"/house/(label+"_original_calibrated.png")
                    path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(img,path)
                    visual["quality_images"].append(dict(path=str(path),kind="unchanged original HM3D calibrated overview used by frozen measurements",black_fraction=(row.get("metrics") or {}).get("black_fraction"),source=img))
            preferred=dominant_metrics(row)
            witnesses=[f for f in row.get("floors",[]) if (f["metrics"].get("placement") or {}).get("found")]
            chosen=preferred if preferred in witnesses else max(witnesses,key=lambda f:f["metrics"]["floor_area_m2"],default=None)
            pose=None
            if chosen:
                w=chosen["metrics"]["placement"]
                pose=dict(camera_m=w["camera_m"],targets=[w["source_1_m"],w["source_2_m"]],kind="placement_witness",floor_id=chosen["floor_id"],witness=w)
            elif row["stage1"]["status"]=="review" or (row.get("human") or {}).get("verdict")=="use":
                pose=reference_pose(renderer,row,p)
            if pose:
                path=out/"demo"/"assets"/house/(label+"_interior.jpg")
                record=renderer.interior(pose["camera_m"],pose["targets"],path)
                record.update({k:v for k,v in pose.items() if k not in {"camera_m","targets"}})
                visual["interiors"].append(record)
                if pose["kind"]=="placement_witness" and pose["witness"]["horizontal_angle_deg"]<55:
                    path2=out/"demo"/"assets"/house/(label+"_interior_2.jpg")
                    record2=renderer.interior(pose["camera_m"],pose["targets"],path2,yaw_offset=math.radians(8))
                    record2.update(kind=pose["kind"],floor_id=pose["floor_id"],witness=pose["witness"])
                    visual["interiors"].append(record2)
            visual["interior_status"]="rendered" if visual["interiors"] else "no justified placement/reference camera available"
            row["demo_visual"]=visual
            print("NATIVE_REGION",house,label,row["stage1"]["status"],len(visual["interiors"]),flush=True)
        data["demo_native_render"]=dict(status="complete",source_glb=str(source),runtime_glb=str(runtime_scene),texture_evidence=texture_evidence,finished_at_sgt=now(),seconds=time.time()-started,horizontal_fov_deg=85,orthographic_cutaway_above_floor_m=1.6,mp3d_quality_policy="merged black_metric on raw calibrated PNG",hm3d_policy="all source metrics and judgments preserved")
        if family=="mp3d":
            nav=rows[0].get("navmesh_identity") or {}
            data["demo_house_gate"]=dict(status="not_run",reason_codes=["FORMAL_HOUSE_AUDIOVISUAL_GATE_NOT_AVAILABLE"],completed_diagnostics=dict(asset_registration="pass",existing_navmesh_read="pass" if nav.get("native_area_m2",0)>0 else "fail",native_textured_rgb="pass"),remaining=["formal MP3D house-level audio/RIR and downstream episode closure"],note="diagnostic RGB/nav checks do not substitute for the stage-4 formal gate")
        else:data["demo_house_gate"]=dict(rows[0].get("stage4",{}),source="unchanged original merged stage4 record")
        write_json(out/"rendered_houses"/(house+".json"),data)
    finally:renderer.close()
    print("NATIVE_HOUSE_DONE",house,round(time.time()-started,2),flush=True)
    return data


def supplement_height_maps(args, data, rt, p):
    """Display missed camera heights; never change a measurement or decision."""
    renderer = Renderer(Path(data["demo_native_render"]["runtime_glb"]), rt)
    started = time.time()
    try:
        anchors = []
        for row in data["rows"]:
            pictures = row.get("demo_visual", {}).get("interiors", [])
            if pictures and pictures[0]["kind"] == "placement_witness":
                pic = pictures[0]
                anchors.append((pic["camera_m"][1]-p["camera_height_m"], row, pic["floor_id"]))
        for ground, anchor_row, anchor_id in sorted(anchors, key=lambda a: a[0]):
            # 0.60m is display coverage within the fixed -0.65m cutaway bottom,
            # not a new admission threshold or a claim about building stories.
            if min(abs(l["floor_y_m"]-ground) for l in data["demo_floor_images"]) <= .60:
                continue
            scopes, labels = [], []
            for row in data["rows"]:
                matching = [f for f in row.get("floors", []) if abs(f["metrics"]["floor_y_m"]-ground) <= .65]
                if row is anchor_row:
                    matching += [f for f in row["floors"] if f["floor_id"] == anchor_id and f not in matching]
                if matching:
                    labels.append(row["room_label"])
                    scopes += [shape(f["metrics"]["floor_polygon"]) for f in matching]
            from tools.rooms.room_screening.geometry import union_projected_polygons
            geometry = union_projected_polygons(scopes)
            if geometry.is_empty:
                raise ValueError("witness height has no source floor footprint")
            level_id = max(l["level_id"] for l in data["demo_floor_images"])+1
            path = args.output/"demo/assets"/data["house"]/(f"level_{level_id}_texture.png")
            entry, record, _, _ = renderer.topdown(geometry, ground, path)
            data["demo_floor_images"].append(dict(level_id=level_id, floor_y_m=ground,
                room_labels=labels, path=str(path), entry=entry, image_record=record,
                kind="witness_camera_height_supplement", anchor_region=anchor_row["room_label"],
                anchor_floor_id=anchor_id, explanation="camera height cutaway only; not a certified additional building story"))
        data["demo_native_render"].update(supplementary_height_maps_at_sgt=now(),
            supplementary_height_map_seconds=time.time()-started)
        write_json(args.output/"rendered_houses"/(data["house"]+".json"),data)
    finally:
        renderer.close()
    print("SUPPLEMENTARY_MAPS_DONE", data["house"], len(data["demo_floor_images"]), flush=True)


def fit_overviews(args, data, rt):
    """Fit every displayed contour; retain original measurement images."""
    from .demo_page import visual_bbox
    from .media import project_xz
    from tools.rooms.room_screening.geometry import union_projected_polygons
    renderer = Renderer(Path(data["demo_native_render"]["runtime_glb"]), rt)
    fitted = []
    try:
        levels = data["demo_floor_images"]
        heights = np.asarray([level["floor_y_m"] for level in levels])
        for index, level in enumerate(levels):
            scopes = []
            for row in data["rows"]:
                parts = [shape(f["metrics"]["floor_polygon"]) for f in row.get("floors", [])
                    if int(np.argmin(abs(heights-f["metrics"]["floor_y_m"]))) == index]
                if not parts and row["room_label"] in level["room_labels"]:
                    bounds, _ = visual_bbox(row)
                    if bounds:
                        g = shape((row.get("native_region") or {}).get("native_floor_polygon", {"type":"GeometryCollection", "geometries":[]}))
                        parts = [g if not g.is_empty else box(*np.asarray(bounds).ravel())]
                scopes.extend(parts)
            geometry = union_projected_polygons(scopes)
            if geometry.is_empty:
                continue
            x0, z0, x1, z1 = geometry.bounds
            pixels = np.asarray(project_xz([(x0,z0),(x0,z1),(x1,z0),(x1,z1)],
                level["floor_y_m"],level["entry"],level["image_record"],(1536,1536)))
            if ((pixels[:,0] >= 0) & (pixels[:,0] < 1536) & (pixels[:,1] >= 44) & (pixels[:,1] < 1536)).all():
                continue
            path = args.output/"demo/assets"/data["house"]/(f"level_{level['level_id']}_expanded_texture.png")
            entry, record, _, _ = renderer.topdown(geometry,level["floor_y_m"],path)
            level.setdefault("previous_display_camera",dict(path=level["path"],entry=level["entry"],image_record=level["image_record"]))
            level.update(path=str(path),entry=entry,image_record=record,
                expanded_display_only=True,expansion_reason="fit all assigned floor and registration-reference contours; original measurement images retained")
            fitted.append(level["level_id"])
        data["demo_native_render"].update(fitted_overviews_at_sgt=now(),fitted_overview_ids=fitted)
        write_json(args.output/"rendered_houses"/(data["house"]+".json"),data)
    finally:
        renderer.close()
    print("OVERVIEWS_FITTED",data["house"],fitted,flush=True)


def render(args):
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    inputs=json.loads((args.output/"inputs.json").read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):raise ValueError("explicit GPU must agree with CUDA_VISIBLE_DEVICES")
    rt=prepare_installed_habitat_runtime(**habitat_runtime_options())
    if args.probe:
        h=next(h for h in inputs["mp3d"] if h["house"]==args.house)
        scene=Path(h["scene_directory"]);source=scene/(scene.name+".glb")
        runtime_scene,texture_evidence=prepare_texture_scene(source,args.output)
        renderer=Renderer(runtime_scene,rt)
        try:
            rows=json.loads((args.output/"jobs"/(args.house+".json")).read_text())["rows"]
            scopes=[box(*np.asarray(r["native_region"]["bbox_xz_m"]).ravel()) for r in rows if r["native_region"]["level_index"]==0]
            from tools.rooms.room_screening.geometry import union_projected_polygons
            floor_y=float(np.median([r["native_region"]["floor_y_m"] for r in rows if r["native_region"]["level_index"]==0]))
            entry,record,image,_=renderer.topdown(union_projected_polygons(scopes),floor_y,args.output/"evidence"/"gpu_probe_texture.png")
            renderer.interior([entry["camera_m"][0],floor_y+1.5,entry["camera_m"][2]],[],args.output/"evidence"/"gpu_probe_interior.jpg")
            write_json(args.output/"evidence/gpu_probe_native.json",dict(status="complete",source_glb=str(source),runtime_glb=str(runtime_scene),texture_evidence=texture_evidence,entry=entry,image_record=record))
            time.sleep(3)
        finally:renderer.close()
        return
    houses=[(h["house"],"hm3d") for h in inputs["hm3d"]]+[(h["house"],"mp3d") for h in inputs["mp3d"]]
    if args.house:houses=[h for h in houses if h[0]==args.house]
    for house,family in houses:
        dest=args.output/"rendered_houses"/(house+".json")
        if args.fit_overviews:
            fit_overviews(args,json.loads(dest.read_text()),rt)
            continue
        if args.supplement_only:
            data = json.loads(dest.read_text())
            supplement_height_maps(args, data, rt, inputs["parameters"])
            continue
        if not args.refresh_visuals and dest.exists() and json.loads(dest.read_text()).get("demo_native_render",{}).get("status")=="complete":
            print("NATIVE_CACHED",house,flush=True);continue
        render_house(args,house,family,rt,inputs)
