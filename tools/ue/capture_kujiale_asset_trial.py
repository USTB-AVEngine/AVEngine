"""Render each registered body/kind with a visible human in three native houses."""
from pathlib import Path
import json
import math
import traceback

import cv2
import numpy as np

from avengine.capture.kujiale_spear_runtime import apply_camera, to_ue
from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry, import_vertices, rotation
from capture_kujiale_admission import write
from avengine.qa.kujiale_native_depth import far_range,visibility,REVISION


def _depth(capture, actors):
    def select():
        capture.components["depth"].set_property_value(
            property_name="PrimitiveRenderMode", property_value="PRM_UseShowOnlyList")
        capture.components["depth"].set_property_value(
            property_name="ShowOnlyActors", property_value=actors)
    capture.frame(select)
    with capture.instance.begin_frame():
        pass
    with capture.instance.end_frame():
        value = capture.depth(capture.components["depth"])
    return value


def _mask(depth, points, view):
    return np.isfinite(depth)&(depth>0)&(depth<far_range(points,view))


def _views(capture, world, human, catalog, geometry):
    body = np.array([human+[x,y,z] for x in [-.45,.45] for y in [0,1.9]
                     for z in [-.35,.35]])
    lo,hi=world.min(0),world.max(0)
    envelope=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
    points = np.concatenate([envelope, body])
    middle = (points.min(0)+points.max(0))/2
    eyes = [np.asarray(x["placement"]["camera_m"])
            for x in capture.house["checked_rooms"][0]["native_witnesses"]]
    eyes += [np.asarray(x["point_m"])+[0,1.5,0] for x in catalog["floor"]]
    choices = []
    tangent = math.tan(math.radians(85)/2)
    for eye in eyes:
        for target_y in [middle[1], .55, .75, 1.0]:
            target = middle.copy();target[1] = target_y
            delta = target-eye
            yaw = math.atan2(delta[2],delta[0])
            pitch = math.atan2(delta[1],np.linalg.norm(delta[[0,2]]))
            forward = np.array([math.cos(pitch)*math.cos(yaw),math.sin(pitch),
                                math.cos(pitch)*math.sin(yaw)])
            side = np.array([-math.sin(yaw),0,math.cos(yaw)])
            up = np.cross(side,forward)
            relative = points-eye;distance = relative@forward
            if (distance<=.15).any():continue
            x = relative@side/(distance*tangent)
            y = relative@up/(distance*tangent*9/16)
            if np.max(np.abs(x))>.96 or np.max(np.abs(y))>.96:continue
            blocked = False
            for location in [world.mean(0),human+[0,.03,0],human+[0,.35,0],human+[0,.8,0],human+[0,1.7,0],human+[.18,.35,0],human+[-.18,.35,0],human+[0,.35,.15],human+[0,.35,-.15]]:
                d=location-eye;n=float(np.linalg.norm(d))
                if geometry.ray(eye,d/n,n-.04) is not None:blocked=True;break
            if not blocked:
                choices.append({"position_m":eye.tolist(),"yaw_deg":math.degrees(yaw),
                                "pitch_deg":math.degrees(pitch),"hfov_deg":85,
                                "score":float(np.mean(distance))})
    return sorted(choices,key=lambda x:x["score"]),body



def _human_options(capture,placement,catalog,geometry):
    from tools.rooms.check_kujiale_native_capsule_contact import query_capsule
    original=capture.plan_human(placement)
    result=[original];origin=np.asarray(placement['root_m']);ordered=[]
    for support in catalog['floor']:
        point=np.asarray(support['point_m']);distance=float(np.linalg.norm((point-origin)[[0,2]]))
        if 1.30<=distance<=2.7:ordered.append((distance,point,support))
    for _,point,support in sorted(ordered,key=lambda t:t[0]):
        if any(np.linalg.norm((point-np.asarray(v['root_m']))[[0,2]])<.4 for v in result):continue
        capsule=query_capsule(geometry.mesh,{'witness_index':0,'role':'trial_human','capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':to_ue(point+[0,.75,0])}})
        if capsule['status']!='pass':continue
        result.append({'reference':original['reference'],'root_m':point.tolist(),'support':support['support_identity'],'native_capsule':capsule})
        if len(result)>=16:break
    return result

def run(capture):
    root=capture.root
    selection=json.loads((root/"inputs/trial_selection.json").read_text())
    assets={x["asset_id"]:x for x in capture.manifest["assets"]}
    geometry=NativeSupportGeometry(capture.ue["native_floor_surface"])
    catalog,_=geometry.support_catalog(capture.house,capture.house["checked_rooms"][0])
    rows=[]
    previous_path=root/"evidence"/("trial_"+capture.args.house_id+".json")
    previous={v["coverage_key"]:v for v in json.loads(previous_path.read_text())} if previous_path.exists() else {}
    for item in selection["shots"]:
        if item["house"]!=capture.args.house_id:continue
        if item["coverage_key"] in previous and previous[item["coverage_key"]]["status"]=="pass":
            rows.append(previous[item["coverage_key"]]);continue
        row=assets[item["asset_id"]]
        result={"asset_id":row["asset_id"],"house":item["house"],"coverage_key":item["coverage_key"],
                "status":"fail","candidates":[],"measurement_revision":REVISION,"occlusion_tolerance_m":.005}
        try:
            imported=json.loads((root/"evidence/import"/capture.revision/(row["asset_id"]+".json")).read_text())
            vertices=import_vertices(imported)
            plan=json.loads((root/"evidence/plans"/capture.revision/item["house"]/(row["asset_id"]+".json")).read_text())
            for attempt,placement in enumerate(plan["placements"]):
                world=vertices@rotation(placement["yaw_rad"]).T+placement["root_m"]
                choices=[]
                for option in _human_options(capture,placement,catalog,geometry):
                    human=np.asarray(option["root_m"]);views,human_points=_views(capture,world,human,catalog,geometry)
                    if views:choices.append((len(views),option,views,human_points))
                if not choices:continue
                _,human_plan,views,human_points=max(choices,key=lambda t:t[0])
                human=np.asarray(human_plan["root_m"])
                with capture.instance.begin_frame():
                    capture.active=capture.spawn_item(row,imported,placement)
                    result["human"]=capture.spawn_human(human_plan)
                with capture.instance.end_frame():pass
                for index,view in enumerate(views[:12]):
                    def setup():
                        apply_camera(capture.camera,view)
                        capture.components["depth"].set_property_value(property_name="PrimitiveRenderMode",property_value="PRM_RenderScenePrimitives")
                        capture.components["depth"].set_property_value(property_name="ShowOnlyActors",property_value=[])
                    capture.frame(setup)
                    for _ in range(240 if not capture.warmed else 24):capture.frame()
                    capture.warmed=True
                    rgb,scene=capture.frame(read=True)
                    isolated=_depth(capture,[x["visual_actor"] for x in capture.active])
                    person=_depth(capture,[capture.human]);capture.frame(setup)
                    measurements=[]
                    for label,depth,points,minimum in [("asset",isolated,world,80),
                                                       ("human",person,human_points,1000)]:
                        mask=_mask(depth,points,view);visible=mask&(np.abs(scene-depth)<=.005)&np.isfinite(scene)
                        count=int(mask.sum());fraction=float(visible.sum()/count) if count else 0
                        crop=bool(mask[:5].any() or mask[-5:].any() or mask[:,:5].any() or mask[:,-5:].any())
                        highlight=float(np.mean(np.all(rgb[visible]>247,axis=1))) if visible.any() else 1.
                        measurements.append({"subject":label,"pixels":count,"visible_fraction":fraction,
                                             "crop_border":crop,"highlight_ratio":highlight,
                                             "status":"pass" if count>=minimum and fraction>=.99 and not crop and highlight<=.02 else "fail"})
                    record={"view":view,"subjects":measurements}
                    result["candidates"].append(record)
                    if all(x["status"]=="pass" for x in measurements):
                        dest=root/"renders/trial"/item["house"]/(row["asset_id"]+".jpg")
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        cv2.imwrite(str(dest),cv2.resize(rgb,(1280,720)),[cv2.IMWRITE_JPEG_QUALITY,92])
                        result.update(status="pass",rgb=str(dest),placement=placement,
                                      native_emitter=capture.read_emitter(row,placement),selected_view=record)
                        break
                capture.cleanup()
                if result["status"]=="pass":break
            if result["status"]!="pass":result["reason"]="no fully visible asset-and-human native view at unchanged 99% and 2% gates"
        except Exception:
            result["error"]=traceback.format_exc()
            if capture.active or capture.human:capture.cleanup()
        rows.append(result)
        write(root/"evidence"/("trial_"+item["house"]+".json"),rows)
        print("TRIAL",item["house"],row["asset_id"],result["status"],flush=True)
    # Cached passes also append rows; persist them after the last new capture.
    write(previous_path,rows)
    return rows
