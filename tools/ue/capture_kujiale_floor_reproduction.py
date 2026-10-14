"""UE-native before/after replay of the retained Kujiale bedroom defect."""
from pathlib import Path
import json
import math

import cv2
import numpy as np

from avengine.capture.kujiale_spear_runtime import apply_camera, from_ue, to_ue
from avengine.backends.spear_ue.research_runtime import read_actor_pose
from capture_kujiale_admission import Capture, write
from avengine.qa.kujiale_native_depth import world_pixels,far_range,REVISION



def native_sole(c,actor,position):
    from tools.rooms.check_v7_kujiale_native_assets import vector_field
    from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry
    g=NativeSupportGeometry(c.ue['native_floor_surface'])
    rows=[]
    with c.instance.begin_frame():
        b=actor.GetActorBounds(bOnlyCollidingComponents=False,bIncludeFromChildActors=True,as_dict=True)
    with c.instance.end_frame():pass
    center=np.asarray(from_ue(vector_field(b,'Origin')));ext=np.abs(np.asarray(from_ue(vector_field(b,'BoxExtent'))))
    for offset,yaw in [([0,1,-3],90),([0,1,3],-90)]:
        eye=np.asarray(position)+offset
        def setup():
            apply_camera(c.camera,{'position_m':eye.tolist(),'yaw_deg':yaw,'pitch_deg':0.,'hfov_deg':85})
            c.components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
            c.components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[actor])
        c.frame(setup);c.frame()
        with c.instance.begin_frame():pass
        with c.instance.end_frame():d=c.depth(c.components['depth'])
        h,w=d.shape;yy,xx=np.indices(d.shape);f=w/(2*math.tan(math.radians(85)/2));D=d
        a=math.radians(yaw);F=np.array([math.cos(a),0,math.sin(a)]);S=np.array([-math.sin(a),0,math.cos(a)]);U=np.array([0,1,0])
        maxd=float(np.linalg.norm(center-eye)+np.linalg.norm(ext)+.1)
        mask=np.isfinite(d)&(d>0)&(d<maxd)
        if not mask.any():rows.append({'status':'fail','reason':'no isolated native human skin depth'});continue
        points=world_pixels(d,{'position_m':eye.tolist(),'yaw_deg':yaw,'pitch_deg':0.,'hfov_deg':85},mask)
        point=points[np.argmin(points[:,1])]
        hit=g.ray(point+[0,.15,0],[0,-1,0],.4)
        gap=None if hit is None else float(point[1]-hit['point_m'][1])
        rows.append({'status':'pass' if hit is not None and '/Meshes/floor/' in hit['material_path'] and abs(gap)<=.02 else 'fail',
                     'lowest_rendered_skin_m':point.tolist(),'gap_m':gap,'support':hit,'tolerance_m':.02,'isolated_pixels':int(mask.sum())})
    return {'status':'pass' if all(x['status']=='pass' for x in rows) else 'fail','views':rows,'method':'Actual native isolated full human skin depth in front and rear, projected sole world point and semantic room-floor ray'}


def run(capture: Capture):
    root = capture.root
    receipt = json.loads((root/"evidence/floor_reproduction.json").read_text())
    references = json.loads((root.parent/"ue_round14/inputs/assets.json").read_text())["assets"]
    results = []
    for case_index, case in enumerate(receipt["results"]):
        reference = next(x for x in references if x["asset_id"] == case["asset_id"])
        row = {"scene_id": case["scene_id"], "asset_id": case["asset_id"],
               "retained_standing_receipt": case["source_receipt"],
               "cpu_floor_rejection": case["rejection_reason"],
               "images": [], "status": "pending"}
        for phase, key in [("before", "old_root_m"), ("after", "corrected_root_m")]:
            position = np.asarray(case[key])
            with capture.instance.begin_frame():
                klass = capture.game.unreal_service.load_class(
                    uclass="AActor", name=reference["ue_generated_class"])
                actor = capture.game.unreal_service.spawn_actor(
                    uclass=klass, spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"})
                actor.SetActorEnableCollision(bNewActorEnableCollision=False)
                actor.K2_GetRootComponent().SetMobility(NewMobility="Movable")
                actor.K2_SetActorLocationAndRotation(
                    NewLocation=dict(zip(["X", "Y", "Z"], to_ue(position))),
                    NewRotation={"Roll": 0., "Pitch": 0., "Yaw": 90.},
                    bSweep=False, bTeleport=True)
            with capture.instance.end_frame():
                pass
            capture.human = actor
            # The retained R1 camera sees the original bed and doorway floor.
            eye = np.array([5.375, 1.5115453898906708, -1.375])
            target = position + [0, .85, 0]
            direction = target-eye
            view = {"position_m": eye.tolist(),
                    "yaw_deg": math.degrees(math.atan2(direction[2], direction[0])),
                    "pitch_deg": math.degrees(math.atan2(direction[1],
                                                       np.linalg.norm(direction[[0, 2]]))),
                    "hfov_deg": 85}
            capture.frame(lambda: apply_camera(capture.camera, view))
            for _ in range(240 if not capture.warmed else 24):
                capture.frame(read=True)
            capture.warmed = True
            rgb, depth = capture.frame(read=True)
            with capture.instance.begin_frame():
                pose = read_actor_pose(actor)
            with capture.instance.end_frame():
                pass
            actual = np.array(from_ue(pose["location_cm"]))
            error = float(np.max(np.abs(actual-position)))
            output = root/"renders/floor_reproduction"/f"case_{case_index}_{phase}.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), cv2.resize(rgb, (1280, 720)),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            row["images"].append({"phase": phase, "rgb": str(output),
                                  "requested_root_m": position.tolist(),
                                  "native_actor_root_m": actual.tolist(),
                                  "native_actor_readback": pose,
                                  "root_readback_error_m": error, "view": view})
            row["images"][-1]["native_sole"]=native_sole(capture,actor,position)
            if phase=="after":
                lowered=position-[0,.1,0]
                capture.frame(lambda:actor.K2_SetActorLocationAndRotation(NewLocation=dict(zip(["X","Y","Z"],to_ue(lowered))),NewRotation={"Roll":0.,"Pitch":0.,"Yaw":90.},bSweep=False,bTeleport=True))
                row["known_bad_10cm_lowered_native_sole"]=native_sole(capture,actor,lowered)
            capture.cleanup()
        row["status"] = ("pass" if all(x["root_readback_error_m"] <= .0005
                                      for x in row["images"])
                         and case["new_original_position_status"] == "rejected"
                         and case["corrected_capsule"]["status"] == "pass"
                         and row["images"][-1]["native_sole"]["status"]=="pass"
                         and row["known_bad_10cm_lowered_native_sole"]["status"]=="fail" else "fail")
        results.append(row)
        write(root/"evidence/floor_native_reproduction.json",
              {"status": "pass" if len(results)==8 and all(x["status"]=="pass" for x in results)
               else "pending", "house": "kujiale_0003", "cases": results, "measurement_revision": REVISION,
               "native_map_path": capture.house["input"]["ue_map_path"],
               "launch_receipt": capture.launch,
               "method": "Original registered UE actor, retained bed coordinates, "
                         "new semantic room-floor gate, actual actor pose and native RGB"})
        print("FLOOR_NATIVE", case_index, row["status"], flush=True)
