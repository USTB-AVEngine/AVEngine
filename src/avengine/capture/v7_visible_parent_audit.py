"""Independent native rechecks at immutable completed parent body transforms.

This adds evidence for corrected typed supports and original foot regions.
Completed videos, audio, roots and registry labels remain immutable.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Point,shape
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.capture.v7_visible_attachment import ray,anchor_distance
from avengine.capture.v7_visible_mesh import indexed_static_mesh
from avengine.capture.v7_visible_native import attachment_template
from avengine.capture.v7_visible_semantic_support import SemanticSupports
from avengine.timeline.continuous_pilot import write_json

REVISION='v7-independent-original-native-support-audit-v3-original-paw-soles'


def audit_completed_parent(recipe,visual,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    house=deepcopy(recipe['house']);assets=recipe['assets'];inputs=recipe['inputs']
    scene={'listener_m':visual['camera_frame_readbacks'][0]['position_m'],'camera_forward':[0.,0.,-1.]}
    video=NativeContinuousVideo(house,scene,inputs,[],source_material_cache_directory=directory/'cache')
    rows=[];errors=[]
    try:
        semantic=SemanticSupports(house)
        runtime=SimpleNamespace(habitat_sim=video.hs,magnum=video.mn)
        actors={a['asset_id']:a for a in visual['frames'][0]['actors']}
        old_checks={r['asset_id']:r for r in visual['visible_attachment_checks']}
        floor=float(house['room']['measured_floor_y_m']);free=shape(house['room']['floor_polygon']).buffer(.01)
        for a in assets:
            mesh=indexed_static_mesh(a['habitat_glb']);_,local,definition=attachment_template(np.asarray(mesh.vertices),a['attachment_surface'],a['kind'],a.get('landmarks_local_m'),mesh)
            matrix=np.asarray(actors[a['asset_id']]['body_to_world_m']);probes=np.asarray(local)@matrix[:3,:3].T+matrix[:3,3]
            normal=np.asarray(old_checks[a['asset_id']]['placement']['normal']);hits=[];deltas=[]
            for probe in probes:
                hit=ray(video.sim,runtime,probe+normal*.1,-normal,.22)
                if hit is None:errors.append({'asset_id':a['asset_id'],'reason':'independent original support ray has no hit'});continue
                hits.append(hit.tolist());deltas.append(float((probe-hit)@normal))
            typed=semantic.qualify_points(a['attachment_surface'],[{'point':p} for p in hits])
            reasons=[]
            if len(hits)!=len(probes):reasons.append('missing_support_raycast')
            if max((abs(x) for x in deltas),default=float('inf'))>.015:reasons.append('attachment_exceeds_15mm')
            if len(typed)!=len(hits):reasons.append('wrong_original_scan_surface_type')
            if a['attachment_surface']=='floor' and any(abs(p[1]-floor)>.08 or not free.covers(Point(p[0],p[2])) for p in hits):reasons.append('not_on_measured_free_floor')
            door=None
            if a.get('object_type')=='doorbell_chime_unit':
                point=old_checks[a['asset_id']]['placement']['support'].get('original_surface_point_m',old_checks[a['asset_id']]['placement']['support']['point'])
                anchors=list(semantic.anchors['door'])
                # An original native ray-measured opening is also evidence;
                # never manufacture an anchor from the requested input seed.
                previous=old_checks[a['asset_id']]['placement']['support'].get('native_door_anchor',{})
                if previous.get('authority','').startswith('native raycast rectangular room-entry doorway:'):anchors.append(previous)
                if anchors:
                    nearest=min(anchors,key=lambda d:anchor_distance(point,d));distance=anchor_distance(point,nearest)
                    door={'distance_3d_m':distance,'threshold_m':1.2,'anchor':nearest}
                    if distance>1.2:reasons.append('doorbell_not_near_actual_3d_door')
                else:reasons.append('missing_original_door_evidence')
            row={'asset_id':a['asset_id'],'status':'fail' if reasons else 'pass','surface':a['attachment_surface'],'support_raycast_points_m':hits,'signed_contact_errors_m':deltas,'worst_error_m':max((abs(x) for x in deltas),default=None),'typed_support_points':typed,'door_proximity':door,'original_body_matrix_m':matrix.tolist(),'original_probe_definition':definition,'failure_reasons':reasons}
            rows.append(row)
            errors.extend({'asset_id':a['asset_id'],'reason':r} for r in reasons)
    finally:video.close()
    result={'revision':REVISION,'status':'fail' if errors else 'pass','checks':rows,'errors':errors,'ground_surface_policy':'original floor/carpet/rug/doormat/floor-mat triangles; measured floor level and free polygon independently mandatory; original 15mm contact threshold unchanged','all_native_media_reused_unchanged':True,'registered_labels_changed':0,'renderer':video.receipt,'scene_id':recipe['row']['scene_id'],'source_registry_hashes':{a['asset_id']:a['source_registry_sha256'] for a in assets}}
    write_json(directory/'result.json',result);return result
