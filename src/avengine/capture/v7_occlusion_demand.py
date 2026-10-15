"""Construct requested foreground relations; final QA still uses native frames."""
from __future__ import annotations
import numpy as np


def relation_roles(scene, asset_id):
    relations=scene.get('native_occlusion_relations',[])
    targets=[r for r in relations if r['target_asset_id']==asset_id]
    foreground=[r for r in relations if r['occluder_asset_id']==asset_id]
    if targets and foreground:
        raise ValueError('one placement cannot be both the demanded target and foreground')
    return targets,foreground


def placed_target(relation, entries):
    return next((e for e in entries if e['declaration']['asset_id']==relation['target_asset_id']),None)


def foreground_geometry(listener, target_root, root):
    listener=np.asarray(listener,float)[[0,2]]
    target=np.asarray(target_root,float)[[0,2]]-listener
    candidate=np.asarray(root,float)[[0,2]]-listener
    length=float(np.linalg.norm(target))
    if length<=.5:raise ValueError('foreground demand needs a separated target')
    direction=target/length
    fraction=float(candidate@direction)/length
    signed_lateral=float(candidate@np.array([direction[1],-direction[0]]))
    return {'depth_fraction':fraction,'lateral_m':abs(signed_lateral),
            'signed_lateral_m':signed_lateral,
            'foreground':.3<=fraction<=.8 and abs(signed_lateral)<=.5}


def proposal_score(scene, entry, support_root, placed):
    _,relations=relation_roles(scene,entry['declaration']['asset_id'])
    if not relations:return None
    relation=relations[0];target=placed_target(relation,placed)
    if target is None:raise ValueError('requested occluder must be placed after its target')
    metrics=foreground_geometry(scene['listener_m'],target['applied_root_input'][:3,3],support_root)
    # A small lateral offset leaves the rear body partly visible; exact overlap
    # would conceal its appearance. These values rank proposals, never QA truth.
    return (not metrics['foreground'],abs(metrics['lateral_m']-.20),abs(metrics['depth_fraction']-.55))


def measure_requested_relations(video,scene,entries):
    """Probe the same real masks used by final delivery, without retaining truth."""
    byid={e['declaration']['asset_id']:e for e in entries}
    relations=[r for r in scene.get('native_occlusion_relations',[])
               if r['target_asset_id'] in byid and r['occluder_asset_id'] in byid]
    if not relations:return {'satisfies_request':True,'relations':[]}
    from avengine.capture.continuous_native_visibility import NativeVisibilityStream
    video._native_pair_pixels(scene,entries)
    modal=np.asarray(video.sim.get_sensor_observations()['semantic']).copy()
    identities=[{'endpoint_id':e['endpoint_id'],'actor_id':e['declaration']['native_instance_id']}
                for e in video.actors]
    stream=NativeVisibilityStream(video)
    truth=stream.capture(0,modal,identities)
    observed=[]
    for relation in relations:
        target=byid[relation['target_asset_id']];occluder=byid[relation['occluder_asset_id']]
        tid=str(target['declaration']['native_instance_id']);oid=str(occluder['declaration']['native_instance_id'])
        frame=next(r for r in stream.occluder_frames if r['target_instance_id']==tid)
        state=truth['per_instance'][tid]['frames'][0]['state']
        satisfies=(state=='visible_occluded' and frame['occluder_instance_ids']==[oid]
                   and frame['unknown_occluder_pixels']==0)
        observed.append({**relation,'native_state':state,**frame,'satisfies_request':satisfies})
    return {'satisfies_request':all(r['satisfies_request'] for r in observed),
            'relations':observed,'native_control_restoration':stream.receipt(),
            'scope':'layout proposal only; final complete 450-frame stream independently measured'}
