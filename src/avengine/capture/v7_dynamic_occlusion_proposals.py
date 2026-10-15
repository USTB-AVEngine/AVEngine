"""Actual moving-body mask probes for requested block-then-clear layouts."""
import numpy as np


def target_root_at(entry,time_s):
    record=entry.get('motion_episode')
    if not record:raise ValueError('dynamic occlusion target has no admitted native motion')
    roots=np.asarray(record['world_roots_m'],float);index=round(float(time_s)*30)
    if not 0<=index<len(roots):raise ValueError('occlusion proposal time outside admitted route')
    return roots[index]


def dynamic_proposal_score(scene,entry,root,placed):
    relations=[r for r in scene.get('native_dynamic_occlusion_relations',[])if r['occluder_asset_id']==entry['declaration']['asset_id']]
    if not relations:return None
    from avengine.capture.v7_occlusion_demand import foreground_geometry,placed_target
    metrics=[]
    for relation in relations:
        target=placed_target(relation,placed)
        if target is None:raise ValueError('dynamic foreground must be placed after the admitted walking target')
        when=relation['query_times_s'][len(relation['query_times_s'])//2]
        metric=foreground_geometry(scene['listener_m'],target_root_at(target,when),root)
        metrics.append((not metric['foreground'],abs(metric['depth_fraction']-relation['foreground_depth_fraction']),
            abs(metric['lateral_m']-relation['preferred_lateral_m'])))
    return tuple(max(m[i]for m in metrics)for i in range(3))


def matches_dynamic_observation(relation,mask,occluder,expected_id):
    if mask['state']!=relation['required_state']:return False
    if occluder['occluder_instance_ids']!=[str(expected_id)]or occluder['unknown_occluder_pixels']:return False
    return float(mask['occlusion_fraction'])+1e-9>=float(relation['minimum_occluded_fraction'])


def measure_dynamic_relations(video,scene,entries):
    byid={e['declaration']['asset_id']:e for e in entries}
    relations=[r for r in scene.get('native_dynamic_occlusion_relations',[])
        if r['target_asset_id']in byid and r['occluder_asset_id']in byid]
    if not relations:return {'satisfies_request':True,'relations':[]}
    from avengine.capture.continuous_native_visibility import NativeVisibilityStream
    stream=NativeVisibilityStream(video);observed=[]
    identities=[{'endpoint_id':e['endpoint_id'],'actor_id':e['declaration']['native_instance_id']}for e in video.actors]
    def probe(when):
        for e in entries:
            if e.get('motion_episode'):video._apply_motion_entry(e,round(when*30))
        video._native_pair_pixels(scene,entries)
        modal=np.asarray(video.sim.get_sensor_observations()['semantic']).copy()
        truth=stream.capture(round(when*15),modal,identities)
        return truth
    try:
        for relation in relations:
            target=byid[relation['target_asset_id']];foreground=byid[relation['occluder_asset_id']]
            tid=str(target['declaration']['native_instance_id']);oid=str(foreground['declaration']['native_instance_id']);samples=[]
            times=relation['query_times_s']
            if len(times)<2 or times[-1]-times[0]+1/15<relation['minimum_stable_seconds']:
                raise ValueError('construction mask probes do not cover the requested duration')
            for when in times:
                truth=probe(when);mask=truth['per_instance'][tid]['frames'][0]
                occ=next(r for r in reversed(stream.occluder_frames)if r['target_instance_id']==tid)
                samples.append({'time_s':when,'mask':mask,'occluder':occ,
                    'satisfies_request':matches_dynamic_observation(relation,mask,occ,oid)})
            later=probe(relation['later_clear_time_s'])['per_instance'][tid]['frames'][0]
            satisfied=all(s['satisfies_request']for s in samples)and later['state']=='visible_clear'
            observed.append({**relation,'samples':samples,'later_native_state':later['state'],'satisfies_request':satisfied})
    finally:
        for e in entries:
            if e.get('motion_episode'):video._apply_motion_entry(e,0)
        video._native_pair_pixels(scene,entries)
    return {'satisfies_request':all(r['satisfies_request']for r in observed),'relations':observed,
        'native_control_restoration':stream.receipt(),'scope':'proposal only; final complete native stream and original QA independently decide'}
