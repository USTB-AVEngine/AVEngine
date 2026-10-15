"""Measured delivery policy for a fixed observer and natural room populations.

These are production/export conditions. QA answer domains and scoring are untouched.
All metrics come from original native readbacks, never requested routes or emitters.
"""
from __future__ import annotations
import hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

REVISION='v7-fixed-observer-natural-layout-measured-speed-v1'
SPEED_RANGES={'human':(.8,1.4),'dog':(.5,1.2),'cat':(.3,.8),'slide':(.2,.5)}


def camera_metrics(readbacks):
    if not readbacks:raise ValueError('missing actual complete camera readbacks')
    p=np.asarray([r['position_m']for r in readbacks],float)
    y=np.degrees(np.unwrap(np.radians([r['yaw_deg']for r in readbacks])))
    if not np.isfinite(p).all()or not np.isfinite(y).all():raise ValueError('nonfinite native camera')
    diameter=max(float(np.linalg.norm(p-v,axis=1).max())for v in p)
    result={'yaw_range_deg':float(np.ptp(y)),'position_diameter_m':diameter,'samples':len(y)}
    forward=[r.get('forward_m')for r in readbacks]
    if all(v is not None for v in forward):
        f=np.asarray(forward,float);f/=np.linalg.norm(f,axis=1)[:,None]
        result['forward_range_deg']=float(np.degrees(np.arccos(np.clip(f@f[0],-1,1))).max())
    orientations=[r.get('orientation_wxyz')for r in readbacks]
    if all(q is not None for q in orientations):
        q=np.asarray(orientations,float);q/=np.linalg.norm(q,axis=1)[:,None]
        result['orientation_range_deg']=float(np.degrees(2*np.arccos(np.clip(np.abs(q@q[0]),0,1))).max())
    result['status']='pass'if result['yaw_range_deg']<=.5+1e-6 and diameter<=.01+1e-9 and result.get('forward_range_deg',0)<=.5+1e-6 and result.get('orientation_range_deg',0)<=.5+1e-6 else'fail'
    return result


def measured_actor_metrics(actors,fps,motion_requests=None):
    results=[]
    for aid,actor in actors.items():
        species=actor.get('species_id');roots=np.asarray(actor.get('root_positions_m',[]),float)
        if species not in {'human','dog','cat','bird'}:continue
        if roots.ndim!=2 or len(roots)<2:raise ValueError('missing actual actor roots: '+str(aid))
        speeds=np.linalg.norm(np.diff(roots[:,[0,2]],axis=0),axis=1)*float(fps)
        moving=speeds>.01
        request=(motion_requests or{}).get(actor.get('asset_id'),{})
        category='slide'if request.get('intentional_slide')or request.get('motion_kind')in {'intentional_slide','slide_no_animation'}or request.get('mode')=='slide_no_animation'else species
        if not moving.any():
            results.append({'actor_id':aid,'asset_id':actor.get('asset_id'),'species':species,'moving_samples':0,'status':'pass'});continue
        bounds=SPEED_RANGES.get(category)
        if bounds is None:raise ValueError('moving species lacks an explicit speed range: '+str(category))
        median=float(np.median(speeds[moving]));lo,hi=bounds
        results.append({'actor_id':aid,'asset_id':actor.get('asset_id'),'species':species,'motion_category':category,
            'moving_samples':int(moving.sum()),'median_root_speed_m_s':median,'required_range_m_s':[lo,hi],
            'status':'pass'if lo-1e-6<=median<=hi+1e-6 else'fail'})
    return results


def layout_metrics(actors,headings=None,listener_positions=None):
    people={str(k):np.asarray(v['root_positions_m'],float)for k,v in actors.items()if v.get('species_id')=='human'}
    animals={str(k):np.asarray(v['root_positions_m'],float)for k,v in actors.items()if v.get('species_id')in {'dog','cat','bird'}}
    minimum=float('inf');animal_min=float('inf')
    keys=list(people)
    for i,k in enumerate(keys):
        for j in keys[i+1:]:minimum=min(minimum,float(np.linalg.norm((people[k]-people[j])[:,[0,2]],axis=1).min()))
        for a in animals.values():animal_min=min(animal_min,float(np.linalg.norm((people[k]-a)[:,[0,2]],axis=1).min()))
    reasons=[]
    if minimum<1.-1e-6:reasons.append('human_centres_under_1m')
    # 0.8m is the construction/export spacing for mixed species; the original
    # complete mesh collision and body-envelope checks still apply separately.
    if animal_min<.8-1e-6:reasons.append('human_animal_centres_under_0_8m')
    natural=None
    if len(keys)>=3:
        if not headings or any(k not in headings for k in keys):reasons.append('missing_actual_human_heading_readbacks')
        else:
            pos=np.asarray([people[k][0,[0,2]]for k in keys]);_,singular,_=np.linalg.svd(pos-pos.mean(axis=0),full_matrices=False)
            row_rms=float(singular[-1]/math.sqrt(len(keys)))
            ys=np.radians([headings[k]for k in keys]);delta=np.angle(np.exp(1j*(ys-ys[0])))
            spread=float(np.degrees(np.ptp(delta)));natural={'line_cross_track_rms_m':row_rms,'heading_spread_deg':spread}
            if spread<30.-1e-6:reasons.append('all_humans_share_one_heading')
            if row_rms<.25 and listener_positions is not None:
                observer=np.asarray(listener_positions)[0,[0,2]];toward=observer-pos
                # Body +Z is the registered person's facing axis.
                actual=np.column_stack((np.sin(ys),np.cos(ys)));cosines=(actual*toward).sum(axis=1)/np.maximum(1e-9,np.linalg.norm(toward,axis=1))
                if np.all(cosines>=math.cos(math.radians(25))):reasons.append('all_humans_in_row_facing_camera')
    return {'status':'fail'if reasons else'pass','minimum_human_centre_distance_m':minimum if math.isfinite(minimum)else None,
        'minimum_human_animal_centre_distance_m':animal_min if math.isfinite(animal_min)else None,'orientation':natural,'reasons':reasons}


def _load(path):return json.loads(Path(path).read_text())


def _native_visuals(parent):
    markers=[parent/'visible_native_completed.json',parent/'visual_v0_completed.json',parent/'visual_v1_completed.json']
    result={}
    for path in markers:
        if not path.is_file():continue
        marker=_load(path);root=parent/marker['attempt']
        for name in ('native_visible_visual_evidence_phase2.json','native_registered_visual_evidence.json','native_visual_evidence.json'):
            candidate=root/name
            if candidate.is_file():
                data=_load(candidate)
                if data.get('frames')and data.get('camera_frame_readbacks'):
                    result['v0'if path.name.startswith('visible')else path.name.split('_')[1]]=(candidate,data);break
    return result


def inspect_parent_policy(pointer_path):
    """Return evidence-rich failures; never modify a producer's seal or media."""
    pointer_path=Path(pointer_path)
    for ancestor in pointer_path.parents:
        if any(ancestor.glob('INVALID*.md')):
            return {'revision':REVISION,'status':'fail','source_seal':str(pointer_path),'reasons':['source_output_explicitly_invalidated'],'native_views':{},'member_listener':{},'original_native_evidence':[]}
    pointer=_load(pointer_path);parent=pointer_path.parent
    # Question-only revisions refer back to original physical native capture.
    recipe=pointer['recipe']['input_recipe'];reissue=recipe.get('question_reissue',{})
    if reissue.get('original_source_seal'):
        return inspect_parent_policy(reissue['original_source_seal'])
    row=recipe.get('row',recipe.get('seed_row',{}));visuals=_native_visuals(parent);members=defaultdict(list);reasons=[];metrics={};proofs=[]
    if not visuals:reasons.append('missing_original_native_camera_and_body_readbacks')
    requests=row.get('motion_requests',row.get('production_scenario',{}).get('motion_requests',{}))
    for view,(path,data)in visuals.items():
        cams=data['camera_frame_readbacks'];cm=camera_metrics(cams)
        times=[float(r.get('time_s',0))for r in cams]
        if min(times)>.1 or max(times)<29.8:reasons.append(view+':incomplete_30_second_camera_readback')
        if cm['status']!='pass':reasons.append(view+':camera_moved')
        frames=data['frames'];actors=defaultdict(lambda:{'root_positions_m':[]});headings={};catalog={a['asset_id']:a for a in recipe.get('assets',[])}
        for frame in frames:
            for actor in frame.get('actors',[]):
                aid=str(actor.get('actor_id',actor.get('endpoint_id')));asset=actor['asset_id'];meta=catalog.get(asset,{})
                species=meta.get('species_id')or('human'if meta.get('kind')=='human'or meta.get('entity_class')=='articulated_human'else None)
                record=actors[aid];record.update(asset_id=asset,species_id=species);record['root_positions_m'].append(actor['position_m'])
                if aid not in headings and actor.get('body_to_world_m')is not None:
                    matrix=np.asarray(actor['body_to_world_m']);headings[aid]=math.degrees(math.atan2(matrix[0,2],matrix[2,2]))
        if not frames:reasons.append(view+':missing_native_actor_frames');continue
        times=[float(f['time_s'])for f in frames];fps=1./float(np.median(np.diff(times)))
        speed=measured_actor_metrics(actors,fps,requests);layout=layout_metrics(actors,headings,[r['position_m']for r in cams])
        if any(s['status']!='pass'for s in speed):reasons.append(view+':measured_root_speed_out_of_range')
        reasons.extend(view+':'+r for r in layout['reasons'])
        metrics[view]={'camera':cm,'actors':speed,'layout':layout}
        proofs.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    # Check actual listener evidence across ALL windows in each member as well.
    for win in pointer['result'].get('windows',[]):
        wid=win['window_id'];member=win.get('member')or wid.rsplit('__',2)[-2]
        marker=parent/f'selected_windows_{member}_completed.json'
        if not marker.is_file():reasons.append(member+':missing_candidate_listener_evidence');continue
        pool=Path(_load(marker)['recipe']['pool_directory'])/wid
        facts=_load(pool/'candidate_audit.json')['facts'];listener=facts['listener']
        window_speed=measured_actor_metrics(facts['actors'],facts['time']['frame_rate_hz'],requests)
        if any(a['status']!='pass'for a in window_speed):reasons.append(wid+':window_measured_root_speed_out_of_range')
        metrics.setdefault('windows',{})[wid]={'actors':window_speed}
        for i,(position,yaw)in enumerate(zip(listener['positions_m'],listener['yaw_deg'])):
            b=listener.get('basis_m3',[]);members[member].append({'position_m':position,'yaw_deg':yaw,**({'forward_m':b[i]['forward']}if b else{})})
    listener_metrics={member:camera_metrics(values)for member,values in members.items()}
    for member,cm in listener_metrics.items():
        if cm['status']!='pass':reasons.append(member+':listener_moved_across_windows')
    if not members:reasons.append('missing_member_listener_readbacks')
    return {'revision':REVISION,'status':'fail'if reasons else'pass','source_seal':str(pointer_path),
        'reasons':sorted(set(reasons)),'native_views':metrics,'member_listener':listener_metrics,'original_native_evidence':proofs}


def require_parent_policy(pointer_path):
    report=inspect_parent_policy(pointer_path)
    if report['status']!='pass':raise ValueError('scene delivery policy failed: '+';'.join(report['reasons']))
    return report


def cached_parent_policy(pointer_path,cache_directory):
    """Cache only immutable native inspections, invalidate on any source stat."""
    pointer_path=Path(pointer_path)
    if any(any(parent.glob('INVALID*.md'))for parent in pointer_path.parents):return inspect_parent_policy(pointer_path)
    key=hashlib.sha256(str(pointer_path.resolve()).encode()).hexdigest()
    path=Path(cache_directory)/(key+'.json');pointer=_load(pointer_path);parent=pointer_path.parent
    sources=[pointer_path]
    for name in ('visible_native_completed.json','visual_v0_completed.json','visual_v1_completed.json'):
        marker=parent/name
        if marker.exists():
            sources.append(marker);m=_load(marker)
            sources.extend(parent/m['attempt']/f for f in m['required_files']if f.endswith('.json')and 'visual_evidence'in f)
    sources.extend(parent.glob('selected_windows_*_completed.json'))
    stamps={str(p):[p.stat().st_size,p.stat().st_mtime_ns]for p in sources if p.is_file()}
    if path.is_file():
        old=_load(path)
        if old.get('revision')==REVISION and old.get('source_stamps')==stamps:return old
    result=inspect_parent_policy(pointer_path);result['source_stamps']=stamps
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(path)
    return result
