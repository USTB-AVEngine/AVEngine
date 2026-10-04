#!/usr/bin/env python3
"""CPU per-room training measurements, retaining previously measured RIRs."""
from __future__ import annotations
import argparse,collections,copy,datetime,json,os,random,sys,time,traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.accept_houses import compact_room,alternate_listener,ray_checks,classify
from tools.acoustics.train_house_followup import read,save,corrected_result
from avengine.acoustics.house_acceptance_metrics import rir_metrics

def prepare(previous,output):
    prior=previous/'followup_auto_only_20261003'
    output.mkdir(parents=True,exist_ok=False)
    for name in ('inputs','evidence','logs','scripts','houses','repeats'): (output/name).mkdir()
    old=read(prior/'report_final/all_house_results.json')
    save(output/'inputs/old_all_house_results.json',old)
    reg=collections.defaultdict(list)
    for s in (prior/'inputs/rooms_registry.jsonl').read_text().splitlines():
        r=json.loads(s)
        if r['stage1']['status']=='pass':reg[r['house']].append(r)
    reserved=set(read(prior/'hm3d_selection.json')['reserved_houses'])
    live=[]
    for p in (previous.parent/'v7_batch2_test_20261003').glob('**/recommended_houses*.json'):
        live.append(str(p));reserved.update(h['house'] for h in read(p).get('houses',[]) if h.get('split') in ('development','test'))
    for p in (previous/'mp3d_rooms_final/houses').glob('*.json'):
        reg[p.stem]=[r for r in read(p)['rows'] if r['stage1']['status']=='pass']
    hm=read(prior/'hm3d_manifest.json');mp=read(previous/'mp3d_manifest.json')
    manifests={f:dict(m,houses=[]) for f,m in [('hm3d',hm),('mp3d',mp)]}
    houses=[];eligible_repeats=[];inventory=[]
    for row in old:
        house=row['house'];family=row['family'];rooms=reg[house]
        if not rooms or house in reserved:continue
        h=copy.deepcopy(row['input']);h.update(house=house,family=family,holdout_reserved=False,split='train',human_labels_filter=False)
        h['old_sampled_rooms']=copy.deepcopy(h['acoustic_rooms'])
        h['acoustic_rooms']=[compact_room(r) for r in rooms]
        h['candidate_room_labels']=[r['room_label'] for r in rooms];h['candidate_room_count']=len(rooms)
        h['candidate_area_sum_m2']=sum(r['metrics']['floor_area_m2'] for r in rooms)
        h['room_rule_records']=rooms
        paths=[Path(row['result_directory'])/'result.json']
        paths += [previous/(family+'_acoustics')/'houses'/house/'result.json']
        paths += [prior/n/'houses'/house/'result.json' for n in ['hm3d_delta','hm3d_blocked','hm3d_current','mp3d_corrected','mp3d_prior_corrected','mp3d_current']]
        sources=[]
        for p in dict.fromkeys(paths):
            if p.exists():
                r=read(p);sources.append((p,r))
                for name in ('measurement_sources',):
                    for q in r.get(name,[]):
                        q=Path(q)
                        if q.exists() and q not in [x[0] for x in sources]:sources.append((q,read(q)))
        merged=dict(house=house,family=family,input=h,known_exposure=h.get('known_exposure',False),rirs=[],leakage=[],occluded_probe_lines=[])
        selected={};leaks={};occlusions={}
        for p,r in reversed(sources):
            for m in r.get('rirs',[]):
                raw,key=m['rir_path'].split('#/',1)
                if Path(raw).exists():selected[m['probe']]=dict(m,reused_result_path=str(p))
            for x in r.get('leakage',[]):leaks[x['room_label'],x['role']]=x
            for x in r.get('occluded_probe_lines',[]):occlusions[x['room_label'],x['source_label']]=x
        merged.update(rirs=list(selected.values()),leakage=list(leaks.values()),occluded_probe_lines=list(occlusions.values()))
        packages=[r.get('acoustic_package_manifest') for p,r in sources if r.get('acoustic_package_manifest') and Path(r['acoustic_package_manifest']).exists()]
        h['reuse_package']=packages[0] if packages else None
        h['cached_results']=str(output/'inputs'/('cache_'+house+'.json'))
        h['previous_visual']=row.get('visual',{})
        h['previous_geometry']=row.get('derived_geometry')
        h['previous_result_directory']=row['result_directory']
        h['previous_reasons']=row['reasons']
        h['semantic_conflict']=any('names two categories' in x for x in row['reasons'])
        save(h['cached_results'],merged)
        houses.append(h);manifests[family]['houses'].append(h)
        labels={m['room_label'] for m in merged['rirs']}
        if row['status']=='pass':eligible_repeats += [(house,label) for label in sorted(labels) if label in h['candidate_room_labels']]
        inventory.append(dict(house=house,family=family,candidate_rooms=len(rooms),cached_rooms=len(labels),cached_rirs=len(merged['rirs']),sources=list(map(str,[p for p,r in sources])),semantic_conflict=h['semantic_conflict'],package=h['reuse_package']))
    repeat=random.Random(20261003).sample(sorted(eligible_repeats),10)
    save(output/'inputs/repeat_rooms.json',dict(seed=20261003,method='uniform sample without replacement from the 281+28 previously accepted-house sampled rooms, before new room outcomes',eligible_rooms=len(eligible_repeats),rooms=[dict(house=h,room_label=l) for h,l in repeat]))
    for f,m in manifests.items():
        save(output/'inputs'/(f+'_manifest.json'),m)
        save(output/'inputs'/(f+'_settings.json'),{k:v for k,v in m.items() if k!='houses'})
    save(output/'inputs/candidates.json',houses)
    (output/'inputs/thresholds.frozen.yaml').write_bytes((prior/'inputs/thresholds.frozen.yaml').read_bytes())
    counts={f:dict(houses=sum(h['family']==f for h in houses),rooms=sum(h['candidate_room_count'] for h in houses if h['family']==f),cached_rooms=sum(x['cached_rooms'] for x in inventory if x['family']==f),cached_rirs=sum(x['cached_rirs'] for x in inventory if x['family']==f)) for f in manifests}
    save(output/'evidence/preparation.json',dict(counts=counts,reserved_houses=sorted(reserved),holdout_sources=live,inventory=inventory,previous=str(prior),criteria_unchanged=hm['criteria']==mp['criteria'],measurement_revision='e3dc246 first arrival + 0.5 ms',compute_device='CPU'))
    assert counts['hm3d']['houses']==146 and counts['hm3d']['rooms']==927,counts
    assert counts['mp3d']['houses']==85 and counts['mp3d']['rooms']==623,counts
    assert len(eligible_repeats)==309,len(eligible_repeats)
    print('PREPARED',json.dumps(counts),flush=True)

def strict_zero_scene(scene,destination):
    """Derive runtime-owned arrays; retain every positive double-area face."""
    objects=[];records=[];omitted={};counts=collections.Counter()
    for obj in scene.objects:
        faces=np.asarray(obj['triangles']);v=np.asarray(obj['vertices'],dtype=np.float64)
        p=v[faces];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);mask=np.all(cross==0,axis=1)
        # Preserve every positive-area face and let the unchanged native
        # uploader decide. A float32 cross product can cancel even when the
        # native float64 product of float32 edge vectors remains nonzero.
        if mask.any():
            omitted[obj['object_id']]=np.flatnonzero(mask)
            records.append(dict(object_id=obj['object_id'],original_faces=len(faces),removed_faces=int(mask.sum()),maximum_removed_area_m2=0.0))
        if mask.all():continue
        derived=dict(obj);derived['triangles']=np.ascontiguousarray(faces[~mask],dtype='<u4')
        derived['triangle_material_ids']=np.ascontiguousarray(obj['triangle_material_ids'][~mask],dtype='<u4')
        objects.append(derived)
        for mat,n in zip(*np.unique(derived['triangle_material_ids'],return_counts=True),strict=True):counts[scene.material_categories[int(mat)]]+=int(n)
    if not objects:raise ValueError('No positive-area acoustic faces')
    receipt=dict(source_package_manifest=str(scene.manifest_path),removed_faces=sum(x['removed_faces'] for x in records),by_object=records,maximum_removed_area_m2=0.0,tolerance_m2=0.0,original_package_modified=False,objects_vertices_and_materials_unchanged=True,operation='派生几何：清退化三角形',qualification='authorized CPU training selection; not a new formally compiled production package')
    if receipt['removed_faces']:
        destination.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(destination/'removed_faces.npz',**omitted)
        # Persist exact derived native inputs rather than only an in-memory view.
        arrays={}
        for i,obj in enumerate(objects):
            for field in ('vertices','triangles','triangle_material_ids'):arrays[f'object_{i}_{field}']=np.asarray(obj[field])
        np.savez_compressed(destination/'native_arrays.npz',**arrays)
        receipt['native_arrays_path']=str(destination/'native_arrays.npz')
        receipt['omitted_indices_path']=str(destination/'removed_faces.npz')
        save(destination/'receipt.json',receipt)
    return replace(scene,objects=tuple(objects),triangle_count_by_material=dict(counts)),receipt

def probe_plan(room):
    p=room['placement'];listener=np.asarray(p['camera_m'],float);far=np.asarray(p['source_1_m'],float)
    axis=far-listener;distance=np.linalg.norm(axis);axis/=distance
    second=np.asarray(p['source_2_m'],float);delta=second-listener;alt=np.asarray(alternate_listener(p))
    return [(name,listener,source) for name,source in [('range_near',listener+axis*min(1.,distance/3)),('range_middle',listener+axis*distance*.65),('source_1',far),('source_2',second),('second_range_near',listener+delta*.35),('second_range_middle',listener+delta*.7)]]+ [('alternate_listener_source_1_m',alt,far),('alternate_listener_source_2_m',alt,second)]

def cached_match(metric,listener,source):
    return np.allclose(metric['listener_m'],listener,rtol=0,atol=1e-7) and np.allclose(metric['source_m'],source,rtol=0,atol=1e-7)

def measure_room(h,room,cache,renderer,bounds,manifest,destination,force=False):
    label=room['room_label'];destination.mkdir(parents=True,exist_ok=False);stamp=time.monotonic()
    single=copy.deepcopy({k:v for k,v in h.items() if k not in ('room_rule_records','acoustic_rooms')});single['acoustic_rooms']=[copy.deepcopy(room)]
    result=dict(house=h['house'],family=h['family'],room_label=label,input=single,status='review',reasons=[],rirs=[],leakage=[],occluded_probe_lines=[],known_exposure=h.get('known_exposure',False),compute_device='CPU',gpu_renderer_created=False,measurement_revision='e3dc246 first arrival + 0.5 ms',acoustic_package_manifest=h['reuse_package'],native_setup=renderer.setup_report,reused_rir_count=0,new_rir_count=0,listener_supplement=False,independent_repeat=force)
    new_arrays={};old={m['probe']:m for m in cache['rirs']};sim=manifest['simulation'];sr=int(sim['sample_rate_hz'])
    try:
        p=room['placement'];origins=[(label,key,p[key]) for key in ('camera_m','source_1_m','source_2_m')]+[(label,'camera_alt_m',alternate_listener(p))]
        for origin in origins:
            existing=next((x for x in cache['leakage'] if x['room_label']==label and x['role']==origin[1] and np.allclose(x['origin_m'],origin[2],rtol=0,atol=1e-7)),None)
            result['leakage'] += [existing] if existing and not force else ray_checks(renderer.context,[origin],bounds,manifest['criteria']['ray_count_per_origin'])
        for name,listener,source in probe_plan(room):
            key=label+'_'+name;distance=float(np.linalg.norm(source-listener));relative=(source-listener)/distance
            blocked=renderer.context.trace_ray_any_hit(listener.tolist(),relative.tolist(),.01,max(.02,distance-.03)).hit
            if blocked:
                result['occluded_probe_lines'].append(dict(room_label=label,source_label=name,listener_m=listener.tolist(),source_m=source.tolist()))
            reuse=old.get(key)
            if reuse and not force and cached_match(reuse,listener,source):
                metrics=copy.deepcopy(reuse);result['reused_rir_count']+=1
            else:
                batch=renderer.render([source.tolist()],listener_position_m=listener.tolist(),listener_orientation_wxyz=[1.,0.,0.,0.])
                ir=batch.samples[0];metrics,_=rir_metrics(ir,sr,distance,sim['max_ir_seconds']);new_arrays[key]=ir
                metrics.update(probe=key,room_label=label,room_type=room['room_type'],listener_m=listener.tolist(),source_m=source.tolist(),distance_m=distance,render_wall_seconds=batch.wall_seconds,native_simulation_wall_seconds=batch.wall_seconds,native_simulation_cpu_seconds=batch.process_cpu_seconds,rir_path=str(destination/'new_rirs.npz')+'#/'+key)
                result['new_rir_count']+=1
            metrics['direct_path_line_clear']=not blocked;result['rirs'].append(metrics)
        if new_arrays:np.savez_compressed(destination/'new_rirs.npz',**new_arrays)
        result=corrected_result(result,manifest,single)
        result['listener_supplement']=result['reused_rir_count']>0 and any('_alternate_listener_' in x for x in new_arrays)
        result['valid_rir_count']=sum(r['status']=='valid' for r in result['rirs'])
        # Keep the shared classifier byte-for-byte unchanged. The owner also
        # explicitly requires six valid RIRs for every admitted room.
        if result['valid_rir_count']<manifest['criteria']['minimum_rirs']:
            result['completion_reasons']=['FEWER_THAN_SIX_VALID_RIRS'];result['status']='fail' if result['status']=='fail' else 'review'
    except Exception as exc:
        if new_arrays and not (destination/'new_rirs.npz').exists():np.savez_compressed(destination/'new_rirs.npz',**new_arrays)
        result.update(status='review',reasons=result.get('reasons',[])+[type(exc).__name__+': '+str(exc)],exception_traceback=traceback.format_exc())
    result['wall_seconds']=time.monotonic()-stamp
    result['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(destination/'result.json',result)
    print('ROOM_DONE',h['house'],label,result['status'],'reuse',result['reused_rir_count'],'new',result['new_rir_count'],round(result['wall_seconds'],2),flush=True)
    return result

def measure_house(output,house):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError("CUDA_VISIBLE_DEVICES must be empty")
    cache=read(output/'inputs'/('cache_'+house+'.json'));h=cache['input']
    manifest=read(output/'inputs'/(h['family']+'_settings.json'))
    dest=output/'houses'/house;dest.mkdir(exist_ok=False)
    repeat={x['room_label'] for x in read(output/'inputs/repeat_rooms.json')['rooms'] if x['house']==house}
    try:
        if h['semantic_conflict']:raise ValueError('SEMANTIC_COLOR_CATEGORY_CONFLICT: original annotations retained; '+ ';'.join(x for x in h['previous_reasons'] if 'names two categories' in x))
        from avengine.acoustics.compiler import compile_hm3d_semantic_research_scene,compile_mp3d_semantic_research_scene
        from avengine.acoustics.runtime import load_compiled_acoustic_scene,RLRSimulationConfig
        from avengine.acoustics.rir_cache import _NativeRIRBatchRenderer
        from tools.acoustics.accept_houses import room_manifest
        if not h['reuse_package']:
            config=dest/'scene_dataset_config.json';save(config,dict(stages=dict(paths={'.glb':[str(Path(h['scene_directory'])/(h['scan_id']+'.glb'))]})))
            rm=dest/'room_manifest.json';save(rm,room_manifest(h,config))
            compiler=compile_hm3d_semantic_research_scene if h['family']=='hm3d' else compile_mp3d_semantic_research_scene
            package,coverage=compiler(room_manifest=rm,material_rules=manifest['material_rules'][h['family']],output=dest/'package',seed=manifest['material_seeds'][h['family']],probe_origins=[r['placement']['camera_m'] for r in h['acoustic_rooms']],probe_direction_count=16)
            h['reuse_package']=str(package)
        scene=load_compiled_acoustic_scene(h['reuse_package'],allow_nonpassing_research_qa=True)
        if scene.manifest['source_room']['room_id']!=h['house']:raise ValueError('Package belongs to another house')
        scene,geometry=strict_zero_scene(scene,dest/'derived_acoustic_geometry')
        first=h['acoustic_rooms'][0]['placement'];renderer=_NativeRIRBatchRenderer(scene,RLRSimulationConfig.from_mapping(manifest['simulation']),batch_size=1,initial_positions_m=[first['source_1_m']],listener_position_m=first['camera_m'],listener_orientation_wxyz=[1.,0.,0.,0.],layout_type='ambisonics',channel_count=4,hrtf_file_path='',source_radius_m=manifest['source_radius_m'],listener_radius_m=manifest['listener_radius_m'],**manifest['runtime'])
        renderer.render([first['source_1_m']])
        v=np.concatenate([o['vertices'] for o in scene.objects]);bounds=np.stack((v.min(0),v.max(0)))
        save(dest/'native_house_setup.json',dict(house=house,acoustic_package_manifest=h['reuse_package'],geometry=geometry,native_setup=renderer.setup_report,simulation=manifest['simulation'],bounds_m=bounds.tolist(),compute_device='CPU'))
        for room in h['acoustic_rooms']:
            result=measure_room(h,room,cache,renderer,bounds,manifest,dest/'rooms'/room['room_label'])
            if room['room_label'] in repeat:
                measure_room(h,room,cache,renderer,bounds,manifest,output/'repeats'/house/room['room_label'],force=True)
    except Exception as exc:
        failure=dict(house=house,status='review',reasons=[type(exc).__name__+': '+str(exc)],exception_traceback=traceback.format_exc(),compute_device='CPU')
        save(dest/'house_failure.json',failure);print('HOUSE_FAILURE',house,failure['reasons'],flush=True)
        for room in h['acoustic_rooms']:
            roomdir=dest/'rooms'/room['room_label']
            if not (roomdir/'result.json').exists():save(roomdir/'result.json',dict(failure,family=h['family'],room_label=room['room_label'],input=dict({k:v for k,v in h.items() if k not in ('room_rule_records','acoustic_rooms')},acoustic_rooms=[room]),rirs=[],leakage=[],new_rir_count=0,reused_rir_count=0,measurement_complete=False))
    save(dest/'finished.json',dict(house=house,rooms=len(h['acoustic_rooms']),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','measure']);p.add_argument('--previous',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--house');a=p.parse_args()
    if a.stage=='prepare':prepare(a.previous,a.output)
    else:measure_house(a.output,a.house)
if __name__=='__main__':main()
