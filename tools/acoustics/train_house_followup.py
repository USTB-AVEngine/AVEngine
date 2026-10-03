#!/usr/bin/env python3
"""Automatic training pool follow-up; immutable historical inputs, CPU only."""
from __future__ import annotations
import argparse, collections, copy, csv, json, sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.accept_houses import choose_rooms,compact_room,classify
from avengine.acoustics.house_acceptance_metrics import rir_metrics,foa_direction,distance_attenuation
from tools.acoustics.summarize_train_house_pool import exposure

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)

def registry(work):
    by=collections.defaultdict(list)
    for line in (work/'inputs/rooms_registry.jsonl').read_text().splitlines():
        r=json.loads(line);by[r['house']].append(r)
    return by

def prepare(output,work):
    reg=registry(work);old=read(output/'hm3d_manifest.json');old_by={h['house']:h for h in old['houses']}
    reserved=set(read(output/'hm3d_selection.json')['reserved_houses'])
    live=[]
    for p in (output.parent/'v7_batch2_test_20261003').glob('**/recommended_houses*.json'):
        live.append(str(p));reserved.update(h['house'] for h in read(p).get('houses',[]) if h.get('split') in ('development','test'))
    houses=[];delta=[];blocked=[];all_rooms=[]
    for house,rs in sorted(reg.items()):
        eligible=[r for r in rs if r['stage1']['status']=='pass']
        if not eligible:continue
        previous=old_by.get(house);gate=rs[0]['stage4'];h=copy.deepcopy(previous) if previous else {}
        # Retain previously sampled rooms, then fill unused slots. Human labels
        # and acoustic outcomes never enter this deterministic sampling rule.
        samples=copy.deepcopy(previous['acoustic_rooms']) if previous else []
        known={r['room_label'] for r in samples}
        samples += [compact_room(r) for r in choose_rooms([r for r in eligible if r['room_label'] not in known])][:3-len(samples)]
        r=eligible[0];sid=house.rsplit('_',1)[-1]
        h.update(house=house,family='hm3d',scan_id=sid,original_split=house.split('_')[1],scene_directory=r['scene_directory'],semantic_source=r['semantic_source'],annotation_source=r['annotation_source'],navmesh_source=r['navmesh_source'],candidate_room_count=len(eligible),candidate_area_sum_m2=sum(r['metrics']['floor_area_m2'] for r in eligible),room_types=sorted({r['room_type'] for r in eligible}),floor_count_proxy=len({round(r['metrics']['placement']['camera_m'][1]/.3) for r in eligible}),acoustic_rooms=samples,registry_source=str(work/'inputs/rooms_registry.jsonl'),candidate_room_labels=[r['room_label'] for r in eligible],historical_house_gate=gate,house_gate=gate,split='train',smy_labels=dict(collections.Counter((r.get('human') or {}).get('verdict') or 'unlabeled' for r in eligible)),human_labels_filter=False,holdout_reserved=house in reserved,sampling_method='retain earlier predetermined samples; fill empty slots with unchanged small/large/type selection; <=3 unique rooms')
        h['exposure']=exposure(output,h);h['known_exposure']=any(h['exposure'][k] for k in ('so_seen','m1_seen','v6_seen'))
        if gate['status']!='pass':blocked.append(h)
        if gate['status']=='pass' and house not in reserved:
            houses.append(h);all_rooms+=eligible
            missing=[r for r in samples if r['room_label'] not in known]
            if missing:
                d=copy.deepcopy(h);d['acoustic_rooms']=missing;d['measurement_scope']='only new sampled rooms; earlier waveforms retained';delta.append(d)
    manifest=copy.deepcopy(old);manifest['houses']=houses;manifest['measurement_revision']='e3dc246 first arrival plus 0.5ms'
    manifest['selection']={'method':'stage1.pass AND stage4.pass; human.verdict descriptive; exclude clean original val and current holdouts','selected_before_acoustic_measurement':True}
    save(work/'hm3d_manifest.json',manifest)
    save(work/'hm3d_delta_manifest.json',dict(manifest,houses=delta))
    save(work/'hm3d_blocked_manifest.json',dict(manifest,houses=blocked))
    save(work/'hm3d_candidate_rooms.json',all_rooms)
    funnels=[]
    for name,pred in [('registered',lambda r:True),('automatic_rooms_only',lambda r:r['stage1']['status']=='pass'),('automatic_and_historical_gate',lambda r:r['stage1']['status']=='pass' and r['stage4']['status']=='pass'),('automatic_gate_and_smy_old',lambda r:r['stage1']['status']=='pass' and r['stage4']['status']=='pass' and (r.get('human') or {}).get('verdict')=='use'),('automatic_train_after_holdouts',lambda r:r['stage1']['status']=='pass' and r['stage4']['status']=='pass' and r['house'] not in reserved),('old_train_after_holdouts',lambda r:r['stage1']['status']=='pass' and r['stage4']['status']=='pass' and (r.get('human') or {}).get('verdict')=='use' and r['house'] not in reserved)]:
        rows=[r for rr in reg.values() for r in rr if pred(r)];funnels.append(dict(stage=name,houses=len({r['house'] for r in rows}),rooms=len(rows)))
    summary=dict(funnel=funnels,reserved_houses=sorted(reserved),holdout_sources=live,delta_houses=len(delta),delta_sampled_rooms=sum(len(h['acoustic_rooms']) for h in delta),new_houses=sorted(set(h['house'] for h in houses)-set(old_by)),new_candidate_rooms=len(all_rooms)-len(read(output/'hm3d_candidate_rooms.json')),blocked_houses=len(blocked),blocked_reserved=sum(h['holdout_reserved'] for h in blocked))
    save(work/'hm3d_selection.json',summary);print('PREPARE',json.dumps(summary,ensure_ascii=False),flush=True)

def corrected_result(result,manifest,house=None,selected_only=False):
    result=copy.deepcopy(result);h=house or result['input'];result['input']=h
    if not result.get('rirs'):
        result['measurement_revision']='e3dc246; no waveform to recompute';return result
    oldrirs=list(result['rirs']);sample_rate=int(manifest['simulation']['sample_rate_hz']);limit=manifest['simulation']['max_ir_seconds']
    labels={r['room_label'] for r in h['acoustic_rooms']};single=len(labels)==1
    if selected_only:
        oldrirs=[m for m in oldrirs if m['room_label'] in labels and (single or not ('_second_range_' in m['probe'] or '_alternate_listener_' in m['probe']))]
        result['leakage']=[r for r in result.get('leakage',[]) if r['room_label'] in labels and (single or r['role']!='camera_alt_m')]
        result['occluded_probe_lines']=[r for r in result.get('occluded_probe_lines',[]) if r['room_label'] in labels]
    arrays={};corrected=[];cached={}
    for m in oldrirs:
        path,key=m['rir_path'].split('#/',1)
        if path not in cached:cached[path]=np.load(path)
        ir=cached[path][key];arrays[m['probe']]=ir
        metrics,_=rir_metrics(ir,sample_rate,m['distance_m'],limit)
        metadata={k:v for k,v in m.items() if k in ('probe','room_label','room_type','listener_m','source_m','distance_m','render_wall_seconds','native_simulation_wall_seconds','native_simulation_cpu_seconds','rir_path','direct_path_line_clear','direct_metrics_validated')}
        metrics.update(metadata);corrected.append(metrics)
    for v in cached.values():v.close()
    result['rirs']=corrected
    occluded={x['room_label']+'_'+x['source_label'] for x in result.get('occluded_probe_lines',[])}
    attenuation=[];directions=[]
    for r in h['acoustic_rooms']:
        names=[r['room_label']+'_'+k for k in ('range_near','range_middle','source_1')]
        rr=[m for m in corrected if m['probe'] in names and m['probe'] not in occluded]
        attenuation.append(dict(room_label=r['room_label'],**distance_attenuation([m['distance_m'] for m in rr],[m.get('direct_energy',0) for m in rr])))
        for name in ('source_1','source_2'):
            key=r['room_label']+'_'+name;m=next((m for m in corrected if m['probe']==key),None)
            if m is None:continue
            forward=np.asarray(m['source_m'])-np.asarray(m['listener_m']);forward[1]=0;forward/=np.linalg.norm(forward)
            yaw=np.radians(20 if name=='source_1' else -20)
            forward=np.array([forward[0]*np.cos(yaw)+forward[2]*np.sin(yaw),0,-forward[0]*np.sin(yaw)+forward[2]*np.cos(yaw)])
            d=foa_direction(arrays[key],m['listener_m'],m['source_m'],forward,sample_rate)
            d.update(direct_line_clear=key not in occluded,probe=key,room_label=r['room_label'],listener_m=m['listener_m'],source_m=m['source_m']);directions.append(d)
    clear=[d for d in directions if d['direct_line_clear']];primary=[d for d in clear if d['probe'].endswith('source_1')]
    checks=(primary+[d for d in clear if d not in primary])[:3]
    if single and len(checks)<3 and directions:
        key=h['acoustic_rooms'][0]['room_label']+'_range_middle';m=next(m for m in corrected if m['probe']==key)
        d=foa_direction(arrays[key],m['listener_m'],m['source_m'],directions[0]['camera_forward'],sample_rate)
        d.update(probe=key,room_label=m['room_label'],listener_m=m['listener_m'],source_m=m['source_m']);checks.append(d)
    result.update(distance_attenuation=attenuation,foa_all_positions=directions,foa_checks=checks,measurement_revision='e3dc246 first arrival plus 0.5ms',historical_status=result.get('status'),historical_reasons=result.get('reasons'),known_exposure=h.get('known_exposure',False))
    classify(result,manifest['criteria']);return result

def calibration(output,work):
    reference=read(work/'inputs/test_all_house_results.json');manifest=read(output/'hm3d_manifest.json');rows=[]
    ids=['hm3d_train_00059_kJxT5qssH4H','hm3d_train_00168_bHKTDQFJxTw']
    for house in ids:
        source=next(r for r in reference if r['house']==house);r=read(Path(source['result_directory'])/'result.json');new=corrected_result(r,manifest)
        dest=work/'calibration'/'houses'/house;save(dest/'result.json',new)
        fields=['status','reasons','max_escape_fraction','max_foa_angle_deg','rt60_valid_fraction','rt60_mid_median_s','distinct_listener_positions','distance_attenuation','foa_checks']
        for k in fields:rows.append(dict(house=house,metric=k,reference=r.get(k),recomputed=new.get(k),exactly_equal=r.get(k)==new.get(k)))
        for a,b in zip(r['rirs'],new['rirs'],strict=True):
            fields=['status','direct_peak_s','direct_gate_samples','direct_energy','direct_delay_error_ms','rt60_mid_s','drr_db','octaves']
            for k in fields:rows.append(dict(house=house,metric=a['probe']+'.'+k,reference=a.get(k),recomputed=b.get(k),exactly_equal=a.get(k)==b.get(k)))
    ok=all(r['exactly_equal'] for r in rows)
    save(work/'evidence/calibration_corrected_exact.json',dict(exactly_equal=ok,reference='v7-fill-test report_final and measurement_final; recompute same saved waveforms with cherry-picked e3dc246',native_propagation_rerun=False,rows=rows))
    print('CORRECTED_CALIBRATION_EXACT',ok,'checks',len(rows),flush=True)
    if not ok:raise ValueError('Corrected calibration reference differs')

def rescore(output,work):
    assert read(work/'evidence/calibration_corrected_exact.json')['exactly_equal']
    jobs=[]
    for family in ('hm3d','mp3d'):
        manifest=read(output/(family+'_manifest.json'))
        for h in manifest['houses']:
            p=output/(family+'_acoustics')/'houses'/h['house']/'result.json'
            if p.exists():jobs.append((p,manifest,work/(family+'_corrected')/'houses'/h['house']))
    first=read(output/'evidence/inputs/batch1_manifest.json')
    prior=read(output/'evidence/inputs/batch1_results.json')
    for r in prior:
        if r['family']=='mp3d':jobs.append((Path(r['result_directory'])/'result.json',first,work/'mp3d_prior_corrected'/'houses'/r['house']))
    rows=[]
    for p,m,dest in jobs:
        old=read(p);new=corrected_result(old,m);new['waveform_source']=str(p)
        save(dest/'result.json',new)
        visual=p.parent/'visual_checks.json'
        if visual.exists():save(dest/'visual_checks.json',read(visual))
        rows.append(dict(house=old['house'],old_status=old['status'],corrected_status=new['status'],rirs=len(new.get('rirs',[])),source=str(p),result=str(dest/'result.json')))
        print('RESCORE',old['house'],old['status'],new['status'],flush=True)
    save(work/'evidence/uniform_rescore.json',dict(houses=len(rows),rirs=sum(r['rirs'] for r in rows),rows=rows))

def combine(output,work):
    manifest=read(work/'hm3d_manifest.json');baseline={h['house'] for h in read(output/'hm3d_manifest.json')['houses']}
    for h in manifest['houses']:
        sources=[]
        if h['house'] in baseline:sources.append(output/'hm3d_acoustics/houses'/h['house']/'result.json')
        delta=work/'hm3d_delta/houses'/h['house']/'result.json'
        if delta.exists():sources.append(delta)
        if not sources:continue
        records=[read(p) for p in sources];result=copy.deepcopy(records[0]);result['rirs']=[];result['leakage']=[];result['occluded_probe_lines']=[]
        for r in records:
            for key in ('rirs','leakage','occluded_probe_lines'):result[key]+=r.get(key,[])
        # Partial native failures remain review, never pass by cherry-picking
        # only the completed probes from a failed measurement.
        if any(not r.get('foa_checks') for r in records):
            result=copy.deepcopy(records[-1]);result.update(input=h,status='review',reasons=list(result.get('reasons',[]))+['DELTA_NATIVE_MEASUREMENT_INCOMPLETE'])
        else:result=corrected_result(result,manifest,h,selected_only=True)
        result['measurement_sources']=list(map(str,sources))
        save(work/'hm3d_current/houses'/h['house']/'result.json',result)
        prev=output/'hm3d_acoustics/houses'/h['house']/'visual_checks.json'
        # A visual check of an earlier sample cannot cover added samples.
        if not delta.exists() and prev.exists():save(work/'hm3d_current/houses'/h['house']/'visual_checks.json',read(prev))
    print('COMBINE_COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','calibration','rescore','combine']);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.output,a.work)
if __name__=='__main__':main()
