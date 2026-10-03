#!/usr/bin/env python3
"""Prepare an all-exposure training pool and verify first-batch CPU repeats.

Acceptance code and thresholds are imported unchanged. Historical exposure is
descriptive for training; clean original HM3D validation houses are reserved.
"""
from __future__ import annotations
import argparse, copy, csv, json
from collections import defaultdict, Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from tools.acoustics.accept_houses import choose_rooms, compact_room
from tools.acoustics.summarize_house_acceptance import collect

def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def prepare(a):
    root=a.root; out=a.output
    inputs=out/'evidence/inputs'
    first=read(inputs/'batch1_manifest.json')
    clean=set(read(inputs/'clean.json'))
    inventory=read(inputs/'inventory.json')
    reg=[json.loads(s) for s in (root/'room_selection_merged_20261003/rooms_registry.jsonl').open()]
    reserved={h for h in clean if h.startswith('hm3d_val_')}
    rec=read(inputs/'batch1_recommended.json')
    reserved.update(h['house'] for h in rec['houses'] if h['split'] in ('test','development'))
    test=read(inputs/'test_order.json') if (inputs/'test_order.json').exists() else {}
    reserved.update(test.get('retained_test',[])+test.get('retained_development',[]))
    # Completed updated holdout recommendations, if available, own their houses.
    for p in (root/'v7_batch2_test_20261003').glob('**/recommended_houses*.json'):
        reserved.update(h['house'] for h in read(p).get('houses',[]) if h.get('split') in ('test','development'))
    by=defaultdict(list)
    for r in reg:
        if r['stage4']['status']=='pass' and r['stage1']['status']=='pass' and (r.get('human') or {}).get('verdict')=='use' and r['house'] not in reserved:
            by[r['house']].append(r)
    houses=[]; eligible_rows=[]
    so=inventory['so_overlap']['houses']['hm3d']
    physical=inventory['our_house_exposure']['physical_house_evidence']
    for house,rs in sorted(by.items()):
        r=rs[0]; sid=house.rsplit('_',1)[-1]
        exposure={'so_seen':sid in so,'m1_seen':False if house in clean else None,
                  'v6_seen':house in physical,'clean_curated':house in clean,
                  'm1_status':'not_identified_in_curated_scope' if house in clean else 'unverified_individual_M1_identity',
                  'sources':[str(inputs/'inventory.json'),str(inputs/'clean.json')]}
        known=exposure['so_seen'] or exposure['v6_seen'] or house not in clean
        rooms=[compact_room(r) for r in choose_rooms(rs)]
        houses.append(dict(house=house,family='hm3d',scan_id=sid,original_split=house.split('_')[1],
            scene_directory=r['scene_directory'],semantic_source=r['semantic_source'],annotation_source=r['annotation_source'],navmesh_source=r['navmesh_source'],
            candidate_room_count=len(rs),candidate_area_sum_m2=sum(r['metrics']['floor_area_m2'] for r in rs),
            room_types=sorted({r['room_type'] for r in rs}),floor_count_proxy=len({round(r['metrics']['placement']['camera_m'][1]/.3) for r in rs}),
            known_exposure=known,exposure=exposure,acoustic_rooms=rooms,registry_source=str(root/'room_selection_merged_20261003/rooms_registry.jsonl'),
            candidate_room_labels=[r['room_label'] for r in rs],house_gate=r['stage4'],human_review_status='smy_use',split='train'))
        eligible_rows.extend(rs)
    manifest=copy.deepcopy(first);manifest['houses']=houses
    manifest['selection']={'method':'all house gate pass AND room rules pass AND smy use; exclude clean original val and current dev/test; no exposure or model-score filter',
        'selected_before_acoustic_measurement':True,'acoustic_room_sampling':'unchanged small/large/type-diverse choose_rooms, <=3'}
    save(out/'hm3d_manifest.json',manifest)
    calibration=copy.deepcopy(first)
    calibration['houses']=[h for h in first['houses'] if h['house'] in ('hm3d_train_00059_kJxT5qssH4H','hm3d_train_00168_bHKTDQFJxTw')]
    assert len(calibration['houses'])==2
    save(out/'calibration_manifest.json',calibration)
    funnel=[]
    for name,predicate in [('registered',lambda r:True),('house_gate_pass',lambda r:r['stage4']['status']=='pass'),('room_rule_pass',lambda r:r['stage4']['status']=='pass' and r['stage1']['status']=='pass'),('smy_use',lambda r:r['stage4']['status']=='pass' and r['stage1']['status']=='pass' and (r.get('human') or {}).get('verdict')=='use'),('holdouts_excluded',lambda r:r['stage4']['status']=='pass' and r['stage1']['status']=='pass' and (r.get('human') or {}).get('verdict')=='use' and r['house'] not in reserved)]:
        rr=[r for r in reg if predicate(r)];funnel.append(dict(stage=name,houses=len({r['house'] for r in rr}),rooms=len(rr)))
    save(out/'hm3d_selection.json',dict(funnel=funnel,reserved_houses=sorted(reserved),house_count=len(houses),room_count=len(eligible_rows),houses=houses))
    # Slim evidence preserves actual rule metrics and the source registry pointers.
    save(out/'hm3d_candidate_rooms.json',[{k:r[k] for k in ['house','room_label','stage1','stage2','stage4','human','floor_selection','metrics']} for r in eligible_rows])
    print(json.dumps(funnel),flush=True)

def compare(a):
    first={r['house']:r for r in read(a.output/'evidence/inputs/batch1_results.json')}
    new=collect(read(a.output/'calibration_manifest.json'),a.output/'calibration')
    rows=[];same=True
    for b in new:
        old=first[b['house']]
        fields=['status','acoustic_status','distinct_listener_positions','rt60_valid_fraction','max_escape_fraction','max_foa_angle_deg','rt60_mid_median_s']
        pairs=[(k,old.get(k),b.get(k)) for k in fields]
        pairs += [('valid_rirs',sum(r.get('status')=='valid' for r in old['rirs']),sum(r.get('status')=='valid' for r in b['rirs'])),('visual_status',old['visual']['status'],b['visual']['status']),('leak_counts',[x['escape_count'] for x in old['leakage']],[x['escape_count'] for x in b['leakage']]),('direct_delays_ms',[x.get('direct_delay_error_ms') for x in old['rirs']],[x.get('direct_delay_error_ms') for x in b['rirs']])]
        pairs += [('distance_exponent_'+x['room_label'],x['pressure_exponent'],y['pressure_exponent']) for x,y in zip(old['distance_attenuation'],b['distance_attenuation'])]
        decision=(old['status']==b['status']=='pass' and old['reasons']==b['reasons'] and old['max_escape_fraction']==b['max_escape_fraction'] and old['distinct_listener_positions']==b['distinct_listener_positions'] and old['rt60_valid_fraction']==b['rt60_valid_fraction'] and all(.7<=r['pressure_exponent']<=1.3 for r in b['distance_attenuation'] if r['status']=='valid') and [abs(r['direct_delay_error_ms'])<=3 for r in old['rirs']]==[abs(r['direct_delay_error_ms'])<=3 for r in b['rirs']] and [r['status'] for r in old['rirs']]==[r['status'] for r in b['rirs']] and not any(r.get('tail_truncated') for r in b['rirs']))
        same &= decision
        for k,x,y in pairs:
            rows.append(dict(house=b['house'],metric=k,batch1=x,repeat=y,numerically_identical=x==y,absolute_difference=abs(x-y) if isinstance(x,(float,int)) and isinstance(y,(float,int)) else None,same_acceptance_decision=decision))
    save(a.output/'evidence/calibration_comparison_v2.json',dict(code='3414629',same_acceptance_decisions=same,numerically_identical=all(r['numerically_identical'] for r in rows),note='8-thread native indirect Monte Carlo values are reported exactly; no numeric matching tolerance changes acceptance.',rows=rows))
    with (a.output/'evidence/calibration_comparison_v2.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('CALIBRATION_ACCEPTANCE_AGREES',same,flush=True)
    if not same:raise SystemExit('Calibration mismatch; new acoustic jobs must not start.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','compare']);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();globals()[a.stage](a)
