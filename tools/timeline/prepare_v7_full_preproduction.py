"""Prepare new TRAIN-only worlds without rendering or opening held-out assets.

This is a seed proposal, not an admission receipt. Every original per-room
qualification and new native placement/motion/direction gate remains required.
"""
import argparse,collections,copy,hashlib,json,math
from pathlib import Path

TEST_HOUSES={'hm3d_val_00808_y9hTuugGdiq','hm3d_val_00877_4ok3usBNeis'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seeded(*parts):return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'big')

def eligible_metadata(catalog,excluded):
    kept=[];reasons=collections.Counter()
    for item in catalog:
        if item['house']in excluded:reasons['heldout_physical_house']+=1;continue
        if item.get('status')!='pass' or not item.get('training_eligible'):reasons['not_original_strict_pass']+=1;continue
        if item['family']=='hm3d'and any(item.get('exposure',{}).get(k)for k in ['so_seen','m1_seen']):reasons['HM3D_SO_or_M1_seen']+=1;continue
        if item['family']not in ['hm3d','mp3d']:reasons['not_authorized_family']+=1;continue
        allowed=[r for r in item['accepted_room_labels']if not item.get('accepted_room_training_tags',{}).get(r)]
        if not allowed:reasons['no_original_strict_room']+=1;continue
        kept.append((item,allowed))
    return kept,dict(reasons)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['catalog','S-seeds','S-houses','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--parents',type=int,default=1500);a=p.parse_args()
    if a.parents!=1500:raise ValueError('owner first preproduction authorization is exactly 1500 new train parents')
    raw=[line for line in a.S_seeds.read_text().splitlines()if line.strip()];originals=[json.loads(x)for x in raw]
    if len(originals)!=500 or collections.Counter(r['split']for r in originals)!={'train':400,'dev':100}:raise ValueError('original fixed S manifest must be 400+100')
    templates=[r for r in originals if r['split']=='train'];prefix=[line for line in raw if json.loads(line)['split']=='train']
    S_houses=json.loads(a.S_houses.read_text());excluded=TEST_HOUSES|{r['house']for r in S_houses['houses']if r['split']!='train'}
    # Filter names and metadata before accessing any house-specific receipt.
    catalog=json.loads(a.catalog.read_text());eligible,reasons=eligible_metadata(catalog,excluded)
    houses=[];rooms=[];source_receipts=[]
    for source,ids in sorted(eligible,key=lambda row:row[0]['house']):
        root=Path(source['result_directory']);setup=root/'native_house_setup.json'
        if not setup.is_file():raise ValueError('selected training house lacks inherited native acoustic identity: '+source['house'])
        setup_data=json.loads(setup.read_text());package=Path(setup_data['acoustic_package_manifest'])
        if not package.is_file():raise ValueError('selected training acoustic package absent: '+str(package))
        selected=[copy.deepcopy(r)for r in source['input']['acoustic_rooms']if r['room_label']in ids]
        if set(r['room_label']for r in selected)!=set(ids):raise ValueError('accepted room metadata is incomplete')
        for room in selected:
            room.pop('human',None)
            placement=room['placement']
            if not placement.get('found'):raise ValueError('strict room lacks actual acoustic placement candidate')
            for key in ['camera_m','source_1_m','source_2_m']:
                if len(placement[key])!=3 or not all(math.isfinite(v)for v in placement[key]):raise ValueError('invalid inherited training placement')
        item={'house':source['house'],'family':source['family'],'split':'train',
            'known_exposure':bool(source['input'].get('known_exposure')),'acoustic_status':source['status'],
            'checked_room_ids':ids,'checked_rooms':selected,'input':copy.deepcopy(source['input']),
            'acoustic_package_manifest':str(package),'acoustic_result':str(root/'result.json'),
            'native_rgb_status':source.get('visual_status'),
            'original_room_evidence':[r for r in source['per_room_evidence']if r['room_label']in ids],
            'selection_scope':'automatic room rules and original strict acoustic pass; no smy-label filter; no 5-to-15-percent review band',
            'inherited_exposure':source['exposure'],'new_native_placements':'pending original episode gates'}
        item['input']['acoustic_rooms']=selected;item['input']['split']='train'
        houses.append(item);source_receipts.append({'house':source['house'],'native_house_setup':str(setup),'native_setup_sha256':digest(setup),'package_manifest':str(package),'package_manifest_sha256':digest(package)})
        for room in selected:rooms.append((item,room))
    if not houses or not rooms:raise ValueError('no strict training houses selected')
    used_seeds={r['scene_seed']for r in originals};used_ids={r['scene_id']for r in originals};new=[]
    for i in range(a.parents):
        house,room=rooms[i%len(rooms)];row=copy.deepcopy(templates[i%400]);index=400+i
        row['scene_id']=f'v7_full_train_{index:06d}';row['scene_index']=index
        seed=seeded('v7-full-20261005',index,house['house'],room['room_label'])
        if row['scene_id']in used_ids or seed in used_seeds:raise ValueError('full world overlaps the S prefix')
        used_seeds.add(seed);used_ids.add(row['scene_id'])
        row.update(scene_seed=seed,seeds={k:seeded(seed,k)for k in row['seeds']},house=house['house'],room_id=room['room_label'],split='train')
        row.pop('selection_coverage_seals',None)
        row['question_quota_base']=index*8
        placement=room['placement'];camera=placement['camera_m'];valid=[]
        for key in ['source_1_m','source_2_m']:
            point=placement[key];yaw=math.degrees(math.atan2(camera[0]-point[0],-(camera[2]-point[2])))
            valid.append({'position_m':point,'body_yaw_deg':yaw,'status':'inherited_acoustic_probe_candidate; full body native placement pending'})
        row['measured_static_layout']={'listener_m':camera,'valid':valid,'room_floor_polygon':room['floor_polygon'],
            'admission_status':'unqualified_new_body_candidate; field name retained for existing production ABI',
            'original_room_receipt':next(r['result_path']for r in house['original_room_evidence']if r['room_label']==room['room_label'])}
        row['preproduction_scope']='TRAIN only; new native body/contact/material/direction/motion/14-question admission still required'
        new.append(row)
    if set(r['house']for r in new)&excluded:raise ValueError('heldout house in seed proposal')
    a.output.mkdir(parents=True,exist_ok=False)
    def write(name,rows):
        (a.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n'for r in rows))
    write('first1500_train_seeds.jsonl',new)
    full=''.join(line+'\n'for line in prefix)+(a.output/'first1500_train_seeds.jsonl').read_text()
    (a.output/'full_train_prefix1900.jsonl').write_text(full)
    (a.output/'houses_train_only.json').write_text(json.dumps({**{k:v for k,v in S_houses.items()if k!='houses'},'houses':houses},ensure_ascii=False)+'\n')
    summary={'status':'prepared_not_production_admitted','parents':len(new),'strict_train_houses':len(houses),'strict_train_rooms':len(rooms),
        'family_houses':dict(collections.Counter(h['family']for h in houses)),'family_parents':dict(collections.Counter(next(h['family']for h in houses if h['house']==r['house'])for r in new)),
        'per_room_parent_counts':dict(sorted(collections.Counter(r['house']+'/'+r['room_id']for r in new).items())),
        'speaker_count_plan':dict(collections.Counter(r['production_scenario']['speaker_count']for r in new)),
        'sound_class_plan':dict(collections.Counter(r['production_scenario'].get('category_focus_class')for r in new)),
        'S_train_prefix_byte_identical':full.splitlines()[:400]==prefix,'S_parent_ID_and_seed_overlap':0,
        'TEST_house_geometry_reads':0,'native_calls':0,'excluded_physical_house_names':sorted(excluded),
        'filter_reasons':reasons,'input_catalog_sha256':digest(a.catalog),'S_seeds_sha256':digest(a.S_seeds),'inherited_train_receipts':source_receipts,
        'execution_blockers':['batch2 complete native RGB/visibility/actions/moving RIR and formal questions certificate','new training house CPU visual/placement smoke','final frozen producer revision and transfer guard'],
        'resource_contract':{'CPU_cores_max':24,'nice':10,'load_over_90_reduce_dispatch':True,'minimum_data_free_bytes':100000000000,'maximum_new_task_bytes':150000000000,'delete_48g_copy_only_after_CCI_SHA256':True},
        'thresholds_changed':False,'smy_labels_used_for_filtering':False}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items()if k not in ['inherited_train_receipts','per_room_parent_counts']},ensure_ascii=False))
if __name__=='__main__':main()
