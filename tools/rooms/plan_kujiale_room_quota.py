#!/usr/bin/env python3
"""Training-only Kujiale budgets bounded by admitted rooms, with nested seeds."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
from copy import deepcopy
from avengine.timeline.v7_production_contract import derived_seed
from tools.rooms.alpha_storage_contract import write_owned_json as write_json


def room_budgets(admission):
    if admission['split']!='train' or admission['render_path']!='ue_spear':raise ValueError('Kujiale must remain native UE and training only')
    rooms=[(h['house'],rid) for h in sorted(admission['houses'],key=lambda h:h['house']) for rid in sorted(h['room_ids'])]
    if len(rooms)!=len(set(rooms)) or len(rooms)!=admission['room_count']:raise ValueError('room admission count or uniqueness differs')
    if len({h for h,_ in rooms})!=admission['house_count']:raise ValueError('house admission count differs')
    return rooms


def quota_rows(admission,template,parents_per_room):
    if parents_per_room not in (4,16):raise ValueError('only authorized 4/16-parent-per-room tiers')
    rooms=room_budgets(admission);rows=[]
    for i,(house,rid) in enumerate(rooms):
        for slot in range(parents_per_room):
            index=9_500_000+16*i+slot;row=deepcopy(template)
            for field in ('measured_static_layout','upstream_room_id','upstream_members','upstream_window_start_ticks','upstream_schema'):
                row.pop(field,None)
            row['schema']='v7-kujiale-room-budget-seed-v1'
            row.update(house=house,room_id=rid,scene_index=index,scene_id=f'v7_kujiale_{house.removeprefix("kujiale_")}_{rid}_p{slot:02d}',scene_seed=derived_seed('train',index,'scene'),split='train',room_parent_slot=slot,render_path='ue_spear',assignment_status='admitted',physical_room_admission=True,
                generation_admission='pending_current_formal_alpha_layout_and_question_pipeline',quota_basis='admitted room count; user round8')
            row['seeds']={name:derived_seed('train',index,name) for name in row['seeds']}
            row['members']=['v0_a0'];row['counterfactual_plan']='single';row['allowed_member_changes']=[]
            rows.append(row)
    return rows


def member_mix(small,full):
    # 1.6 expected members means ~20% four-member parents + ~80% singles.
    # The shared selector accepts complete quartets or singles, so nearest
    # integral legal totals are185/743, rather than an invented two-member group.
    def order(row):return hashlib.sha256(row['scene_id'].encode()).hexdigest()
    sq=round(len(small)*.2);fq=round(len(full)*.2);selected={r['scene_id'] for r in sorted(small,key=order)[:sq]}
    extra=sorted([r for r in full if r['scene_id'] not in {x['scene_id'] for x in small}],key=order)
    selected.update(r['scene_id'] for r in extra[:fq-sq])
    for rows in [small,full]:
        for index,row in enumerate(rows):
            if row['scene_id'] in selected:
                row.update(members=['v0_a0','v0_a1','v1_a0','v1_a1'],counterfactual_plan='appearance_x_emitter_association',allowed_member_changes=['actor_appearance_assignment','dry_bus_to_emitter_assignment'])
            row['question_quota_base']=(row['scene_index']-9_500_000)*8
    return small,full


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['admission','seed-template','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();admission=json.loads(a.admission.read_text());template=json.loads(a.seed_template.read_text())
    small,full=member_mix(quota_rows(admission,template,4),quota_rows(admission,template,16));a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for name,rows in [('v7_S',small),('full',full)]:
        (a.output/(name+'_train.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        members=sum(len(r['members']) for r in rows);results.append({'tier':name,'houses':admission['house_count'],'rooms':admission['room_count'],'parents_per_room':4 if name=='v7_S' else 16,'parents':len(rows),'expected_members_at_1_6':len(rows)*1.6,'legal_rounded_members':members,'quartet_parents':sum(len(r['members'])==4 for r in rows),'single_parents':sum(len(r['members'])==1 for r in rows),'windows':members*2,'logical_questions_at_14_per_window':members*28,'offered_question_forms_at_current_alpha':members*28,'offered_form_scope':'open only in the measured canonical alpha; MCQ not emitted, no doubling','generation_status':'planned_not_generated','member_mix_status':'explicit feasible proposal pending common pipeline quota incorporation','native_map_imports_already_exist':True})
    write_json(a.output/'budgets.json',{'schema':'kujiale_room_budget_round8_v1','tiers':results,'test_or_dev_Kujiale_parents':0,'old_2300_4800_parent_range_revoked':True,'room_ids':room_budgets(admission),'benchmark_replicas_count_towards_training':False})
    with (a.output/'room_budgets.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['house','room_id','render_path','split','v7_S_parent_limit','full_parent_limit']);w.writeheader()
        w.writerows({'house':h,'room_id':r,'render_path':'ue_spear','split':'train','v7_S_parent_limit':4,'full_parent_limit':16} for h,r in room_budgets(admission))
    print(json.dumps(results,ensure_ascii=False))
if __name__=='__main__':main()
