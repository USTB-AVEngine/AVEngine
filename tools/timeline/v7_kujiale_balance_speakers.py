"""Version pending Kujiale requests to balance actual native 1–4 speakers.

Preserves original seeds, already attempted inputs, and zero/five-to-eight
coverage. These are requests only; the original native and QA gates decide
which worlds are admitted.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

RUN=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round14')
OLD=RUN/'inputs/kujiale_v7_S_train.jsonl'
NEW=RUN/'inputs/kujiale_v7_S_train_round18_balanced.jsonl'
EVIDENCE=RUN/'evidence/round18_speaker_balance.json'

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()

def select_spread(rows,number,area):
    """Choose donors across houses and rooms, preferring larger floors."""
    picked=[];byhouse=Counter();byroom=Counter()
    candidates=list(rows)
    while len(picked)<number:
        if not candidates:raise ValueError('insufficient donor rows')
        row=min(candidates,key=lambda r:(byhouse[r['house']],byroom[(r['house'],r['room_id'])],-area[r['scene_id']],r['scene_id']))
        candidates.remove(row);picked.append(row)
        byhouse[row['house']]+=1;byroom[(row['house'],row['room_id'])]+=1
    return picked

def main():
    from shapely.geometry import shape
    from avengine.timeline.v7_registered_sampling import _placements,choose_assets
    if NEW.exists() or EVIDENCE.exists():raise FileExistsError('balanced request snapshot already exists')
    rows=[json.loads(s) for s in OLD.read_text().splitlines() if s.strip()]
    state=json.loads((RUN/'PRODUCTION_STATUS.json').read_text())
    if state['status']!='stopped_not_complete' or len(state['attempted_parent_ids'])!=21:
        raise ValueError('balance only the exact stopped 21-attempt checkpoint')
    attempted=set(state['attempted_parent_ids'])
    if len(rows)!=132 or len({r['scene_id'] for r in rows})!=132:raise ValueError('fixed request identity differs')
    pending=[r for r in rows if r['scene_id'] not in attempted]
    eligible=[r for r in pending if 1<=r['production_scenario']['speaker_count']<=4]
    before=Counter(r['production_scenario']['speaker_count'] for r in pending)
    counts=Counter(r['production_scenario']['speaker_count'] for r in eligible)
    if len(pending)!=111 or sum(counts.values())!=76 or counts!=Counter({1:23,2:40,3:9,4:4}):
        raise ValueError('pending cohort changed; redesign balance, do not silently reassign')
    target=19
    area={r['scene_id']:float(shape(r['measured_static_layout']['room_floor_polygon']).area) for r in eligible}
    donors=select_spread([r for r in eligible if r['production_scenario']['speaker_count']==1],counts[1]-target,area)
    donors+=select_spread([r for r in eligible if r['production_scenario']['speaker_count']==2],counts[2]-target,area)
    if len(donors)!=(target-counts[3])+(target-counts[4]):raise ValueError('balance arithmetic differs')
    # Four speakers need five visible registered bodies. Spread these requests
    # across rooms and prefer more floor area; native checks still decide pass.
    current4=Counter((r['house'],r['room_id']) for r in eligible if r['production_scenario']['speaker_count']==4)
    to4=[];pool=list(donors)
    while len(to4)<target-counts[4]:
        row=min(pool,key=lambda r:(current4[(r['house'],r['room_id'])],-area[r['scene_id']],r['scene_id']))
        pool.remove(row);to4.append(row);current4[(row['house'],row['room_id'])]+=1
    assignment={r['scene_id']:4 for r in to4}
    assignment.update({r['scene_id']:3 for r in pool})
    assets=json.loads((RUN/'inputs/assets.json').read_text())['assets']
    updated=[];changes=[]
    for original in rows:
        r=deepcopy(original);old=r['production_scenario']['speaker_count'];new=assignment.get(r['scene_id'],old)
        if r['scene_id'] in attempted and new!=old:raise AssertionError('attempted request changed')
        if new!=old:
            if r['production_scenario'].get('asset_ids') or r['production_scenario'].get('semantic_scenario_id'):
                raise ValueError('explicit native role plan cannot be reallocated')
            r['production_scenario']['speaker_count']=new
            selected=choose_assets(assets,r)
            if len(selected)<new+1 or len(_placements(r['measured_static_layout'],len(selected)))<len(selected):
                raise ValueError('CPU separated candidate/actor catalog deficit')
            changes.append({'scene_id':r['scene_id'],'house':r['house'],'room_id':r['room_id'],'before':old,'after':new,'floor_area_m2':area[r['scene_id']]})
        if r['scene_seed']!=original['scene_seed'] or r['seeds']!=original['seeds']:
            raise AssertionError('fixed seeds changed')
        updated.append(r)
    after=Counter(r['production_scenario']['speaker_count'] for r in updated if r['scene_id'] not in attempted)
    if any(after[n]!=target for n in range(1,5)) or any(after[n]!=before[n] for n in [0,5,6,7,8]):
        raise AssertionError('speaker allocation failed')
    NEW.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in updated))
    evidence={'schema':'kujiale_round18_pending_speaker_balance_v1','status':'request_only_native_qualification_pending',
              'original_inputs':str(OLD),'original_sha256':digest(OLD),'balanced_inputs':str(NEW),'balanced_sha256':digest(NEW),
              'attempted_fixed_unchanged':sorted(attempted),'pending_count':len(pending),'pending_before':dict(sorted(before.items())),
              'pending_after':dict(sorted(after.items())),'changed_count':len(changes),'changes':changes,
              'unchanged_seed_and_qa_threshold':True,'zero_and_5_to_8_coverage_preserved':True,
              'CPU_separated_placement_and_catalog_checked':True,'UE_contact_and_formal_QA_pending':True}
    EVIDENCE.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ['pending_count','pending_before','pending_after','changed_count','original_sha256','balanced_sha256']},ensure_ascii=False))

if __name__=='__main__':main()
