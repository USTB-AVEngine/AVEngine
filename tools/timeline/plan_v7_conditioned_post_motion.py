#!/usr/bin/env python3
"""Construct one native v7 post-sound parent using the existing v3 solver."""
import argparse,json,subprocess
from pathlib import Path
from copy import deepcopy
from avengine.timeline.v7_conditioned_post_motion import (
    POST_SOUND_FAMILIES,registered_human_motion_registry,plan_post_sound_motion,build_conditioned_post_scene)
from avengine.timeline.v7_phase2_sources import build_phase2_scene
from avengine.timeline.v7_demand import read_demand_state,demand_snapshot,FAMILY_SCENE_REQUIREMENTS

def load(path):return json.loads(Path(path).read_text())

def prepare(manifest_path,*,source_slot,scene_id,qa_id=None):
    m=load(manifest_path);root=Path(m['output_root']);state=read_demand_state(root/'inputs/demand_state.json')
    if qa_id is None:
        qa_id=min(POST_SOUND_FAMILIES,key=lambda family:(state['family_counts'].get(family,0),family))
    if any(t['row']['scene_id']==scene_id for t in m['parents']):
        raise ValueError('scene ID already planned; preserve its old recipe')
    source=m['parents'][source_slot];previous=root/'inputs/planned'/f'{source_slot:03d}.json'
    row=deepcopy(load(previous) if previous.exists() else source['row'])
    row.update(scene_id=scene_id,independent_parent_scene_id=scene_id)
    catalog=load(m['assets_cpu'])['assets'];lookup={a['asset_id']:a for a in catalog}
    humans=[lookup[aid] for aid in row['asset_ids'] if lookup[aid]['kind']=='human' and aid not in row.get('silent_asset_ids',[])]
    target=next((a for a in humans if a.get('walk_allowed')),None)
    if target is None or len(humans)<2:raise ValueError('actual admitted walking speaker and static companion required')
    companion=next(a for a in humans if a['asset_id']!=target['asset_id'])
    runtime=load(m['runtime_cpu']);semantic=load(runtime['semantic_manifest'])
    # Keep a real authored standing companion, plus the walking recorded voice.
    gender=companion['visual_gender'];candidates=[s for s in semantic['sounds']
        if s['voice_reference']['gender']==gender and .8<=s['synthesis_readback']['duration_s']<=2.3]
    voices=sorted({s['voice_reference']['speaker_id'] for s in candidates})
    chosen_voice=voices[len(state.get('parents',{}))%len(voices)]
    speech=min((s for s in candidates if s['voice_reference']['speaker_id']==chosen_voice),
        key=lambda s:(s['synthesis_readback']['duration_s'],s['sound_asset_id']))
    row.update(semantic_scenario_id=speech['scenario_id'],semantic_sound_ids=[speech['sound_asset_id']],
        semantic_human_asset_ids=[companion['asset_id']],semantic_start_offsets_s=[0.],motion_requests={})
    for key in ('phase2_recorded_speech_schedule_s','phase2_recorded_sound_ids','proposed_sound_asset_ids'):
        row.get(key,{}).pop(companion['asset_id'],None)
    row['scene_demand']={**demand_snapshot(state,next_questions=70),'focus':'conditioned_post_sound',
        'requested_conditions':list(FAMILY_SCENE_REQUIREMENTS[qa_id]['conditions']),
        'required_asset_ids':list(row['asset_ids']),'required_independent_family_counts':{qa_id:3}}
    house=next(h for h in load(m['houses_cpu'])['houses'] if h['house']==row['house'])
    contract=load(m['contract']);source_admission=load(contract['inputs']['source_admission']['path'])
    _,scene,program,assets=build_phase2_scene(row,house,source_admission['records'],catalog,semantic)
    records={r['sound_asset_id']:r for r in program['records']};bindings={b['event_id']:b for b in program['events']}
    records_by_asset={}
    for aid in (target['asset_id'],companion['asset_id']):
        ep=next(a['endpoint_id'] for a in assets if a['asset_id']==aid)
        events=sorted((e for e in scene['events'] if e['endpoint_id']==ep),key=lambda e:e['start_tick'])
        records_by_asset[aid]=[records[bindings[e['event_id']]['sound_asset_id']] for e in events]
    registry=registered_human_motion_registry(humans,runtime,house)
    row=plan_post_sound_motion(row,qa_id=qa_id,target=target['asset_id'],companion=companion['asset_id'],
        assets=assets,registry=registry,records_by_asset=records_by_asset)
    # Resolve the resulting real full clips once more before publishing the plan.
    _,resolved,resolved_program,_=build_conditioned_post_scene(row,house,source_admission['records'],catalog,semantic)
    slot=len(m['parents']);m['parents'].append({'slot':slot,'family':source['family'],'device':'cpu',
        'profile':'phase2','focus':'human_motion','row':row})
    m['code_commit']=subprocess.check_output(['git','-C',m['worktree'],'rev-parse','HEAD'],text=True).strip()
    evidence=root/'evidence'/('CONDITIONED_POST_'+str(slot).zfill(3)+'.json')
    evidence.write_text(json.dumps({'row':row,'events':resolved['events'],'program':resolved_program,
        'status':'compiled_proposal_native_not_yet_admitted'},ensure_ascii=False,indent=2)+'\n')
    (root/'inputs/planned'/f'{slot:03d}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    Path(manifest_path).write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'slot':slot,'scene_id':scene_id,'qa_id':qa_id,'evidence':str(evidence),
        'motion_intervals_s':row['motion_requests'][target['asset_id']]['intervals_s'],
        'target_event_starts_s':row['phase2_recorded_speech_schedule_s'][target['asset_id']],
        'standing_TTS_voice':chosen_voice}),flush=True)
    return m,slot

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source-slot',type=int,required=True);p.add_argument('--scene-id',required=True)
    p.add_argument('--qa-id',choices=POST_SOUND_FAMILIES);p.add_argument('--run',action='store_true')
    a=p.parse_args();m,slot=prepare(a.manifest,source_slot=a.source_slot,scene_id=a.scene_id,qa_id=a.qa_id)
    if a.run:
        from tools.timeline.run_v7_acceptance import run_worker
        raise SystemExit(run_worker(m,slot))
if __name__=='__main__':main()
