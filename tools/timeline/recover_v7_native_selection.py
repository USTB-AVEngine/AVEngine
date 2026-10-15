#!/usr/bin/env python3
"""Finish a native parent's selection in a fresh owned attempt without rendering.

Source RGB/actions/audio and every original QA truth replay must pass. Only
quota selection resumes; native observations, answers and gates are unchanged.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,datetime,fcntl,json,os,subprocess,time,uuid
from avengine.timeline.v7_candidate_generation import stage,read_json
from avengine.timeline.continuous_pilot import write_json


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def recover(manifest_path,source_parent):
    from avengine.timeline.v7_phase2_generation import native_checks,cpu_seconds,PILOT_PROFILE
    from avengine.timeline.v7_full_question_delivery import finalize_parent_question_pools
    from avengine.timeline.v7_registered_generation import _verify_window
    from avengine.qa.v7_validation import validate_v6_items,validate_content_items
    from avengine.qa.v7_visible_truth import validate_additional_items
    from avengine.acoustics.moving_rir_acceptance import qualify_energy
    from avengine.timeline.v7_question_independence import deduplicate_parent_delivery
    from avengine.timeline.v7_s_batch_checks import open_delivered_media
    from avengine.timeline.v7_demand import record_selected_parent,SELECTION_FEEDBACK_ENV
    from tools.timeline.export_v7_trial import parent_payload
    m=read_json(manifest_path);output=Path(m['output_root']).resolve();source_parent=Path(source_parent).resolve()
    allowed=[output,*[Path(p).resolve()for p in m.get('peer_output_roots',[])]]
    if not any(source_parent.is_relative_to(p)for p in allowed):raise ValueError('source parent outside authorized evidence roots')
    recipe=read_json(source_parent/'input_recipe.json');row=recipe['row'];scene=row['scene_id']
    if row['split']!='train'or row.get('acceptance_allow_partial_families')is True:raise ValueError('complete train-only scene required')
    if any(v.get('action','').startswith('removed_')for v in row.get('element_degradations',[])):raise ValueError('required body was removed')
    if any(row.get('source_classes',{}).get(aid)!=['footsteps']for aid in row.get('phase2_include_footsteps_asset_ids',[])):raise ValueError('human has multiple sound sources')
    if row['members']!=['v0_a0']:raise ValueError('this recovery only supports complete single-member physical motion parents')
    windows=read_json(source_parent/'windows_v0_a0_completed.json');pool=source_parent/windows['attempt']
    audio=read_json(source_parent/'audio_v0_a0_completed.json');adir=source_parent/audio['attempt'];raw_checks=[]
    for ep in adir.glob('*_moving_energy.json'):
        data=read_json(ep);checks=[c for x in data.get('detailed_midpoint_checks',[])for c in x.get('raw_checks',[])]
        if not checks or qualify_energy(checks)['status']!='pass':raise ValueError('original absolute raw RIR comparison fails: '+str(ep))
        raw_checks.append({'evidence':str(ep),'native_independent_comparisons':len(checks),'unchanged_qualify_energy':'pass'})
    proof=read_json(pool/windows['result']['windows'][0]['window_id']/'evidence.json')
    visual=read_json(proof['native_visual_evidence']);checks=native_checks(proof['scene'],recipe['assets'],visual,audio['result'])
    if checks['status']!='pass'or checks['RIR_direction']['status']!='pass':raise ValueError('original native machine checks fail')
    original_replays=[];starts=[]
    for win in windows['result']['windows']:
        wid=win['window_id'];folder=pool/wid;q=read_json(folder/'questions.json');g=read_json(folder/'private_answers.json');e=read_json(folder/'evidence.json');audit=read_json(folder/'candidate_audit.json')
        old=read_json(folder/'independent_v6_truth.json')
        if old['status']!='pass'or old['checked_forms']!=len(q)or old['mismatches']:raise ValueError('source full-pool replay incomplete')
        parts=[validate_v6_items(q,g,audit,seed=str(row['scene_seed'])),validate_additional_items(q,g,e['source_qa_evidence']),validate_content_items(q,g)]
        if sum(p['checked_forms']for p in parts)!=len(q)or any(p['mismatches']for p in parts):raise ValueError('original complete QA replay failed during recovery')
        original_replays.append({'window_id':wid,'checked_forms':len(q),'panels':parts,'status':'pass'})
        starts.append(float(e['source_qa_evidence']['window_start_s']))
    if starts!=[0.,5.,10.,15.,20.]:raise ValueError('owner half-overlap crop clock missing')
    attempt=output/'native_reselections'/uuid.uuid4().hex/'parents'/scene;attempt.mkdir(parents=True)
    save(attempt/'input_recipe.json',recipe)
    evidence={'source_parent':str(source_parent),'original_raw_RIR_recheck':raw_checks,'native_checks':checks,
        'complete_original_QA_replays':original_replays,'new_native_renders':0,'required_assets_preserved':[a['asset_id']for a in recipe['assets']]}
    save(attempt/'selection_recovery_evidence.json',evidence)
    with(output/'inputs/plan.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);fresh=read_json(manifest_path)
        if any(t.get('native_selection_recovery_source')==str(source_parent)for t in fresh['parents']):raise ValueError('same physical source already recovered; do not count it twice')
        slot=len(fresh['parents']);task={'slot':slot,'family':'hm3d'if row['house'].startswith('hm3d')else'mp3d','device':'cpu','profile':'phase2','focus':'animal_motion','row':row,'launch_disabled':True,'native_selection_recovery_source':str(source_parent)}
        fresh['parents'].append(task);save(manifest_path,fresh)
    started=time.monotonic();cpu=cpu_seconds();os.environ[SELECTION_FEEDBACK_ENV]=str(output/'inputs/demand_state.json')
    record={'slot':slot,'status':'running','house':row['house'],'room_id':row['room_id'],'scene_id':scene,'pid':os.getpid(),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'origin_helper':'motion','origin_helper_receipt':str(source_parent),'selection_recovery':str(attempt/'selection_recovery_evidence.json')}
    save(output/'evidence/jobs'/f'{slot:03d}.json',record)
    try:
        members,selection=finalize_parent_question_pools(row,[{'member':'v0_a0','windows_directory':str(pool),
            'windows':windows['result']['windows'],'audio_directory':str(adir),'visual_directory':str(Path(proof['native_visual_evidence']).parent)}],attempt)
        def seal(dest):
            required=[];stats=[];selected=Path(members[0]['windows_directory']);private=dest/'private';private.mkdir()
            for win in members[0]['windows']:
                wid=win['window_id'];target=dest/'windows'/wid;target.mkdir(parents=True)
                for name in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json']:
                    os.link(selected/wid/name,target/name);required.append(str((target/name).relative_to(dest)))
                for old,new in [('private_answers.json',wid+'_answers.json'),('evidence.json',wid+'_evidence.json')]:
                    os.link(selected/wid/old,private/new);required.append('private/'+new)
                qs=read_json(target/'questions.json');logical={q['logical_question_id']:q for q in qs}
                stats.append({'window_id':wid,'media_readback':_verify_window(target),'logical_questions':len(logical),
                    'family_counts':dict(Counter(q['qa_id']for q in logical.values()))})
            for name,value in [('machine_checks.json',checks),('assets.json',recipe['assets']),('dry_program.json',recipe['program']),
                    ('dry_receipt.json',read_json(source_parent/'dry_completed.json')['result']),('selection_recovery.json',evidence)]:
                write_json(private/name,value);required.append('private/'+name)
            result={'status':'integration_pilot_admitted','profile':PILOT_PROFILE,'scene_id':scene,'house':row['house'],'family':task['family'],
                'room_id':row['room_id'],'windows':stats,'machine_checks':checks,'pilot_windows':5,'pilot_logical_questions':70,
                'delivery_window_starts_s':starts,'new_native_renders_in_selection_recovery':0,'original_native_parent':str(source_parent),
                'wall_seconds':time.monotonic()-started,'cpu_seconds_including_children':cpu_seconds()-cpu}
            write_json(dest/'pilot_manifest.json',result);required.append('pilot_manifest.json');return result,required
        folder,result,_=stage(attempt,'integration_seal',{'input_recipe':recipe,'native_selection_recovery':str(source_parent)},seal)
        delivered=deduplicate_parent_delivery(row,result,attempt,folder)
        info={'slot':slot,'family':task['family'],'status':'passed','house':row['house'],'scene_id':scene,
            'parent_directory':str(attempt),'seal_path':str(attempt/'integration_seal_completed.json')}
        rows,exported=parent_payload(info,output/'export',output,read_only_source_roots=m.get('peer_output_roots',[]))
        for r in rows:r['dataset_role']='acceptance_only'
        table=output/'export/parent_rows'/f'{slot:03d}.jsonl';table.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows))
        reader_root=output/'reader_checks_v2'/f'{slot:03d}';media=[]
        for win in exported['windows']:
            wid=win['window_id'];relative=Path('parents')/scene/'windows'/wid
            for name in ['video.mp4','audio_foa.wav','audio_mono.wav']:
                target=reader_root/relative/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(output/'export'/relative/name,target)
            media.append({'window_id':wid,'video':str(relative/'video.mp4'),'audio_foa':str(relative/'audio_foa.wav'),'audio_mono':str(relative/'audio_mono.wav')})
        (reader_root/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in media));reader=open_delivered_media(reader_root)
        if reader['status']!='pass':raise ValueError('final recovered media reader fails')
        record.update(status='passed',producer_result=delivered,export=exported,reader=reader,independent_questions=exported['independent_questions'],
            raw_questions=exported['raw_export_questions'],wall_seconds=time.monotonic()-started,cpu_seconds=cpu_seconds()-cpu)
        record_selected_parent(output/'inputs/demand_state.json',{**delivered,'windows':[{'family_counts':exported['question_family_counts'],'logical_questions':len(rows)}]})
    except Exception as error:
        import traceback
        record.update(status='failed',reason=str(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started,cpu_seconds=cpu_seconds()-cpu)
    save(output/'evidence/jobs'/f'{slot:03d}.json',record)
    with(output/'PROGRESS.md').open('a')as log:log.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' 原生完整池新attempt恢复='+scene+'；状态='+record['status']+'；独立题='+str(record.get('independent_questions',0))+'；原绝对RIR/全池QA复验、原14题/reader；实体和原媒体未改。\n')
    print(json.dumps({k:v for k,v in record.items()if k in ['slot','scene_id','status','reason','independent_questions']},ensure_ascii=False),flush=True)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source-parent',type=Path,required=True);a=p.parse_args()
    record=recover(a.manifest,a.source_parent)
    raise SystemExit(0 if record['status']=='passed'else 1)
if __name__=='__main__':main()
