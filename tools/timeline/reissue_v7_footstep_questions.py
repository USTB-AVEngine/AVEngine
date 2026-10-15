#!/usr/bin/env python3
"""Re-question admitted native footstep parents without rendering or changing media.

Each revision has a new owned pool/seal/export. After every gate and reader pass,
atomically replace the active slot and retain the old slot/table as audit history.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,time,uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path
from avengine.timeline.v7_candidate_generation import stage,read_json
from avengine.timeline.continuous_pilot import write_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def prefix_media(rows,prefix):
    result=deepcopy(rows)
    for row in result:
        for field in ('video','audio_foa','audio_mono','private_evidence'):
            row[field]=str(Path(prefix)/row[field])
        row['dataset_role']='acceptance_only'
    return result


def reissue(manifest_path,slot):
    from avengine.timeline.v7_footstep_bouts import REVISION,footstep_bout_question_inputs
    from avengine.timeline.v7_full_question_delivery import collect_native_pool,finalize_parent_question_pools
    from avengine.timeline.v7_phase2_generation import native_checks,PILOT_PROFILE
    from avengine.timeline.v7_registered_generation import _verify_window
    from avengine.qa.v7_validation import validate_v6_items,validate_content_items
    from avengine.qa.v7_visible_truth import validate_additional_items
    from avengine.timeline.v7_question_independence import deduplicate_parent_delivery
    from avengine.timeline.v7_s_batch_checks import open_delivered_media
    from avengine.timeline.v7_english_export import validate_english_model_text
    from avengine.acoustics.moving_rir_acceptance import qualify_energy
    from avengine.timeline.v7_frozen_code import source_commit
    from tools.timeline.export_v7_trial import parent_payload
    started=time.monotonic();m=read_json(manifest_path);output=Path(m['output_root']).resolve()
    jobfile=output/'evidence/jobs'/f'{slot:03d}.json';old=read_json(jobfile)
    if old['status']!='passed':raise ValueError('only a previously fully admitted slot can be re-questioned')
    if old.get('question_annotation_revision')==REVISION:return old
    pointer_path=Path(old['export']['source_seal']);pointer=read_json(pointer_path)
    source=pointer_path.parent;recipe=pointer['recipe']['input_recipe'];row=deepcopy(recipe['row'])
    permitted=[output,*[Path(p).resolve()for p in m.get('peer_output_roots',[])]]
    if not any(source.resolve().is_relative_to(p)for p in permitted):raise ValueError('source outside authorized read-only evidence roots')
    if row['members']!=['v0_a0']or row['split']!='train':raise ValueError('complete single-member native CPU footstep parent required')
    if row['house']in read_json(m['contract']).get('test_houses',[]):raise ValueError('held-out house')
    if not row.get('phase2_include_footsteps_asset_ids'):raise ValueError('parent does not declare real footstep emitters')
    if any(row.get('source_classes',{}).get(a)!=['footsteps']for a in row['phase2_include_footsteps_asset_ids']):raise ValueError('footstep person cannot also speak')
    selected=read_json(source/'selected_windows_v0_a0_completed.json');pool=Path(selected['recipe']['pool_directory'])
    audio_pointer=read_json(source/'audio_v0_a0_completed.json');audio_dir=source/audio_pointer['attempt']
    raw_rir=[]
    for path in audio_dir.glob('*_moving_energy.json'):
        data=read_json(path);checks=[c for p in data.get('detailed_midpoint_checks',[])for c in p.get('raw_checks',[])]
        if not checks or qualify_energy(checks)['status']!='pass':raise ValueError('unchanged raw RIR gate failed: '+str(path))
        raw_rir.append({'path':str(path),'sha256':sha(path),'native_checks':len(checks),'status':'pass'})
    wid0=pointer['result']['windows'][0]['window_id'];first=read_json(pool/wid0/'evidence.json')
    visual=read_json(first['native_visual_evidence']);physical=first['scene']
    machine=native_checks(physical,recipe['assets'],visual,audio_pointer['result'])
    if machine['status']!='pass'or machine['RIR_direction']['status']!='pass':raise ValueError('original native checks failed')
    token=uuid.uuid4().hex;root=output/'question_reissues'/token/'parents'/row['scene_id'];root.mkdir(parents=True)
    row['question_annotation_revision']=REVISION
    updated_recipe={**deepcopy(recipe),'row':row,'question_reissue':{'revision':REVISION,'original_source_seal':str(pointer_path),'original_source_seal_sha256':sha(pointer_path),'new_native_renders':0}}
    write_json(root/'input_recipe.json',updated_recipe)
    destination=root/'native_question_pool';destination.mkdir();wins=[];media_proofs=[];bout_stats=[]
    os.environ.pop('AVENGINE_V7_STORAGE_BUDGET',None)
    for win in pointer['result']['windows']:
        wid=win['window_id'];oldfolder=pool/wid;folder=destination/wid;folder.mkdir()
        evidence=read_json(oldfolder/'evidence.json');start=float(evidence['source_qa_evidence']['window_start_s'])
        question_scene,proof=footstep_bout_question_inputs(evidence['scene'],evidence['source_qa_evidence'])
        if proof.get('footstep_bouts',{}).get('revision')!=REVISION:raise ValueError('actual raw footstep events missing')
        media={}
        for field,name in [('video','video.mp4'),('audio_foa','audio_foa.wav'),('audio_mono','audio_mono.wav')]:
            os.link(oldfolder/name,folder/name);media[field]=str(folder/name)
            media_proofs.append({'source':str(oldfolder/name),'destination':str(folder/name),'sha256':sha(folder/name),'identical_inode':(oldfolder/name).stat().st_ino==(folder/name).stat().st_ino})
        index=int(wid.rsplit('w',1)[1]);q,g,audit=collect_native_pool(question_scene,start,wid,visual,media,proof,row['question_quota_base']+index,14)
        panels=[validate_v6_items(q,g,audit,seed=str(row['scene_seed'])),validate_additional_items(q,g,proof),validate_content_items(q,g)]
        replay={'status':'pass','checked_forms':sum(p['checked_forms']for p in panels),'passed_forms':sum(p['passed_forms']for p in panels),'mismatches':[r for p in panels for r in p['mismatches']],'panels':panels}
        if replay['checked_forms']!=len(q)or replay['mismatches']:raise ValueError('new logical-bout original truth replay failed')
        for name,value in [('questions.json',q),('private_answers.json',g),('candidate_audit.json',audit),('independent_v6_truth.json',replay)]:write_json(folder/name,value)
        write_json(folder/'evidence.json',{**evidence,'seed_row':row,'scene':question_scene,'physical_scene_before_question_annotation':evidence['scene'],'source_qa_evidence':proof,'independent_v6_truth':replay,'question_annotation_revision':REVISION})
        bout_stats.append({'window_id':wid,'raw_step_count':proof['footstep_bouts']['raw_step_count'],'bout_count':proof['footstep_bouts']['bout_count'],'bouts':proof['footstep_bouts']['bouts']});wins.append({'window_id':wid})
    starts=[float(read_json(destination/w['window_id']/'evidence.json')['source_qa_evidence']['window_start_s'])for w in wins]
    if starts!=[0.,5.,10.,15.,20.]:raise ValueError('owner five-window clock differs')
    provenance={'revision':REVISION,'original_slot':slot,'source_parent':str(source),'original_source_seal_sha256':sha(pointer_path),'new_native_renders':0,'media':media_proofs,'original_raw_RIR_recheck':raw_rir,'bout_evidence':bout_stats}
    write_json(root/'question_reissue_evidence.json',provenance)
    members,selection=finalize_parent_question_pools(row,[{'member':'v0_a0','windows_directory':str(destination),'windows':wins,'audio_directory':str(audio_dir),'visual_directory':str(Path(first['native_visual_evidence']).parent)}],root)
    def seal(dest):
        required=[];stats=[];selected_dir=Path(members[0]['windows_directory']);private=dest/'private';private.mkdir()
        for win in members[0]['windows']:
            wid=win['window_id'];folder=dest/'windows'/wid;folder.mkdir(parents=True)
            for name in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json']:
                os.link(selected_dir/wid/name,folder/name);required.append(str((folder/name).relative_to(dest)))
            for original,new in [('private_answers.json',wid+'_answers.json'),('evidence.json',wid+'_evidence.json')]:os.link(selected_dir/wid/original,private/new);required.append('private/'+new)
            qs=read_json(folder/'questions.json');logical={q['logical_question_id']:q for q in qs}
            stats.append({'window_id':wid,'media_readback':_verify_window(folder),'logical_questions':len(logical),'family_counts':dict(Counter(q['qa_id']for q in logical.values()))})
        for name,value in [('machine_checks.json',machine),('assets.json',recipe['assets']),('dry_program.json',recipe['program']),('dry_receipt.json',read_json(source/'dry_completed.json')['result']),('footstep_bout_reissue.json',provenance)]:write_json(private/name,value);required.append('private/'+name)
        result={**pointer['result'],'windows':stats,'machine_checks':machine,'new_native_renders':0,'question_annotation_revision':REVISION,'wall_seconds_original_parent':pointer['result'].get('wall_seconds'),'wall_seconds':time.monotonic()-started,'raw_physical_class_event_counts':pointer['result'].get('class_event_counts',{}),'class_event_counts':dict(Counter(e['sound_class']for e in question_scene['events']))}
        write_json(dest/'pilot_manifest.json',result);required.append('pilot_manifest.json');return result,required
    sealed,result,_=stage(root,'integration_seal',{'input_recipe':updated_recipe},seal)
    delivered=deduplicate_parent_delivery(row,result,root,sealed)
    prefix=str(Path('question_reissues')/token);new_export=output/'export'/prefix
    info={'slot':slot,'family':m['parents'][slot]['family'],'status':'passed','house':row['house'],'scene_id':row['scene_id'],'parent_directory':str(root),'seal_path':str(root/'integration_seal_completed.json')}
    rows,exported=parent_payload(info,new_export,output,read_only_source_roots=m.get('peer_output_roots',[]))
    for r in rows:validate_english_model_text(r)
    new_export.mkdir(parents=True,exist_ok=True);(new_export/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    reader=open_delivered_media(new_export)
    if reader['status']!='pass'or reader['actual_decoded_windows']!=5:raise ValueError('new export reader did not verify all five windows')
    rows=prefix_media(rows,prefix)
    replacement={**old,'producer_result':delivered,'export':exported,'reader':reader,'independent_questions':exported['independent_questions'],'raw_questions':exported['raw_export_questions'],'question_annotation_revision':REVISION,'delivery_export_prefix':prefix,'question_reissue_provenance':str(root/'question_reissue_evidence.json'),'question_reissue_code_commit':source_commit(Path(__file__).resolve().parents[2]),'question_reissue_wall_seconds':time.monotonic()-started}
    with(output/'evidence/report_build.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if sha(jobfile)!=hashlib.sha256(json.dumps(old,ensure_ascii=False,indent=2).encode()).hexdigest():
            # Byte formatting may differ; the semantic receipt must stay exact.
            if read_json(jobfile)!=old:raise ValueError('active parent changed while re-questioning')
        history=output/'evidence/question_reissue_history'/token;history.mkdir(parents=True)
        os.link(jobfile,history/f'{slot:03d}.json')
        table=output/'export/parent_rows'/f'{slot:03d}.jsonl';os.link(table,history/f'{slot:03d}.jsonl')
        temp=table.with_name(table.name+'.'+token+'.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows));temp.replace(table)
        atomic(jobfile,replacement)
    with(output/'PROGRESS.md').open('a')as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' FOOTSTEP_BOUT_REISSUE '+row['scene_id']+' old='+str(old['independent_questions'])+' new='+str(len(rows))+' families='+json.dumps(exported['question_family_counts'])+' renders=0 evidence='+str(root/'question_reissue_evidence.json')+'\n')
    return replacement


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--slot',type=int,action='append',required=True);a=p.parse_args()
    for slot in a.slot:
        result=reissue(a.manifest,slot);print(json.dumps({'slot':slot,'scene_id':result['scene_id'],'independent_questions':result['independent_questions'],'family_counts':result['export']['question_family_counts'],'revision':result['question_annotation_revision']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
