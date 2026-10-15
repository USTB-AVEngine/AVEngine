#!/usr/bin/env python3
"""Finalize a task-scoped unattended batch with independent modality quotas.

Reads automatically produced receipts and general registries. It never reads
an agent's scene-construction scripts, changes a seal, or starts a renderer.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,datetime,fcntl,hashlib,json,os
from avengine.timeline.v7_admitted_pool_reselection import balanced_batch,record_batch_selection,save
from avengine.timeline.v7_english_export import validate_english_model_text
from avengine.timeline.v7_s_batch_checks import open_delivered_media


def admitted_receipts(manifest):
    root=Path(manifest['output_root']);receipts=[];excluded=[]
    for task in manifest['parents']:
        path=root/'evidence/jobs'/f"{task['slot']:03d}.json"
        if not path.exists():continue
        receipt=json.loads(path.read_text())
        if receipt.get('status')!='passed':continue
        if receipt.get('reader',{}).get('status')!='pass':raise ValueError('passed parent lacks completed actual media reader')
        pointer=json.loads(Path(receipt['export']['source_seal']).read_text());recipe=pointer['recipe']['input_recipe'];row=recipe.get('row',recipe.get('seed_row'))
        reasons=[]
        if any(x.get('action','').startswith('removed_')for x in row.get('element_degradations',[])):reasons.append('required_element_removed')
        if any(row.get('source_classes',{}).get(aid)!=['footsteps']for aid in row.get('phase2_include_footsteps_asset_ids',[])):reasons.append('mixed_human_sound_sources')
        # Candidate/modal allocation is a sampling observation. Every delivered
        # question still passed original native truth, English and scoring.
        # Owner 21:0x forbids invalidating a parent for this allocation flag.
        motion=pointer['result'].get('machine_checks',{}).get('motion',{}).get('instances',[])
        if any(x.get('motion_kind')=='static_after_failed_route'for x in motion):reasons.append('motion_demand_became_static')
        from avengine.timeline.v7_export_scene_policy import cached_parent_policy
        try:
            policy=cached_parent_policy(receipt['export']['source_seal'],root/'evidence/scene_delivery_policy')
            reasons.extend(policy['reasons'])
        except Exception as error:
            reasons.append('scene_policy_evidence_error:'+str(error))
        if reasons:excluded.append({'scene_id':receipt['scene_id'],'slot':receipt['slot'],'reasons':reasons})
        else:receipts.append(receipt)
    return receipts,excluded


def delivery_fingerprint(manifest):
    root=Path(manifest['output_root']);parents=[]
    for task in manifest['parents']:
        path=root/'evidence/jobs'/f"{task['slot']:03d}.json"
        if not path.is_file():continue
        data=path.read_bytes();receipt=json.loads(data)
        if receipt.get('status')=='passed':parents.append((task['slot'],hashlib.sha256(data).hexdigest()))
    inputs={'parents':parents,'code_commit':manifest['code_commit'],
        'motion_quartet_final_targets':manifest.get('motion_quartet_final_targets',{}),
        'scene_delivery_policy':__import__('avengine.timeline.v7_export_scene_policy',fromlist=['REVISION']).REVISION,
        'post_dedup_modality_balance':manifest.get('post_dedup_modality_balance',False)}
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()


def finalize(manifest_path):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text());root=Path(manifest['output_root'])
    if manifest.get('dataset_role')!='acceptance_only'or manifest.get('training_dataset_merge')is not False or manifest.get('CCI_transfer')is not False:
        raise ValueError('explicit acceptance-only output scope is required')
    with(root/'evidence/report_build.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);receipts,excluded=admitted_receipts(manifest)
        target=manifest.get('motion_quartet_final_targets',{});minimums={f:int(target.get('minimum_independent_questions_per_family',30))for f in target.get('qa_ids',[])}
        rows,selection=balanced_batch(receipts,root,read_only_source_roots=manifest.get('peer_output_roots',[]),family_minimums=minimums)
        for row in rows:validate_english_model_text(row)
        if len({r['question_id']for r in rows})!=len(rows):raise ValueError('duplicate final question IDs')
        index=root/'export/train.jsonl';temporary=index.with_name(index.name+f'.{os.getpid()}.tmp')
        temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows));temporary.replace(index)
        reader=open_delivered_media(root/'export')
        if reader['status']!='pass':raise ValueError('final post-dedup model reader failed')
        from avengine.timeline.v7_source_usage import for_manifest
        source_usage=for_manifest(manifest,receipts,rows)
        if source_usage is not None:save(root/'evidence/SOURCE_USAGE.json',source_usage)
        record_batch_selection(root/'inputs/demand_state.json',rows,source_usage=source_usage)
        result={'status':'pass','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'input_fingerprint':delivery_fingerprint(manifest),'dataset_role':'acceptance_only','independent_questions':len(rows),'native_admitted_parents':len(receipts),
            'houses':len({r['house']for r in rows}),'modality_counts':dict(Counter(r['modality']for r in rows)),
            'family_counts':dict(Counter(r['qa_id']for r in rows)),'selection':{k:v for k,v in selection.items()if k!='parents'},
            'excluded_historical_parents':excluded,'actual_media_reader':reader,'model_text_CJK_rows':0,'registration_name_rows':0,
            'missing_question_en_rows':0,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'model_table_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'code_commit':manifest['code_commit'],
            'new_native_renders':0,'original_native_seals_changed':False}
        save(root/'evidence/FINAL_MODEL_READER.json',result)
        print(json.dumps({k:result[k]for k in ['status','independent_questions','houses','native_admitted_parents','modality_counts','model_table_sha256']},ensure_ascii=False),flush=True)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();finalize(a.manifest)
if __name__=='__main__':main()
