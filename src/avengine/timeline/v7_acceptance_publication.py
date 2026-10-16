from avengine.timeline.v7_post_dedup_selection import manifest_house_minimums
"""Publish each newly admitted parent's English rows without waiting for a batch."""
from pathlib import Path
from collections import Counter
import datetime,fcntl,hashlib,json,os
from avengine.timeline.v7_generation_scope import assert_new_generation_output
from avengine.timeline.v7_english_export import validate_english_model_text
from avengine.timeline.v7_export_scene_policy import REVISION,cached_parent_policy


def publish_current_parents(root, *, blocking=True):
    root=Path(root);assert_new_generation_output(root)
    manifest=json.loads((root/'inputs/ACCEPTANCE_PLAN.json').read_text())
    if manifest.get('publication_closed'):return json.loads((root/'PILOT_SUMMARY.json').read_text())
    with(root/'evidence/report_build.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX if blocking else fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'pending_retry','reason':'batch publication already in progress; native parent remains admitted'}
        parents=[];rows=[]
        for path in sorted((root/'evidence/jobs').glob('[0-9][0-9][0-9].json')):
            receipt=json.loads(path.read_text())
            if receipt.get('status')!='passed':continue
            if receipt.get('scene_id')in manifest.get('delivery_excluded_parent_scene_ids',{}):continue
            source_policy=cached_parent_policy(receipt['export']['source_seal'],root/'evidence/scene_delivery_policy')
            from avengine.timeline.v7_export_window_policy import scope_receipt_windows
            receipt,policy=scope_receipt_windows(receipt,source_policy,enabled=manifest.get('drop_invalid_speed_windows',True))
            prior=receipt['export'].get('scene_delivery_policy',{})
            if prior.get('revision')!=REVISION or policy['status']!='pass':
                from avengine.timeline.v7_admitted_pool_reselection import save
                history=root/'evidence/policy_history'/(path.stem+'_'+str(prior.get('revision','unknown'))+'.json')
                if not history.exists():save(history,receipt)
                receipt['export']['scene_delivery_policy']=policy
                if policy['status']!='pass':
                    receipt.update(status='failed',error_type='ValueError',reason='scene delivery policy failed: '+';'.join(policy['reasons']),latest_owner_policy_rejected=True)
                save(path,receipt)
            if policy['status']!='pass':
                from avengine.timeline.v7_construction_retry import queue_construction_retry
                queue_construction_retry(root/'inputs/ACCEPTANCE_PLAN.json',receipt)
                continue
            if receipt.get('reader',{}).get('status')!='pass':raise ValueError('cannot publish before actual parent media readback')
            assert_new_generation_output(receipt['export']['source_seal'])
            table=root/'export/parent_rows'/f"{receipt['slot']:03d}.jsonl"
            allowed_windows={w['window_id']for w in receipt['export']['windows']}
            rows.extend(r for s in table.read_text().splitlines()if s.strip()for r in [json.loads(s)]if r['window_id']in allowed_windows);parents.append(receipt)
        for row in rows:validate_english_model_text(row)
        ids=[r['question_id']for r in rows]
        if len(set(ids))!=len(ids):raise ValueError('duplicate physical parent publication rows')
        if manifest.get('supplement_answer_branch_candidates')and parents:
            from avengine.timeline.v7_admitted_pool_reselection import balanced_batch
            try:
                rows,_=balanced_batch(parents,root,read_only_source_roots=manifest.get('peer_output_roots',[]),
                    minimum_houses_per_family=manifest_house_minimums(manifest),
                    family_protocol_panels=manifest.get('family_protocol_panels',{}),
                    family_minimums=__import__('avengine.timeline.v7_post_dedup_selection',fromlist=['manifest_family_minimums']).manifest_family_minimums(manifest))
            except ValueError as error:
                if not any(text in str(error)for text in ('30/25/45 batch','modality budget cannot preserve')):raise
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows);content_sha=hashlib.sha256(content.encode()).hexdigest()
        publication=root/'evidence/PUBLICATION_CURRENT.json'
        table=root/'export/train.jsonl'
        if publication.exists()and table.exists():
            cache=json.loads(publication.read_text())
            if cache.get('revision')==REVISION and cache.get('model_table_sha256')==content_sha and hashlib.sha256(table.read_bytes()).hexdigest()==content_sha:
                return json.loads((root/'PILOT_SUMMARY.json').read_text())
        temporary=table.with_name(table.name+f'.{os.getpid()}.tmp')
        temporary.write_text(content);temporary.replace(table)
        from avengine.timeline.v7_source_usage import for_manifest
        from avengine.timeline.v7_admitted_pool_reselection import record_batch_selection,save
        manifest=json.loads((root/'inputs/ACCEPTANCE_PLAN.json').read_text())
        usage=for_manifest(manifest,parents,rows)
        if usage is not None:
            usage['model_table_sha256']=content_sha
            save(root/'evidence/SOURCE_USAGE.json',usage)
        record_batch_selection(root/'inputs/demand_state.json',rows,source_usage=usage)
        family=Counter(r['qa_id']for r in rows);modality=Counter(r['modality']for r in rows)
        result={'status':'partial_parents_individually_passed','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'dataset_role':'acceptance_only','independent_questions':len(rows),'passed_parents':len(parents),'houses':len({r['house']for r in rows}),
            'question_family_counts':dict(family),'modality_counts':dict(modality),'families_at_least_three':sum(n>=3 for n in family.values()),
            'model_visible_CJK_rows':0,'registration_name_rows':0,'internal_terminology_rows':0,'scene_policy_revision':REVISION,
            'batch_modality_selection':'pending_final_30_25_45_selection','aggregate_reader_status':'pending_final_aggregate; each native parent reader passed',
            'old_media_or_seals_reused':False,'training_dataset_merge':False,'CCI_transfer':False}
        target=root/'PILOT_SUMMARY.json';temporary=target.with_name(target.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(target)
        publication.write_text(json.dumps({'revision':REVISION,'model_table_sha256':content_sha})+'\n')
        with(root/'PROGRESS.md').open('a')as stream:stream.write(result['generated_utc']+' PARTIAL_EXPORT_READY 独立题='+str(len(rows))+' 房子='+str(result['houses'])+' 达到3题族='+str(result['families_at_least_three'])+'/33；'+str(table)+'；各父四项门禁/reader通过，批次模态补选待最终汇总。\n')
        return result


def publish_admitted_parent(root,receipt_path):
    """Keep a native/reader admission when an unrelated batch update needs retry.

    The batch table still requires every original gate. A newer scene-policy
    rejection already written by publication is never restored to passed.
    """
    import traceback
    from avengine.timeline.v7_admitted_pool_reselection import save
    root=Path(root);receipt_path=Path(receipt_path)
    assert_new_generation_output(root)
    if not receipt_path.resolve().is_relative_to(root.resolve()):raise ValueError('parent receipt escaped the owned output')
    try:return publish_current_parents(root,blocking=False)
    except Exception as error:
        failure={'status':'pending_retry','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc()}
        current=json.loads(receipt_path.read_text())
        if current.get('status')=='passed':
            current['batch_publication']=failure;save(receipt_path,current)
        with(root/'evidence/batch_publication_errors.jsonl').open('a')as stream:
            stream.write(json.dumps({'parent_receipt':str(receipt_path),**failure},ensure_ascii=False)+'\n')
        with(root/'PROGRESS.md').open('a')as stream:
            stream.write(failure['time_utc']+' BATCH_PUBLICATION_PENDING parent='+str(current.get('scene_id'))+' error='+type(error).__name__+':'+str(error)+'; original parent admission retained; no unvalidated batch rows added\n')
        return failure
