"""Publish each newly admitted parent's English rows without waiting for a batch."""
from pathlib import Path
from collections import Counter
import datetime,fcntl,json,os
from avengine.timeline.v7_generation_scope import assert_new_generation_output
from avengine.timeline.v7_english_export import validate_english_model_text
from avengine.timeline.v7_export_scene_policy import REVISION,cached_parent_policy


def publish_current_parents(root):
    root=Path(root);assert_new_generation_output(root)
    with(root/'evidence/report_build.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);parents=[];rows=[]
        for path in sorted((root/'evidence/jobs').glob('[0-9][0-9][0-9].json')):
            receipt=json.loads(path.read_text())
            if receipt.get('status')!='passed':continue
            policy=cached_parent_policy(receipt['export']['source_seal'],root/'evidence/scene_delivery_policy')
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
            rows.extend(json.loads(s)for s in table.read_text().splitlines()if s.strip());parents.append(receipt)
        for row in rows:validate_english_model_text(row)
        ids=[r['question_id']for r in rows]
        if len(set(ids))!=len(ids):raise ValueError('duplicate physical parent publication rows')
        table=root/'export/train.jsonl';temporary=table.with_name(table.name+f'.{os.getpid()}.tmp')
        temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows));temporary.replace(table)
        from avengine.timeline.v7_source_usage import for_manifest
        from avengine.timeline.v7_admitted_pool_reselection import record_batch_selection,save
        manifest=json.loads((root/'inputs/ACCEPTANCE_PLAN.json').read_text())
        usage=for_manifest(manifest,parents,rows)
        if usage is not None:save(root/'evidence/SOURCE_USAGE.json',usage)
        record_batch_selection(root/'inputs/demand_state.json',rows,source_usage=usage)
        family=Counter(r['qa_id']for r in rows);modality=Counter(r['modality']for r in rows)
        result={'status':'partial_parents_individually_passed','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'dataset_role':'acceptance_only','independent_questions':len(rows),'passed_parents':len(parents),'houses':len({r['house']for r in rows}),
            'question_family_counts':dict(family),'modality_counts':dict(modality),'families_at_least_three':sum(n>=3 for n in family.values()),
            'model_visible_CJK_rows':0,'registration_name_rows':0,'internal_terminology_rows':0,'scene_policy_revision':REVISION,
            'batch_modality_selection':'pending_final_30_25_45_selection','aggregate_reader_status':'pending_final_aggregate; each native parent reader passed',
            'old_media_or_seals_reused':False,'training_dataset_merge':False,'CCI_transfer':False}
        target=root/'PILOT_SUMMARY.json';temporary=target.with_name(target.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(target)
        with(root/'PROGRESS.md').open('a')as stream:stream.write(result['generated_utc']+' PARTIAL_EXPORT_READY 独立题='+str(len(rows))+' 房子='+str(result['houses'])+' 达到3题族='+str(result['families_at_least_three'])+'/33；'+str(table)+'；各父四项门禁/reader通过，批次模态补选待最终汇总。\n')
        return result
