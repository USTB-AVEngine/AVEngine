"""Actual passed exports supply demand, class and voice rotation feedback."""
from collections import Counter,defaultdict
from pathlib import Path
import json


def from_passed_exports(roots):
    seen=set();families=Counter();parents=defaultdict(Counter);classes=Counter();sources=[];voices=Counter();voice_parents=set();observed_houses=set()
    pending=list(map(Path,roots));visited=set();answers=defaultdict(Counter)
    while pending:
        root=pending.pop(0).resolve()
        from avengine.timeline.v7_generation_scope import invalidation_marker
        if invalidation_marker(root):continue
        if root in visited:continue
        visited.add(root)
        for marker in root.glob('*/AUTOMATION_RUN.json'):
            if json.loads(marker.read_text()).get('schema')=='v7-automatic-run-v1':pending.append(marker.parent)
        for path in sorted((root/'evidence/jobs').glob('*.json')):
            job=json.loads(path.read_text())
            if job.get('house'):observed_houses.add(job['house'])
            if job.get('status')!='passed':continue
            table=Path(job['export_rows_path'])if job.get('export_rows_path')else root/'export/parent_rows'/(path.stem+'.jsonl')
            if not table.exists():continue  # Imported receipts need their actual exported table.
            sources.append(str(table))
            for line in table.read_text().splitlines():
                row=json.loads(line);parent=row.get('independent_parent_scene_id');logical=row.get('logical_question_id')
                if not parent or not logical:raise ValueError('actual feedback requires independent parent and logical question IDs')
                if (parent,logical)in seen:continue
                seen.add((parent,logical));families[row['qa_id']]+=1;parents[parent][row['qa_id']]+=1
                if row['qa_id']=='V7-CLASS34':classes[str(row['canonical_answer'])]+=1
                if row['qa_id']in {'QA-08','QA-20','QA-24','STARSS-FRUSTUM'}:
                    value=row['canonical_answer']
                    if isinstance(value,dict):value=value.get('visibility_state',value.get('final_visibility_state',value.get('answer',value.get('value',value))))
                    value=str(value).strip().lower().replace(' ','_').replace('-','_')
                    if row['qa_id']=='QA-20':value='none_of_visible'if value in {'none','none_of_visible','none_of_them'}else'visible_person'
                    answers[row['qa_id']][value]+=1
            pid=job.get('requested_scene_id',job.get('scene_id'))
            if pid and pid not in voice_parents and job.get('actual_voice_ids'):
                voices.update(set(job['actual_voice_ids']));voice_parents.add(pid)
    state={'revision':'v7-native-export-feedback-v1','logical_questions':sum(families.values()),
        'family_counts':dict(families),'parents':{k:{'scene_id':k,'logical_questions':sum(v.values()),'family_counts':dict(v)}for k,v in parents.items()},
        'class34_positive_counts':dict(classes),'authored_voice_counts':dict(voices),
        'answer_counts':{key:dict(value)for key,value in answers.items()},
        'source_export_tables':sources,'observed_houses':sorted(observed_houses),'count_authority':'unique passed exported parent+logical_question_id; no requested or candidate quota counted'}
    from avengine.timeline.v7_demand import demand_snapshot
    demand_snapshot(state)
    return state
