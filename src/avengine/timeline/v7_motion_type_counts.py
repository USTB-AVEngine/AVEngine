"""Count exported logical questions by their actual bound physical target."""
import json,re
from collections import Counter,defaultdict
from pathlib import Path
from avengine.timeline.v7_motion_source_roles import classify_source_type
FAMILIES={'QA-06','QA-15','QA-16','QA-17'}

def question_type_counts(export_root,row_file):
    counts=defaultdict(Counter);unknown=[];seen=set();cache={}
    for line in Path(row_file).read_text().splitlines():
        row=json.loads(line);logical=row['logical_question_id']
        if logical in seen or row['qa_id']not in FAMILIES:continue
        seen.add(logical)
        witness=row.get('target_verification',{}).get('result',{}).get('question_id','')
        match=re.search(r'__target_(\d+)__',witness)
        if not match:unknown.append({'logical_question_id':logical,'witness':witness});continue
        path=Path(export_root)/row['private_evidence']
        if path not in cache:cache[path]=json.loads(path.read_text())
        evidence=cache[path];specs=[s for s in evidence['scene']['sources'].values()if str(s['actor_id'])==match.group(1)]
        active=[s for s in specs if s.get('sound_class')or s.get('content_kind')not in {None,'silent'}]
        if not active:unknown.append({'logical_question_id':logical,'reason':'bound target has no registered physical sound spec'});continue
        counts[classify_source_type(active[0])][row['qa_id']]+=1
    return {'counts_by_actual_question_target_source_type':{kind:dict(c)for kind,c in counts.items()},
            'unattributed_motion_logical_questions':unknown,'unique_motion_logical_question_count':len(seen),
            'method':'original independent v6 scorer target ID -> retained physical source declaration; logical question IDs counted once'}
