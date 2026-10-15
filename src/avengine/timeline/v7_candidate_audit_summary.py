"""Small diagnostic summaries keyed by actual immutable native audit bytes."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,os


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def candidate_window_summary(folder,cache_root):
    folder=Path(folder);audit_path=folder/'candidate_audit.json';questions=folder/'questions.json'
    ah=digest(audit_path);qh=digest(questions);key=hashlib.sha256((ah+qh).encode()).hexdigest()
    cached=Path(cache_root)/(key+'.json')
    if cached.exists():return json.loads(cached.read_text())
    public=json.loads(questions.read_text());logical={q['logical_question_id']:q for q in public};audit=json.loads(audit_path.read_text());rejected=defaultdict(set)
    for d in audit.get('deferred',[]):
        if d.get('qa_id'):rejected[d['qa_id']].add(d.get('code','unknown'))
    realization=audit.get('demand_realization',{})
    result={'candidate_audit_sha256':ah,'questions_sha256':qh,
        'legal_family_counts':dict(Counter(q['qa_id']for q in logical.values())),
        'deferred_codes':{f:sorted(codes)for f,codes in rejected.items()},
        'outside_class13_excluded':bool(audit.get('out_of_domain_activity_count_exclusions')),
        'measured_moving_actor_ids':realization.get('native_moving_actor_ids',realization.get('moving_actor_ids',[])),
        'measured_wet_support_event_ids':realization.get('wet_support_event_ids',[]),
        'native_evidence_and_QA_predicates_changed':False}
    cached.parent.mkdir(parents=True,exist_ok=True);tmp=cached.with_name(cached.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(cached)
    return result
