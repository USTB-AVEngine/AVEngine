from pathlib import Path
import importlib.util,json,hashlib
import pytest
from avengine.timeline.v7_storage_resume import initialize_resume

def tool(name):
    p=Path(__file__).resolve().parents[2]/'tools/timeline'/name
    spec=importlib.util.spec_from_file_location(name.replace('.','_'),p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

gate=tool('wait_v7_space_resume.py');account=tool('v7_storage_accounting.py')

def state(headroom=120_000_000_000):return {'status':'pass','global_age_seconds':0,'own_age_seconds':0,'estimated_owner_allocated_bytes':1_920_000_000_000-headroom,'quota_bytes':2_000_000_000_000,'reserve_bytes':80_000_000_000}

def test_120GB_inclusive_boundary():
    assert gate.enough_space(state());assert not gate.enough_space(state(119_999_999_999));assert gate.enough_space(state(120_000_000_001))

@pytest.mark.parametrize('key,value',[('status','confirmed_full'),('status','awaiting_inventory'),('global_age_seconds',901),('own_age_seconds',61),('own_age_seconds',None),('estimated_owner_allocated_bytes',True),('estimated_owner_allocated_bytes',-1),('reserve_bytes',79_000_000_000)])
def test_unknown_stale_or_changed_quota_never_enqueues(key,value):
    s=state();s[key]=value;assert not gate.enough_space(s)

def test_cannot_lower_owner_gate():
    with pytest.raises(ValueError):gate.enough_space(state(),minimum=1)

def test_queue_transform_preserves_other_bytes():
    before=b'/other/p.json 2 # untouched\n/old/p.json 2 # ours\n# leave this\n'
    after,count=gate.transform_queue(before,'/old/p.json','/new/p.json');assert count==1
    assert after.splitlines()[0].startswith(b'/new/p.json 2')
    assert b'/other/p.json 2 # untouched\n'in after and after.endswith(b'# leave this\n')
    assert b'# R19 hold own space-stopped'in after

def test_duplicate_queue_or_repeat_new_is_rejected():
    with pytest.raises(ValueError):gate.transform_queue(b'/old/p 2\n/old/p 2\n','/old/p')
    with pytest.raises(ValueError):gate.transform_queue(b'/new/p 2\n','/old/p','/new/p')

def test_concurrent_queue_writer_is_preserved(tmp_path):
    q=tmp_path/'q';q.write_bytes(b'external latest\n')
    with pytest.raises(ValueError,match='concurrently'):gate.compare_and_replace(q,b'old\n',b'our update\n')
    assert q.read_bytes()==b'external latest\n'

def fixture(tmp_path):
    root=tmp_path/'task';work=root/'production_r12';work.mkdir(parents=True)
    rows=[{'scene_id':x,'members':['v0_a0'],'window_count':1,'scene_seed':i}for i,x in enumerate('abcd')]
    seeds=root/'seeds.jsonl';seeds.write_text(''.join(json.dumps(r)+'\n'for r in rows))
    for row in rows:
        p=work/'parents'/row['scene_id'];p.mkdir(parents=True);(p/'input_recipe.json').write_text(json.dumps({'seed_row':row}))
    p=work/'parents/a';attempt=p/'alpha_seal_attempt_test';attempt.mkdir();(attempt/'media').write_bytes(b'hermetic file presence fixture; not native evidence')
    marker={'attempt':attempt.name,'recipe':{'input_recipe':{'seed_row':rows[0]}},'result':{'status':'formal_alpha_admitted','alpha_formal_windows':1,'alpha_formal_logical_questions':14},'required_files':['media']};(p/'alpha_seal_completed.json').write_text(json.dumps(marker))
    (work/'parents/b/failed_test.json').write_text(json.dumps({'reason':'actual fixture rejection'}))
    checkpoint={'schema':'v7-space-resume-r19-v1','seed_SHA256':hashlib.sha256(seeds.read_bytes()).hexdigest(),'sealed_ids':['a'],'rejected_ids':['b'],'interrupted_ids':['c'],'unstarted_ids':['d']};cp=root/'checkpoint.json';cp.write_text(json.dumps(checkpoint))
    return {'root':str(root),'seeds':str(seeds),'resume_checkpoint':str(cp)},rows,work,marker,checkpoint

def test_resume_only_unstarted_and_interrupted_with_original_seal(tmp_path):
    p,rows,w,_m,_c=fixture(tmp_path);before=set(w.rglob('*'));done,rejected,pending=initialize_resume(p,rows,w)
    assert len(done)==len(rejected)==1 and [r['scene_id']for _,r in pending]==['c','d'];assert rejected[0]['redispatched']is False
    assert set(w.rglob('*'))==before

@pytest.mark.parametrize('failure',['partition_duplicate','wrong_seed','missing_media','escape','wrong_count','wrong_recipe','rejection_without_receipt'])
def test_resume_bad_seals_or_partition_are_rejected(tmp_path,failure):
    p,rows,w,m,c=fixture(tmp_path)
    if failure=='partition_duplicate':c['unstarted_ids'].append('a')
    if failure=='wrong_seed':c['seed_SHA256']='0'*64
    if failure=='missing_media':m['required_files']=['absent']
    if failure=='escape':m['required_files']=['../input_recipe.json']
    if failure=='wrong_count':m['result']['alpha_formal_logical_questions']=13
    if failure=='wrong_recipe':m['recipe']['input_recipe']['seed_row']={'scene_id':'other'}
    if failure=='rejection_without_receipt':(w/'parents/b/failed_test.json').unlink()
    Path(p['resume_checkpoint']).write_text(json.dumps(c));(w/'parents/a/alpha_seal_completed.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):initialize_resume(p,rows,w)

def test_accounting_deduplicates_formal_hardlinks_and_never_deletes(tmp_path):
    import os
    root=tmp_path/'source';seal=root/'parents/a/alpha_seal_attempt_test/windows/w';seal.mkdir(parents=True)
    original=seal/'audio_foa.wav';original.write_bytes(b'x'*8192)
    work=root/'parents/a/windows_v0_a0_attempt_test/w';work.mkdir(parents=True);os.link(original,work/'audio_foa.wav')
    cache=root/'parents/a/rir_cache';cache.mkdir();(cache/'rir.npz').write_bytes(b'z'*8192)
    # Parent is not a production fixture; no admission marker is fabricated.
    summary=account.inspect({'task':str(root)},tmp_path/'out')
    assert summary['unique_file_inodes']==2 and summary['file_paths']==3 and summary['files_deleted']==0
    assert summary['proposal_only_release_upper_bound_bytes']==0;assert original.is_file()and(cache/'rir.npz').is_file()

def test_accounting_external_link_has_zero_reclamation_credit(tmp_path):
    import os
    root=tmp_path/'source';folder=root/'parents/a/dry_attempt_test';folder.mkdir(parents=True);f=folder/'voice.npy';f.write_bytes(b'x'*8192);os.link(f,tmp_path/'outside_retained.npy')
    summary=account.inspect({'task':str(root)},tmp_path/'out');proposal=json.loads((tmp_path/'out/DELETION_PROPOSAL_ONLY.json').read_text())
    assert summary['proposal_only_release_upper_bound_bytes']==0 and proposal[0]['possible_external_links']


def test_owner_pause_does_not_mutate_queue(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/"tools/timeline/wait_v7_space_resume.py"
    spec=importlib.util.spec_from_file_location("paused_space_wait",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    watch=tmp_path/"watch";watch.mkdir();(watch/"PAUSE_SPACE_RESUME_R19").write_text("owner pause")
    q=tmp_path/"queue";q.write_bytes(b"other_plan 2\n")
    monkeypatch.setattr(m,"validate_scope",lambda c:(tmp_path,watch))
    result=m.enqueue_ready({"queue":str(q),"new_plan":str(tmp_path/"new"),"profile":str(tmp_path/"profile")},{"status":"space_ready"})
    assert result=={"status":"paused_owner_marker","queue_mutated":False}
    assert q.read_bytes()==b"other_plan 2\n"
