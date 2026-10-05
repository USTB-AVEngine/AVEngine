"""Publish actual sealed v7-S parents after the original attempt naturally ends.

This P19 CPU entry never stops a previous coordinator, retries its failed seeds,
edits answers, or calls a native renderer. Failed parents get fresh deterministic
supplement seeds in the same house and quota stratum.
"""
from __future__ import annotations
import argparse,collections,copy,hashlib,json,os,re,socket,sys,time,uuid
from pathlib import Path

def load(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def qualified_rows(rows,work):
    good=[];failed=[]
    for row in rows:
        parent=Path(work)/'parents'/row['scene_id'];mark=parent/'alpha_seal_completed.json'
        if mark.is_file():
            d=load(mark)
            if d['recipe']['input_recipe']['seed_row']!=row:raise ValueError('sealed original seed changed')
            if d['result']['status']!='formal_alpha_admitted':raise ValueError('unqualified seal pointer')
            if d['result']['alpha_formal_logical_questions']!=14*d['result']['alpha_formal_windows']:raise ValueError('sealed 14-question count differs')
            good.append(row)
        else:
            receipts=sorted(parent.glob('failed_*.json'))
            if not receipts:raise ValueError('parent has neither a qualified seal nor a failure receipt: '+row['scene_id'])
            d=load(receipts[-1]);failed.append({'scene_id':row['scene_id'],'reason':str(d.get('reason',d.get('error','unknown'))),'failure_receipt':str(receipts[-1]),'split':row['split'],'house':row['house'],'speaker_count':row['production_scenario']['speaker_count'],'category_focus_class':row['production_scenario'].get('category_focus_class')})
    return good,failed
def supplement_rows(rows,failed):
    byid={r['scene_id']:r for r in rows};used={r['scene_seed']for r in rows};result=[]
    for failure in failed:
        original=byid[failure['scene_id']];row=copy.deepcopy(original)
        row['scene_id']='v7_s_supp1_'+original['scene_id'];row['replacement_for_parent_scene_id']=original['scene_id']
        row['replacement_policy']='same physical house, room, speaker/count/class/family quota; fresh independent scene and component seeds'
        seed=int.from_bytes(hashlib.sha256(('v7S-r16-supp1:'+original['scene_id']+':'+str(original['scene_seed'])).encode()).digest()[:4],'big')
        while seed in used:seed=(seed+1)%(1<<32)
        used.add(seed);row['scene_seed']=seed
        row['seeds']={k:int.from_bytes(hashlib.sha256((str(seed)+':'+k).encode()).digest()[:4],'big')for k in original['seeds']}
        result.append(row)
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generator-profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['production','validate'],default='production');a=p.parse_args()
    profile=load(a.generator_profile);root=Path(profile['root']).resolve();a.output.mkdir(parents=True,exist_ok=True)
    os.environ.update(profile['cpu_env']);os.environ['CUDA_VISIBLE_DEVICES']=''
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    code=Path(profile['parent_generator_code']).resolve()
    if not code.is_relative_to(root/'code'):raise ValueError('native generator code escapes task frozen code')
    sys.path.insert(0,str(code/'src'));sys.path.insert(1,str(code))
    rows=[json.loads(l)for l in Path(profile['seeds']).read_text().splitlines()if l.strip()]
    if len(rows)!=500 or len({r['scene_id']for r in rows})!=500:raise ValueError('exact original 500-parent manifest required')
    houses={h['house']:h for h in load(Path(profile['inputs'])/'houses.json')['houses']}
    for r in rows:
        h=houses[r['house']]
        if r['split']not in {'train','dev'} or {'development':'dev'}.get(h['split'],h['split'])!=r['split']:raise ValueError('physical split crossed')
    if a.mode=='validate':atomic(a.output/'metrics.json',{'status':'validated','source_parents':len(rows),'test_house_access':0});return
    match=re.fullmatch(r'eval_job_a[0-9]{2}_(job_[A-Za-z0-9_.-]+)',a.output.name)
    active=Path(profile['owner_root'])/'hardbank/tmp/p19_active'/profile['plan_name']
    if not match or not a.output.resolve().is_relative_to(root/profile['p19_run_directory'])or not active.is_file()or active.read_text().split()[:2]!=[match.group(1),socket.gethostname()]:raise ValueError('publisher lacks exact P19 job/host authority')
    # Wait for natural completion, even if this publisher leases another rank.
    # A stale cross-host lock must be handled by the renderer's P19 restart ABI,
    # never stolen by the publisher. No STOP handoff is used.
    started=time.monotonic()
    while (root/'coordinator.lock').exists():
        atomic(root/'qualified_publication_status.json',{'status':'waiting_previous_natural_exit','previous_owner':load(root/'coordinator.lock/owner.json'),'pid':os.getpid(),'hostname':socket.gethostname(),'wait_seconds':time.monotonic()-started,'STOP_handoff':False})
        time.sleep(30)
    progress=load(root/'PROGRESS.json')
    if progress.get('active_parents')or progress.get('remaining_parents')!=0:raise ValueError('original production did not attempt all parents; resume original production first')
    work=root/profile['production_work_directory'];good,failed=qualified_rows(rows,work)
    if not good or not all(any(r['split']==s for r in good)for s in ('train','dev')):raise ValueError('qualified train and development parents required')
    report=root/'qualified_r16_reports';report.mkdir(exist_ok=True)
    seedfile=report/'qualified_seeds.jsonl';supp=report/'supplement_1_seeds.jsonl'
    for dest,data in [(seedfile,good),(supp,supplement_rows(rows,failed))]:
        text=''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in data)
        if dest.exists()and dest.read_text()!=text:raise ValueError('immutable qualified/replacement seed manifest changed')
        if not dest.exists():dest.write_text(text)
    reasons=collections.Counter(f['reason']for f in failed)
    failure_summary={'original_attempts':len(rows),'qualified_parents':len(good),'failed_parents':len(failed),'reason_counts':[{'reason':k,'parents':v,'percent_of_failed':100*v/len(failed)}for k,v in reasons.most_common()],'failures':failed,'replacement_seed_manifest':str(supp),'replacement_seed_sha256':sha(supp),'no_gate_changed':True}
    atomic(report/'failure_summary.json',failure_summary)
    final=root/'release_r16_qualified_v1'
    if not (final/'DELIVERY_MANIFEST.json').exists():
        from avengine.timeline.v7_alpha_delivery import export_alpha
        from avengine.timeline.v7_s_batch_checks import check_critical_delivery,open_delivered_media
        stage=root/'tmp'/('qualified_r16_export_'+uuid.uuid4().hex)
        atomic(root/'qualified_publication_status.json',{'status':'exporting_qualified_seals','qualified_parents':len(good),'failed_parents':len(failed),'pid':os.getpid(),'hostname':socket.gethostname()})
        summary=export_alpha(seedfile,[work],stage,expected_parents=len(good))
        # The parent gates and original 14-question allocation all remain in
        # force. Full 500-parent coverage is reported separately, not falsified.
        coverage=check_critical_delivery(stage,require_full=False)
        full_coverage=check_critical_delivery(stage,require_full=True)
        if coverage['status']!='pass':raise ValueError('qualified delivery fails unchanged per-window checks')
        media=open_delivered_media(stage)
        if media['status']!='pass':raise ValueError('actual CCI media readback failed')
        splits=collections.Counter(r['split']for r in good)
        manifest={'status':'sealed','dataset':'v7-S','release_policy':'owner round16: publish all qualified parents from the original attempt; failed seeds are a separate supplement','source_requested_parents':500,'independent_parent_scenes':len(good),'parent_split_counts':dict(splits),'formal_windows':summary['formal_windows'],'formal_logical_questions':summary['formal_logical_questions'],'failed_parents_excluded':len(failed),'train_jsonl':str(final/'train.jsonl'),'dev_jsonl':str(final/'dev.jsonl'),'coverage':coverage,'original_full_batch_coverage':full_coverage,'media_readback':media,'generator_commit':profile['native_generator_commit'],'publisher_commit':profile['publisher_commit'],'test_house_access':0,'GPU_started':False,'supplement_seed_manifest':str(supp),'failure_summary':str(report/'failure_summary.json'),'no_gate_changed':True,'consumer_contract':{'same_alpha_train_jsonl_schema':True,'parent_count_is_actual_qualified_subset':True,'old_consumer_500_parent_guard_requires_owner_round16_adapter':True}}
        manifest['file_sha256']={n:sha(stage/n)for n in ['train.jsonl','dev.jsonl','summary.json','parents.csv','windows.csv','rejections.json']}
        atomic(stage/'DELIVERY_MANIFEST.json',manifest)
        os.rename(stage,final)
    manifest=load(final/'DELIVERY_MANIFEST.json')
    atomic(root/'HANDOFF.json',{'status':'sealed','release_directory':str(final),'independent_parent_scenes':manifest['independent_parent_scenes'],'formal_windows':manifest['formal_windows'],'formal_logical_questions':manifest['formal_logical_questions'],'owner_round16_partial_delivery':True,'manifest_sha256':sha(final/'DELIVERY_MANIFEST.json'),'consumer_old_500_parent_contract_detected':True})
    atomic(a.output/'metrics.json',{**manifest,'status':'complete'})
    atomic(a.output/'run.json',{'status':'complete','kind':'qualified simulated data publication, no model evaluation','count':len(good)})
    (a.output/'input_manifest.jsonl').write_text(''.join(json.dumps({'scene_id':r['scene_id']})+'\n'for r in good))
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps({'scene_id':r['scene_id'],'status':'formal_alpha_admitted'})+'\n'for r in good))
    atomic(root/'qualified_publication_status.json',{'status':'complete','release_directory':str(final),'pid':os.getpid(),'hostname':socket.gethostname(),'manifest':manifest})
    print(json.dumps({'status':'sealed','release_directory':str(final),'parents':len(good),'windows':manifest['formal_windows'],'questions':manifest['formal_logical_questions']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
