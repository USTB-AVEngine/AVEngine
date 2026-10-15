"""Bounded CPU workers for independent native windows of one parent only.

The original collector, candidate budgets, wording checks and private evidence
are unchanged. Each worker writes its own window, then returns a small receipt.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing,os
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_candidate_generation import read_json


def _pin(cpu_ids):
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('native question workers require nice >=10')
    identity=multiprocessing.current_process()._identity[-1]-1
    os.sched_setaffinity(0,{cpu_ids[identity%len(cpu_ids)]})
    os.environ['CUDA_VISIBLE_DEVICES']=''


@lru_cache(maxsize=2)
def _visual(path):
    path=Path(path);visual=read_json(path)
    visual['native_appearance_review']=read_json(path.with_name('native_appearance_review.json'))['review']
    return visual


def publish_window(job):
    from avengine.timeline.v7_full_question_delivery import collect_native_pool
    visual=_visual(job['visual_path']);ev=job['evidence'];proof=job['proof'];wid=job['window_id'];dest=Path(job['directory'])
    q,g,audit=collect_native_pool(ev['scene'],ev['start_s'],wid,visual,job['media'],proof,job['quota'],job['count'])
    if job.get('current_native_environment'):
        from avengine.qa.kujiale_current_color import filter_current_scene_colors
        q,g,color_audit=filter_current_scene_colors(q,g,audit,visual,ev['start_s'])
        audit['current_scene_colour_gate']=color_audit
    if job.get('frozen_dialogue'):
        from avengine.qa.v7_dialogue_binding import audit_semantic_wording_exchange
        semantic=audit.get('semantic_manifest');facts=audit.get('family_fact_overrides',{}).get('QA-26',audit['facts'])
        wording=audit_semantic_wording_exchange(facts,semantic,seed=str(job['scene_seed']),variant_id=wid) if semantic else {'status':'blocked','reason':'no_registered_semantic_manifest'}
        if wording['status']!='pass':raise ValueError('actual dialogue standard/alternate wording exchange failed: '+str(wording))
        audit['native_semantic_wording_exchange']=wording
        ev['registry_resolution']=job['registry_resolution']
    else:ev['candidate_family_counts']=dict(Counter(x['qa_id']for x in q if x['form']=='open'))
    ev['source_qa_evidence']=proof
    for name,value in [('questions.json',q),('private_answers.json',g),('candidate_audit.json',audit),('evidence.json',ev)]:write_json(dest/name,value)
    write_json(dest/'question_worker.json',{'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'window_id':wid,'candidate_collector':'original collect_native_pool, all original gates'})
    names=['video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','candidate_audit.json','evidence.json','question_worker.json']
    return {'window_id':wid,'logical_candidates':len({x['logical_question_id']for x in q})},[f'{wid}/{name}'for name in names]


def collect_content_window(job):
    from copy import deepcopy
    from avengine.qa.v7_content_binding import collect_content_binding
    bundle=job['bundle'];wid=job['window_id'];seed=job['seed'];media=job['media']
    pub,priv,audit=collect_content_binding(bundle,wid,media,seed=seed)
    alternate=deepcopy(bundle)
    for record in alternate['queries']:record['wording_variant']='alternate'
    ap,ag,_=collect_content_binding(alternate,wid,pub[0]['media']if pub else media,seed=seed)
    signature=lambda qs,gs:([(q['question_id'],q['qa_id'],q['form'],q.get('choices'),q['answer_type'])for q in qs],[(g['question_id'],g['canonical_answer'],g['evidence']['answer_domain'])for g in gs])
    if signature(pub,priv)!=signature(ap,ag):raise ValueError('actual content wording exchange changes IDs, answers, domains or options')
    audit['wording_exchange']={'status':'pass','standard_logical':len(pub)//2,'alternate_logical':len(ap)//2,'facts_identical':True}
    return pub,priv,audit


class RegisteredQuestionWorkers:
    def __init__(self,inputs):
        requested=os.environ.get('AVENGINE_REGISTERED_QUESTION_WORKERS')
        self.workers=int(requested or inputs.get('registered_question_workers',1));self.enabled=self.workers>1
        if not 1<=self.workers<=5:raise ValueError('registered question workers must be between one and five')
        assigned=os.environ.get('AVENGINE_REGISTERED_QUESTION_CPUS')
        self.cpu_ids=tuple(int(x)for x in (assigned.split(',') if assigned else inputs.get('registered_question_cpu_ids',range(8,8+self.workers))))
        if len(self.cpu_ids)<self.workers or len(set(self.cpu_ids))!=len(self.cpu_ids):raise ValueError('independent CPU IDs required')
        self.futures=[];self.pool=None
    def __enter__(self):
        if self.enabled:self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_pin,initargs=(self.cpu_ids,))
        return self
    def submit(self,job):self.futures.append(self.pool.submit(publish_window,job))
    def submit_content(self,job):return self.pool.submit(collect_content_window,job)
    def results(self):return [f.result()for f in self.futures]
    def __exit__(self,*exc):
        if self.pool:self.pool.shutdown(wait=True)
