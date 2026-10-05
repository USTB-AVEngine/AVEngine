"""AFS personal allocation inventory with restartable, bounded background reads.

Shared-mount df is intentionally never interpreted as personal quota.
Unknown inventory pauses new writes; transient errors do not kill producers.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,time,uuid


def allocated_inventory(root):
    start=time.perf_counter()
    p=subprocess.run(['du','-s','-B1',str(root)],capture_output=True,text=True,timeout=1200)
    if p.returncode:raise RuntimeError('du inventory failed: '+p.stderr[:300])
    value=int(p.stdout.split()[0])
    if value<0:raise ValueError('negative allocation inventory')
    return {'bytes':value,'wall_seconds':time.perf_counter()-start}


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('x')as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


class CachedOwnerQuota:
    """Global snapshot + conservative own growth since before that snapshot.

    The own baseline is measured BEFORE the slow owner scan; growth during
    the scan can be double-counted but cannot be subtracted from the owner.
    A failed/expired snapshot never grants fresh dispatch. Only a newly
    measured owner allocation can confirm exhausted quota; an upper-bound
    estimate instead requests refresh while in-flight work keeps receipts.
    """
    def __init__(self,owner_root,task_root,cache,quota,reserve,*,reader=allocated_inventory,clock=time.time):
        self.owner=Path(owner_root).resolve();self.task=Path(task_root).resolve();self.cache=Path(cache)
        if self.owner==self.task or not self.task.is_relative_to(self.owner):raise ValueError('quota task must stay in the personal owner root')
        if not self.cache.resolve().is_relative_to(self.task):raise ValueError('quota cache escaped its own task')
        self.quota=int(quota);self.reserve=int(reserve);self.reader=reader;self.clock=clock
        if not 0<self.reserve<self.quota:raise ValueError('invalid owner quota/reserve')
        self.closed=False;self.pool=ThreadPoolExecutor(max_workers=2);self.global_future=None;self.own_future=None
        self.global_checked_this_run=False;self.global_value=None;self.own_value=None;self.errors={};self.attempts={'global':0,'own':0};self.next_try={'global':0.,'own':0.}
        if self.cache.is_file():
            d=json.loads(self.cache.read_text())
            if d.get('schema')!='v7-owner-quota-cache-r18-v1' or d['owner_root']!=str(self.owner) or d['task_root']!=str(self.task):raise ValueError('quota cache identity mismatch')
            self.global_value=d.get('global');self.own_value=d.get('own')
        self._schedule()

    def _global_read(self):
        before=self.reader(self.task)
        if type(before['bytes'])is not int or before['bytes']<0:raise ValueError('invalid task baseline')
        result=self.reader(self.owner)
        if type(result['bytes'])is not int or result['bytes']<0:raise ValueError('invalid owner allocation')
        return {**result,'task_baseline_bytes':before['bytes'],'measured_unix':self.clock(),'measurement':'du -s -B1 owner; task baseline precedes scan'}

    def _own_read(self):
        result=self.reader(self.task)
        if type(result['bytes'])is not int or result['bytes']<0:raise ValueError('invalid own allocation')
        return {**result,'measured_unix':self.clock()}

    def _save(self):
        atomic(self.cache,{'schema':'v7-owner-quota-cache-r18-v1','owner_root':str(self.owner),'task_root':str(self.task),
            'global':self.global_value,'own':self.own_value,'errors':self.errors,'saved_unix':self.clock()})

    def _schedule(self,force_global=False):
        if self.closed:return
        now=self.clock()
        if self.global_future is None and now>=self.next_try['global'] and (force_global or self.global_value is None or now-self.global_value['measured_unix']>=900):self.global_future=self.pool.submit(self._global_read)
        if self.own_future is None and now>=self.next_try['own'] and (self.own_value is None or now-self.own_value['measured_unix']>=30):self.own_future=self.pool.submit(self._own_read)

    @property
    def check_in_progress(self):return self.global_future is not None

    def poll(self,expected_bytes=0):
        now=self.clock();changed=False
        for key,attr in [('global','global_future'),('own','own_future')]:
            future=getattr(self,attr)
            if future is None or not future.done():continue
            try:
                result=future.result();setattr(self,key+'_value',result);self.errors.pop(key,None);self.attempts[key]=0
                if key=='global':self.global_checked_this_run=True
                self.next_try[key]=now+(900 if key=='global'else 30)
            except Exception as exc:
                self.attempts[key]+=1;self.next_try[key]=now+min(600,30*2**min(self.attempts[key]-1,5));self.errors[key]=type(exc).__name__+': '+str(exc)
            setattr(self,attr,None);changed=True
        if changed:self._save()
        global_age=None if self.global_value is None else max(0,now-self.global_value['measured_unix'])
        own_age=None if self.own_value is None else max(0,now-self.own_value['measured_unix'])
        growth=None;estimate=None;expected=max(0,int(expected_bytes))
        if self.global_value is not None and self.own_value is not None:
            growth=max(0,self.own_value['bytes']-self.global_value['task_baseline_bytes']);estimate=self.global_value['bytes']+growth
        if self.global_value is None or self.own_value is None:status='awaiting_inventory'
        elif global_age>900 or own_age>60:status='awaiting_fresh_inventory'
        elif self.global_value['bytes']+self.reserve>=self.quota:status='confirmed_full' if self.global_checked_this_run else 'awaiting_headroom_refresh'
        elif estimate+expected+self.reserve>=self.quota:status='awaiting_headroom_refresh'
        else:status='pass'
        self._schedule(force_global=status=='awaiting_headroom_refresh')
        return {'status':status,'new_dispatch_allowed':status=='pass','confirmed_full':status=='confirmed_full',
            'owner_cached_allocated_bytes':None if self.global_value is None else self.global_value['bytes'],
            'own_growth_bytes':growth,'estimated_owner_allocated_bytes':estimate,'expected_new_bytes':expected,
            'quota_bytes':self.quota,'reserve_bytes':self.reserve,'global_age_seconds':global_age,'own_age_seconds':own_age,
            'global_scan_running':self.global_future is not None,'own_scan_running':self.own_future is not None,
            'global_inventory_wall_seconds':None if self.global_value is None else self.global_value['wall_seconds'],
            'own_inventory_wall_seconds':None if self.own_value is None else self.own_value['wall_seconds'],
            'retry_not_before_unix':dict(self.next_try),'transient_errors':dict(self.errors),
            'cache':str(self.cache),'slow_query_policy':'pause new dispatch; retain active parents; bounded retry 30..600s; no producer failure',
            'scope':'personal owner allocation; shared-mount df never substitutes'}

    def close(self):
        self.closed=True
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.poll()
