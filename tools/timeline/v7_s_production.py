"""CPU-only P19 coordinator for the frozen recorded v7-S critical batch.

One coordinator owns all 500 parents. Parent subprocesses persist immutable
stages; a host restart can rerun this entry with the same profile. Formal
delivery is published only after measured count/class and original gates pass.
"""
from __future__ import annotations
import argparse,json,os,re,signal,socket,statistics,subprocess,sys,time,uuid
from pathlib import Path


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)



def native_process_credentials(root):
    # Use the actual task directory owner for native children only. AFS may
    # expose UID 10740 even when a P19 coordinator starts as root; Mesa checks
    # XDG_RUNTIME_DIR against the native process UID. Never chown AFS or /tmp.
    root=Path(root);root.mkdir(parents=True,exist_ok=True);owner=root.stat()
    if owner.st_uid==os.geteuid():return {}
    if os.geteuid()!=0:raise ValueError('native task owner differs and this coordinator cannot delegate credentials')
    return {'user':owner.st_uid,'group':owner.st_gid,'extra_groups':[]}


def private_mesa_runtime(root,runtime_uid=None):
    # Mesa otherwise falls back to a shared /tmp/xdg-runtime-mesa-UID. P19
    # containers can inherit a directory they cannot use. Own this temporary
    # directory under the task root; never chmod or remove a shared directory.
    root=Path(root).resolve()
    runtime=root/'tmp'/('mesa_runtime_'+str(os.getpid())+'_'+uuid.uuid4().hex)
    runtime.mkdir(parents=True,mode=0o700,exist_ok=False)
    info=runtime.stat()
    if info.st_uid!=(os.geteuid() if runtime_uid is None else runtime_uid) or info.st_mode & 0o777 != 0o700:
        raise ValueError('private Mesa runtime ownership or mode mismatch')
    os.environ['XDG_RUNTIME_DIR']=str(runtime)
    atomic(runtime/'runtime_owner.json',{'pid':os.getpid(),'hostname':socket.gethostname(),'creator_uid':os.geteuid(),'native_uid':info.st_uid,'mode':'0700','shared_tmp_touched':False})
    return runtime


def read_tail(path,limit=512000):
    with Path(path).open('rb') as f:
        f.seek(max(0,Path(path).stat().st_size-limit));data=f.read().decode(errors='replace')
    result=[]
    for line in data.splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass
    return result


def training_sample(paths):
    result={}
    for path in paths:
        p=Path(path)
        if not p.is_file() or time.time()-p.stat().st_mtime>300:continue
        rows=[r for r in read_tail(p) if isinstance(r.get('elapsed_ms'),(int,float)) and r['elapsed_ms']>0]
        # Compare the same presentation. Mixed audio-only examples are cheaper.
        groups={}
        for r in rows[-150:]:groups.setdefault(r.get('presentation','unknown'),[]).append(r['elapsed_ms'])
        medians={k:statistics.median(v) for k,v in groups.items() if len(v)>=12}
        if medians:result[path]={'median_ms_by_presentation':medians,'sample_step':rows[-1].get('sample_step')}
    return result


def slowdown(baseline,current):
    ratios=[]
    for path,b in baseline.items():
        c=current.get(path)
        if not c or c['sample_step']==b['sample_step']:continue
        for k,ms in b['median_ms_by_presentation'].items():
            if k in c['median_ms_by_presentation']:ratios.append(ms/c['median_ms_by_presentation'][k])
    return 1-min(ratios) if ratios else None


def owner_bytes(root):
    output=subprocess.check_output(['du','-s','-B1',str(root)],text=True,timeout=300)
    return int(output.split()[0])



def native_cpu_probe(profile,output):
    selector=profile['cpu_env'].get('LD_PRELOAD','')
    if not selector or str(Path(selector).resolve()) not in Path('/proc/self/maps').read_text():
        raise ValueError('CPU EGL selector was not preloaded at process start; launch with frozen profile cpu_env before Python')
    from avengine.acoustics.runtime import load_habitat_runtime
    inputs=json.loads((Path(profile['inputs'])/'runtime.json').read_text())
    hs,_runtime=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
    cfg=hs.SimulatorConfiguration();cfg.scene_id='NONE';cfg.gpu_device_id=-1;cfg.enable_physics=False
    camera=hs.CameraSensorSpec();camera.uuid='rgb';camera.sensor_type=hs.SensorType.COLOR;camera.resolution=[32,32]
    agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[camera]
    sim=hs.Simulator(hs.Configuration(cfg,[agent]))
    try:
        rgb=sim.get_sensor_observations()['rgb'];maps=Path('/proc/self/maps').read_text()
        software=any(v in maps for v in ('swrast_dri.so','libgallium'))
        if list(rgb.shape[:2])!=[32,32] or not software or 'libEGL_nvidia' in maps or 'libnvidia-gl' in maps:
            raise ValueError('native CPU context probe did not read an exclusively software RGB frame')
    finally:sim.close()
    result={'status':'pass','actual_RGB_shape':list(rgb.shape),'software_driver_mapped':software,
        'XDG_RUNTIME_DIR':os.environ['XDG_RUNTIME_DIR'],'hostname':socket.gethostname(),
        'scene_id':'NONE','house_access':0,'GPU_started':False,'native_uid':os.geteuid(),'native_gid':os.getegid()}
    atomic(Path(output)/'native_cpu_readback.json',result)
    print(json.dumps(result),flush=True)


def record_cpu_sample(root,profile,workers,active,previous):
    """Container-visible CPU counters; retain cgroup quota with host context."""
    stat=Path('/proc/stat').read_text();load=Path('/proc/loadavg').read_text().strip()
    counters=[int(v) for v in stat.splitlines()[0].split()[1:]]
    # guest and guest_nice are already included in user/nice; do not double count.
    total=sum(counters[:8]);idle=counters[3]+counters[4]
    now=time.monotonic();fraction=None;cores=None;delta=None
    if previous is not None:
        before_total,before_idle,before_time=previous;dt=now-before_time
        delta=total-before_total
        if delta>0 and dt>0:
            busy=delta-(idle-before_idle);fraction=busy/delta
            cores=busy/os.sysconf('SC_CLK_TCK')/dt
    cgroup={}
    for name in ('cpu.max','cpu.stat','cpuset.cpus.effective'):
        p=Path('/sys/fs/cgroup')/name
        if p.is_file():cgroup[name]=p.read_text().strip()
    sample={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hostname':socket.gethostname(),
        'pid':os.getpid(),'workers_limit':workers,'active_parents':active,
        'proc_stat':stat,'proc_loadavg':load,'visible_logical_cpus':os.cpu_count(),
        'task_affinity_cpus':sorted(os.sched_getaffinity(0)),'owner_P19_cpu_allocation':88,
        'aggregate_busy_fraction':fraction,'aggregate_busy_core_equivalent':cores,
        'cgroup':cgroup,'scope':'container-visible /proc counters; host aggregate may exceed P19 88-core allocation'}
    folder=Path(root)/'cpu_monitor';folder.mkdir(exist_ok=True)
    with (folder/('cpu_'+profile['plan_name']+'.jsonl')).open('a') as f:
        f.write(json.dumps(sample,ensure_ascii=False)+'\n');f.flush()
    return total,idle,now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator-profile',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',choices=['production','smoke','validate','runtime_probe'],default='production')
    args=parser.parse_args();profile=json.loads(args.generator_profile.read_text());root=Path(profile['root']);code=Path(profile['code']);args.output.mkdir(parents=True,exist_ok=True)
    # P19 may lease a GPU to its rank; this entry clears it before native imports.
    os.environ.update(profile['cpu_env']);os.environ['CUDA_VISIBLE_DEVICES']=''
    credentials=native_process_credentials(root)
    private_mesa_runtime(root,runtime_uid=credentials.get('user',os.geteuid()))
    for folder in reversed(profile['cpu_env']['PYTHONPATH'].split(os.pathsep)):
        if folder not in sys.path:sys.path.insert(0,folder)
    if os.environ.get('LIBGL_ALWAYS_SOFTWARE')!='1' or os.environ.get('LP_NUM_THREADS')!='1':raise ValueError('CPU software-rendering contract missing')
    cpus=sorted(os.sched_getaffinity(0));cpu_limit=int(profile.get('cpu_affinity_limit',40))
    if not 1<=cpu_limit<=40:raise ValueError('CPU affinity exceeds owner-authorized 40-core maximum')
    os.sched_setaffinity(0,set(cpus[:cpu_limit]))
    if args.mode=='runtime_probe':
        if credentials:
            subprocess.run([sys.executable,*sys.argv],check=True,env=os.environ.copy(),**credentials)
        else:native_cpu_probe(profile,args.output)
        return
    if args.mode=='production':
        match=re.fullmatch(r'eval_job_a[0-9]{2}_(job_[A-Za-z0-9_.-]+)',args.output.name)
        run_root=root/profile.get('p19_run_directory','p19_run')
        if run_root.resolve().parent!=root.resolve():raise ValueError('P19 run directory must stay directly within this task root')
        if not match or not args.output.resolve().is_relative_to(run_root.resolve()):raise ValueError('production output lacks the P19 eval ABI')
        os.environ['AVENGINE_V7_P19_OWNER_JOB']=match.group(1)
    if args.mode=='production':
        job=os.environ.get('AVENGINE_V7_P19_OWNER_JOB','')
        active=Path(profile['owner_root'])/'hardbank/tmp/p19_active'/profile['plan_name']
        if not job or not active.is_file() or active.read_text().split()[:2]!=[job,socket.gethostname()]:raise ValueError('P19 active ownership does not match this host/job')
    rows=[json.loads(l) for l in Path(profile['seeds']).read_text().splitlines() if l.strip()]
    if args.mode=='smoke':rows=rows[:profile.get('smoke_parents',1)]
    if args.mode=='production' and len(rows)!=500:raise ValueError('production requires exact frozen 500-parent manifest')
    if len({r['scene_id'] for r in rows})!=len(rows):raise ValueError('duplicate physical parents')
    inputs=Path(profile['inputs']);houses=json.loads((inputs/'houses.json').read_text())['houses'];byh={h['house']:h for h in houses}
    for r in rows:
        h=byh[r['house']]
        if h['split']=='test' or {'development':'dev'}.get(h['split'],h['split'])!=r['split']:raise ValueError('house split crossed')
    if args.mode=='validate':
        atomic(args.output/'metrics.json',{'status':'profile_validated','parents':len(rows),'test_house_access':0,'GPU_started':False});return
    final=root/('release' if args.mode=='production' else 'smoke_delivery')
    if (final/'DELIVERY_MANIFEST.json').is_file():
        delivered=json.loads((final/'DELIVERY_MANIFEST.json').read_text())
        if delivered['generator_commit']!=profile['generator_commit'] or delivered['independent_parent_scenes']!=len(rows):raise ValueError('completed release belongs to a different immutable profile')
        (args.output/'input_manifest.jsonl').write_text(''.join(json.dumps({'scene_id':r['scene_id']})+'\n' for r in rows))
        (args.output/'predictions.jsonl').write_text(''.join(json.dumps({'scene_id':r['scene_id'],'status':'formal_alpha_admitted'})+'\n' for r in rows))
        atomic(args.output/'run.json',{'status':'complete','count':len(rows),'kind':'CPU generation receipts, not model predictions'})
        return
    work=root/(profile.get('production_work_directory','production') if args.mode=='production' else 'cci_smoke')
    if work.resolve().parent!=root.resolve():raise ValueError('parent work directory must stay directly within this task root')
    work.mkdir(exist_ok=True)
    # AFS atomic mkdir gives one writer; restart release is through explicit host
    # identity verification rather than timeouts that could steal a live job.
    if (root/'STOP').exists():raise RuntimeError('this task STOP is present')
    subprocess.run([sys.executable,str(code/'tools/timeline/v7_s_production.py'),
        '--generator-profile',str(args.generator_profile),'--mode','runtime_probe',
        '--output',str(args.output/'cpu_runtime_probe')],check=True,env=os.environ.copy(),cwd=code,**credentials)
    lock=root/('coordinator.lock' if args.mode=='production' else 'smoke.lock')
    try:lock.mkdir()
    except FileExistsError:
        old=json.loads((lock/'owner.json').read_text())
        job=os.environ.get('AVENGINE_V7_P19_OWNER_JOB','')
        active=Path(profile['owner_root'])/'hardbank/tmp/p19_active'/profile['plan_name']
        authority=active.read_text().split() if active.is_file() else []
        if old['hostname']==socket.gethostname():
            try:os.kill(old['pid'],0)
            except ProcessLookupError:pass
            else:raise ValueError('a live coordinator already owns this output')
        else:
            if not job or old.get('job_id')==job or authority[:2]!=[job,socket.gethostname()]:raise ValueError('previous host lock lacks P19-verified restart handoff')
            atomic(root/('restart_handoff_'+str(time.time_ns())+'.json'),{'old':old,'new_job_id':job,'P19_active_authority':str(active),'owner':authority})
        (lock/'owner.json').unlink();lock.rmdir();lock.mkdir()
    atomic(lock/'owner.json',{'pid':os.getpid(),'hostname':socket.gethostname(),'mode':args.mode,'job_id':os.environ.get('AVENGINE_V7_P19_OWNER_JOB'),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    children={};stopping=False;failures=[];done=[];started=time.monotonic();monitor_time=0
    def stop(sig,frame):
        nonlocal stopping
        stopping=True
        for p,_log,_row in children.values():
            # Each child was started by this coordinator in its own session.
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    paths=profile.get('training_step_files',[]);baseline=training_sample(paths)
    active_limit=min(cpu_limit,int(profile.get('workers',40)));protected_limit=active_limit;last_control=None;cpu_previous=None;cpu_monitor_time=0.;last_storage=time.monotonic();storage_limit=int(profile.get('owner_quota_bytes',2_000_000_000_000));reserve=int(profile.get('owner_reserve_bytes',80_000_000_000))
    pending=list(enumerate(rows))
    try:
        if owner_bytes(profile['owner_root'])+reserve>=storage_limit:raise ValueError('owner 2 TB quota reserve insufficient')
        while pending or children:
            if (root/'STOP').exists():stop(signal.SIGTERM,None)
            if time.monotonic()-last_storage>900:
                if owner_bytes(profile['owner_root'])+reserve>=storage_limit:stop(signal.SIGTERM,None);failures.append({'reason':'owner_quota_reserve_exhausted'})
                last_storage=time.monotonic()
            if time.monotonic()-monitor_time>120:
                current=training_sample(paths);loss=slowdown(baseline,current)
                threshold=float(profile.get('training_slowdown_threshold',.35))
                if not 0<threshold<=.35:raise ValueError('training slowdown threshold exceeds owner rule')
                if loss is not None and loss>threshold:
                    active_limit=max(2,active_limit//2);protected_limit=active_limit
                atomic(root/'training_throughput_monitor.json',{'baseline':baseline,'current':current,'worst_comparable_throughput_drop':loss,'active_limit':active_limit,'rule':'more than 35 percent => halve new dispatch; in-flight parents finish','threshold':threshold,'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});monitor_time=time.monotonic()
            control_path=root/'dispatch_control.json'
            if control_path.is_file():
                control=json.loads(control_path.read_text());requested=int(control['workers'])
                if not 1<=requested<=cpu_limit:raise ValueError('online dispatch control exceeds authorized affinity')
                if control!=last_control:
                    active_limit=requested;protected_limit=requested;last_control=control
                    atomic(root/('dispatch_control_applied_'+str(time.time_ns())+'.json'),{'control':control,'workers':active_limit,'pid':os.getpid(),'hostname':socket.gethostname()})
            if time.monotonic()-cpu_monitor_time>=60:
                cpu_previous=record_cpu_sample(root,profile,active_limit,len(children),cpu_previous)
                cpu_monitor_time=time.monotonic()
            while pending and len(children)<active_limit and not stopping:
                index,row=pending.pop(0);sid=row['scene_id'];seed=work/(sid+'_seed.jsonl')
                if not seed.exists():seed.write_text(json.dumps(row,ensure_ascii=False)+'\n')
                logfile=(root/'logs'/(sid+'_'+str(time.time_ns())+'.log')).open('x')
                cmd=[sys.executable,str(code/'tools/timeline/generate_v7_registered.py'),'--houses',str(inputs/'houses.json'),'--seeds',str(seed),'--runtime-inputs',str(inputs/'runtime.json'),'--assets',str(inputs/'assets.json'),'--source-admission',str(inputs/'source_admission.json'),'--human-admission',str(inputs/'human_admission.json'),'--output',str(work),'--workers','1','--limit','1','--render-backend','cpu_llvmpipe']
                p=subprocess.Popen(cmd,env=os.environ.copy(),stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True,cwd=code,**credentials);children[sid]=(p,logfile,row)
            for sid,(p,logfile,row) in list(children.items()):
                if p.poll() is None:continue
                logfile.close();del children[sid];parent=work/'parents'/sid;seal=parent/'alpha_seal_completed.json'
                if p.returncode==0 and seal.is_file():
                    mark=json.loads(seal.read_text());result=mark['result']
                    if result['status']!='formal_alpha_admitted':raise ValueError('unqualified parent seal')
                    done.append(result)
                    # This task's generated material cache is rebuildable; retain
                    # source inputs, exact RIRs, full native evidence and media.
                    cache=parent/'source_material_cache'
                    files=[f for f in cache.rglob('*.glb') if f.is_file()] if cache.is_dir() else []
                    if files:
                        atomic(parent/'discarded_material_cache.json',{'paths':[str(f) for f in files],'reason':'parent seal complete; original source assets and material checks retained'})
                        for f in files:f.unlink()
                else:failures.append({'scene_id':sid,'returncode':p.returncode,'failure_receipts':[str(f) for f in parent.glob('failed_*.json')]})
            atomic(root/'PROGRESS.json',{'status':'stopped' if stopping else 'running','formal_parents':len(done),'formal_windows':sum(r['alpha_formal_windows'] for r in done),'formal_questions':sum(r['alpha_formal_logical_questions'] for r in done),'active_parents':list(children),'remaining_parents':len(pending),'failures':failures,'workers_limit':active_limit,'elapsed_seconds':time.monotonic()-started,'pid':os.getpid(),'hostname':socket.gethostname(),'GPU_started':False})
            if stopping and not children:break
            time.sleep(5)
        if stopping or failures:raise RuntimeError('production incomplete; failures retained; rerun resumes immutable stages')
        from avengine.timeline.v7_alpha_delivery import export_alpha
        staging=root/'tmp'/('export_attempt_'+uuid.uuid4().hex)
        release=staging
        summary=export_alpha(profile['seeds'],[work],release,expected_parents=len(rows))
        from avengine.timeline.v7_s_batch_checks import check_critical_delivery,open_delivered_media
        coverage=check_critical_delivery(release,require_full=args.mode=='production')
        media=open_delivered_media(release)
        if coverage['status']!='pass' or media['status']!='pass':raise ValueError('critical final delivery checks failed')
        final=root/('release' if args.mode=='production' else 'smoke_delivery')
        atomic(release/'DELIVERY_MANIFEST.json',{'status':'sealed','independent_parent_scenes':len(rows),'train_jsonl':str(final/'train.jsonl'),'dev_jsonl':str(final/'dev.jsonl'),'coverage':coverage,'media_readback':media,'generator_commit':profile['generator_commit'],'GPU_started':False})
        os.rename(release,final)
        release=final
        (args.output/'input_manifest.jsonl').write_text(''.join(json.dumps({'scene_id':r['scene_id']})+'\n' for r in rows))
        (args.output/'predictions.jsonl').write_text(''.join(json.dumps({'scene_id':r['scene_id'],'status':'formal_alpha_admitted'})+'\n' for r in rows))
        atomic(args.output/'run.json',{'status':'complete','count':len(rows),'kind':'CPU generation receipts, not model predictions'})
        atomic(args.output/'metrics.json',{'status':'complete','parents':len(rows),'delivery_root':str(release),'windows':summary['formal_windows'],'questions':summary['formal_logical_questions']})
    finally:
        # A failed coordinator must not leave its own renderers running.
        if children:
            stop(signal.SIGTERM,None)
            deadline=time.monotonic()+20
            for p,logfile,_row in children.values():
                try:p.wait(timeout=max(.1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
                    p.wait(timeout=5)
                logfile.close()
            children.clear()
        if (lock/'owner.json').is_file():
            (lock/'owner.json').unlink();lock.rmdir()


if __name__=='__main__':main()
