"""Run one bounded owned job with GPU admission, process identity and resource accounting."""
import argparse,datetime,json,os,pathlib,subprocess,time,signal,fcntl
from v7_object_progress import append as progress

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(root,record):
    with (root/'JOBS.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);j=root/'JOBS.json';rows=json.loads(j.read_text()) if j.exists() else []
        existing=next((r for r in rows if r['name']==record['name']),None)
        if existing and existing.get('supervisor_pid')!=record['supervisor_pid']:
            raise ValueError('Owned job name already registered; inspect existing PID and choose a new evidence batch name')
        prior=existing.get('status') if existing else None;rows=[r for r in rows if r['name']!=record['name']];rows.append(record);tmp=j.with_suffix('.tmp');tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');tmp.replace(j)
        if prior!=record['status']:progress(root,'作业 '+record['name']+'：'+record['status'],str(j)+'；'+record.get('log',''),'按已登记批次继续；完成后合并机器验收')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--name',required=True);p.add_argument('--gpu',type=int);p.add_argument('--timeout',type=int,default=18000);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if a.gpu is not None and a.gpu not in [-1,0,1]:raise SystemExit('Only physical GPU 0 or 1 allowed; -1 selects an eligible one')
    budget=a.root/'budget.json'
    if budget.exists():
        deadline=datetime.datetime.fromisoformat(json.loads(budget.read_text())['gpu_generation_cutoff_utc']);a.timeout=min(a.timeout,max(1,int((deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds())))
    free=os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize;used=sum(q.stat().st_size for q in a.root.rglob('*') if q.is_file())
    if free<150*1024**3 or used>60_000_000_000:raise SystemExit('Disk budget stops job')
    logs=a.root/'logs';logs.mkdir(exist_ok=True);log=logs/(a.name+'.log');started=time.monotonic();record={'name':a.name,'machine':'48g','supervisor_pid':os.getpid(),'command':cmd,'log':str(log),'started_at':now(),'status':'waiting_gpu' if a.gpu is not None else 'starting','stop':'核对 /proc/'+str(os.getpid())+'/cmdline 后 kill -TERM '+str(os.getpid())};save(a.root,record)
    proc=None;cancelled=False
    def terminate(sig,frame):
        nonlocal cancelled
        cancelled=True
        if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    gpu=None;env=os.environ.copy();env.update(TMPDIR=str(a.root/'cache/tmp'),XDG_CACHE_HOME=str(a.root/'cache/xdg'),CUDA_CACHE_PATH=str(a.root/'cache/cuda'),TORCHINDUCTOR_CACHE_DIR=str(a.root/'cache/torchinductor'),__GL_SHADER_DISK_CACHE_PATH=str(a.root/'cache/gl'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
    if a.gpu is not None:
        while not cancelled:
            info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True);candidates=[line.split(', ') for line in info.splitlines() if int(line.split(',')[0]) in ([0,1] if a.gpu==-1 else [a.gpu])];gpu=max(candidates,key=lambda x:int(x[2]));record['last_gpu_free_mib']={x[0]:int(x[2]) for x in candidates};save(a.root,record)
            if int(gpu[2])>=30720:break
            if time.monotonic()-started>a.timeout:cancelled=True;record['reason']='GPU gate wait deadline';break
            print('WAIT_GPU',record['last_gpu_free_mib'],flush=True);time.sleep(15)
        if cancelled:record.update(status='stopped_before_gpu_launch',ended_at=now(),wall_seconds=time.monotonic()-started,gpu_allocated_seconds=0);save(a.root,record);return 75
        env['CUDA_VISIBLE_DEVICES']=gpu[1];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    else:env['CUDA_VISIBLE_DEVICES']=''
    launch=time.monotonic()
    with log.open('w') as f:
        proc=subprocess.Popen(['nice','-n','10',*cmd],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);record.update(pid=proc.pid,physical_gpu=int(gpu[0]) if gpu else None,gpu_uuid=gpu[1] if gpu else None,free_gpu_mib_before_launch=int(gpu[2]) if gpu else None,status='running',launched_at=now());save(a.root,record);print(json.dumps(record),flush=True)
        try:code=proc.wait(timeout=max(1,a.timeout-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
        record.update(status='completed' if code==0 else 'failed',returncode=code,ended_at=now(),wall_seconds=time.monotonic()-started,gpu_allocated_seconds=time.monotonic()-launch if gpu else 0);save(a.root,record);print(json.dumps(record),flush=True);return code
if __name__=='__main__':raise SystemExit(main())
