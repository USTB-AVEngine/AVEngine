"""Run matching CPU baseline/final suites and record exact failure/error test IDs."""
import os,json,sys,time,resource,threading
from pathlib import Path
# These controls affect only this test process, never a shared environment.
try:
 import torch
 torch.set_num_threads(1); torch.set_num_interop_threads(1)
except ImportError: pass
try:
 import cv2
 cv2.setNumThreads(1)
except ImportError: pass
try:
 import onnxruntime as ort
 old_session=ort.InferenceSession
 def bounded_session(*args,**kwargs):
  options=kwargs.get('sess_options')
  if options is None and len(args)<2:
   options=ort.SessionOptions();kwargs['sess_options']=options
  elif len(args)>=2:options=args[1]
  if options is not None:options.intra_op_num_threads=1;options.inter_op_num_threads=1
  return old_session(*args,**kwargs)
 ort.InferenceSession=bounded_session
except ImportError: pass
import pytest
label=sys.argv[1]; out=Path(sys.argv[2]); reports=[]; peak=0; stop=threading.Event()
def monitor():
 global peak
 while not stop.wait(.1):
  count=len(list(Path('/proc/self/task').iterdir())); peak=max(peak,count)
  if count>32:
   (out/f'{label}_thread_violation.json').write_text(json.dumps({'pid':os.getpid(),'threads':count,'timestamp':time.time()}))
   os._exit(125)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
class Receipts:
 def pytest_runtest_logreport(self,report):
  reports.append({'nodeid':report.nodeid,'when':report.when,'outcome':report.outcome})
 def pytest_collectreport(self,report):
  if report.failed:reports.append({'nodeid':report.nodeid,'when':'collect','outcome':'failed'})
tick=time.monotonic()
args=['tests','-q','-m','not integration and not canary and not native_habitat and not rlr_audio and not blender_assets and not media_readback and not release_canary','--basetemp',str(out/(label+'_tmp')),'-o','cache_dir='+str(out/(label+'_cache'))]
code=pytest.main(args,plugins=[Receipts()]);stop.set();thread.join()
usage=resource.getrusage(resource.RUSAGE_SELF)
summary={'label':label,'commit':os.environ['TEST_COMMIT'],'command':['python','test_runner.py',label,str(out)],'pytest_args':args,'scope':'all tests; native/integration/media/canary markers deselected; CPU only','exit_code':int(code),'failure_ids':sorted(set(r['nodeid'] for r in reports if r['when']=='call' and r['outcome']=='failed')),'error_ids':sorted(set(r['nodeid'] for r in reports if r['when']!='call' and r['outcome']=='failed')),'passed':sum(r['when']=='call' and r['outcome']=='passed' for r in reports),'skipped':sum(r['outcome']=='skipped' for r in reports),'wall_seconds':time.monotonic()-tick,'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,'os_threads_peak':peak,'private_runtime_caps':['torch intra/inter 1','cv2 1','onnxruntime intra/inter 1','BLAS/OMP/numba 1'],'reports':reports}
(out/(label+'.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('JSON_SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='reports'}),flush=True)
sys.exit(int(code))
