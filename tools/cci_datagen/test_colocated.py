"""CPU-only tests of colocation boundaries, dynamic AFS space and own-PID stops."""
import copy,json,os,shutil,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from resources import GIB,cgroup_spare
from colocated_resources import capacity,observe
from space_budget import AFSSpace,pool_allowance
from gpu_guard import snapshot,require_headroom,GPUWatch

def fixture(private=False):
 def report(rank):
  limits=[{'path':'/shared','canonical_path':'/shared','cpu_quota_cores':88,'memory_limit_bytes':920*GIB,'memory.current':0}]
  if private:limits.append({'path':f'/shared/rank{rank}','canonical_path':f'/shared/rank{rank}','cpu_quota_cores':4,'memory_limit_bytes':16*GIB,'memory.current':0})
  return {'pid':100+rank,'host':'one_node','affinity_cpus':list(range(88)),'effective_cpu_cores':4 if private else 88,'memory_info':{'MemTotal':920*GIB,'MemAvailable':920*GIB},'memory_available_bytes':16*GIB if private else 920*GIB,'cgroup_limits':limits}
 return {'observer':report(0),'rank_observations':[report(i) for i in range(8)],'observed_loader_counts':{},'scheduler_lease':{'parent_pid':100,'local_world_size':8,'gpu':'2'}}

class ResourceTests(unittest.TestCase):
 def test_exited_peer_is_unobserved_and_does_not_abort_probe(self):
  env={'LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'2','WORLD_SIZE':'2','RANK':'0','LADDER_JOB_ID':'exit-race'}
  with patch('colocated_resources.process_env',return_value=env),patch('colocated_resources.read',return_value='/owned/p19_worker.py'),patch('colocated_resources.probe',return_value={'ranks':env.copy()}),patch('colocated_resources.worker_processes',return_value=[(999999,env)]),patch('colocated_resources.cgroup_limits',return_value=('',[])),patch('colocated_resources.os.sched_getaffinity',side_effect=ProcessLookupError),patch('colocated_resources.child_loader_counts',return_value={}):
   r=observe('/unused',False);self.assertEqual(r['scheduler_lease']['unobserved_local_ranks'],2);self.assertEqual(r['rank_observations'],[])
 def test_clean_inactive_afs_cache_does_not_consume_cpu_memory_allowance(self):
  row={'memory_limit_bytes':32*GIB,'memory.current':str(31*GIB),'reclaimable_inactive_file_bytes':24*GIB}
  self.assertEqual(cgroup_spare(row),25*GIB)
 def test_one_rank_uses_remaining_shared_cores(self):
  c=capacity(fixture(),{});self.assertEqual(c['cpu_processes'],51);self.assertEqual(c['generation_scheduler_ranks'],1);self.assertEqual(c['torchrun_children_launched'],0);self.assertEqual(c['other_training_ranks'],7);self.assertEqual(c['gpu_lease'],'2')
 def test_isolated_rank_quota_is_not_node_quota(self):
  c=capacity(fixture(True),{});self.assertEqual(c['cpu_processes'],3)
 def test_more_observed_loaders_reduce_cpu(self):
  f=fixture();f['observed_loader_counts']={101:8};self.assertEqual(capacity(f,{})['cpu_processes'],23)
 def test_world_size_is_runtime(self):
  f=fixture();f['rank_observations']=f['rank_observations'][:4];f['scheduler_lease']['local_world_size']=4;self.assertEqual(capacity(f,{})['cpu_processes'],67)
 def test_low_memory_caps_processes(self):
  f=fixture();f['observer']['memory_available_bytes']=10*GIB;self.assertEqual(capacity(f,{})['cpu_processes'],2)
 def test_unknown_ranks_are_reserved(self):
  f=fixture();f['rank_observations']=f['rank_observations'][:1];self.assertEqual(capacity(f,{})['cpu_processes'],51)

class SpaceTests(unittest.TestCase):
 def test_owner_budget_decimal_two_tb(self):
  self.assertEqual(pool_allowance(1_483_910_266_880,0),392_089_733_120)
 def test_cpu_pool_not_fixed_eight_worlds(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);worlds=[{'world_id':str(i),'scratch_bytes_peak':4*GIB,'delivery_bytes_upper':128*1024**2} for i in range(64)];s=AFSSpace(root,root/'run',392_089_733_120,worlds);active=[]
   for i in range(51):self.assertTrue(s.can_admit(str(i),active));active.append(str(i))
   self.assertGreater(s.snapshot(active)['headroom_bytes'],100_000_000_000)
 def test_retained_delivery_reduces_admission(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);run=root/'run';(run/'worlds/old').mkdir(parents=True)
   with (run/'worlds/old/media').open('wb') as f:f.truncate(380_000_000_000)
   worlds=[{'world_id':str(i),'scratch_bytes_peak':4*GIB,'delivery_bytes_upper':128*1024**2} for i in range(64)];s=AFSSpace(root,run,392_089_733_120,worlds)
   self.assertTrue(s.can_admit('0',[]));self.assertTrue(s.can_admit('1',['0']));self.assertFalse(s.can_admit('2',['0','1']))
 def test_world_peak_is_checked(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);run=root/'run';(run/'scratch/x').mkdir(parents=True);(run/'scratch/x/large').write_bytes(b'12345')
   s=AFSSpace(root,run,100,[{'world_id':'x','scratch_bytes_peak':4,'delivery_bytes_upper':1}]);self.assertRaises(RuntimeError,s.check,['x'])
 def test_paths_cannot_leave_private_workspace(self):
  with tempfile.TemporaryDirectory() as t:self.assertRaises(ValueError,AFSSpace,Path(t),Path(t).parent,100,[])

class GPUTests(unittest.TestCase):
 def test_nvml_read_only_lease_selector(self):
  xml='<nvidia_smi_log><gpu><uuid>GPU-x</uuid><product_name>test</product_name><fb_memory_usage><total>81920 MiB</total><used>38000 MiB</used><free>43920 MiB</free></fb_memory_usage><processes><process_info><pid>1</pid><type>G</type><used_memory>10 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'
  result=type('Result',(),{'returncode':0,'stdout':xml,'stderr':''})()
  with patch('gpu_guard.subprocess.run',return_value=result) as call:
   r=snapshot('2');self.assertEqual(call.call_args[0][0],['nvidia-smi','--id=2','-q','-x']);self.assertEqual(r['processes'][0]['type'],'G')
 def test_selected_profile_memory_budget_takes_effect(self):
  with patch.dict(os.environ,{'CCI_RENDER_CAP_GIB':'4','CCI_RENDER_TRAINING_RESERVE_GIB':'1'}),patch('gpu_guard.snapshot',return_value={'free_bytes':6*GIB}):self.assertEqual(require_headroom('2')['free_bytes'],6*GIB)
  with patch.dict(os.environ,{'CCI_RENDER_CAP_GIB':'4','CCI_RENDER_TRAINING_RESERVE_GIB':'1'}),patch('gpu_guard.snapshot',return_value={'free_bytes':4*GIB}):self.assertRaises(RuntimeError,require_headroom,'2')
 def test_multiple_gpu_selector_is_rejected(self):self.assertRaises(ValueError,snapshot,'0,1')
 def test_insufficient_free_memory_refuses_before_context(self):
  with patch('gpu_guard.snapshot',return_value={'free_bytes':15*GIB}):self.assertRaises(RuntimeError,require_headroom,'2')
 def test_unobservable_own_memory_is_not_reported_as_zero(self):
  child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)'],start_new_session=True)
  try:
   with tempfile.TemporaryDirectory() as t,patch('gpu_guard.snapshot',return_value={'free_bytes':80*GIB,'processes':[]}):
    p=Path(t)/'gpu.json';watch=GPUWatch('unit_test_gpu',child,p).start();child.wait(timeout=5);self.assertRaises(RuntimeError,watch.close);r=json.loads(p.read_text());self.assertIsNone(r['own_renderer_peak_bytes']);self.assertFalse(r['own_renderer_memory_observed'])
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=5)
 def test_renderer_memory_certificate_survives_scratch_cleanup(self):
  from generate import execute
  with tempfile.TemporaryDirectory() as t,patch.dict(os.environ,{'CCI_REQUIRE_GPU_GUARD':'1'}),patch('gpu_guard.snapshot',return_value={'free_bytes':80*GIB,'processes':[{'pid':999999,'used_bytes':2*GIB}]}),patch('gpu_guard.owned_descendant',return_value=True):
   root=Path(t);spec={'argv':['{python}','-c','import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(\"ok\")','{attempt}/marker'],'outputs':['{attempt}/marker']}
   r=execute(root,'unit_world','render',spec,{},sys.executable,'unit_job','unit_manifest','unit_test_gpu',[],19)
   shutil.rmtree(root/'scratch/unit_world');memory=Path(r['gpu_memory_log']);self.assertTrue(memory.is_file());self.assertTrue(memory.is_relative_to(root/'logs'));self.assertEqual(r['gpu_memory_summary']['own_renderer_peak_bytes'],2*GIB)
 def test_only_own_renderer_pid_is_stopped(self):
  renderer=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True);training=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
  try:
   with tempfile.TemporaryDirectory() as t,patch('gpu_guard.snapshot',return_value={'free_bytes':3*GIB,'processes':[{'pid':renderer.pid,'used_bytes':1*GIB}], 'used_bytes':39*GIB}):
    watch=GPUWatch('2',renderer,Path(t)/'gpu.json').start();renderer.wait(timeout=5);self.assertIsNone(training.poll());self.assertRaises(RuntimeError,watch.close);self.assertIsNotNone(json.loads((Path(t)/'gpu.json').read_text())['monitor_error'])
  finally:
   for child in (renderer,training):
    if child.poll() is None:child.terminate();child.wait(timeout=5)

if __name__=='__main__':unittest.main()
