#!/usr/bin/env python3
"""CPU tests of capacity, sharding, verified resume, corruption and cleanup."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import signal
import unittest
from resources import GIB,allocate,write_json
from generate import sha,files_ok,storage_usage
from wav_metadata import canonical_wav_sha256
from remap_configs import mapped

class ResourceTests(unittest.TestCase):
 def reports(self,cpu=88,memory=920,n=8):
  return [{'host':'p19-fixture','affinity_cpus':list(range(cpu)),'effective_cpu_cores':cpu,'memory_limit_bytes':int(memory*GIB),'memory_available_bytes':int(memory*GIB),'ranks':{'RANK':str(i),'LOCAL_WORLD_SIZE':str(n)}} for i in range(n)]
 def test_shared_node_not_multiplied_by_rank(self):
  a=allocate(self.reports())['p19-fixture'];self.assertEqual(a['total_cpu_workers'],79);self.assertEqual(a['reserved_cpu_cores'],9)
  flat=[cpu for row in a['worker_affinity'].values() for cpu in row];self.assertEqual(len(flat),len(set(flat)))
 def test_training_loaders_and_memory_reserved(self):
  a=allocate(self.reports(),mode='mixed',training_ranks=6,loader_workers=4)['p19-fixture'];self.assertEqual(a['total_cpu_workers'],55)
  self.assertLessEqual(a['memory_spare_bytes'],int((920*.9-6*33)*GIB))
 def test_developer_quota_despite_many_host_cpus(self):
  r=self.reports(cpu=192,memory=32,n=1);r[0]['effective_cpu_cores']=2
  self.assertEqual(allocate(r)['p19-fixture']['total_cpu_workers'],1)
 def test_zero_capacity_is_not_oversubscribed(self):
  a=allocate(self.reports(cpu=8,n=8),mode='mixed',training_ranks=2,loader_workers=4)['p19-fixture'];self.assertEqual(a['total_cpu_workers'],0)
 def test_disjoint_rank_affinity(self):
  r=self.reports(cpu=4,n=2);r[0]['affinity_cpus']=[0,1];r[1]['affinity_cpus']=[2,3]
  for row in r:row['effective_cpu_cores']=2
  a=allocate(r)['p19-fixture'];self.assertEqual(a['node_cpu_cores'],4);self.assertEqual(a['total_cpu_workers'],3)
 def test_distinct_rank_quotas_share_a_node_parent(self):
  r=self.reports()
  for i,row in enumerate(r):
   row['effective_cpu_cores']=11
   row['cgroup_limits']=[{'path':'/sys/fs/cgroup','canonical_path':'/job','cpu_quota_cores':88},{'path':'/sys/fs/cgroup/rank'+str(i),'canonical_path':'/job/rank'+str(i),'cpu_quota_cores':11}]
  self.assertEqual(allocate(r)['p19-fixture']['total_cpu_workers'],79)
 def test_shared_parent_quota_remains_a_node_limit(self):
  r=self.reports(cpu=192,n=8)
  for i,row in enumerate(r):
   row['effective_cpu_cores']=1
   row['cgroup_limits']=[{'path':'/sys/fs/cgroup','canonical_path':'/job','cpu_quota_cores':2},{'path':'/sys/fs/cgroup/rank'+str(i),'canonical_path':'/job/rank'+str(i),'cpu_quota_cores':1}]
  self.assertEqual(allocate(r)['p19-fixture']['total_cpu_workers'],1)
 def test_parent_memory_quota_and_current_are_shared(self):
  r=self.reports(cpu=88,memory=16,n=8)
  for i,row in enumerate(r):
   row['memory_available_bytes']=16*GIB
   row['cgroup_limits']=[{'path':'/sys/fs/cgroup','canonical_path':'/job','memory_limit_bytes':32*GIB,'memory.current':str(20*GIB)},{'path':'/sys/fs/cgroup/rank'+str(i),'canonical_path':'/job/rank'+str(i),'memory_limit_bytes':16*GIB,'memory.current':'0'}]
  a=allocate(r)['p19-fixture'];self.assertEqual(a['total_cpu_workers'],3)
 def test_memory_limits_concurrency(self):
  self.assertEqual(allocate(self.reports(cpu=88,memory=16),worker_gib=4)['p19-fixture']['total_cpu_workers'],3)

class ControllerTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.base=Path(self.tmp.name);self.root=self.base/'run';self.manifest=self.base/'manifest.json';self.script=self.base/'fixture.py'
  self.script.write_text('''import json,sys
from pathlib import Path
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({"fixture":sys.argv[2]}))
''')
  render=self.base/'render.bin';render.write_bytes(b'CPU fixture; not actual Habitat media')
  pre={'attempt':str(self.base),'files':[{'path':str(render),'bytes':render.stat().st_size,'sha256':sha(render)}]}
  worlds=[]
  for i in range(2):
   stages={}
   for phase in ['plan','render','audio','deliver']:
    output='{world_root}/ARTIFACTS.json' if phase=='deliver' else '{attempt}/stage.json'
    stages[phase]={'argv':['{python}',str(self.script),output,phase], 'outputs':[output]}
   worlds.append({'world_id':f'w{i}','split':'train','scratch_bytes_peak':1024**2,'delivery_bytes_upper':1024,'stages':stages,'precompleted':{'render':pre}})
  write_json(self.manifest,{'schema':'cci_datagen_stages_v1','train_only':True,'formal_frozen_test':False,'worlds':worlds})
 def tearDown(self):self.tmp.cleanup()
 def run_controller(self,job,expected=0,extra=()):
  env=dict(os.environ);env.update(DATAGEN_LAUNCH_ID=job,RANK='0',WORLD_SIZE='1',LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',CUDA_VISIBLE_DEVICES='')
  args=[sys.executable,'-B',str(Path(__file__).with_name('generate.py')),'--manifest',str(self.manifest),'--run-root',str(self.root),'--python',sys.executable,'--mode','cpu','--worker-gib','.1','--timeout-seconds','25','--prune-owned-intermediates',*extra]
  result=subprocess.run(args,env=env,text=True,capture_output=True,timeout=35)
  self.assertEqual(result.returncode,expected,result.stdout+result.stderr);return result
 def test_one_scheduler_rank_with_shared_afs_allowance_completes_and_resumes(self):
  assigned=self.base/'single_rank_capacity.json';write_json(assigned,{'affinity_cpus':sorted(os.sched_getaffinity(0))[:2]})
  extra=['--resource-assignment',str(assigned),'--generation-space-budget-bytes',str(16*1024**2),'--storage-workspace',str(self.base)]
  self.run_controller('shared-afs-first',extra=extra)
  self.assertEqual(json.loads((self.root/'ALL_DONE.json').read_text())['worlds'],2)
  self.assertFalse(list(self.root.glob('scratch/*/*')))
  self.run_controller('shared-afs-resume',extra=extra)
  self.assertEqual(len(list(self.root.glob('receipts/*/deliver.json'))),2)
 def test_real_subprocesses_resume_and_prune_only_owned_scratch(self):
  self.run_controller('first');first=list(self.root.glob('receipts/*/deliver.json'));before={str(p):p.read_bytes() for p in first};self.assertEqual(len(first),2)
  self.assertFalse(list(self.root.glob('scratch/*/*')));self.assertTrue((self.base/'render.bin').is_file())
  self.run_controller('resume');self.assertEqual(before,{str(p):p.read_bytes() for p in first})
 def test_corruption_is_detected_and_recomputed(self):
  self.run_controller('first');p=self.root/'worlds/w0/ARTIFACTS.json';p.write_text('corrupt')
  self.run_controller('recover');self.assertEqual(json.loads(p.read_text())['fixture'],'deliver')
 def test_interrupted_own_stage_stops_and_can_resume(self):
  self.script.write_text(self.script.read_text().replace("p.write_text", "import time\nif sys.argv[2]=='audio':time.sleep(30)\np.write_text"))
  env=dict(os.environ);env.update(DATAGEN_LAUNCH_ID='interrupt',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',CUDA_VISIBLE_DEVICES='')
  args=[sys.executable,'-B',str(Path(__file__).with_name('generate.py')),'--manifest',str(self.manifest),'--run-root',str(self.root),'--mode','cpu','--worker-gib','.1','--timeout-seconds','45','--prune-owned-intermediates']
  child=subprocess.Popen(args,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   end=time.monotonic()+25;pid=None
   while time.monotonic()<end:
    matches=list(self.root.glob('scratch/*/audio-*/process.json'))
    if matches:pid=json.loads(matches[0].read_text())['pid'];break
    time.sleep(.1)
   self.assertIsNotNone(pid)
   child.send_signal(signal.SIGTERM);self.assertEqual(child.wait(timeout=10),143)
   with self.assertRaises(ProcessLookupError):os.kill(pid,0)
   self.assertFalse((self.root/'ALL_DONE.json').exists())
   self.script.write_text(self.script.read_text().replace("if sys.argv[2]=='audio':time.sleep(30)","if sys.argv[2]=='audio':pass"))
   self.run_controller('after-interruption');self.assertTrue((self.root/'ALL_DONE.json').is_file())
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=10)
 def test_formal_test_is_rejected(self):
  data=json.loads(self.manifest.read_text());data['formal_frozen_test']=True;write_json(self.manifest,data);self.run_controller('test-rejected',expected=1)
 def test_manifest_storage_budget_is_enforced_before_launch(self):
  self.run_controller('quota-rejected',expected=1,extra=['--media-budget-gib','0.00000001']);self.assertFalse((self.root/'receipts').exists())
 def test_runtime_mapping_preserves_hashes_and_redirects_selected_configs(self):
  value={'asset':'/data/assets/a.glb','sha256':'abc','profile':'/data/config/profile.json','relative':'mesh/a.glb'}
  result=mapped(value,Path('/mnt/afs/test/mirror'),{'/data/config/profile.json':Path('/mnt/afs/test/config/profile.json')})
  self.assertEqual(result['asset'],'/mnt/afs/test/mirror/data/assets/a.glb');self.assertEqual(result['profile'],'/mnt/afs/test/config/profile.json');self.assertEqual(result['sha256'],'abc');self.assertEqual(result['relative'],'mesh/a.glb')
 def test_mapped_config_links_and_json_fragments_are_checked_after_all_writes(self):
  mirror=self.base/'mirror';(mirror/'data').mkdir(parents=True)
  write_json(mirror/'data/first.json',{'other':'/data/second.json#/value'})
  write_json(mirror/'data/second.json',{'value':1})
  sources=self.base/'sources.json';write_json(sources,['/data/first.json','/data/second.json']);dest=self.base/'mapped'
  result=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('remap_configs.py')),'--source-jsons',str(sources),'--mirror-root',str(mirror),'--output-root',str(dest)],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)
  self.assertEqual(json.loads((dest/'data/first.json').read_text())['other'],str(dest/'data/second.json')+'#/value')
  self.assertTrue(all(not r['missing_referenced_paths'] for r in json.loads((dest/'PATH_MAPPING.json').read_text())['rows']))
 def test_wav_peak_timestamp_does_not_change_canonical_audio_hash(self):
  import struct
  prefix=b'RIFF'+struct.pack('<I',32)+b'WAVEPEAK'+struct.pack('<I',8)+struct.pack('<I',1)
  a=self.base/'a.wav';b=self.base/'b.wav';a.write_bytes(prefix+struct.pack('<I',10));b.write_bytes(prefix+struct.pack('<I',20))
  self.assertNotEqual(sha(a),sha(b));self.assertEqual(canonical_wav_sha256(a)['canonical_sha256'],canonical_wav_sha256(b)['canonical_sha256'])
  self.assertEqual(canonical_wav_sha256(a)['peak_timestamps'],[10])
 def test_interrupted_attempts_count_toward_storage_budget(self):
  stale=self.root/'scratch/w0/old_attempt';stale.mkdir(parents=True);(stale/'file').write_bytes(b'x'*2048)
  with self.assertRaises(RuntimeError):storage_usage(self.root,1024,1024)
  self.assertTrue((stale/'file').is_file())
 def test_changed_manifest_cannot_reuse_old_run(self):
  self.run_controller('first');data=json.loads(self.manifest.read_text());data['scope']='different';write_json(self.manifest,data);self.run_controller('changed',expected=1)

if __name__=='__main__':unittest.main(verbosity=2)
