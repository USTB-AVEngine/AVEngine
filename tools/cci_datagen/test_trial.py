import json,os,signal,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
from trial_limits import slots,identity,verified_stop,monitor_generation
from space_budget import AFSSpace
from generate import sha
from v7_infra_stage import check_request
from avengine.capture.continuous_native_video import NativeContinuousVideo

class TrialTests(unittest.TestCase):
 def test_budget_accounts_for_all_future_deliveries(self):
  self.assertEqual(slots(51),12)
  self.assertEqual(slots(7),7)
  self.assertEqual(slots(51,limit=20_000_000_000),0)
  self.assertLessEqual(11*8*1024**3+64*512*1024**2+2_000_000_000,140_000_000_000)
 def test_held_out_house_and_production_id_rejected(self):
  for split in ('dev','test'):
   with self.assertRaises(ValueError):check_request({'scope':'infrastructure_trial','training_eligible':False,'row':{'split':split,'scene_id':'cci_infra_r4_x'},'house':{'split':split}})
 def test_v4_does_not_fall_back(self):
  with self.assertRaises(ValueError):check_request({'scope':'infrastructure_trial','training_eligible':False,'row':{'split':'train','scene_id':'cci_infra_r4_x','schema':'v7-production-inputs-20261004-v4','members':['v0_a0']},'house':{'split':'train'}})
 def test_gpu_requires_the_exact_explicit_infrastructure_lease(self):
  for trial,visible in (('0','2'),('1','3'),('1','2,3')):
   with mock.patch.dict(os.environ,{'CCI_INFRA_RENDER_GPU':'2','CCI_INFRASTRUCTURE_TRIAL':trial,'CUDA_VISIBLE_DEVICES':visible},clear=True):
    with self.assertRaises(RuntimeError):NativeContinuousVideo({}, {}, {}, [])
 def test_cpu_default_still_requires_software_selector(self):
  with mock.patch.dict(os.environ,{},clear=True):
   with self.assertRaises(RuntimeError):NativeContinuousVideo({}, {}, {}, [])
 def test_per_world_delivery_bound_is_enforced(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'worlds/x').mkdir(parents=True);(root/'worlds/x/data').write_bytes(b'12345')
   space=AFSSpace(root,root,1000,[{'world_id':'x','scratch_bytes_peak':20,'delivery_bytes_upper':4}])
   with self.assertRaises(RuntimeError):space.check(['x'])
 def test_reject_signal_when_identity_differs(self):
  p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'],start_new_session=True)
  try:
   expected=identity(p.pid);expected['start_ticks']+=1
   with self.assertRaises(RuntimeError):verified_stop(p,expected)
   self.assertIsNone(p.poll())
  finally:
   expected=identity(p.pid);verified_stop(p,expected);p.wait(timeout=5)
 def test_actual_own_sigterm_resume_and_receipt_readback(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);root=base/'workspace';root.mkdir()
   world=root/'worlds/x';world.mkdir(parents=True);payload=world/'payload';payload.write_text('stable')
   record={'path':str(payload),'bytes':payload.stat().st_size,'sha256':sha(payload)}
   (world/'COMPLETE.json').write_text(json.dumps({'files':[record]}))
   profile=base/'profile.json';profile.write_text(json.dumps({'base':str(base),'trial_stop_limit_bytes':150000000000,'workspace_operating_limit_bytes':140000000000,'restart_drill_after_completed_worlds':1}))
   argv=[sys.executable,'-c','import time; time.sleep(30)']
   p=subprocess.Popen(argv,start_new_session=True)
   try:
    with mock.patch.dict(os.environ,{'CCI_TRIAL_PROFILE':str(profile)}):result=monitor_generation(p,root,base/'first.log',argv)
    self.assertEqual(p.wait(timeout=5),-signal.SIGTERM);self.assertEqual(result['type'],'verified_own_SIGTERM_then_resume')
    p=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
    with mock.patch.dict(os.environ,{'CCI_TRIAL_PROFILE':str(profile)}):self.assertIsNone(monitor_generation(p,root,base/'resume.log',[sys.executable,'-c','pass']))
    self.assertEqual(p.wait(timeout=5),0);self.assertEqual(json.loads((root/'OWN_RESTART_READBACK.json').read_text())['status'],'pass')
   finally:
    if p.poll() is None:verified_stop(p,identity(p.pid));p.wait(timeout=5)

if __name__=='__main__':unittest.main()
