#!/usr/bin/env python3
"""Two-house native RLR + FOA convolution calibration; no render, no dataset QA.

It is not a complete v6/v7 world or a frozen test member. Both machines run
identical explicit single-thread settings and analytic calibration dry sound.
"""
import argparse,json,os,time
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'src'))
from resources import write_json
from generate import sha
from avengine.acoustics.runtime import load_compiled_acoustic_scene,RLRSimulationConfig,RuntimeAnchor,simulate_compiled_acoustic_scene

p=argparse.ArgumentParser();p.add_argument('--mirror-root',default='/');p.add_argument('--native-root',default='/data/avengine_external');p.add_argument('--output',required=True);p.add_argument('--simulation',required=True);a=p.parse_args()
mirror=Path(a.mirror_root);native=Path(a.native_root);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
b=mirror/'data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003'
selected=json.loads((b/'report_final/recommended_houses.json').read_text())['houses'][:2]
settings=json.loads(Path(a.simulation).read_text())
# Accept the existing request's explicit simulation mapping, not a new acoustic preset.
raw=settings.get('simulation',settings);raw=dict(raw);raw['thread_count']=1;raw['temporal_coherence']=False;raw['channel_layout']={'type':'ambisonics','channel_count':4}
config=RLRSimulationConfig.from_mapping(raw);config.validate()
runtime={'runtime_mode':'current-installed','runtime_prefix':str(native/'runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z'),'magnum_python_site':str(native/'runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages'),'rlr_sdk_root':str(native/'rlr-sdk/RLRAudioPropagationPkg')}
worlds=[];rate=int(config.sample_rate_hz)
for h in selected:
 started=time.monotonic();root=output/h['house'];root.mkdir(exist_ok=True)
 measure=json.loads((b/'measurement_final/houses'/h['house']/'result.json').read_text());room=measure['input']['acoustic_rooms'][0];placement=room['placement']
 scene=load_compiled_acoustic_scene(b/'measurement_final/houses'/h['house']/'package/manifest.json',allow_nonpassing_research_qa=True)
 listener=RuntimeAnchor('listener',tuple(placement['camera_m']))
 mixed=np.zeros((rate*10,4),dtype=np.float64);reports=[]
 for i,key in enumerate(['source_1_m','source_2_m']):
  source=RuntimeAnchor(f'source{i+1}',tuple(placement[key]))
  t=time.monotonic();result=simulate_compiled_acoustic_scene(scene,config,source=source,listener=listener,**runtime)
  ir=np.asarray(result.samples,dtype=np.float32)
  if ir.shape[0]==4:ir=ir.T
  if ir.ndim!=2 or ir.shape[1]!=4:raise RuntimeError('unexpected FOA IR shape: '+str(ir.shape))
  sf.write(root/f'ir{i+1}.wav',ir,rate,subtype='FLOAT')
  time_axis=np.arange(rate,dtype=np.float64)/rate
  dry=.05*np.sin(2*np.pi*(220+220*i)*time_axis)*np.sin(np.pi*time_axis)**2
  onset=(1+3*i)*rate
  for channel in range(4):
   wet=fftconvolve(dry,ir[:,channel],mode='full');end=min(len(mixed)-onset,len(wet));mixed[onset:onset+end,channel]+=wet[:end]
  reports.append({'source':key,'native_rir_and_setup_seconds':time.monotonic()-t,'ir_shape':list(ir.shape),'ir_file_sha256':sha(root/f'ir{i+1}.wav'),'runtime_configuration':result.runtime.get('configuration_readback'),'indirect_ray_efficiency':result.indirect_ray_efficiency})
 sf.write(root/'foa.wav',mixed.astype(np.float32),rate,subtype='FLOAT')
 write_json(root/'calibration.json',{'house':h['house'],'room':room['room_label'],'simulation':config.to_dict(),'sources':reports,'scope':'CPU acoustic calibration only; no Habitat RGB, no actual v6 QA'})
 worlds.append({'house':h['house'],'foa':str(root/'foa.wav'),'foa_sha256':sha(root/'foa.wav'),'foa_samples_sha256':__import__('hashlib').sha256(mixed.astype('<f4').tobytes()).hexdigest(),'calibration':str(root/'calibration.json'),'wall_seconds':time.monotonic()-started,'native_pairs':2,'video':'not_run_no_authorized_gpu','questions':'not_run_no_capture'})
write_json(output/'SUMMARY.json',{'status':'acoustic_calibration_complete','worlds':worlds,'thread_count':1,'duration_seconds':10,'schema':'cci_cpu_calibration_v1','is_complete_v6_world':False})
print(json.dumps(worlds,ensure_ascii=False))
