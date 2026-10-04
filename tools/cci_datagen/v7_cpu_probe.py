#!/usr/bin/env python3
"""One CPU native RIR + real v7 formant dry audio, no EGL/CUDA context."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from v7_infra_stage import check_request

def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--source-paths',action='store_true');a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('native correctness probe must have no CUDA lease')
 os.nice(19);os.sched_setaffinity(0,[max(os.sched_getaffinity(0))]);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 request=json.loads(Path(a.request).read_text());b0='/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex'
 def source(v):
  if isinstance(v,dict):return {k:source(x) for k,x in v.items()}
  if isinstance(v,list):return [source(x) for x in v]
  if isinstance(v,str):
   for root,replacement in ((b0+'/native/data','/data'),(b0+'/mirror/data','/data')):
    if v.startswith(root+'/'):return replacement+v[len(root):]
  return v
 if a.source_paths:request=source(request)
 working,scene=check_request(request)
 from avengine.timeline.continuous_pilot import FormantSpeech,PositionRIRCache
 from tools.timeline.pilot_continuous_scenes import native_renderer
 from avengine.timeline.continuous_scene import partitioned_convolve,raw_world_to_camera_sn3d
 from avengine.spatial_audio.audio import write_float32_wav
 start=time.monotonic();tts=FormantSpeech();dry=tts.synthesize('The person is speaking in this room.','en-us+m1')
 np.save(out/'dry.npy',dry,allow_pickle=False)
 renderer,identity=native_renderer(request['runtime_inputs'],working,threads=1)
 (out/'native_RLR_setup.json').write_text(json.dumps(renderer.setup_report,indent=2)+'\n')
 cache=PositionRIRCache(out/'rir_cache',renderer,identity)
 rir=cache.get(scene['sources']['voice1']['trajectory'][0]['position_m']);cache.db.close()
 foa=raw_world_to_camera_sn3d(partitioned_convolve(dry,[rir],[0]),scene['camera_forward'])
 write_float32_wav(out/'foa.wav',foa,16000,channel_axis=0)
 result={'status':'pass','dry_shape':list(dry.shape),'dry_float64_sha256':digest(dry),'rir_shape':list(rir.shape),'rir_sha256':digest(rir),'foa_shape':list(foa.shape),'foa_float32_sha256':digest(foa.astype(np.float32)),'rlr_threads_requested':1,'tts_receipt':tts.receipt,'wall_seconds':time.monotonic()-start,'full_native_worlds':0,'rendering':'not_run_CPU_only','scientific_admission':False,'source_path_mapping':a.source_paths}
 (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
