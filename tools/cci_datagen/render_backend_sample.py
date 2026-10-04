#!/usr/bin/env python3
"""Native RGB/depth/semantic samples; CPU-only full rendering is also supported."""
import argparse,json,os,time,hashlib,subprocess,resource,statistics
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=24);p.add_argument('--gpu');p.add_argument('--full',action='store_true');p.add_argument('--capture-references',action='store_true');p.add_argument('--source-paths',action='store_true');a=p.parse_args()
 os.nice(19);allowed=sorted(os.sched_getaffinity(0));cpus=int(os.environ.get('CCI_RENDER_CPU_COUNT','1'));os.sched_setaffinity(0,allowed[-cpus:])
 req=json.loads(Path(a.request).read_text());base='/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex'
 def remap(v):
  if isinstance(v,dict):return {k:remap(x) for k,x in v.items()}
  if isinstance(v,list):return [remap(x) for x in v]
  if isinstance(v,str):
   for prefix in [base+'/native',base+'/mirror']:
    if v.startswith(prefix+'/'):return v[len(prefix):]
  return v
 if a.source_paths:req=remap(req)
 if a.gpu:
  if 'CCI_INFRA_HABITAT_DEVICE_ID' not in os.environ or 'CCI_INFRA_HABITAT_PHYSICAL_UUID' not in os.environ:raise ValueError('explicit UUID-proved CUDA ordinal mapping is required')
  if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu:raise ValueError('renderer is not on its exact lease')
  os.environ.update(CCI_INFRA_RENDER_GPU=a.gpu,CCI_INFRASTRUCTURE_TRIAL='1')
 elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('software rendering must hide CUDA')
 from v7_infra_stage import check_request
 from avengine.capture.continuous_native_video import NativeContinuousVideo
 import numpy as np
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);house,scene=check_request(req)
 start=time.monotonic();video=NativeContinuousVideo(house,scene,req['runtime_inputs'],req['assets'],source_material_cache_directory=out/'source_material_cache');setup=time.monotonic()-start
 try:
  if a.full:
   result=video.encode(scene,out/'scene_video.mp4');elapsed=time.monotonic()-start
   record={'scope':'render_only_infrastructure_diagnostic','training_eligible':False,'formal_dataset_count':0,'full_30s_render':True,'frames':result['frame_count'],'setup_seconds':setup,'render_encode_seconds':result['wall_seconds'],'total_seconds':elapsed,'video_bytes':(out/'scene_video.mp4').stat().st_size,'renderer':video.receipt}
  else:
   indices=sorted(set([0,149,300,449]+np.linspace(0,449,a.samples,dtype=int).tolist()));rows=[]
   clip=out/'sample.mp4';cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate','15','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(clip)]
   encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
   try:
    for index in indices:
     t=time.monotonic();video._apply(scene,index);obs=video.sim.get_sensor_observations();rgb=np.asarray(obs['rgb'])[:,:,:3];depth=np.asarray(obs['depth']);semantic=np.asarray(obs['semantic'])
     if not np.isfinite(depth).all() or rgb.std()<2:raise ValueError('invalid native diagnostic frame')
     encoder.stdin.write(np.ascontiguousarray(rgb).tobytes());wall=time.monotonic()-t
     rows.append({'frame_index':index,'wall_seconds':wall,'rgb_sha256':hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),'depth_sha256':hashlib.sha256(np.ascontiguousarray(depth).tobytes()).hexdigest(),'semantic_sha256':hashlib.sha256(np.ascontiguousarray(semantic).tobytes()).hexdigest()})
     if index in (0,149,300,449):np.savez_compressed(out/f'frame_{index:04d}.npz',rgb=rgb,depth=depth,semantic=semantic)
    encoder.stdin.close()
    if encoder.wait():raise RuntimeError('sample encoding failed')
   finally:
    if encoder.poll() is None:
     if not encoder.stdin.closed:encoder.stdin.close()
     encoder.wait()
   costs=[r['wall_seconds'] for r in rows];record={'scope':'render_only_infrastructure_diagnostic','training_eligible':False,'formal_dataset_count':0,'full_30s_render':False,'sample_frames':len(rows),'setup_seconds':setup,'sample_render_encode_seconds':sum(costs),'sample_frame_seconds_median':statistics.median(costs),'sample_frame_seconds_mean':statistics.mean(costs),'estimated_450_frame_seconds':setup+450*statistics.mean(costs),'estimate_note':'sparse pose samples including encoding; setup included once; full 450-frame run is still required','sample_video_bytes':clip.stat().st_size,'estimated_30s_video_bytes':clip.stat().st_size*450/len(rows),'rows':rows,'renderer':video.receipt}
  if a.capture_references:
   refs=[];reference_start=time.monotonic()
   for index in (0,149,300,449):
    t=time.monotonic();video._apply(scene,index);obs=video.sim.get_sensor_observations();rgb=np.asarray(obs['rgb'])[:,:,:3];depth=np.asarray(obs['depth']);semantic=np.asarray(obs['semantic']);wall=time.monotonic()-t
    if not np.isfinite(depth).all() or rgb.std()<2:raise ValueError('invalid diagnostic raw reference')
    np.savez_compressed(out/f'frame_{index:04d}.npz',rgb=rgb,depth=depth,semantic=semantic);refs.append({'frame_index':index,'render_only_seconds':wall,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'depth_sha256':hashlib.sha256(depth.tobytes()).hexdigest(),'semantic_sha256':hashlib.sha256(semantic.tobytes()).hexdigest()})
   record.update(reference_frames=refs,reference_capture_seconds=time.monotonic()-reference_start,reference_note='four deterministic pose replays after measured 450-frame encode; replay/NPZ storage excluded from total_seconds')
 finally:video.close()
 record.update(cpu_affinity=list(os.sched_getaffinity(0)),LP_NUM_THREADS=os.environ.get('LP_NUM_THREADS'),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 paths=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(x in line for x in ('libEGL','libOpenGL','libGL','libnvidia','libLLVM','libgallium','swrast')) and line.split()[-1].startswith('/')))
 record.update(loaded_graphics_libraries=paths,cpu_seconds_whole_process_self=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,cpu_seconds_children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,cpu_time_note='whole process includes imports/setup/reference NPZ; child time includes single-thread encoder; wall total_seconds excludes imports and post-encode reference replay')
 (out/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
