"""Replay every native frame with bounded resource lifetime; unchanged gates."""
import argparse,json,pathlib,sys,subprocess,time,os
from avengine.assets.animal_motion.native import run_habitat
from avengine.assets.animal_motion.checks import frame_checks,direction_negative_controls


def main():
 p=argparse.ArgumentParser();p.add_argument('--blender',type=pathlib.Path,default=pathlib.Path(os.environ.get('AVENGINE_BLENDER_BINARY','blender')));p.add_argument('--inventory',type=pathlib.Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--house',type=pathlib.Path,required=True);p.add_argument('--runtime',type=pathlib.Path,required=True);p.add_argument('--route',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--speed',type=float,default=.03);p.add_argument('--period',type=float,default=1.6);p.add_argument('--seconds',type=float,default=4.8);p.add_argument('--block-size',type=int,default=12);p.add_argument('--mode',choices=['walk_animation','slide_no_animation'],default='walk_animation');p.add_argument('--worker-indices');p.add_argument('--pose-stream',type=pathlib.Path);p.add_argument('--skin-stream',type=pathlib.Path);p.add_argument('--max-frames',type=int);a=p.parse_args()
 record=next(x for x in json.loads(a.inventory.read_text())['assets'] if x['asset_id']==a.asset)
 if a.worker_indices:
  result=run_habitat(record,json.loads(a.house.read_text()),json.loads(a.runtime.read_text()),json.loads(a.route.read_text()),a.output,speed=a.speed,period=a.period,seconds=a.seconds,mode=a.mode,frame_indices=[int(x) for x in a.worker_indices.split(',')],pose_stream=a.pose_stream,skin_stream=a.skin_stream)
  if result['status']=='failed':raise SystemExit(2)
  return
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'textures').mkdir();(a.output/'material_cache').mkdir();start=time.monotonic();n=round(a.seconds*30)+1
 all_indices=list(range(n if a.max_frames is None else min(a.max_frames,n)));chunks=[all_indices[i:i+a.block_size] for i in range(0,len(all_indices),a.block_size)]
 if len(chunks[-1])==1:chunks[-2].extend(chunks.pop())
 results=[];shape_rows=[];error=None;pose_stream=a.output/'working_pose.glb';skin_stream=a.output/'working_skin_stream.npz'
 for k,indices in enumerate(chunks):
  block=a.output/f'block_{k:03d}';block.mkdir();(block/'textures').symlink_to(a.output/'textures',target_is_directory=True);(block/'material_cache').symlink_to(a.output/'material_cache',target_is_directory=True)
  cmd=[sys.executable,str(pathlib.Path(__file__).resolve()),'--blender',str(a.blender),'--inventory',str(a.inventory),'--asset',a.asset,'--house',str(a.house),'--runtime',str(a.runtime),'--route',str(a.route),'--output',str(block),'--speed',str(a.speed),'--period',str(a.period),'--seconds',str(a.seconds),'--mode',a.mode,'--worker-indices',','.join(map(str,indices)),'--pose-stream',str(pose_stream),'--skin-stream',str(skin_stream)]
  with (block/'native.log').open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
  if not (block/'native_motion.json').exists():error='native_worker_did_not_write_receipt';break
  res=json.loads((block/'native_motion.json').read_text());results.append(res)
  print('BLOCK_NATIVE',k,'status',res['status'],'frames',len(res.get('frames',[])),flush=True)
  if rc or res['status']=='failed' or len(res.get('frames',[]))!=len(indices):error=res.get('error','native_worker_rejected');break
  qa=block/'skin_qa';tool=pathlib.Path(__file__).with_name('blender_check_cat_native_skin.py');case=a.inventory.parent
  with (block/'skin_qa.log').open('w') as f:rc=subprocess.run([str(a.blender),'--background','--factory-startup','--threads','1','--python-exit-code','2','--python',str(tool),'--','--case',str(case),'--stream',str(skin_stream),'--output',str(qa)],stdout=f,stderr=subprocess.STDOUT).returncode
  if not (qa/'skin_checks.json').exists():error='native_skin_qa_did_not_write_receipt';break
  qs=json.loads((qa/'skin_checks.json').read_text());shape_rows.extend(qs['frames'])
  print('BLOCK_SKIN',k,qs['status'],qs['failures'],flush=True)
  if rc or qs['status']!='pass':error='actual_native_skin_rejected:'+','.join(qs['failures']);break
  (a.output/'progress.json').write_text(json.dumps({'completed_frames':sum(len(x['frames']) for x in results),'shape_frames':len(shape_rows),'total_frames':n,'blocks_complete':k+1,'last_completed_utc':time.time()},indent=2))
 frames=[f for r in results for f in r.get('frames',[])];result={k:v for k,v in results[0].items() if k not in ('frames','checks','direction_negative_controls','images')} if results else {'asset_id':a.asset}
 result.update(frames=frames,images=[x for r in results for x in r.get('images',[])],frame_count=len(frames),native_skin_checks={'status':'pass' if error is None and len(shape_rows)==n and all(x['status']=='pass' for x in shape_rows) else 'fail','frames':shape_rows},bounded_replay={'block_size':a.block_size,'worker_count':len(results),'indices':[f['frame_index'] for f in frames],'pose_storage':'Only five keyframes are retained; each other drawable was loaded under a distinct path, checked against actual native resource bounds, and its private backing inode was subsequently overwritten. Stream links are not archived poses.','skin_storage':'One bounded private skin stream; per-frame hashes, rest baseline and quantitative checks retained in each block/skin_qa/skin_checks.json.'})
 if error is not None:result.update(status='failed',error=error)
 elif [f['frame_index'] for f in frames]!=list(range(n)):result.update(status='failed',error='incomplete_full_native_replay')
 else:
  checks=frame_checks(frames,native=True,mode=a.mode);neg=direction_negative_controls(frames)
  if not all(f['complete_silhouette_in_frame'] for f in frames):checks['status']='fail';checks['failures'].append('native_silhouette_clipped_or_under_100_pixels')
  result.update(status='geometry_pass_audio_pending' if checks['status']==neg['status']=='pass' else 'geometry_fail',checks=checks,direction_negative_controls=neg,max_emitter_error_m=max(f['native_emitter_readback_error_m'] for f in frames),min_visible_pixels=min(f['visible_pixels'] for f in frames))
 result['elapsed_s']=time.monotonic()-start;(a.output/'native_motion.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k in ('asset_id','status','error','frame_count','elapsed_s')},flush=True)
 if result['status']!='geometry_pass_audio_pending':raise SystemExit(2)
if __name__=='__main__':main()
