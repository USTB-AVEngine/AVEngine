"""Use MIT TRELLIS.2 only after three recorded Pixal3D seed failures or a recorded 30-minute setup failure."""
from v7_object_progress import append as progress
import argparse,json,os,sys,time,traceback,threading,shutil
from v7_object_memory import is_cuda_oom
from pathlib import Path

def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--setup-failure',type=Path);p.add_argument('--force-512',action='store_true');p.add_argument('--output-name',default='trellis');a=p.parse_args();T=a.task_root;requests=json.loads(a.plan.read_text())['requests'];previous=json.loads((T/'pixal_results.json').read_text()) if (T/'pixal_results.json').exists() else [];eligible={r['asset_id'] for r in previous if r['status']=='fail' and r.get('fallback_eligible',False) and len({attempt['seed'] for attempt in r.get('attempts',[]) if attempt.get('status')=='fail' and attempt.get('reason')!='CUDA_OOM' and not any(word in str(attempt.get('reason','')) for word in ['ImportError','ModuleNotFoundError','flash_attn','jsonschema'])})==3};setup=None
 if a.setup_failure:
  setup=json.loads(a.setup_failure.read_text());assert setup['stage']=='pixal_setup' and setup['status']=='fail' and setup['elapsed_seconds']>=1800;eligible={r['asset_id'] for r in requests}
 rows=[r for r in requests if r['asset_id'] in eligible];dump(T/(a.output_name+'_eligibility.json'),{'eligible_ids':sorted(eligible),'source':str(T/'pixal_results.json'),'rule':'three unique failed declared Pixal seeds, excluding OOM and import/backend setup failures; or recorded >=1800s setup failure'})
 if not rows:print('TRELLIS_NOT_ELIGIBLE');return
 os.environ.update(ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HUGGINGFACE_HUB_CACHE='/data/models/hub',TORCH_HOME=str(T/'cache/torch'),FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(T/'cache/trellis_autotune.json'),TRITON_CACHE_DIR=str(T/'cache/triton'),TORCH_EXTENSIONS_DIR=str(T/'cache/torch_extensions'));sys.path.insert(0,str(a.code));sys.path.insert(0,str(T/'cache/python_packages'))
 from pixal3d.pipelines import rembg
 for name in dir(rembg):
  if not name.startswith('_') and isinstance(getattr(rembg,name),type):setattr(rembg,name,lambda *_a,**_k:None)
 import torch,numpy as np,o_voxel
 from PIL import Image
 from pixal3d.pipelines import Trellis2ImageTo3DPipeline
 from v7_object_prepare import prepare_one,wrapped
 from v7_object_pixal_worker import within_budget
 from measure_static_upright_correction import load_glb
 setup_start=time.monotonic()
 def setup_timeout():
  dump(T/(a.output_name+'_setup_failure.json'),{'status':'fail','elapsed_seconds':time.monotonic()-setup_start,'reason':'TRELLIS setup exceeded 30 minutes'});os._exit(124)
 timer=threading.Timer(1800,setup_timeout);timer.daemon=True;timer.start()
 pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(a.model));pipeline.low_vram=False;pipeline.cuda();timer.cancel();loaded=time.monotonic()-setup_start;dump(T/(a.output_name+'_worker_load.json'),{'status':'pass','load_seconds_once':loaded,'model':str(a.model)});results=[];prepared=json.loads((T/'prepared_new.json').read_text()) if (T/'prepared_new.json').exists() else [];reduced=a.force_512
 for row in rows:
  if not within_budget(T):break
  D=T/'new'/row['asset_id'];image=D/'input_rgba.png'
  if (D/'trellis.json').exists():
   prior=json.loads((D/'trellis.json').read_text())
   if not (a.force_512 and 'out of memory' in prior.get('reason','').lower()):results.append(prior);continue
   archive=D/'trellis_before_512_retry.json'
   if not archive.exists():shutil.copyfile(D/'trellis.json',archive)
  if not image.exists():continue
  start=time.monotonic();seed=row['seed']+300000;raw=D/('trellis_seed'+str(seed)+'.glb')
  retry=True
  while retry:
   retry=False;resolution=512 if reduced else 1024;raw=D/('trellis_seed'+str(seed)+'_r'+str(resolution)+'.glb')
   try:
    mesh_list,(shape,tex,res)=pipeline.run(Image.open(image),seed=seed,pipeline_type='512' if reduced else '1024_cascade',return_latent=True)
    mesh=mesh_list[0];glb=o_voxel.postprocess.to_glb(vertices=mesh.vertices,faces=mesh.faces,attr_volume=mesh.attrs,coords=mesh.coords,attr_layout=pipeline.pbr_attr_layout,grid_size=res,aabb=[[-.5,-.5,-.5],[.5,.5,.5]],decimation_target=60000,texture_size=2048,remesh=True,remesh_band=1,remesh_project=0,use_tqdm=True);glb.export(raw,extension_webp=False);v,f=load_glb(raw);M=np.eye(4);M[:3,:3]=np.array([[0,0,-1],[0,1,0],[1,0,0]])*row['target_dimensions_m'][1]/float(np.ptp(v[:,1]));normalized=D/'trellis_heading_scaled.glb';wrapped(raw,normalized,M);record={'asset_id':row['asset_id'],'identity':{'object_type':row['object_type']},'realized_attributes':{'form_factor':row['form_factor'],'body_color':row['intended_color']},'geometry':{'resting_pose':{'attachment_surface':row['attachment_surface']}}};meta=prepare_one(record,T,None,source=normalized);meta['license']['geometry_model']='TRELLIS.2 MIT';meta['license']['source_model_license_path']=str(a.model/'LICENSE');meta.update(counting_kind=row['counting_kind'],generation={'pipeline':'flux2_trellis2_static_fallback','model':str(a.model),'license':'MIT','seed':seed,'resolution':resolution,'resident_worker':True,'worker_load_seconds_once':loaded,'coordinate_authority':str(T/'coordinate_authority.json'),'fallback_reason':'three_recorded_pixal_failures' if not setup else 'recorded_30min_pixal_setup_failure'});dump(T/'assets'/row['asset_id']/'asset.json',meta);status='pass_preflight' if not meta['preflight_issues'] else 'fail'
    if status=='pass_preflight':prepared=[r for r in prepared if r['asset_id']!=row['asset_id']]+[meta]
    rec={'asset_id':row['asset_id'],'status':status,'seed':seed,'issues':meta['preflight_issues'],'wall_seconds':time.monotonic()-start}
   except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    if not reduced:reduced=True;retry=True;continue
    rec={'asset_id':row['asset_id'],'status':'fail','reason':'TRELLIS_OOM_after_one_512_retry','seed':seed,'wall_seconds':time.monotonic()-start};dump(T/(a.output_name+'_fatal_oom.json'),rec)
   except Exception as e:
    traceback.print_exc()
    if is_cuda_oom(e):
     torch.cuda.empty_cache()
     if not reduced:
      dump(D/'trellis_oom_1024.json',{'asset_id':row['asset_id'],'status':'fail','reason':str(e),'seed':seed,'resolution':1024});reduced=True;retry=True;continue
     rec={'asset_id':row['asset_id'],'status':'fail','reason':'TRELLIS_OOM_after_one_512_retry: '+str(e),'seed':seed,'resolution':512,'wall_seconds':time.monotonic()-start};dump(T/(a.output_name+'_fatal_oom.json'),rec)
    else:rec={'asset_id':row['asset_id'],'status':'fail','reason':type(e).__name__+':'+str(e),'seed':seed,'wall_seconds':time.monotonic()-start}
  rec['resolution']=resolution;rec['oom_retry_source']=str(D/'trellis_before_512_retry.json') if (D/'trellis_before_512_retry.json').exists() else (str(D/'trellis_oom_1024.json') if (D/'trellis_oom_1024.json').exists() else None);dump(D/'trellis.json',rec);results.append(rec);dump(T/(a.output_name+'_results.json'),results);dump(T/(a.output_name+'_prepared.json'),prepared);print('TRELLIS',row['asset_id'],rec['status'],flush=True);progress(T,'TRELLIS 后备 '+row['asset_id']+'：'+rec['status'],str(D/'trellis.json'),'合并派生资产并做原生验收')
  if (T/(a.output_name+'_fatal_oom.json')).exists():break
 dump(T/(a.output_name+'_results.json'),results);dump(T/(a.output_name+'_prepared.json'),prepared)
if __name__=='__main__':main()
