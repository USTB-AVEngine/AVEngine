"""Run a resident Pixal3D batch with bounded setup, three seeds, one lower-resolution OOM retry and resource checks."""
from v7_object_progress import append as progress
import argparse,datetime,json,os,sys,time,traceback,threading
from v7_object_memory import is_cuda_oom
from pathlib import Path

def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def within_budget(T):
    budget=json.loads((T/'budget.json').read_text());free=os.statvfs('/data');free=free.f_bavail*free.f_frsize;used=sum(p.stat().st_size for p in T.rglob('*') if p.is_file());reason=None
    if free<budget['minimum_data_free_bytes']:reason='data_free_below_150GiB'
    elif used>budget['maximum_added_bytes']:reason='task_added_bytes_exceed_60GB'
    elif datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(budget['gpu_generation_cutoff_utc']):reason='generation_time_cutoff'
    if reason:dump(T/'generation_budget_stop.json',{'reason':reason,'data_free_bytes':free,'task_used_bytes':used,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return reason is None

def wait_cutout(T,D,follow):
    while True:
        image=D/'input_rgba.png';proof=D/'segmentation.json'
        if image.exists() and proof.exists():return json.loads(proof.read_text()).get('status')=='pass'
        if not follow or not within_budget(T):return False
        jobs=json.loads((T/'JOBS.json').read_text());active=any(r['name'] in ['cutout_batch','flux_batch'] and r['status'] in ['running','starting','waiting_gpu'] for r in jobs)
        if not active:return False
        time.sleep(3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--only');p.add_argument('--follow-cutouts',action='store_true');p.add_argument('--partition-index',type=int,default=0);p.add_argument('--partition-count',type=int,default=1);p.add_argument('--output-name',default='pixal');a=p.parse_args();T=a.task_root
    for name in ['tmp','xdg','cuda','torchinductor','triton','torch_extensions']:(T/'cache'/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',TMPDIR=str(T/'cache/tmp'),XDG_CACHE_HOME=str(T/'cache/xdg'),CUDA_CACHE_PATH=str(T/'cache/cuda'),TORCHINDUCTOR_CACHE_DIR=str(T/'cache/torchinductor'),HF_HUB_CACHE='/data/models/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HUGGINGFACE_HUB_CACHE='/data/models/hub',FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(T/'cache'/(a.output_name+'_autotune.json')),TORCH_HOME=str(T/'cache/torch'),TORCH_EXTENSIONS_DIR=str(T/'cache'/(a.output_name+'_torch_extensions')),TRITON_CACHE_DIR=str(T/'cache'/(a.output_name+'_triton')),MAX_JOBS='4');sys.path.insert(0,str(a.code));sys.path.insert(0,str(T/'cache/python_packages'))
    setup_start=time.monotonic()
    def setup_timeout():
        dump(T/(a.output_name+'_setup_failure.json'),{'stage':'pixal_setup','status':'fail','elapsed_seconds':time.monotonic()-setup_start,'reason':'resident model setup exceeded 30 minutes'});os._exit(124)
    timer=threading.Timer(1800,setup_timeout);timer.daemon=True;timer.start();print('PIXAL_SETUP_START',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
    try:
        from pixal3d.pipelines import rembg
        for name in dir(rembg):
            if not name.startswith('_') and isinstance(getattr(rembg,name),type):setattr(rembg,name,lambda *_a,**_k:None)
        import torch,numpy as np
        from v7_object_pixal_entry import init_pipeline,run_inference
        from v7_object_prepare import prepare_one,wrapped
        from measure_static_upright_correction import load_glb
        torch.set_num_threads(4);pipeline=init_pipeline(str(a.model),low_vram=False)
    except Exception as e:
        timer.cancel();dump(T/(a.output_name+'_setup_error.json'),{'stage':'pixal_setup','status':'fail','elapsed_seconds':time.monotonic()-setup_start,'reason':type(e).__name__+':'+str(e)});raise
    timer.cancel();loaded=time.monotonic()-setup_start;dump(T/(a.output_name+'_worker_load.json'),{'status':'pass','load_seconds_once':loaded,'model':str(a.model),'resident_worker':True});print('PIXAL_SETUP_READY',loaded,flush=True)
    rows=json.loads(a.plan.read_text())['requests'];rows=[r for r in rows if not a.only or a.only in r['asset_id']];assert 0<=a.partition_index<a.partition_count;rows=rows[a.partition_index::a.partition_count]
    if a.limit:rows=rows[:a.limit]
    results=[];prepared=[];oom_reduced=False;fatal_oom=False
    for row in rows:
        if not within_budget(T):break
        D=T/'new'/row['asset_id'];image=D/'input_rgba.png';attempts=[]
        if not wait_cutout(T,D,a.follow_cutouts):
            results.append({'asset_id':row['asset_id'],'status':'not_run','reason':'valid_ISNet_cutout_missing'});dump(T/(a.output_name+'_results.json'),results);continue
        ready=T/'assets'/row['asset_id']/'asset.json'
        if ready.exists():
            existing=json.loads(ready.read_text())
            if not existing.get('preflight_issues'):
                prepared.append(existing);results.append({'asset_id':row['asset_id'],'status':'pass_preflight','attempts':existing.get('generation',{}).get('attempts',[]),'reused_own_derivation':True});dump(T/(a.output_name+'_prepared.json'),prepared);dump(T/(a.output_name+'_results.json'),results);continue
        accepted=None
        for trial in range(3):
            seed=row['seed']+trial*100000;retry=True
            while retry:
                retry=False;resolution=512 if oom_reduced else 1024;raw=D/('mesh_seed'+str(seed)+'_r'+str(resolution)+'.glb');start=time.monotonic()
                try:
                    if not raw.exists():run_inference(str(image),str(raw),seed=seed,model_path=str(a.model),manual_fov=.2,resolution=resolution,pipeline=pipeline)
                    vertices,faces=load_glb(raw);height=float(np.ptp(vertices[:,1]));M=np.eye(4);M[:3,:3]=np.array([[0,0,-1],[0,1,0],[1,0,0]])*float(row['target_dimensions_m'][1])/height
                    normalized=D/('heading_scaled_seed'+str(seed)+'_r'+str(resolution)+'.glb');wrapped(raw,normalized,M)
                    record={'asset_id':row['asset_id'],'identity':{'object_type':row['object_type']},'realized_attributes':{'form_factor':row['form_factor'],'body_color':row['intended_color']},'geometry':{'resting_pose':{'attachment_surface':row['attachment_surface']}}}
                    meta=prepare_one(record,T,None,source=normalized);attempt={'seed':seed,'status':'pass' if not meta['preflight_issues'] else 'fail','resolution':resolution,'mesh':str(raw),'issues':meta['preflight_issues'],'wall_seconds':time.monotonic()-start};attempts.append(attempt)
                    meta.update(counting_kind=row['counting_kind'],generation={'pipeline':'flux2_pixal3d_static_v1','reference':str(image),'flux_manifest':str(D/'flux.json'),'pixal_model':str(a.model),'pixal_code_readonly':str(a.code),'seed':seed,'resolution':resolution,'worker_load_seconds_once':loaded,'resident_worker':True,'attention_backend':'sdpa','heading_authority':'same source -Z to +X yaw as historical AVEngine static finalizer; support-plane tilt measured after export','attempts':attempts});dump(ready,meta)
                    if not meta['preflight_issues']:accepted=meta
                except torch.cuda.OutOfMemoryError:
                    attempts.append({'seed':seed,'status':'fail','reason':'CUDA_OOM','resolution':resolution,'wall_seconds':time.monotonic()-start});torch.cuda.empty_cache()
                    if oom_reduced:fatal_oom=True
                    else:oom_reduced=True;retry=True
                except Exception as e:
                    traceback.print_exc()
                    if is_cuda_oom(e):
                        attempts.append({'seed':seed,'status':'fail','reason':'CUDA_OOM','extension_message':str(e),'resolution':resolution,'wall_seconds':time.monotonic()-start});torch.cuda.empty_cache()
                        if oom_reduced:fatal_oom=True
                        else:oom_reduced=True;retry=True
                    else:attempts.append({'seed':seed,'status':'fail','reason':type(e).__name__+':'+str(e),'resolution':resolution,'wall_seconds':time.monotonic()-start})
            if accepted or fatal_oom or not within_budget(T):break
        failed_seeds={r['seed'] for r in attempts if r['status']=='fail' and r.get('reason')!='CUDA_OOM'}
        result={'asset_id':row['asset_id'],'status':'pass_preflight' if accepted else 'fail','attempts':attempts,'max_seeds':3,'fallback_eligible':not accepted and len(failed_seeds)==3,'fallback_status':'pending_TRELLIS_after_three_failed_seeds' if not accepted and len(failed_seeds)==3 else 'not_needed'};dump(D/'pixal.json',result);results.append(result)
        if accepted:prepared.append(accepted)
        dump(T/(a.output_name+'_results.json'),results);dump(T/(a.output_name+'_prepared.json'),prepared);print('PIXAL',row['asset_id'],result['status'],flush=True);progress(T,'3D '+row['asset_id']+'：'+result['status'],str(D/'pixal.json'),'原生 MP3D/HM3D；三种子失败才转 TRELLIS.2')
        if fatal_oom:dump(T/(a.output_name+'_fatal_oom.json'),{'reason':'second OOM after one lower-resolution retry','asset_id':row['asset_id'],'attempts':attempts});break
    dump(T/(a.output_name+'_results.json'),results);dump(T/(a.output_name+'_prepared.json'),prepared)
if __name__=='__main__':main()
