"""Persistent Pixal3D batch using July runtime, pinned local DINO and NAF inputs."""
import argparse,gc,hashlib,json,os,sys,time,traceback,shutil,subprocess
from pathlib import Path

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--pixal-root',type=Path,required=True);p.add_argument('--selection-file',type=Path);p.add_argument('--status-name',default='mesh_status.json');p.add_argument('--normalize-gate',action='store_true');p.add_argument('--max-seconds',type=float,default=5.1*3600);a=p.parse_args();selection={r['asset_id']:r for r in json.loads(a.selection_file.read_text())} if a.selection_file else None;plan=json.loads((a.task_root/'plan.json').read_text());sys.path.insert(0,str(a.pixal_root));from pixal3d.pipelines import rembg
    rembg.BiRefNet=lambda *args,**kwargs:None
    import v7_animal_variants_pixal_inference as inf
    import torch
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    def local_naf(self):
        if self.naf_model is None:
            device=next(self.model.parameters()).device
            self.naf_model=torch.hub.load('/data/models/torch/hub/valeoai_NAF_main','naf',source='local',pretrained=False,device=device)
            self.naf_model.load_state_dict(torch.load('/data/models/torch/hub/checkpoints/naf_release.pth',map_location=device,weights_only=True));self.naf_model.eval();self.naf_model.requires_grad_(False)
    DinoV3ProjFeatureExtractor._load_naf=local_naf
    dino=Path('/data/models/hub/models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots')/plan['dino_revision'];model=Path('/data/models/hub/models--TencentARC--Pixal3D/snapshots')/plan['mesh_revision']
    for cfg in inf.IMAGE_COND_CONFIGS.values():cfg['model_name']=str(dino)
    status={'started_unix':time.time(),'backend':'Pixal3D','python':sys.executable,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'model':str(model),'dino':str(dino),'model_load_seconds':None,'jobs':[]};dump(a.task_root/'evidence'/a.status_name,status);t0=time.time()
    try:pipe=inf.init_pipeline(str(model),low_vram=False)
    except BaseException as e:status.update(status='environment_failed',error=repr(e),traceback=traceback.format_exc());dump(a.task_root/'evidence'/a.status_name,status);raise
    status['model_load_seconds']=time.time()-t0;status['status']='running';dump(a.task_root/'evidence'/a.status_name,status);torch.cuda.reset_peak_memory_stats()
    for item in plan['animals']:
        if item['mode']!='generated_static' or (selection is not None and item['asset_id'] not in selection):continue
        d=a.task_root/'assets'/item['asset_id'];receipt=d/'reference/segmentation_receipt.json'
        while not receipt.exists():
            if time.time()-status['started_unix']>1800:raise RuntimeError('matting deadline exceeded')
            time.sleep(3)
        seg=json.loads(receipt.read_text());rec={'asset_id':item['asset_id'],'attempts':[]}
        if seg['status']!='pass':rec['status']='rejected_matting';status['jobs'].append(rec);dump(a.task_root/'evidence'/a.status_name,status);continue
        for ordinal,seed in enumerate(item['mesh_seeds']):
            if selection is not None and ordinal<selection[item['asset_id']].get('start_ordinal',0):continue
            existing=d/'generation'/f'seed_{seed}'/'generation_receipt.json'
            if existing.exists():raise RuntimeError('refusing to overwrite an already tried object seed: '+str(seed))
            if shutil.disk_usage(a.task_root).free<150*1024**3:raise RuntimeError('disk reserve exhausted')
            if time.time()-status['started_unix']>a.max_seconds:raise RuntimeError('generation budget exhausted')
            out=d/'generation'/f'seed_{seed}'/'raw.glb';out.parent.mkdir(parents=True,exist_ok=True);start=time.time();attempt={'seed':seed,'ordinal':ordinal,'backend':'Pixal3D','model_revision':plan['mesh_revision'],'dino_revision':plan['dino_revision'],'raw_glb':str(out),'parameters':{'resolution':1024,'manual_fov':.2,'low_vram':False,'max_num_tokens':49152},'input_rgba_sha256':seg['rgba_sha256'],'model_reused':True}
            try:
                inf.run_inference(image_path=str(d/'reference/input_rgba_isnet.png'),output_path=str(out),seed=seed,model_path=str(model),manual_fov=.2,resolution=1024,low_vram=False,pipeline=pipe)
                attempt.update(status='generated_pending_native_qa',raw_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),raw_bytes=out.stat().st_size)
                if a.normalize_gate:
                    dump(out.parent/'generation_receipt.json',attempt)
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
                    result_name='normalization_'+item['asset_id']+'_'+str(seed)+'.json'
                    command=['nice','-n','10','/data/jzy/envs/room-selection-merged-20261003/bin/python',str(Path(__file__).with_name('v7_animal_variants_normalize.py')),'--task-root',str(a.task_root),'--only',item['asset_id'],'--seed',str(seed),'--result-name',result_name]
                    with (out.parent/'normalization.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=700)
                    nr=json.loads((a.task_root/'evidence'/result_name).read_text())[0]
                    attempt['normalization_qualification']=nr
                    if nr['status']!='pass_pending_native':attempt.update(status='failed_geometry_qualification',error=nr.get('error'))

            except BaseException as e:
                attempt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
                if 'out of memory' in str(e).lower():attempt['oom']='batch is already one; cannot reduce batch; stopped';attempt['seconds']=time.time()-start;rec['attempts'].append(attempt);status['jobs'].append(rec);status['status']='oom_stopped';dump(out.parent/'generation_receipt.json',attempt);dump(a.task_root/'evidence'/a.status_name,status);raise
            attempt['seconds']=time.time()-start;rec['attempts'].append(attempt);dump(out.parent/'generation_receipt.json',attempt)
            if attempt['status'].startswith('generated'):rec.update(status='generated_pending_native_qa',selected_seed=seed,selected_glb=str(out));break
            gc.collect();torch.cuda.empty_cache()
        if not rec.get('status'):rec['status']='three_pixal_seeds_failed_trellis_eligible'
        status['jobs'].append(rec);dump(a.task_root/'evidence'/a.status_name,status);print('PIXAL_DONE',item['asset_id'],rec['status'],time.time()-start,flush=True)
    status.update(status='completed_generation',finished_unix=time.time(),peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576);dump(a.task_root/'evidence'/a.status_name,status)
if __name__=='__main__':main()
