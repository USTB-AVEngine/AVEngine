"""Generate the declared animal references with one resident FLUX model."""
import argparse,hashlib,json,os,time,shutil
from pathlib import Path
import torch
from diffusers import Flux2KleinPipeline

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=int,default=28);a=p.parse_args();plan=json.loads((a.task_root/'plan.json').read_text());t0=time.time()
    pipe=Flux2KleinPipeline.from_pretrained(str(a.model),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');load=time.time()-t0;torch.cuda.reset_peak_memory_stats();rows=[]
    for item in plan['animals']:
        if item['mode']!='generated_static':continue
        free=shutil.disk_usage(a.task_root).free
        if free<150*1024**3:raise RuntimeError(f'disk reserve exhausted: {free}')
        d=a.task_root/'assets'/item['asset_id']/'reference';d.mkdir(parents=True,exist_ok=True);out=d/'candidate.png';start=time.time()
        im=pipe(image=None,prompt=item['prompt'],width=1024,height=1024,num_inference_steps=a.steps,guidance_scale=1,max_sequence_length=512,generator=torch.Generator('cuda').manual_seed(item['reference_seed'])).images[0].convert('RGB');im.save(out)
        receipt={'asset_id':item['asset_id'],'model':'black-forest-labs/FLUX.2-klein-4B','revision':plan['reference_revision'],'snapshot':str(a.model),'prompt':item['prompt'],'seed':item['reference_seed'],'parameters':{'steps':a.steps,'guidance_scale':1,'width':1024,'height':1024,'max_sequence_length':512},'license':'Apache-2.0','image':str(out),'image_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seconds':time.time()-start,'model_load_seconds':load,'model_reused':True,'offload':False,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576}
        (d/'generation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');rows.append(receipt);(a.task_root/'evidence/reference_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print('FLUX_DONE',item['asset_id'],receipt['seconds'],flush=True)
    (a.task_root/'evidence/reference_complete.json').write_text(json.dumps({'count':len(rows),'seconds':time.time()-t0,'model_load_seconds':load})+'\n')
if __name__=='__main__':main()
