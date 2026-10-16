"""Generate declared static object references with one resident FLUX.2-klein pipeline."""
from v7_object_progress import append as progress
import argparse,json,os,time
from pathlib import Path

def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--only');p.add_argument('--output-name',default='flux');a=p.parse_args();T=a.task_root
    for name in ['tmp','xdg','cuda','torchinductor']:(T/'cache'/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(T/'cache/tmp'),XDG_CACHE_HOME=str(T/'cache/xdg'),CUDA_CACHE_PATH=str(T/'cache/cuda'),TORCHINDUCTOR_CACHE_DIR=str(T/'cache/torchinductor'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_CACHE='/data/models/hub')
    import torch
    from diffusers import Flux2KleinPipeline
    torch.set_num_threads(4);start=time.monotonic();pipe=Flux2KleinPipeline.from_pretrained(str(a.model),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');loaded=time.monotonic()-start;rows=json.loads(a.plan.read_text())['requests'];rows=[r for r in rows if not a.only or a.only in r['asset_id']]
    if a.limit:rows=rows[:a.limit]
    results=[]
    for row in rows:
        D=T/'new'/row['asset_id'];D.mkdir(parents=True,exist_ok=True);target=D/'candidate.png';proof=D/'flux.json'
        if target.exists() and proof.exists():results.append(json.loads(proof.read_text()));continue
        start=time.monotonic()
        try:
            image=pipe(prompt=row['prompt'],height=1024,width=1024,num_inference_steps=28,guidance_scale=1.0,max_sequence_length=512,generator=torch.Generator(device='cuda').manual_seed(row['seed'])).images[0];image.save(target);rec={'status':'pass','asset_id':row['asset_id'],'image':str(target),'prompt':row['prompt'],'seed':row['seed'],'model':str(a.model),'steps':28,'guidance_scale':1.0,'resident_worker':True,'wall_seconds':time.monotonic()-start,'load_seconds_once':loaded,'license':'Apache-2.0'}
        except torch.cuda.OutOfMemoryError:
            rec={'status':'fail','asset_id':row['asset_id'],'reason':'FLUX OOM at batch 1; no lower batch available','wall_seconds':time.monotonic()-start};dump(proof,rec);results.append(rec);break
        except Exception as e:rec={'status':'fail','asset_id':row['asset_id'],'reason':type(e).__name__+':'+str(e),'wall_seconds':time.monotonic()-start}
        dump(proof,rec);results.append(rec);dump(T/(a.output_name+'_results.json'),results);print('FLUX',row['asset_id'],rec['status'],rec['wall_seconds'],flush=True);progress(T,'出图 '+row['asset_id']+'：'+rec['status'],str(proof),'ISNet 抠图，然后 Pixal3D')
    dump(T/(a.output_name+'_results.json'),results)
if __name__=='__main__':main()
