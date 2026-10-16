"""Generate fixed-seed canary references using the installed FLUX.2-klein-4B."""
import argparse, importlib.metadata, json, os, time
from pathlib import Path
import torch
from diffusers import Flux2KleinPipeline
def main():
    p=argparse.ArgumentParser();p.add_argument("--task-root",type=Path,required=True);p.add_argument("--model",type=Path,required=True);a=p.parse_args()
    d=a.task_root/"assets/canary/reference";d.mkdir(parents=True,exist_ok=True)
    prompt="Studio product photograph of one beautiful adult domestic yellow canary, pure warm yellow feathers, slender small finch body, one rounded head, one small short straight conical pale pink beak, two dark alert eyes, anatomically correct two folded wings and a complete slender yellow tail. The bird stands upright with both distinct pink clawed feet gripping a simple smooth horizontal wooden perch. The perch is attached to one slim wooden vertical stem and one round flat stable wooden base, all clearly visible, realistic freestanding indoor pet bird stand. Bird body from beak to tail is about one third the total stand height. Bird and complete freestanding perch only, three-quarter front view showing both feet and the complete tail, whole subject fully inside the image with generous blank margins, neutral relaxed pose, photorealistic feather detail, soft even lighting, plain light gray background. Avoid: cage, extra birds, duplicated head, extra beak, extra wings, extra legs, human, text, watermark, cropped tail, cropped base, metallic sheen, parrot hooked beak."
    pipe=Flux2KleinPipeline.from_pretrained(str(a.model),torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for seed in [20261012,20261013,20261014]:
        start=time.time();path=d/f"candidate_{seed}.png"
        im=pipe(image=None,prompt=prompt,width=1024,height=1024,num_inference_steps=28,guidance_scale=1,max_sequence_length=512,generator=torch.Generator("cuda").manual_seed(seed)).images[0].convert("RGB")
        im.save(path)
        rows.append({"seed":seed,"image":str(path),"seconds":time.time()-start})
        data={"model":"black-forest-labs/FLUX.2-klein-4B","snapshot":str(a.model),"prompt":prompt,"parameters":{"width":1024,"height":1024,"steps":28,"guidance_scale":1,"max_sequence_length":512},"gpu_visible":os.environ.get("CUDA_VISIBLE_DEVICES"),"offload":False,"candidates":rows,"peak_allocated_mib":torch.cuda.max_memory_allocated()/1048576,"peak_reserved_mib":torch.cuda.max_memory_reserved()/1048576,"packages":{x:importlib.metadata.version(x) for x in ["torch","diffusers","Pillow"]}}
        (d/"generation.json").write_text(json.dumps(data,indent=2)+"\n");print("CANDIDATE",seed,rows[-1]["seconds"],flush=True)
if __name__=="__main__":main()
