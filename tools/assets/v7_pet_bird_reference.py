"""Task-owned FLUX.2 pet-bird reference; 3D remains owner-review pending."""
from pathlib import Path
import json,os,hashlib,importlib.metadata,sys
import torch
from diffusers import Flux2KleinPipeline
T=Path(sys.argv[-1]);R=T/'round2';dest=R/'assets/bird_2d';dest.mkdir(parents=True,exist_ok=True)
snapshot=Path('/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294')
prompt="Studio product photograph of one beautiful adult pet budgerigar parrot, bright yellow head, lime green chest, black scalloped wing markings, blue cheek patches, small curved pale beak, two dark alert eyes, anatomically correct folded wings and one long blue-green tail. The bird stands upright with both distinct clawed feet gripping a simple smooth horizontal wooden perch. The perch is attached to one wooden vertical stem and one round flat stable wooden base, all clearly visible. Bird and complete freestanding perch only, three-quarter front view, whole subject fully inside the image, neutral relaxed pose, photorealistic feathers, soft even lighting, plain light gray background. Avoid: cage, extra birds, duplicated head, extra beak, extra wings, extra legs, human, text, watermark, cropped tail, cropped base, metallic sheen."
pipe=Flux2KleinPipeline.from_pretrained(str(snapshot),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
torch.cuda.reset_peak_memory_stats()
image=pipe(image=None,prompt=prompt,width=1024,height=1024,num_inference_steps=28,guidance_scale=1.0,max_sequence_length=512,generator=torch.Generator('cuda').manual_seed(20261004)).images[0].convert('RGB')
image.save(dest/'candidate.png')
data={'schema':'avengine_v7_pet_bird_flux2_candidate_v1','asset_id':'v7_pet_budgerigar_yellow_green_perch_01','model':'black-forest-labs/FLUX.2-klein-4B','revision':snapshot.name,'model_snapshot':str(snapshot),'prompt':prompt,'seed':20261004,'parameters':{'width':1024,'height':1024,'num_inference_steps':28,'guidance_scale':1.0,'max_sequence_length':512},'low_vram':False,'offload':False,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'max_memory_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'max_memory_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'candidate':str(dest/'candidate.png'),'candidate_sha256':hashlib.sha256((dest/'candidate.png').read_bytes()).hexdigest(),'status':'pending_project_owner_canonical_image_review','packages':{k:importlib.metadata.version(k)for k in ['torch','diffusers','Pillow']},'upstream_runner':'USTB-AVEngine/3d@213ff79:tools/run_human_flux2_single.py','adaptation':'own output root and pet-bird prompt; same native FLUX.2 direct GPU residency'}
(dest/'generation.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))
