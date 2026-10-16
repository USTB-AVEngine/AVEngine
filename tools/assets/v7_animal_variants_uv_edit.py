"""One bounded FLUX-guided UV repair of a rejected texture; preserve every geometric buffer."""
import argparse,copy,hashlib,io,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--build-only',action='store_true');p.add_argument('--build-tag',default='v1');a=p.parse_args();items=json.loads(a.selection.read_text());plan=json.loads((a.task_root/'plan.json').read_text());start=time.time();records=[]
    if not a.build_only:
        import torch
        from diffusers import Flux2KleinPipeline
        pipe=Flux2KleinPipeline.from_pretrained(str(a.model),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');load=time.time()-start
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from avengine.assets.source_materials import read_glb,image_bytes,prepare_source_glb,inspect_source_glb
        from v7_human_roster_materials import append_image,write_glb,geometry_animation_digest
    targets={'black':'uniform natural deep black fur or feathers, subtle dark shading','white':'uniform bright neutral white fur or feathers, dominant pure white, minimal gray shading; never cream, green or yellow','gray':'natural light neutral silver-gray fur, a light gray coat, not black, not blue or purple','brown':'natural rich medium warm brown fur, clearly chestnut brown, not black, not gray','green':'vivid natural grass-green parrot body feathers; keep existing dark wing markings and natural yellow face; main body plumage strongly green','blue':'natural bright sky-blue parrot body feathers with a clean white face; keep existing dark wing markings; no green or yellow feathers','yellow':'uniform bright natural canary-yellow feathers, dominant saturated yellow, no gray or green areas'}
    for index,item in enumerate(items):
        D=a.task_root/'assets'/item['asset_id'];attempt=int(item.get('attempt_index',1));assert 1<=attempt<=2
        E=D/'appearance_repair'/('flux_uv_one_attempt' if attempt==1 else 'flux_uv_attempt_2');E.mkdir(parents=True,exist_ok=True);source_texture=Path(item['source_texture']);seed=item['uv_seed'];prompt='Edit this exact flat UV albedo texture atlas of '+item['body_name']+'. Keep every UV island, edge and feature in exactly the same position. Change only the coat/plumage to '+item.get('target_description',targets[item['label_color']])+'. Preserve small eyes, nose, beak, mouth, paw pads, claws and skin details in their original positions. Preserve realistic fur or feather detail. Output the same flat UV atlas, no animal photograph, no pose, no background composition, no text. '+item.get('biological_color_note','')
        if not a.build_only:
            if (E/'generation_receipt.json').exists():raise RuntimeError('UV repair already tried; refusing another draw')
            t=time.time();image=Image.open(source_texture).convert('RGB');edited=pipe(image=[image.resize((1024,1024),Image.Resampling.LANCZOS)],prompt=prompt,width=1024,height=1024,num_inference_steps=item.get('steps',28),guidance_scale=1,max_sequence_length=512,generator=torch.Generator('cuda').manual_seed(seed)).images[0].convert('RGB');edited.save(E/'edited_uv.png');receipt={'asset_id':item['asset_id'],'model':'black-forest-labs/FLUX.2-klein-4B','revision':plan['reference_revision'],'snapshot':str(a.model),'license':'Apache-2.0','method':'bounded FLUX reference-guided UV edit; at most two UV seeds plus original reference, no RGB multiplication','attempt_index':attempt,'original_reference_plus_uv_seed_count':1+attempt,'source_texture':str(source_texture),'source_texture_sha256':sha(source_texture),'prompt':prompt,'seed':seed,'parameters':{'steps':item.get('steps',28),'guidance_scale':1,'width':1024,'height':1024,'max_sequence_length':512},'output_texture':str(E/'edited_uv.png'),'output_texture_sha256':sha(E/'edited_uv.png'),'seconds':time.time()-t,'model_reused':True,'persistent_load_seconds':load,'reason':item['reason'],'before_native_receipts':item.get('before_native_receipts',[])};dump(E/'generation_receipt.json',receipt);print('UV_EDIT',item['asset_id'],flush=True);records.append({'asset_id':item['asset_id'],'status':'generated_texture_pending_native'});continue
        meta=json.loads((D/'asset.json').read_text());image=Image.open(source_texture).convert('RGB');edited=Image.open(E/'edited_uv.png').convert('RGB').resize(image.size,Image.Resampling.LANCZOS);buf=io.BytesIO();edited.save(buf,format='PNG');payload=buf.getvalue();proof=[]
        for filename in ['runtime.glb','body_only.glb','motion_runtime.glb']:
            source=D/filename
            if not source.exists():continue
            backup=E/('before_'+a.build_tag)/filename;backup.parent.mkdir(exist_ok=True)
            if not backup.exists():backup.write_bytes(source.read_bytes())
            d,b=read_glb(backup);before=geometry_animation_digest(d,b);doc=copy.deepcopy(d);animal_materials=[i for i,m in enumerate(doc['materials']) if not m.get('name','').startswith(('OwnedTaskMatteWood','OwnedTask'))];indices=sorted({doc['textures'][doc['materials'][i]['pbrMetallicRoughness']['baseColorTexture']['index']]['source'] for i in animal_materials if 'baseColorTexture' in doc['materials'][i].get('pbrMetallicRoughness',{})})
            if len(indices)!=1:raise RuntimeError('ambiguous animal UV image selection:'+filename)
            blob=append_image(doc,b,indices[0],payload);staged=E/('edited_'+a.build_tag+'_'+filename);write_glb(staged,doc,blob);prepared,matrec=prepare_source_glb(staged,asset_id=meta['asset_id'],entity_class='articulated_animal',output_directory=E/(filename+'_'+a.build_tag+'_materials'));source.write_bytes(Path(prepared).read_bytes());nd,nb=read_glb(source);assert geometry_animation_digest(nd,nb)==before;proof.append({'part':filename,'before_glb':str(backup),'before_sha256':sha(backup),'after_sha256':sha(source),'before_geometry_skin_all_animation_sha256':before,'after_geometry_skin_all_animation_sha256':geometry_animation_digest(nd,nb),'geometry_rig_all_clips_unchanged':True,'material_receipt':matrec})
        meta.update(status='pending_native_acceptance',runtime_sha256=sha(D/'runtime.glb'),glb_bytes=(D/'runtime.glb').stat().st_size,material_inspection=inspect_source_glb(D/'runtime.glb'),appearance_repair={'method':'bounded FLUX-guided UV edit after machine rejection; all geometry and clips unchanged','attempt_index':attempt,'receipt':str(E/'generation_receipt.json'),'proof':str(E/('equivalence_'+a.build_tag+'.json')),'intended_label':item['label_color']});dump(E/('equivalence_'+a.build_tag+'.json'),proof);dump(D/'asset.json',meta);records.append({'asset_id':item['asset_id'],'status':'texture_repaired_pending_native'});print('UV_BUILD',item['asset_id'],flush=True)
    dump(a.task_root/'evidence'/('uv_edit_build_results.json' if a.build_only else 'uv_edit_generation_results.json'),records)
if __name__=='__main__':main()
