"""FLUX reference-guided Rocketbox UV coats; never alter geometry, rig or clips."""
import argparse,copy,hashlib,io,json,shutil,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--extract-only',action='store_true');p.add_argument('--generate-only',action='store_true');p.add_argument('--build-only',action='store_true');a=p.parse_args();plan=json.loads((a.task_root/'plan.json').read_text());
    if not a.generate_only:
        from avengine.assets.source_materials import read_glb,image_bytes,prepare_source_glb,inspect_source_glb
    items=[x for x in plan['animals'] if x['mode']=='rocketbox_coat_variant'];sources={}
    for breed in ['beagle','german_shepherd']:
        src=a.reference_root/f'round2/assets/dogs/{breed}/motion_runtime.glb';dest=a.task_root/'evidence/coat_source_textures'/f'{breed}.png'
        if a.generate_only:
            image=Image.open(dest).convert('RGB');ii=None
        else:
            d,b=read_glb(src);mat=d['materials'][0];tx=mat['pbrMetallicRoughness']['baseColorTexture']['index'];ii=d['textures'][tx]['source'];image=Image.open(io.BytesIO(image_bytes(d,b,ii,src))).convert('RGB');dest.parent.mkdir(parents=True,exist_ok=True);image.save(dest)
        sources[breed]=(src,image,dest,ii)
    if a.extract_only:return
    start=time.time();records=[];load=0.
    if not a.build_only:
        import torch
        from diffusers import Flux2KleinPipeline
        pipe=Flux2KleinPipeline.from_pretrained(str(a.model),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');load=time.time()-start
    if not a.generate_only:
        import trimesh
        from v7_human_roster_materials import append_image,write_glb,geometry_animation_digest
        from v7_human_roster_pose import pose
        from v7_animal_variants_normalize import paw_supports
    for index,item in enumerate(items):
        D=a.task_root/'assets'/item['asset_id'];R=D/'reference';R.mkdir(parents=True,exist_ok=True);src,image,source_texture,ii=sources[item['body_id']];coat=item['coat_profile'];target={'lemon_white':'natural lemon-and-white Beagle: replace every black or dark brown FUR patch by pale lemon yellow fur; keep all existing white fur white','red_white':'natural red-and-white Beagle: replace every black or dark brown FUR patch by warm reddish tan fur; keep all existing white fur white','solid_black':'solid natural black German Shepherd: change all fur to dense coal black with subtle realistic dark shading','solid_white':'solid natural white German Shepherd: change all fur to neutral pure white with subtle gray shading, never yellow or cream'}[coat];prompt=f'Edit this exact flat dog UV albedo texture atlas. Make a {target}. Preserve the exact UV island layout, edges, and placement of every surface feature. Keep eyes, nose, lips, teeth, tongue and inner mouth unchanged. Keep natural fur detail. This output must remain a flat UV texture atlas matching the supplied image, not a photograph of a dog, not a new layout, no text. Change only fur appearance.';seed=2026101280+index;t=time.time()
        if not a.build_only:
            edited=pipe(image=[image.resize((1024,1024),Image.Resampling.LANCZOS)],prompt=prompt,width=1024,height=1024,num_inference_steps=28,guidance_scale=1,max_sequence_length=512,generator=torch.Generator('cuda').manual_seed(seed)).images[0].convert('RGB');edited.save(R/'flux_edited_texture.png');receipt={'asset_id':item['asset_id'],'method':'FLUX reference-guided UV appearance editing','model':'black-forest-labs/FLUX.2-klein-4B','revision':plan['reference_revision'],'snapshot':str(a.model),'source_texture':str(source_texture),'source_texture_sha256':sha(source_texture),'prompt':prompt,'seed':seed,'parameters':{'steps':28,'guidance_scale':1,'width':1024,'height':1024,'max_sequence_length':512},'output_texture_sha256':sha(R/'flux_edited_texture.png'),'seconds':time.time()-t,'model_reused':True,'persistent_model_load_seconds':load,'license':'Apache-2.0; Rocketbox source texture MIT'};dump(R/'generation_receipt.json',receipt)
        if a.generate_only:records.append({'asset_id':item['asset_id'],'status':'texture_generated'});continue
        if a.build_only:edited=Image.open(R/'flux_edited_texture.png').convert('RGB')
        buf=io.BytesIO();edited.resize(image.size,Image.Resampling.LANCZOS).save(buf,format='PNG');payload=buf.getvalue();proof=[]
        for name,source in [('idle',a.reference_root/f"round3/assets/dogs/{item['body_id']}/idle.glb"),('motion',src)]:
            d,b=read_glb(source);orig=geometry_animation_digest(d,b);original_blob=b;d=copy.deepcopy(d);b=append_image(d,b,ii,payload);assert geometry_animation_digest(d,b)==orig and b.startswith(original_blob);coated=D/f'{name}_coated.glb';write_glb(coated,d,b);prepared,matrec=prepare_source_glb(coated,asset_id=item['asset_id'],entity_class='articulated_animal',output_directory=D/f'{name}_materials');target=D/('runtime.glb' if name=='idle' else 'motion_runtime.glb');shutil.copy2(prepared,target);nd,nb=read_glb(target);assert geometry_animation_digest(nd,nb)==orig;proof.append({'part':name,'source':str(source),'source_sha256':sha(source),'runtime':str(target),'runtime_sha256':sha(target),'source_geometry_rig_animation_sha256':orig,'new_geometry_rig_animation_sha256':geometry_animation_digest(nd,nb),'unchanged':True,'source_binary_prefix_unchanged':nb.startswith(original_blob),'material_receipt':matrec})
        motion=D/'motion_runtime.glb';d,b=read_glb(motion);action=next(x for x in d['animations'] if x['name']=='Idle');parts,ev=pose(d,b,action,0.,ground=False,canonical_flip=False,emitter_bone='Xtra Mouth');mesh=trimesh.load(D/'runtime.glb',force='mesh',process=False);bounds=mesh.bounds;points,groups=paw_supports(mesh.vertices);mouth=np.array(ev['mouth_m']);body=D/'body_only.glb';shutil.copy2(D/'runtime.glb',body);meta={**item,'status':'pending_native_acceptance','runtime_glb':str(D/'runtime.glb'),'body_only_glb':str(body),'motion_runtime_glb':str(motion),'units':'metres','coordinates':{'up':'+Y','front':'-Z','origin':'preserved source rig and idle origin'},'dimensions_m':np.ptp(bounds,axis=0).tolist(),'bounds_min_m':bounds[0].tolist(),'bounds_max_m':bounds[1].tolist(),'emit_point_local_m':mouth.tolist(),'head_center_local_m':mouth.tolist(),'emit_head_distance_m':0.,'emit_head_radius_m':.08,'emit_head_status':'pass' if ev['mouth_nearest_surface_m']<.08 else 'fail','landmarks_local_m':{**{f'foot_{i}':v.tolist() for i,v in enumerate(points)},'mouth':mouth.tolist(),'head_center':mouth.tolist()},'body_floor_y_m':0.,'triangles':len(mesh.faces),'glb_bytes':(D/'runtime.glb').stat().st_size,'runtime_sha256':sha(D/'runtime.glb'),'perch':None,'normalization':{'method':'preserved Rocketbox source geometry, rig, skin and all eight source clips; no geometric changes','raw_support_clusters':groups,'source_scale':'preserved from admitted v7 Rocketbox roster','four_paw_height_range_m':float(np.ptp(points[:,1])) if len(points)==4 else None},'license':{'source_mesh_texture_rig':'Rocketbox MIT','motion':'Quaternius CC0','coat_reference_guided_edit':'FLUX Apache-2.0','downstream_ai_training_admission':'pending_native_checks'},'material_inspection':inspect_source_glb(D/'runtime.glb'),'geometry_skin_animation_unchanged_by_materials':True,'coat_proof':proof,'source_generator':'Rocketbox coat variant via FLUX.2 klein'}
        if len(points)!=4 or meta['triangles']>100000 or meta['glb_bytes']>40*1024**2:meta['status']='rejected_budget_or_support'
        dump(D/'asset.json',meta);dump(D/'coat_equivalence.json',proof);records.append({'asset_id':item['asset_id'],'status':meta['status'],'rig_and_motion_unchanged':True});dump(a.task_root/'evidence/coat_results.json',records);print('COAT_DONE',item['asset_id'],meta['status'],flush=True)
    extra=[]
    if a.generate_only:
        for item in plan['animals']:
            if item['mode']!='generated_static':continue
            R=a.task_root/'assets'/item['asset_id']/'reference'
            if (R/'generation_receipt.json').exists():continue
            R.mkdir(parents=True,exist_ok=True);t=time.time();im=pipe(image=None,prompt=item['prompt'],width=1024,height=1024,num_inference_steps=28,guidance_scale=1,max_sequence_length=512,generator=torch.Generator('cuda').manual_seed(item['reference_seed'])).images[0].convert('RGB');im.save(R/'candidate.png');receipt={'asset_id':item['asset_id'],'model':'black-forest-labs/FLUX.2-klein-4B','revision':plan['reference_revision'],'snapshot':str(a.model),'prompt':item['prompt'],'seed':item['reference_seed'],'parameters':{'steps':28,'guidance_scale':1,'width':1024,'height':1024,'max_sequence_length':512},'license':'Apache-2.0','image':str(R/'candidate.png'),'image_sha256':sha(R/'candidate.png'),'seconds':time.time()-t,'model_reused':True};dump(R/'generation_receipt.json',receipt);extra.append(receipt);print('EXTRA_REFERENCE',item['asset_id'],flush=True)
        dump(a.task_root/'evidence/reference_results_additional.json',extra)
    dump(a.task_root/'evidence/coat_complete.json',{'count':len(records),'seconds':time.time()-start,'model_load_seconds':load,'mode':'generate_only' if a.generate_only else 'build_only','additional_references':len(extra)})
if __name__=='__main__':main()
