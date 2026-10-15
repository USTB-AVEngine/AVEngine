#!/usr/bin/env python3
"""New input snapshot/UE project/seed namespace; no old scene media or seals."""
import argparse,copy,datetime,hashlib,json,os,shutil
from pathlib import Path
from avengine.registry.kujiale_ue_measurements import sha256
from avengine.timeline.v7_kujiale_manifest import atomic_write


def snapshot_declared_inputs(document,inputs):
    for key,declaration in document.get('inputs',{}).items():
        if key.startswith('room_')and key!='room_kujiale':continue  # other helpers own those room authorities
        if not isinstance(declaration,dict)or not declaration.get('path'):continue
        source=Path(declaration['path'])
        if source.suffix!='.json'or not source.is_file():continue
        target=Path(inputs)/'declared_authorities'/(key+'_'+source.name);target.parent.mkdir(parents=True,exist_ok=True)
        if target.resolve()!=source.resolve():shutil.copy2(source,target)
        if declaration.get('sha256')and sha256(target)!=declaration['sha256']:raise ValueError('declared source input hash changed')
        declaration.update(path=str(target),sha256=sha256(target))
    return document


def bootstrap(source_manifest,output):
    source_manifest=Path(source_manifest);old=json.loads(source_manifest.read_text());oldroot=Path(old['output_root']);output=Path(output).resolve();inputs=output/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    if (inputs/'GAP_PLAN.json').exists():raise ValueError('fresh campaign manifest already exists; do not overwrite')
    w=Path(old['worktree']);houses=json.loads(Path(old['houses_ue']).read_text());runtime=json.loads(Path(old['runtime_ue']).read_text());project=Path(houses['houses'][0]['input']['ue_config']['uproject']);new_project=output/'ue_v1/runtime/SpearSim'/project.name
    # UE package/plugin/config inputs only. Saved capture outputs never enter.
    ignored=shutil.ignore_patterns('Saved','Intermediate','DerivedDataCache','__pycache__','.git')
    shutil.copytree(project.parent,new_project.parent,ignore=ignored,symlinks=True)
    snapshot_root=oldroot/'input_snapshots'
    if snapshot_root.exists():shutil.copytree(snapshot_root,output/'input_snapshots',symlinks=True)
    def remap(value):
        if isinstance(value,str):return value.replace(str(oldroot/'input_snapshots'),str(output/'input_snapshots'))
        if isinstance(value,list):return[remap(x)for x in value]
        if isinstance(value,dict):return{remap(k):remap(v)for k,v in value.items()}
        return value
    new=copy.deepcopy(old);new.update(output_root=str(output),parents=[],campaign_id='v7_kujiale_fixed_v2',scene_seed_base=261015915000,owner_generation_policy='owner-fixed-camera-natural-layout-20261015-v2',invalid_predecessor_root=str(oldroot),old_scene_media_reused=False,old_seals_reused=False)
    for key in ['contract','assets_ue','human_ue']:
        value=remap(json.loads(Path(old[key]).read_text()))
        if key=='contract':value=snapshot_declared_inputs(value,inputs)
        target=inputs/(key+'.json');atomic_write(target,value);new[key]=str(target)
    for house in houses['houses']:
        config=house['input']['ue_config'];config.update(uproject=str(new_project),output_root=str(output),unreal_editor=str(output/'scripts/ue_gpu2'))
        for key in ['native_lighting_audit','native_room_visual_qa']:
            source=Path(config[key])
            if key=='native_room_visual_qa':
                record=json.loads(source.read_text());source=Path(record.get('prior_admission_provenance',{}).get('qa',source))
            target=inputs/'original_house_authority'/house['house']/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);config[key]=str(target)
    atomic_write(inputs/'houses_ue.json',houses);new['houses_ue']=str(inputs/'houses_ue.json');runtime['houses']=houses['houses'];runtime=remap(runtime)
    for key in ['semantic_manifest','dry_source_admission','animal_motion_admission','kujiale_ue_asset_admission']:
        if runtime.get(key):
            target=inputs/'source_authorities'/Path(runtime[key]).name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(runtime[key],target);runtime[key]=str(target)
    receipt=json.loads(Path(runtime['native_registered_content_receipt']).read_text());original_project=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_v1/runtime/SpearSim')
    for item in receipt['files']:
        for prefix in [original_project,project.parent]:
            path=Path(item['path'])
            if path.is_relative_to(prefix):item['path']=str(new_project.parent/path.relative_to(prefix));break
        if sha256(item['path'])!=item['sha256']:raise ValueError('fresh source UE content differs from original registered bytes')
    atomic_write(inputs/'native_registered_content_copy.json',receipt);runtime.update(native_registered_content_receipt=str(inputs/'native_registered_content_copy.json'),production_input_contract=new['contract'],source_material_cache_directory=str(output/'cache/source_materials_ue'),registered_question_workers=1,registered_question_cpu_ids=[12],registered_ue_capture_scale=1,suspend_unused_registered_capture_sensors=True,registered_offscreen_viewport_resolution_hw=[64,64],registered_actor_yaw_offset_deg=-90.,construct_registered_dialogue_camera=False)
    runtime.pop('registered_camera_evidence_policy',None);atomic_write(inputs/'runtime_ue.json',runtime);new['runtime_ue']=str(inputs/'runtime_ue.json')
    source_registration=w/'registrations/v7_kujiale_ue_native_measurements_20261015.json';formal=json.loads(source_registration.read_text());formal['revision']='20261015-fixed-natural-v2';formal['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();formal['technical_input_provenance']={'kind':'native coordinate basis/package identity only; no old scene media or seal','source':str(source_registration),'sha256':sha256(source_registration)}
    formal['rooms']=[{k:r[k]for k in ['house','room_id','native_floor_surface','room_legality_source']}|{'measurement_status':'not_run','smoke_status':'not_run','measurements':[],'independent_questions':0,'reason':'new fixed-camera natural-layout parent pending'}for r in formal['rooms']]
    registration=w/'registrations/v7_kujiale_ue_native_measurements_20261015_v2.json';atomic_write(registration,formal);registration.with_suffix('.json.sha256').write_text(sha256(registration)+'  '+registration.name+'\n');runtime['kujiale_native_measurement_registration']=str(registration);atomic_write(inputs/'runtime_ue.json',runtime);new['formal_kujiale_registration']=str(registration)
    for name in ['ue_tmp','ue_shm','ue_config','scripts','logs','evidence']:(output/name).mkdir(exist_ok=True)
    policy=json.loads((oldroot/'kujiale_native_gpu_policy.json').read_text());atomic_write(output/'kujiale_native_gpu_policy.json',policy);scope=json.loads((oldroot/'inputs/task_scope.json').read_text());scope['host_pss_admission_cutover_unix']=1792043687;atomic_write(inputs/'task_scope.json',scope)
    env=(oldroot/'scripts/env.sh').read_text().replace(str(oldroot),str(output));(output/'scripts/env.sh').write_text(env)
    atomic_write(inputs/'GAP_PLAN.json',new);atomic_write(output/'evidence/FRESH_CAMPAIGN_INPUTS.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_input_manifest':str(source_manifest),'new_manifest':str(inputs/'GAP_PLAN.json'),'new_project':str(new_project),'scene_seed_base':new['scene_seed_base'],'old_parent_requests_copied':0,'old_scene_media_copied':0,'old_seals_copied':0,'source_assets_only':True})
    return new


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();value=bootstrap(a.source_manifest,a.output_root);print(json.dumps({'output_root':value['output_root'],'old_scene_media_reused':value['old_scene_media_reused'],'old_seals_reused':value['old_seals_reused']}))
if __name__=='__main__':main()
