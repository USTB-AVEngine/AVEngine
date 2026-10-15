"""Finish selection on an immutable completed new parent, never rerender it."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,datetime

NATIVE_STAGES=('visible_native','dry','audio_v0_a0','windows_v0_a0')


def prepare_completed_parent(source,output,authorized_root):
    from avengine.timeline.v7_generation_scope import invalidation_marker
    source=Path(source).resolve();output=Path(output).resolve();authorized_root=Path(authorized_root).resolve()
    if not source.is_relative_to(authorized_root)or not output.is_relative_to(authorized_root):raise ValueError('selection resume escaped its declared own output root')
    if invalidation_marker(source)or invalidation_marker(output):raise ValueError('invalidated old media cannot be resumed')
    if output.exists():raise ValueError('preserve previous resume output; choose a fresh own output path')
    manifest=json.loads((source/'inputs/generated_manifest.json').read_text());row=json.loads((source/'inputs/planned/000.json').read_text())
    if row.get('trial_only')is not True or row['members']!=['v0_a0']:raise ValueError('selection-only resume requires the declared single-member acceptance parent')
    budget=Path(manifest.get('storage_budget_root',source)).resolve()
    if not source.is_relative_to(budget)or not output.is_relative_to(budget)or not budget.is_relative_to(authorized_root):raise ValueError('selection source and result must share their explicit own task budget root')
    parent=source/'native/parents'/row['scene_id'];originals=[parent/'input_recipe.json',*[parent/(n+'_completed.json')for n in NATIVE_STAGES]]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in originals}
    newparent=output/'native/parents'/row['scene_id'];newparent.mkdir(parents=True)
    for marker in originals[1:]:
        item=json.loads(marker.read_text());item['attempt']=str((parent/item['attempt']).resolve())
        if not Path(item['attempt']).is_relative_to(budget):raise ValueError('native checkpoint attempt escaped declared own task root')
        (newparent/marker.name).write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n')
    selected_row=deepcopy(row);selected_row['acceptance_allow_partial_families']=True
    manifest=deepcopy(manifest);manifest['output_root']=str(output);manifest['parents'][0]['row']=selected_row
    manifest['storage_admission_prechecked']=False
    (output/'inputs/planned').mkdir(parents=True);(output/'inputs/planned/000.json').write_text(json.dumps(selected_row,ensure_ascii=False,indent=2)+'\n')
    (output/'inputs/demand_state.json').write_text((source/'inputs/demand_state.json').read_text())
    return manifest,row,before


def resume_completed_selection(source,output,cfg):
    from avengine.timeline.v7_unattended_resources import ResourcePolicy,admission
    from avengine.timeline import v7_candidate_generation as candidate,v7_phase2_generation as phase2
    from tools.timeline import export_v7_trial
    from tools.timeline.run_v7_acceptance import run_worker
    output=Path(output).resolve();snapshot=admission(ResourcePolicy(**cfg.get('resources',{})))
    if snapshot['status']!='admitted':return {'status':'paused_before_new_selection_job','resource_admission':snapshot}
    manifest,row,before=prepare_completed_parent(source,output,cfg['authorized_output_root']);budget=Path(manifest['storage_budget_root']).resolve()
    (output/'AUTOMATION_RUN.json').write_text(json.dumps({'schema':'v7-automatic-run-v1','source_commit':manifest['code_commit'],
        'same_parent_selection_only':True,'source_attempt':str(Path(source).resolve()),'new_independent_parent':False,
        'handwritten_scene_configuration_inputs_read':[],'sampling_policy':'actual original legal candidates; all batch family deficits remain required'},indent=2)+'\n')
    original_stage=candidate.stage;original_phase=phase2.stage;original_payload=export_v7_trial.parent_payload;reused=[]
    def guarded(root,name,recipe,run):
        if name in NATIVE_STAGES:
            def never_render(attempt):raise RuntimeError('selection resume requires the exact completed native stage; rerender forbidden: '+name)
            answer=original_stage(root,name,recipe,never_render);reused.append(name);return answer
        return original_stage(root,name,recipe,run)
    def owned_payload(info,delivery,owned_root):
        if not Path(info['parent_directory']).resolve().is_relative_to(budget):raise ValueError('selection parent escaped declared own root')
        return original_payload(info,delivery,budget)
    try:
        candidate.stage=guarded;phase2.stage=guarded;export_v7_trial.parent_payload=owned_payload
        code=run_worker(manifest,0)
    finally:
        candidate.stage=original_stage;phase2.stage=original_phase;export_v7_trial.parent_payload=original_payload
    after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in before}
    if before!=after:raise ValueError('original native checkpoint changed')
    proof={'status':'passed'if code==0 else'failed','source_attempt':str(source),'output':str(output),
        'same_independent_parent_id':row['scene_id'],'new_independent_parent':False,'native_stages_exactly_resumed':reused,
        'original_checkpoint_sha256':before,'original_checkpoints_unchanged':True,'native_truth_and_scorers_changed':False,
        'completed_sgt':datetime.datetime.now().isoformat()}
    (output/'SELECTION_RESUME_RECEIPT.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    return proof
