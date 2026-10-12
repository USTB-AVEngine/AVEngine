"""Reuse immutable stationary native media only under identical physical inputs."""
from pathlib import Path
import json,os,shutil
from avengine.timeline.continuous_pilot import write_json


def validate_physical_reuse(candidate,original):
    for key in ('house','inputs','assets'):
        if candidate[key]!=original[key]:raise ValueError('native reuse physical input differs: '+key)
    ignored={'scene_id','semantic_start_offsets_s','reuse_completed_native_parent','question_quota_base','timing_proposal_evidence'}
    left={k:v for k,v in candidate['row'].items() if k not in ignored}
    right={k:v for k,v in original['row'].items() if k not in ignored}
    if left!=right:raise ValueError('native reuse camera, placement or asset row differs')
    if any(a.get('pilot_walk_allowed') or a.get('allowed_pose_for_this_task')!='stand' for a in candidate['assets']):
        raise ValueError('only registered stationary bodies can reuse native media')


def reuse_completed_native(parent,recipe,attempt,scene,audit_root):
    from avengine.timeline.v7_visible_generation import native_checks
    parent=Path(parent).resolve();audit_root=Path(audit_root).resolve();attempt=Path(attempt)
    if parent.parent.parent.parent!=audit_root or parent.parent.name!='parents' or not parent.parent.parent.name.startswith('pilot_v'):
        raise ValueError('native reuse requires a completed parent in this task directory')
    original=json.loads((parent/'input_recipe.json').read_text());validate_physical_reuse(recipe,original)
    certificate=audit_root/'post_seal_support_v3'/parent.name/'result.json'
    if not certificate.is_file() or json.loads(certificate.read_text())['status']!='pass':
        raise ValueError('native reuse requires the independent original support certificate')
    pointer=json.loads((parent/'visible_native_completed.json').read_text());source=parent/pointer['attempt']
    if any(not (source/r).is_file() for r in pointer['required_files']):raise ValueError('native reuse source stage is incomplete')
    visual=json.loads((source/'native_visible_visual_evidence.json').read_text())
    checks=native_checks(scene,recipe['assets'],visual)
    if checks['status']!='pass':raise ValueError('native reuse failed physical drawable or highlight gates')
    # Metadata is copied to avoid modifying any old inode through later QA
    # namespace receipts. Immutable media are hardlinked, never rerendered.
    for f in source.rglob('*'):
        if not f.is_file() or f.name in {'native_checks.json','registry_namespace_readback.json'}:continue
        target=attempt/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        if f.suffix=='.json':shutil.copy2(f,target)
        else:os.link(f,target)
    write_json(attempt/'native_checks.json',checks)
    write_json(attempt/'native_reuse_provenance.json',{'source_parent':str(parent),'source_stage_pointer':str(parent/'visible_native_completed.json'),'independent_support_certificate':str(certificate),'physical_inputs_identical':True,'native_video_rerendered':False,'new_audio_timing_requires_new_dry_and_RIR_audio':True})
    return {'status':'pass','native_checks':checks,'reused_stationary_native_parent':str(parent)},[*pointer['required_files'],'native_reuse_provenance.json']
