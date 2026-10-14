"""Borrow an explicit object material-quality prior; never borrow answer truth."""
from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import subprocess
from avengine.timeline.continuous_pilot import write_json

REQUIRED_NATIVE_FILES=['scene_video.mp4','native_visible_visual_evidence_phase2.json','native_color_checks.json','native_checks.json','source_material_checks.json','native_visibility_evidence.json','acoustic_placement_admission.json','standing/standing_admission.json','standing/native_support_survey.json']


def same_object_material_class(target,peer):
    if target.get('kind')!='object' or peer.get('kind')!='object':
        return False
    if target.get('attachment_surface')!=peer.get('attachment_surface'):
        return False
    return (bool(target.get('object_type'))and target['object_type']==peer.get('object_type')) or bool(
        set(target.get('allowed_classes',[]))&set(peer.get('allowed_classes',[])))


def qualify_missing_materials(root,assets,visual):
    """Annotate a borrowed quality reference, preserving every original measure."""
    checks=visual['source_material_checks']['checks']
    measured={r['asset_id']for r in checks if r['status']in {'pass','fail'}}
    missing=[a for a in assets if a['asset_id']not in measured and a.get('kind')=='object']
    if not missing:
        return visual
    peers=[]
    for parent in sorted(Path(root).parent.iterdir())[:64]:
        recipe=parent/'input_recipe.json'
        if not recipe.is_file():
            continue
        declarations=json.loads(recipe.read_text()).get('assets',[])
        by_id={a['asset_id']:a for a in declarations if a.get('kind')=='object'}
        for path in sorted(parent.glob('visible_native_attempt_*/source_material_checks.json')):
            data=json.loads(path.read_text());rows=data.get('checks',[])
            failed={r['asset_id']for r in rows if r['status']=='fail'}
            for record in rows:
                aid=record['asset_id']
                if record['status']=='pass'and aid in by_id and aid not in failed:
                    peers.append((by_id[aid],record,path))
    priors=[]
    for target in missing:
        candidates=[p for p in peers if same_object_material_class(target,p[0])]
        candidates.sort(key=lambda p:(p[0]['asset_id']!=target['asset_id'],p[0].get('object_type')!=target.get('object_type')))
        if not candidates:
            continue
        peer,check,path=candidates[0]
        priors.append({'asset_id':target['asset_id'],'status':'borrowed_measured_object_material_quality',
            'actual_asset_was_measured':False,'reference_asset_id':peer['asset_id'],
            'reference_registered_glb':peer.get('habitat_glb'),'reference_measurement_path':str(path),
            'reference_measurement':check,'same_registered_function_and_attachment_class':True,
            'policy':'owner-material-quality-prior-20261014',
            'target_appearance_color_position_and_answer_truth_borrowed':False,
            'target_rendered_asset_changed':False})
    if not priors:
        return visual
    result=deepcopy(visual);result['owner_material_quality_fallbacks']=priors
    return result


def recover_complete_native_attempt(root,attempt,scene,assets,checker):
    """Recheck complete, immutable media from this exact parent's failed stage."""
    root=Path(root);attempt=Path(attempt)
    for source in sorted(root.glob('visible_native_attempt_*'),key=lambda p:p.stat().st_mtime,reverse=True):
        if source==attempt or not all((source/f).is_file()for f in REQUIRED_NATIVE_FILES):
            continue
        visual=json.loads((source/'native_visible_visual_evidence_phase2.json').read_text())
        if visual.get('frame_count')!=450:
            continue
        annotated=qualify_missing_materials(root,assets,visual)
        if not annotated.get('owner_material_quality_fallbacks'):
            continue
        checks=checker(scene,assets,annotated,pending_feet=True)
        if checks['status']!='pass':
            continue
        decoded=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,duration,width,height','-of','json',str(source/'scene_video.mp4')],text=True))
        stream=decoded['streams'][0]
        if int(stream['nb_read_frames'])!=450 or abs(float(stream['duration'])-30.)>.01:
            continue
        shutil.copytree(source,attempt,copy_function=os.link,dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('native_checks.json','native_visible_visual_evidence_phase2.json','owner_material_quality_recovery.json'))
        write_json(attempt/'native_visible_visual_evidence_phase2.json',annotated)
        write_json(attempt/'native_checks.json',checks)
        proof={'status':'rechecked_complete_native_media_with_marked_material_quality_prior',
            'source_attempt':str(source),'exact_parent_input_recipe':str(root/'input_recipe.json'),
            'actual_decoded_video':decoded,'all_original_measurements_retained':True,
            'source_files_rewritten':False,'rendered_geometry_camera_and_emitter_positions_changed':False,
            'borrowed_object_quality_references':annotated['owner_material_quality_fallbacks']}
        write_json(attempt/'owner_material_quality_recovery.json',proof)
        return {'status':'pass','native_checks':checks,'owner_quality_recovery':proof},[*REQUIRED_NATIVE_FILES,'owner_material_quality_recovery.json']
    return None
