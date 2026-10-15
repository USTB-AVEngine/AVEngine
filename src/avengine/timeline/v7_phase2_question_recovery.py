"""Rebuild QA from immutable completed native RGB and audio in a fresh root."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,shutil,time


def attach_completed_native_evidence(source_parent,target_parent):
    """Keep exact original readbacks beside the new QA for export hard gates."""
    from avengine.timeline.v7_native_stage_resume import _link_tree
    source=Path(source_parent).resolve();target=Path(target_parent).resolve()
    if source==target or target.is_relative_to(source):raise ValueError('native evidence target must be separate')
    target.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ('visible_native','audio_v0_a0','dry'):
        marker=source/(name+'_completed.json');saved=json.loads(marker.read_text());attempt=saved['attempt']
        if Path(attempt).name!=attempt or not saved['required_files']or any(Path(f).is_absolute()or'..'in Path(f).parts for f in saved['required_files']):raise ValueError('invalid native stage paths')
        if any(not(source/attempt/f).is_file()for f in saved['required_files']):raise ValueError('missing original native artifact')
        _link_tree(source/attempt,target/attempt)
        destination=target/marker.name
        if destination.exists():
            if destination.read_bytes()!=marker.read_bytes():raise ValueError('native stage evidence differs')
        else:shutil.copyfile(marker,destination)
        records.append({'stage':name,'original_marker_sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),'copied_readbacks':'immutable hardlinks','new_native_renders':0})
    return records


def regenerate_questions(source_parent,output_native,*,requested_conditions=None):
    from avengine.timeline.v7_phase2_generation import finish_phase2_delivery,cpu_seconds
    from avengine.timeline.v7_production_inputs import bind_semantic_registry
    from avengine.assets.v7_visible_color import rebuild_appearance_review
    source=Path(source_parent).resolve();output=Path(output_native).resolve()
    recipe=json.loads((source/'input_recipe.json').read_text());row=deepcopy(recipe['row'])
    if row.get('members')!=['v0_a0']or row.get('split')!='train':
        raise ValueError('native question recovery requires one train member')
    root=output/'parents'/row['scene_id']
    if root.exists():raise FileExistsError('question recovery requires a fresh attempt root: '+str(root))
    if root==source or root.is_relative_to(source):raise ValueError('recovery must not modify its native source')
    def completed(name):
        mark=source/(name+'_completed.json');value=json.loads(mark.read_text());directory=source/value['attempt']
        if Path(value['attempt']).name!=value['attempt']or not value['required_files']or any(not(directory/p).is_file()for p in value['required_files']):
            raise ValueError('incomplete original native stage: '+name)
        return value,directory,hashlib.sha256(mark.read_bytes()).hexdigest()
    visible,vd,vsha=completed('visible_native');audio,ad,asha=completed('audio_v0_a0');dry,dd,dsha=completed('dry')
    scene=deepcopy(audio['recipe']['scene']);assets=recipe['assets'];program=recipe['program'];receipt=dry['result']
    visual=json.loads((vd/'native_visible_visual_evidence_phase2.json').read_text())
    if requested_conditions is not None:
        if not isinstance(requested_conditions,list)or any(not isinstance(c,str)for c in requested_conditions):raise ValueError('requested conditions must be text metadata only')
        row.setdefault('scene_demand',{})['requested_conditions']=list(requested_conditions)
        scene.setdefault('scene_demand',{})['requested_conditions']=list(requested_conditions)
    visual['native_appearance_review'],eligibility=rebuild_appearance_review(visual['visible_color_checks'],assets,(recipe['inputs'].get('animal_color_thresholds')or{}).get('min_pixels',200))
    semantic=None
    if row.get('semantic_scenario_id'):
        semantic=bind_semantic_registry(json.loads(Path(recipe['inputs']['semantic_manifest']).read_text()),assets)
    # No waveform, pose, event, camera, actual mask or answer is substituted.
    physical_scene=deepcopy(scene);physical_scene.pop('scene_demand',None)
    original_physical=deepcopy(audio['recipe']['scene']);original_physical.pop('scene_demand',None)
    if physical_scene!=original_physical:raise ValueError('recovery changed physical native scene')
    root.mkdir(parents=True)
    native_evidence=attach_completed_native_evidence(source,root)
    source_foot=source/'resolved_footstep_contacts.json';foot=root/source_foot.name
    if not source_foot.is_file():raise ValueError('original resolved footstep evidence missing')
    shutil.copyfile(source_foot,foot)
    recovered_recipe={**deepcopy(recipe),'row':row}
    (root/'input_recipe.json').write_text(json.dumps(recovered_recipe,ensure_ascii=False,indent=2)+'\n')
    evidence={'retained_native_evidence':native_evidence,'source_parent':str(source),'source_scene_id':row['scene_id'],'new_native_renders':0,'actual_scene_unchanged_except_requested_condition_metadata':True,'original_requested_conditions':recipe['row'].get('scene_demand',{}).get('requested_conditions'),'requested_conditions':requested_conditions,'source_completed_stage_sha256':{'visible_native':vsha,'audio_v0_a0':asha,'dry':dsha},'original_native_visual_directory':str(vd),'original_native_audio_directory':str(ad),'old_questions_or_seals_imported':False,'truth_scorer_and_answer_thresholds_changed':False}
    (root/'native_question_recovery.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    result=finish_phase2_delivery(row,recipe['house'],recipe['inputs'],assets,program,semantic,root,time.perf_counter(),cpu_seconds(),recovered_recipe,scene,visual,vd,ad,audio['result'],receipt,eligibility,foot)
    for name,expected in evidence['source_completed_stage_sha256'].items():
        if hashlib.sha256((source/(name+'_completed.json')).read_bytes()).hexdigest()!=expected:raise ValueError('native predecessor marker changed during recovery')
    return {**result,'native_question_recovery':evidence,'new_native_renders':0,'parent_directory':str(root),'seal_path':str(root/'integration_seal_completed.json')}
