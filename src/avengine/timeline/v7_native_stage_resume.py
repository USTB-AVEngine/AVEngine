"""Reuse immutable native media/RIR stages while rebuilding QA in a fresh root.

Only a task-owned, stopped predecessor may be provided by the coordinator.
The generator independently checks every imported stage's exact recipe before
using it. Old question pools, answers, selection, audits and seals are excluded.
"""
from pathlib import Path
import hashlib,json,os,shutil,sqlite3
from avengine.timeline.continuous_pilot import write_json

STAGES=('dry','visual_v0','visual_v1','audio_v0_a0','audio_v0_a1','audio_v1_a0','audio_v1_a1')

def _link_tree(source,target):
    target.mkdir(parents=True,exist_ok=True)
    for item in source.iterdir():
        # This is a derived QA certificate, not an original RGB/RIR stage.
        if item.name=="registry_namespace_readback.json":continue
        dest=target/item.name
        if item.is_symlink():raise ValueError('native predecessor must not contain symlinked artifacts')
        if item.is_dir():_link_tree(item,dest)
        elif not dest.exists():os.link(item,dest)
        elif not os.path.samefile(item,dest):
            if hashlib.sha256(item.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():
                raise ValueError('native reuse target differs from immutable predecessor')

def reuse_native_parent(source_work,target_work,row,*,allow_selection_baseline_change=False):
    source=Path(source_work)/'parents'/row['scene_id'];target=Path(target_work)/'parents'/row['scene_id']
    pointer=target/'native_stage_reuse.json'
    if pointer.is_file():
        saved=json.loads(pointer.read_text())
        if saved['source_parent']!=str(source) or saved['scene_id']!=row['scene_id']:raise ValueError('native reuse provenance changed')
        return saved
    recipe_path=source/'input_recipe.json'
    if not recipe_path.is_file():return {'status':'no_completed_predecessor','scene_id':row['scene_id']}
    original=json.loads(recipe_path.read_text())
    previous=dict(original['seed_row']);requested=dict(row)
    if allow_selection_baseline_change:
        previous.pop('selection_coverage_seals',None);requested.pop('selection_coverage_seals',None)
    if previous!=requested:raise ValueError('native reuse seed differs; no physical metadata substitution allowed')
    if target.resolve().is_relative_to(source.resolve()) or source.resolve()==target.resolve():raise ValueError('fresh QA root must be distinct from its predecessor')
    target.mkdir(parents=True,exist_ok=True);stages=[]
    for name in STAGES:
        marker=source/(name+'_completed.json')
        if not marker.is_file():continue
        saved=json.loads(marker.read_text());attempt=source/saved['attempt']
        if Path(saved['attempt']).name!=saved['attempt'] or any(not (attempt/f).is_file() for f in saved['required_files']):
            raise ValueError('predecessor native stage is incomplete')
        _link_tree(attempt,target/saved['attempt'])
        dest=target/marker.name
        if dest.exists() and dest.read_bytes()!=marker.read_bytes():raise ValueError('native stage marker changed')
        if not dest.exists():shutil.copyfile(marker,dest)
        stages.append({'stage':name,'marker_sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),
                       'files':len(saved['required_files']),'recipe_revalidated_by_generator':True})
    cache=source/'rir_cache';dest=target/'rir_cache'
    if cache.is_dir():
        dest.mkdir(exist_ok=True)
        for f in cache.iterdir():
            if f.name=='positions.sqlite3':
                if not (dest/f.name).exists():
                    with sqlite3.connect(f'file:{f}?mode=ro',uri=True) as a,sqlite3.connect(dest/f.name) as b:a.backup(b)
            elif f.name=='configuration.json' or f.suffix=='.npy':
                if not (dest/f.name).exists():os.link(f,dest/f.name)
    result={'status':'reused_completed_native_only','scene_id':row['scene_id'],'source_parent':str(source),
            'stages':stages,'old_QA_pools_imported':False,'old_answers_modified':False,
            'SQLite':'independent backup; never hardlinked mutable databases',
            'native_attempt_files':'immutable hardlinks; original byte evidence retained'}
    write_json(pointer,result);return result


def compatible_static_native_recipe(root,name,requested,row):
    """Retain the actual revision of a completed, unchanged static render.

    The v7 renderer changed animation geometry caching and silent furniture
    controls. Neither is present in an admitted standing-only predecessor.
    Every remaining physical input must still match exactly.
    """
    root=Path(root);reuse=root/'native_stage_reuse.json';mark=root/(name+'_completed.json')
    if not reuse.is_file()or not mark.is_file():return requested
    if row.get('production_scenario',{}).get('motion_requests'):return requested
    saved=json.loads(mark.read_text());previous=saved['recipe']
    old=previous.get('visual_revision');new=requested.get('visual_revision')
    if old==new:return requested
    if old!='registered-contact-movement-native-emitter-v6' or new!='registered-contact-movement-native-emitter-v7-exact-target-mask':
        raise ValueError('unreviewed native visual revision transition')
    if previous.get('scene',{}).get('production_scenario',{}).get('motion_requests'):
        raise ValueError('moving or seated native stages cannot use static compatibility')
    a=dict(previous);b=dict(requested);a.pop('visual_revision',None);b.pop('visual_revision',None)
    if a!=b:raise ValueError('static native physical inputs changed')
    for filename in saved['required_files']:
        if not(root/saved['attempt']/filename).is_file():raise ValueError('completed static native artifact missing')
    write_json(root/(name+'_static_native_compatibility.json'),{
        'status':'reused_exact_static_physical_inputs','actual_native_visual_revision':old,
        'requested_generator_visual_revision':new,'animation_or_seating_present':False,
        'completed_marker_sha256':hashlib.sha256(mark.read_bytes()).hexdigest(),
        'new_native_render_claim':False,'thresholds_changed':False})
    return previous


def retain_exact_json_receipt(path, measured):
    """Recompute original checks, then reuse only an identical saved receipt."""
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=measured:
            raise ValueError('retained native receipt differs from current measurements: '+str(path))
        return False
    write_json(path,measured)
    return True


def require_completed_registered_media(root,row):
    """Resume mode may consume native stages but must never start a render."""
    root=Path(root)
    names=['dry',*[f'visual_{v}'for v in sorted({m.split('_')[0]for m in row['members']})],*[f'audio_{m}'for m in row['members']]]
    for name in names:
        pointer=root/(name+'_completed.json')
        if not pointer.is_file():raise ValueError('resume-only missing original native stage: '+name)
        saved=json.loads(pointer.read_text());attempt=saved['attempt']
        if Path(attempt).name!=attempt or not saved['required_files']or any(not(root/attempt/f).is_file()for f in saved['required_files']):
            raise ValueError('resume-only original native stage incomplete: '+name)
    return names
