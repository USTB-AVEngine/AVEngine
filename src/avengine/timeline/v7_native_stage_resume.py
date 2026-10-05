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

def reuse_native_parent(source_work,target_work,row):
    source=Path(source_work)/'parents'/row['scene_id'];target=Path(target_work)/'parents'/row['scene_id']
    pointer=target/'native_stage_reuse.json'
    if pointer.is_file():
        saved=json.loads(pointer.read_text())
        if saved['source_parent']!=str(source) or saved['scene_id']!=row['scene_id']:raise ValueError('native reuse provenance changed')
        return saved
    recipe_path=source/'input_recipe.json'
    if not recipe_path.is_file():return {'status':'no_completed_predecessor','scene_id':row['scene_id']}
    original=json.loads(recipe_path.read_text())
    if original['seed_row']!=row:raise ValueError('native reuse seed differs; no metadata substitution allowed')
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
