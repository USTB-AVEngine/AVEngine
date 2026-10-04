"""Storage boundaries for immutable native alpha seal and derived bundles."""
from pathlib import Path
import json


def owned_parent(parent,task_root):
    supplied=Path(parent)
    if supplied.is_symlink():raise ValueError('parent symlink refused')
    parent=supplied.resolve();scope=Path(task_root).resolve()/'ue_round8'
    parent.relative_to(scope)
    if parent.parent.name!='parents':raise ValueError('not an owned alpha parent')
    recipe=json.loads((parent/'input_recipe.json').read_text())
    if recipe['seed_row']['scene_id']!=parent.name or recipe['house']['input']['visual_backend']!='ue_spear':
        raise ValueError('native alpha parent ownership differs')
    return parent,recipe


def relative_file(root,name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('artifact path escapes its root')
    path=Path(root)
    for part in relative.parts:
        path=path/part
        if path.is_symlink():raise ValueError('artifact symlink refused')
    path.resolve().relative_to(Path(root).resolve())
    if not path.is_file():raise ValueError('required artifact missing: '+str(path))
    return path


def native_seal(parent,task_root):
    parent,recipe=owned_parent(parent,task_root)
    pointer=json.loads(relative_file(parent,'alpha_seal_completed.json').read_text())
    name=pointer['attempt']
    if Path(name).name!=name or not name.startswith('alpha_seal_attempt_'):raise ValueError('seal escaped its parent')
    seal=parent/name
    if seal.is_symlink() or not seal.is_dir():raise ValueError('native seal missing or symlink')
    if pointer['result']['status']!='formal_alpha_admitted':raise ValueError('unadmitted alpha seal')
    for name in pointer['required_files']:relative_file(seal,name)
    if any(f.is_symlink() for f in seal.rglob('*')):raise ValueError('seal contains a symlink')
    return parent,recipe,pointer,seal


def write_owned_json(path,value):
    """Publish owned metadata once, without retaining successful temp links."""
    import os,uuid
    path=Path(path)
    if 'ue_round8' not in path.parts or path.is_symlink():raise ValueError('JSON publication must stay in own round8')
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    pending=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.partial')
    with pending.open('x') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    try:os.link(pending,path)
    finally:pending.unlink()
