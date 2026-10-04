"""Seal self-contained task-owned windows before freeing parent intermediates.

Only this seed generator's explicit parent directory can be compacted. The
scheduler guarantees one writer. No global scan, raw input, prior-round output
or external cache is considered for deletion.
"""
from __future__ import annotations
from pathlib import Path
import json,os,shutil,uuid
from avengine.timeline.continuous_pilot import write_json


def _read(path):return json.loads(Path(path).read_text())


def load_sealed(root, recipe):
    root=Path(root);pointer=root/'delivery_completed.json'
    if not pointer.is_file():return None
    sealed=_read(pointer)
    if sealed['recipe']!=recipe:raise ValueError('sealed parent recipe changed')
    directory=root/sealed['attempt']
    if not directory.is_dir() or any(not (directory/p).is_file() for p in sealed['required_files']):
        raise ValueError('sealed delivery lost required files')
    return sealed


def compact_parent(root, receipt):
    root=Path(root).resolve();recipe=_read(root/'input_recipe.json')
    if root.name!=recipe['seed_row']['scene_id'] or root.parent.name!='parents':
        raise ValueError('compaction lacks seed-parent ownership')
    current=load_sealed(root,recipe)
    if current is not None:return current
    members=recipe['seed_row']['members'];pointers={};stages=[]
    window_stages={m['member']:m.get('window_stage_name','windows_'+m['member']) for m in receipt['members']}
    for name in sorted({f'visual_{m[:2]}' for m in members}|{f'audio_{m}' for m in members}|{window_stages[m] for m in members}):
        pointer=root/(name+'_completed.json');stage=_read(pointer);attempt=root/stage['attempt']
        if attempt.parent!=root or not attempt.name.startswith(name+'_attempt_') or attempt.is_symlink():
            raise ValueError('intermediate stage escaped owned parent')
        if any(not (attempt/p).is_file() for p in stage['required_files']):raise ValueError('cannot compact incomplete stage')
        pointers[name]=stage;stages.append((pointer,attempt))
    # Check all removals before publication, including incomplete attempts from
    # a previous killed worker. Symbolic links are never followed or deleted.
    removable=[]
    for p in root.iterdir():
        if p.name=='rir_cache' or p.name.startswith('dry_attempt_') or any(p.name.startswith(name+'_attempt_') for name in pointers):
            if p.is_symlink() or (p.is_dir() and any(x.is_symlink() for x in p.rglob('*'))):
                raise ValueError('owned intermediate contains a symbolic link')
            removable.append(p)
    final=root/('delivery_attempt_'+uuid.uuid4().hex);final.mkdir();required=[];updated=[]
    for member in receipt['members']:
        name=member['member'];wp=pointers[window_stages[name]];source=root/wp['attempt']
        vp=pointers['visual_'+name[:2]];ap=pointers['audio_'+name]
        visual=_read(root/vp['attempt']/'native_visual_evidence.json')
        setup=_read(root/ap['attempt']/'native_RLR_setup.json')
        metadata=final/('provenance_'+name+'.json')
        # Readback poses/masks remain metadata. Dense arrays/full media are not
        # needed to evaluate a delivered window or reconstruct its world state.
        appearance={}
        if member['windows']:
            evidence_path=source/member['windows'][0]['window_id']/'evidence.json'
            appearance_path=_read(evidence_path).get('native_appearance_evidence')
            if appearance_path:
                resolved=Path(appearance_path).resolve()
                resolved.relative_to(root)
                appearance=_read(resolved)
        write_json(metadata,{'native_visual_readback':visual,'native_RLR_setup':setup,
                             'native_audio_evidence':ap['result'],'native_appearance_readback':appearance,'input_recipe':recipe,'member':name})
        required.append(metadata.name)
        for window in member['windows']:
            wid=window['window_id'];destination=final/wid;destination.mkdir()
            for file in (source/wid).iterdir():
                if file.name.startswith('.'):continue
                if file.name=='evidence.json':
                    evidence=_read(file);evidence.pop('native_visual_evidence',None);evidence.pop('native_audio_evidence',None);evidence.pop('native_appearance_evidence',None)
                    evidence['self_contained_provenance']='../'+metadata.name;write_json(destination/file.name,evidence)
                else:os.link(file,destination/file.name)
                required.append(str((destination/file.name).relative_to(final)))
        updated.append({**{k:v for k,v in member.items() if k!='question_pool_directory'},'windows_directory':str(final)})
    result={**receipt,'members':updated,'compacted':True}
    sealed={'schema':'v7-sealed-candidate-v1','recipe':recipe,'attempt':final.name,'required_files':required,'result':result}
    write_json(root/'delivery_completed.json',sealed)
    # Everything above is durable before any unlink. On a crash during cleanup
    # the sealed pointer already owns all required final artifacts.
    for p in removable:
        if p.is_dir():shutil.rmtree(p)
        else:p.unlink()
    for pointer,_ in stages:pointer.unlink()
    dry_pointer=root/'dry_completed.json'
    if dry_pointer.is_file():dry_pointer.unlink()
    return sealed
