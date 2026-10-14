#!/usr/bin/env python3
"""Remove only a completed round8 alpha parent's verified intermediates."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
from tools.rooms.alpha_storage_contract import write_owned_json as write_json
from tools.rooms.alpha_storage_contract import native_seal


def compact_alpha_parent(parent,task_root):
    parent,recipe,pointer,delivery=native_seal(parent,task_root);row=recipe['seed_row']
    record=parent/'alpha_compaction.json'
    if record.exists():return json.loads(record.read_text())
    retained=[];witness=parent/'witness';witness.mkdir(exist_ok=True)
    removed=[];files=[]
    for f in sorted(delivery.rglob('*')):
        if f.is_file():files.append({'path':str(f.relative_to(parent)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    # Keep one representative frame per rendered appearance, all native contact
    # readbacks and media/provenance. The dense frame/mask intermediates go away.
    variants=sorted({m[:2] for m in row['members']})
    for v in variants:
        visual_pointer=parent/('visual_'+v+'_completed.json');p=json.loads(visual_pointer.read_text());source=parent/p['attempt']
        target=witness/v;target.mkdir(exist_ok=True)
        for name in ['native_frame_0000.png','native_frame_0090.png','native_frame_0449.png','standing/standing_admission.json']:
            src=source/name
            if not src.is_file():raise ValueError('necessary witness missing: '+str(src))
            dst=target/Path(name).name
            if dst.exists():
                if hashlib.sha256(src.read_bytes()).hexdigest()!=hashlib.sha256(dst.read_bytes()).hexdigest():raise ValueError('existing witness differs')
            else:os.link(src,dst)
            retained.append(str(dst.relative_to(parent)))
    prefixes=('visual_v0_attempt_','visual_v1_attempt_','audio_','windows_','selected_windows_','dry_attempt_')
    for p in parent.iterdir():
        if p==delivery or not p.is_dir():continue
        if p.name=='rir_cache' or p.name.startswith(prefixes):
            if p.is_symlink() or any(f.is_symlink() for f in p.rglob('*')):raise ValueError('owned intermediate contains a symlink')
            removed.append(p)
    # Explicit new-parent paths only; no cross-round caches or source packages.
    result={'schema':'kujiale_round8_alpha_compaction_v1','status':'sealed_ready_to_compact','parent':str(parent),
            'delivery':str(delivery),'preserved_delivery_files':files,'retained_witness_files':retained,
            'intermediate_directories':[str(p) for p in removed],'logical_intermediate_bytes':sum(f.stat().st_size for p in removed for f in p.rglob('*') if f.is_file()),'sample_epoch':time.time()}
    preflight=parent/'alpha_compaction_preflight.json'
    if preflight.exists():
        old=json.loads(preflight.read_text())
        if old['preserved_delivery_files']!=files:raise ValueError('seal changed since compaction preflight')
        result=old
    else:write_json(preflight,result)
    for p in removed:shutil.rmtree(p)
    for p in parent.glob('*_completed.json'):
        if p.name!='alpha_seal_completed.json':p.unlink()
    result['status']='compacted';write_json(record,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();print(json.dumps(compact_alpha_parent(a.parent,a.task_root),ensure_ascii=False))
if __name__=='__main__':main()
