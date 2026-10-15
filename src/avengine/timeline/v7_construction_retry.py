"""Bounded automatic reconstruction; required elements and the base seed stay."""
from copy import deepcopy
from pathlib import Path
import fcntl,json,os


def spatial_construction_failure(receipt):
    reason=str(receipt.get('reason',''))
    evidence_collision=receipt.get('error_type')=='FileExistsError'and ('attachment_rejection.json'in reason or 'occlusion_proposal_'in reason)
    if evidence_collision:return True
    return receipt.get('error_type')=='ValueError'and any(s in reason for s in (
        'person_camera_clearance', 'scene delivery policy failed:', 'no strict native ', 'final registered scene occludes',
        'requested registered occlusion', 'room cannot supply requested count',
        'no strict native floor', 'bounded native NavMesh', 'no fully qualified new-house walking route'))


def queue_construction_retry(manifest_path,receipt):
    if not spatial_construction_failure(receipt):return None
    manifest_path=Path(manifest_path)
    with(manifest_path.parent/'plan.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);m=json.loads(manifest_path.read_text());old=m['parents'][receipt['slot']]
        if old.get('constructor')!='automatic_family':return None
        if any(t.get('retry_of_slot')==receipt['slot']for t in m['parents']):return None
        attempt=int(old.get('construction_attempt',0))+1
        if attempt>=int(m.get('maximum_automatic_attempts_per_parent',24)):return None
        task=deepcopy(old);base=old.get('construction_lineage_id',old['row']['scene_id']);rooms=[old.get('initial_room_id',old['row']['room_id']),*old.get('alternative_room_ids',[])]
        task.update(slot=len(m['parents']),construction_attempt=attempt,construction_lineage_id=base,initial_room_id=rooms[0],retry_of_slot=receipt['slot'],
            snapshot_worktree=m['worktree'],snapshot_commit=m['code_commit'])
        task['row'].update(scene_id=base+'__attempt_'+str(attempt).zfill(2),room_id=rooms[(attempt//4)%len(rooms)])
        task['automatic_retry_reason']=receipt['reason'];task['automatic_retry_preserved_base_seed']=old['row']['scene_seed']
        m['parents'].append(task);temporary=manifest_path.with_name(manifest_path.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');temporary.replace(manifest_path)
        return task['slot']


def queue_unmet_family_retry(manifest_path,receipt):
    """Retain a valid parent; reconstruct only a genuinely absent target family.

    Existing legal candidates belong to cumulative re-selection, not another
    render. A new physical construction keeps the base seed and all role counts.
    """
    if receipt.get('status')!='passed':return None
    manifest_path=Path(manifest_path);root=manifest_path.parent.parent
    initial=json.loads(manifest_path.read_text());old=initial['parents'][receipt['slot']]
    if old.get('constructor')!='automatic_family':return None
    family=old.get('target_family')
    if not family or receipt.get('export',{}).get('question_family_counts',{}).get(family,0):return None
    seal=Path(receipt['export']['source_seal'])
    if not seal.resolve().is_relative_to(root.resolve()):return None
    pointer=json.loads(seal.read_text())
    for window in pointer['result']['windows']:
        wid=window['window_id'];member=window.get('member')or wid.rsplit('__',2)[-2]
        marker=json.loads((seal.parent/f'selected_windows_{member}_completed.json').read_text())
        pool=Path(marker['recipe']['pool_directory'])/wid/'questions.json'
        if any(q['qa_id']==family for q in json.loads(pool.read_text())):return None
    with(manifest_path.parent/'plan.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);m=json.loads(manifest_path.read_text())
        if m.get('publication_closed')or any(t.get('condition_retry_of_slot')==receipt['slot']for t in m['parents']):return None
        state_path=root/'inputs/demand_state.json'
        if state_path.exists()and json.loads(state_path.read_text()).get('family_counts',{}).get(family,0)>=3:return None
        old=m['parents'][receipt['slot']];attempt=int(old.get('construction_attempt',0))+1
        if attempt>=int(m.get('maximum_automatic_attempts_per_parent',24)):return None
        task=deepcopy(old);base=old.get('construction_lineage_id',old['row']['scene_id'])
        task.update(slot=len(m['parents']),construction_attempt=attempt,construction_lineage_id=base,
            condition_retry_of_slot=receipt['slot'],snapshot_worktree=m['worktree'],snapshot_commit=m['code_commit'],
            launch_disabled=False,automatic_retry_reason='requested family has no legal original native candidate: '+family)
        task['row']['scene_id']=base+'__condition_'+str(attempt).zfill(2)
        m['parents'].append(task);temporary=manifest_path.with_name(manifest_path.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');temporary.replace(manifest_path)
        return task['slot']
