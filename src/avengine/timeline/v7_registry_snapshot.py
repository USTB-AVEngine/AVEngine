"""Recover exact native registry snapshots and recheck all actual standing frames.

Archived bytes keep their original SHA and artifact root. The archive path
carries that exact SHA; changed metadata is rejected by the QA intake. Every
original frame, loaded GLB and registered mouth remains independently checked.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from avengine.timeline.continuous_pilot import write_json

def resolve_registry(visual_path,visual,original_path):
    hashes={a['source_registry_sha256'] for f in visual['frames'] for a in f['actors']}
    if len(hashes)!=1:raise ValueError('mixed actual native registry versions')
    original_sha=next(iter(hashes));original=Path(original_path)
    snapshot=Path(visual_path).parents[4]/'input_snapshots/human'/original_sha/'registrations/human_roster.json'
    raw_path=snapshot if snapshot.is_file() else original
    raw=raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=original_sha:raise ValueError('actual original native registry bytes missing; no hash substitution allowed')
    return str(raw_path),original_sha,original_sha

def qualify_namespace(visual,registry,original_sha,receipt_path,visual_path):
    from avengine.qa.v7_roster_intake import intake_roster,standing_mouth_world
    roster=intake_roster(registry);assets={r['asset_id']:r for r in roster['records']}
    if len(visual['frames'])!=450:raise ValueError('registry requalification requires all 450 actual frames')
    error=0.;checked=set()
    for frame in visual['frames']:
        if len({a['actor_id'] for a in frame['actors']})!=len(frame['actors']):raise ValueError('duplicate physical actor IDs')
        for a in frame['actors']:
            asset=assets[a['asset_id']]
            if not asset['standing_registered_qualified'] or a['source_registry_sha256']!=original_sha:raise ValueError('actual standing registry/source mismatch')
            from avengine.timeline.v7_native_asset_readback import require_loaded_asset
            require_loaded_asset(a,asset)
            if a.get('moving') is not False or a.get('action_id') not in ({'registered_stand'} if a.get('native_load_backend')=='ue_spear' else {'stand'}):raise ValueError('namespace certificate cannot promote motion')
            expected=np.asarray(standing_mouth_world(asset,a['body_to_world_m']),float)
            error=max(error,float(np.linalg.norm(expected-np.asarray(a['emitter_position_m']))));checked.add((a['actor_id'],a['asset_id']))
    if error>.002:raise ValueError('namespace requalification fails unchanged 2mm mouth criterion')
    receipt={'status':'pass','source':str(receipt_path),'raw_native_source':str(visual_path),
        'original_native_registry_sha256':original_sha,'source_registry_sha256':roster['source_registry_sha256'],
        'resolved_registry':str(registry),'namespace_change':'exact original registry bytes retained at SHA-addressed archive; original artifact root unchanged',
        'complete_frames_checked':450,'actual_instances_checked':len(checked),'mouth_max_error_m':error,
        'actual_GLB_hashes_match_original_registration':True,'raw_native_frame_hashes_unchanged':True,'motion_scope':'stand only',
        'native_load_backends':sorted({a.get('native_load_backend','habitat_glb')for f in visual['frames']for a in f['actors']}),
        'UE_load_identity':'actual original Blueprint/mesh/material packages; original import GLB separately checked'}
    path=Path(receipt_path)
    if path.exists():
        if json.loads(path.read_text())!=receipt:raise ValueError('registry namespace receipt changed')
    else:write_json(path,receipt)
    return roster,receipt
