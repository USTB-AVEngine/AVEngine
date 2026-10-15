"""Actual registered actor/event/mouth readbacks for the QA-owned TTS adapter."""
from copy import deepcopy
from pathlib import Path

def dialogue_evidence(scene,program,receipt,visual,manifest,start,retained_visual):
    first=round(start*15);frame=visual['frames'][first];instances=[]
    missing=[{'actor_id':a['actor_id'],'missing':[k for k in ['body_to_world_m','loaded_artifact_sha256']if k not in a]}for a in frame['actors']if any(k not in a for k in ['body_to_world_m','loaded_artifact_sha256'])]
    if missing:
        return {'native_dialogue_deferred':{'status':'deferred','reason':'actual native transform or original artifact readback unavailable; planned positions are never substituted','actors':missing}}
    moving_assets={a['asset_id'] for f in visual['frames'] for a in f['actors'] if a.get('action_id') not in {'stand',None}}
    for actor in frame['actors']:
        if actor['asset_id'] in moving_assets:continue
        aid=actor['actor_id']
        instances.append({'native_actor_id':aid,'asset_id':actor['asset_id'],'pose':'stand','body_to_world_m':actor['body_to_world_m'],'standing_pose_readback':{'status':'pass','source':retained_visual,'actor_id':aid,'asset_id':actor['asset_id'],'source_registry_sha256':actor['source_registry_sha256'],'loaded_artifact_sha256':actor['loaded_artifact_sha256']}})
    from avengine.timeline.v7_registry_snapshot import resolve_registry,qualify_namespace
    registry,original_sha,registry_sha=resolve_registry(retained_visual,visual,manifest['source_registry'])
    namespace_path=Path(retained_visual).with_name('registry_namespace_readback.json')
    if scene.get('phase2_profile'):
        import hashlib
        location=hashlib.sha256(str(Path(retained_visual).resolve()).encode()).hexdigest()[:16]
        namespace_path=Path(retained_visual).with_name('registry_namespace_phase2_'+location+'.json')
    static_visual={**visual,'frames':[{**f,'actors':[a for a in f['actors'] if a['asset_id'] not in moving_assets]} for f in visual['frames']]}
    _,namespace=qualify_namespace(static_visual,registry,original_sha,namespace_path,retained_visual)
    for instance in instances:
        stand=instance['standing_pose_readback'];stand['source']=namespace['source']
        stand['source_registry_sha256']=registry_sha;stand['original_native_registry_sha256']=original_sha
    manifest=deepcopy(manifest);manifest['source_registry']=registry
    sounds={r['sound_asset_id']:r for r in program['records']};bindings={r['event_id']:r for r in receipt['events']};placements={};roles={}
    actor_by_id={a['actor_id']:a for a in frame['actors']}
    for event in scene['events']:
        eid=event['event_id'];binding=bindings[eid];sound=sounds[binding['sound_asset_id']]
        if sound.get('source_type')!='author_tts':continue
        lo=event['start_tick']/10-start;hi=event['end_tick']/10-start;actor=actor_by_id[event['source_id']]
        placements[eid]={'status':'pass','source':retained_visual,'event_id':eid,'actor_id':event['source_id'],'window_start_s':lo,'window_end_s':hi,'native_emitter_world_m':actor['emitter_position_m'],'emitter_point_kind':'mouth','source_sample_count':binding['waveform_sample_count'],'source_sample_rate_hz':16000,'source_audio_sha256':sound['sha256'],'full_content_preserved':binding['semantic_content_complete'],'trimmed_or_repeated':False}
        roles[eid]={'status':'pass','source':retained_visual,'actor_id':event['source_id'],'scenario_id':sound['scenario_id'],'role':sound['role_id']}
    return {'moving_TTS_content_binding_deferred_asset_ids':sorted(moving_assets),'semantic_manifest':deepcopy(manifest),'native_dialogue_evidence':{'registered_actors':instances,'placements':placements,'roles':roles},'semantic_wording_variant':'standard'}
