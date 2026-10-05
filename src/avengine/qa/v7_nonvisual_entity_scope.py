"""Measured non-drawable audio nodes are not possible visual candidates.

This scope never supplies pixel states, removes an audio event, or interprets
missing visibility as absence. Legacy facts without the native registry keep
the original strict visibility requirements.
"""
from __future__ import annotations
from collections.abc import Mapping
import hashlib,json
import numpy as np

SCHEMA='v7-native-nonvisual-source-domain-r17-v1'


def validate_native_nonvisual_registry(raw):
    registry=raw.get('native_nonvisual_source_registry')
    if registry is None:return None
    if registry.get('schema')!=SCHEMA or registry.get('authority')!='observed_Habitat_audio_only_SceneGraph_child':
        raise ValueError('unverified native nonvisual source registry')
    count=raw['clock']['frame_count'];actors=raw['actors'];tracks=raw['frame_readbacks']['actors']
    if count!=150:raise ValueError('nonvisual native scope requires all 150 window frames')
    records=registry.get('records');ids=[];assets={};digests={}
    if not isinstance(records,list):raise ValueError('native nonvisual registry must enumerate records')
    expected={str(k)for k,a in actors.items()if a.get('entity_class')=='environmental_audio_point'}
    for record in records:
        sid=str(record['actor_id']);ep=record['endpoint_id'];actor=actors.get(sid)
        if sid in ids or not actor or not isinstance(ep,str):raise ValueError('native nonvisual source identity invalid')
        if (actor.get('entity_class')!='environmental_audio_point' or
            actor.get('asset_id')!='native_audio_point:'+ep or
            actor.get('visual_geometry')!='none; no rendered person/object is claimed' or
            actor.get('native_registration_source')!='environmental_emitter_readbacks'):
            raise ValueError('a drawable or unregistered entity cannot enter the nonvisual domain')
        if sid in (raw.get('pixel_visibility_truth')or{}).get('per_instance',{}):
            raise ValueError('native source is also in the actual drawable pixel registry')
        if sid in (raw.get('appearance_review')or{}).get('actors',{}):
            raise ValueError('a reviewed visual actor cannot be declared nonvisual')
        frames=tracks.get(sid)
        if not isinstance(frames,list)or len(frames)!=count:raise ValueError('native nonvisual track incomplete')
        for i,frame in enumerate(frames):
            if (frame.get('frame_index')!=i or str(frame.get('actor_id'))!=sid or
                frame.get('native_source_kind')!='observed_Habitat_audio_only_SceneGraph_child' or
                frame.get('joint_state')!='no_joint_DOFs_no_drawable' or
                frame.get('action_id')!='static_audio_point' or frame.get('moving')is not False or
                'native_object_id'in frame or 'semantic_id'in frame):
                raise ValueError('nonvisual source lacks the complete actual native-node lineage')
            m=np.asarray(frame['body_to_world_m'],float);p=np.asarray(frame['position_m'],float)
            if (m.shape!=(4,4)or p.shape!=(3,)or not np.isfinite(m).all()or
                not np.array_equal(m[:3,3],p)or not np.array_equal(p,np.asarray(frame['emitter_position_m'],float))):
                raise ValueError('native nonvisual emitter transform differs from observed position')
        if any(e.get('native_source_endpoint_id')!=ep for e in raw['audio_program']['events']if str(e.get('actor_id'))==sid):
            raise ValueError('native nonvisual event identity differs from physical endpoint')
        ids.append(sid);assets[sid]=actor['asset_id'];digests[sid]=hashlib.sha256(json.dumps(frames,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if set(ids)!=expected:raise ValueError('native nonvisual registry omits or invents audio-only entities')
    return {'schema':SCHEMA,'status':'validated_actual_native_no_drawable_domain','actor_ids':sorted(ids),
        'asset_ids':assets,'native_track_SHA256':digests,'frame_count':count,
        'pixel_states_invented':False,'audio_events_removed':False}


def nonvisual_entity_ids_for_qa(facts):
    scope=facts.get('native_nonvisual_entity_scope')
    if scope is None:return frozenset()
    if (scope.get('schema')!=SCHEMA or scope.get('status')!='validated_actual_native_no_drawable_domain' or
        scope.get('frame_count')!=facts['time']['frame_count']):raise ValueError('invalid nonvisual visual-query domain')
    ids=scope.get('actor_ids',[])
    for sid in ids:
        actor=facts['actors'].get(sid)
        if (not actor or actor.get('asset_id')!=scope['asset_ids'].get(sid)or
            not str(actor['asset_id']).startswith('native_audio_point:')or
            actor.get('species_id')!='environmental_audio_point'or actor.get('appearance')is not None or
            sid in facts.get('visibility',{})):
            raise ValueError('visual entity or unobserved replacement cannot use a nonvisual scope')
    return frozenset(ids)
