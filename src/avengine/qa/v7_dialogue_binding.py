"""Attach actual batch speech to registered standing actors and native events.

Author plans do not establish native placement. The producer supplies complete
source placement, actual standing transforms and role evidence. v6 semantics,
answers and scorers are delegated unchanged.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from avengine.qa.v7_roster_intake import intake_roster, standing_mouth_world

def _retained(proof):
    return (isinstance(proof,dict) and proof.get('status')=='pass'
            and bool(proof.get('source'))
            and Path(str(proof['source']).partition('#')[0]).is_file())

def worded_dialogues(dialogues, variant='standard'):
    if variant not in {'standard','alternate'}:
        raise ValueError('unknown semantic wording variant')
    result=deepcopy(dialogues)
    for scene in result['scenarios']:
        if variant=='alternate':
            scene['question_zh']=scene.get('question_zh_alternate','请根据人物的话回答：'+scene['question_zh'])
            scene['question_en']='From what the person says, answer: '+scene['question_en']
            scene['reverse_question_zh']='说出“{answer}”这个意思的是哪一位？请用画面中的外观描述回答。'
            scene['reverse_question_en']='Describe the appearance of the person whose words mean {answer}.'
            scene['direction_question_zh']='说出“{answer}”这个意思的那位，开口时在你的哪个方向？'
            scene['direction_question_en']='At the moment the person begins the statement meaning {answer}, what direction are they in?'
        # Short author scripts contain a generic deictic stem. Replace it with
        # the authority's *observed* appearance, never an author role or voice ID.
        for language,key in [('zh','question_zh'),('en','question_en')]:
            text=scene[key]
            if '{appearance}' not in text:
                if language=='zh':
                    text=text.replace('这个人','{appearance}').replace('这位人物','{appearance}')
                    if '{appearance}' not in text:text='{appearance}说的话中，'+text
                else:
                    text=text.replace('this person','{appearance}').replace('the person','{appearance}')
                    if '{appearance}' not in text:text='{appearance}: '+text
            scene[key]=text
    return result

def attach_dialogue_sources(facts, manifest, native_registered_actors, placements, role_evidence,
                            *, roster=None, wording_variant='standard'):
    """Return a window-specific manifest, plus exact deferrals.

    placements and role_evidence are keyed by event_id, not reusable source
    sound IDs. Repeated use of one source clip must get its own native proof.
    """
    if roster is None:
        path=manifest.get('source_registry')
        if not path:raise ValueError('batch speech has no formal human registry')
        roster=intake_roster(path)
    by_asset={r['asset_id']:r for r in roster['records']}
    by_sound={r['sound_asset_id']:r for r in manifest.get('sounds',[])}
    actors={str(a['native_actor_id']):a for a in native_registered_actors}
    if len(actors)!=len(native_registered_actors):
        raise ValueError('duplicate physical actor IDs in native registered instances')
    authored_roles={(s['id'],r['role_id']):r for s in manifest['dialogues']['scenarios']
                    for r in s.get('roles',[])}
    eligible=[];deferred=[];voice_by_actor={};role_actors=defaultdict(dict)
    sr=facts['time']['sample_rate_hz'];duration=facts['time']['sample_count']/sr
    for event in facts['events']:
        sid=event.get('sound_asset_id');original=by_sound.get(sid)
        if original is None:continue
        eid=event['event_id'];aid=str(event.get('actor_id'))
        instance=actors.get(aid);reasons=[]
        asset=by_asset.get((instance or {}).get('asset_id'))
        placement=deepcopy(placements.get(eid) or {})
        role=deepcopy(role_evidence.get(eid) or {})
        if not instance or not asset or not asset['standing_registered_qualified']:
            reasons.append('native_registered_standing_actor_missing')
        else:
            if instance.get('pose')!='stand':reasons.append('dialogue_actor_not_standing')
            if not asset['dialogue_voice_binding_eligible'] or asset['visual_gender']!=original.get('voice_gender'):
                reasons.append('registered_visual_and_voice_gender_mismatch')
            stand=instance.get('standing_pose_readback') or {}
            if not _retained(stand) or str(stand.get('actor_id'))!=aid or stand.get('asset_id')!=asset['asset_id']:
                reasons.append('actual_native_standing_resource_receipt_missing')
            elif stand.get('source_registry_sha256')!=roster['source_registry_sha256']:
                reasons.append('native_standing_registration_receipt_is_stale')
            if not _retained(placement) or str(placement.get('actor_id'))!=aid or placement.get('event_id')!=eid:
                reasons.append('event_specific_native_placement_receipt_missing')
            point=None
            requirement=authored_roles.get((original.get('scenario_id'),original.get('role_id')),{}).get('required_asset_capability','standing_human')
            action=role.get('native_action_readback') or {}
            asset_kind=asset['capability']['asset_kind']
            if requirement in {'phone_call','megaphone_speaker'} and asset_kind!=requirement:
                if (not _retained(action) or str(action.get('actor_id'))!=aid
                        or action.get('asset_kind')!=requirement or not action.get('device_id')):
                    reasons.append('actual_registered_role_action_or_prop_missing')
            try:
                mouth=np.asarray(standing_mouth_world(asset,instance['body_to_world_m']),float)
                point=mouth
                if requirement=='megaphone_speaker' or asset_kind=='megaphone_speaker':
                    if (not _retained(action) or str(action.get('actor_id'))!=aid
                            or action.get('asset_kind')!='megaphone_speaker' or not action.get('device_id')):
                        raise ValueError('real megaphone driver evidence missing')
                    point=np.asarray(action['native_driver_world_m'],float)
                    expected_kind='megaphone_driver'
                else:
                    expected_kind='mouth'
                if point.shape!=(3,) or not np.isfinite(point).all():
                    raise ValueError('registered actual source point is not finite')
                observed=np.asarray(placement['native_emitter_world_m'],float)
                if observed.shape!=(3,) or not np.isfinite(observed).all() or np.linalg.norm(point-observed)>.002:
                    reasons.append('native_FOA_source_not_at_registered_mouth')
                if placement.get('emitter_point_kind')!=expected_kind:
                    reasons.append('standing_dialogue_source_part_does_not_match_actual_mechanism')
            except (KeyError,TypeError,ValueError):
                reasons.append('registered_mouth_native_attachment_readback_missing')
            actual=facts.get('actors',{}).get(aid) or {}
            stand=instance.get('standing_pose_readback') or {}
            actual_asset=actual.get('asset_id')
            glb_sha=asset['actual_artifact_sha256'].get(asset['habitat_glb'])
            if (actual_asset!=asset['asset_id'] and
                    (not actual_asset or stand.get('actual_loaded_asset_id')!=actual_asset
                     or stand.get('loaded_artifact_sha256')!=glb_sha)):
                reasons.append('native_loaded_actor_resource_not_bound_to_formal_asset')
            try:
                roots=np.asarray(actual['root_positions_m'],float)
                emitters=np.asarray(actual['emitter_positions_m'],float)
                expected_root=np.asarray(instance['body_to_world_m'],float)[:3,3]
                frame_count=facts['time']['frame_count']
                actual_source=actual.get('source') or {}
                if (roots.shape!=(frame_count,3) or emitters.shape!=(frame_count,3)
                        or not np.isfinite(roots).all() or not np.isfinite(emitters).all()
                        or actual_source.get('root_readbacks') is not True
                        or actual_source.get('emitter_readbacks') is not True
                        or np.max(np.linalg.norm(roots-expected_root,axis=1))>.002
                        or np.max(np.linalg.norm(emitters-point,axis=1))>.002):
                    reasons.append('actual_native_frames_not_registered_standing_mouth')
            except (KeyError,TypeError,ValueError,UnboundLocalError):
                reasons.append('actual_native_standing_frames_missing')
            placement.update(visual_gender=asset['visual_gender'],registered_asset_id=asset['asset_id'],
                  mouth_local_glb_m=asset['mouth_local_glb_m'],source_registry_sha256=roster['source_registry_sha256'])
        if not _retained(role) or str(role.get('actor_id'))!=aid or role.get('scenario_id')!=original.get('scenario_id') or role.get('role')!=original.get('role_id'):
            reasons.append('native_role_actor_scenario_binding_missing')
        if event.get('transcript')!=original.get('transcript'):
            reasons.append('native_event_authored_transcript_mismatch')
        # Placement is a complete source, with optional following zero padding.
        # A tick-rounded event duration may exceed a complete waveform; cropping
        # the last word or repeating it is never accepted.
        proof=original.get('synthesis_readback') or {}
        try:
            source_duration=proof['sample_count']/proof['sample_rate_hz']
            event_start,event_end=float(event['start_s']),float(event['end_s'])
            start,end=float(placement['window_start_s']),float(placement['window_end_s'])
            valid=(math.isfinite(start) and math.isfinite(end)
                   and abs(start-event_start)<=1/sr and abs(end-event_end)<=1/sr
                   and 0<=start<end<=duration+1/sr
                   and end-start+1/sr>=source_duration
                   and placement.get('source_sample_count')==proof['sample_count']
                   and placement.get('source_sample_rate_hz')==proof['sample_rate_hz']
                   and placement.get('source_audio_sha256')==original.get('sha256')
                   and placement.get('full_content_preserved') is True
                   and placement.get('trimmed_or_repeated') is False)
            if not valid:reasons.append('full_authored_utterance_does_not_match_native_window')
        except (KeyError,TypeError,ValueError,ZeroDivisionError):
            reasons.append('complete_native_source_clock_readback_missing')
        voice=original['voice_reference']['speaker_id']
        if aid in voice_by_actor and voice_by_actor[aid]!=voice:
            reasons.append('same_physical_actor_voice_identity_changed')
        voice_by_actor[aid]=voice
        group=role_actors[original['scenario_id']]
        if original['role_id'] in group and group[original['role_id']]!=aid:
            reasons.append('one_authored_role_has_multiple_native_actors')
        group[original['role_id']]=aid
        if reasons:
            deferred.extend({'qa_id':q,'event_id':eid,'code':code}
                for q in ('QA-26','QA-27','QA-28') for code in sorted(set(reasons)))
            continue
        sound=deepcopy(original)
        # Unique per-event key allows the same authored clip to be used more
        # than once without borrowing another event's native placement.
        sound['source_sound_asset_id']=sid
        sound['sound_asset_id']=sid+'__native_event__'+str(eid)
        event['sound_asset_id']=sound['sound_asset_id']
        sound['native_placement_readback']=placement;sound['dialogue_role_evidence']=role
        sound['registered_actor_evidence']={'asset_id':asset['asset_id'],'pose':'stand',
            'source_registry_sha256':roster['source_registry_sha256'],'formal_static_authority':True,
            'complete_episode_admitted':False}
        eligible.append(sound)
    # Distinct author roles are distinct physical actors AND source voices.
    for scene, roles in role_actors.items():
        if len(set(roles.values()))!=len(roles):
            eligible=[s for s in eligible if s['scenario_id']!=scene]
            deferred.extend({'qa_id':q,'scenario_id':scene,'code':'different_dialogue_roles_share_one_physical_actor'}
                            for q in ('QA-26','QA-27','QA-28'))
        heard=[s['voice_reference']['speaker_id'] for s in eligible if s['scenario_id']==scene]
        actor_voice={str(s['dialogue_role_evidence']['actor_id']):s['voice_reference']['speaker_id']
                     for s in eligible if s['scenario_id']==scene}
        if len(set(actor_voice.values()))!=len(actor_voice):
            eligible=[s for s in eligible if s['scenario_id']!=scene]
            deferred.extend({'qa_id':q,'scenario_id':scene,'code':'different_dialogue_actors_share_one_voice'}
                            for q in ('QA-26','QA-27','QA-28'))
    result=deepcopy(manifest);result['sounds']=eligible
    result['dialogues']=worded_dialogues(manifest['dialogues'],wording_variant)
    result['registered_native_attachment_count']=len(eligible)
    return result,deferred

def audit_semantic_wording_exchange(facts, manifest, *, seed='v7', variant_id='wording_audit'):
    """Run the unchanged v6 generator with both wording variants."""
    from avengine.qa.semantic_questions import generate_semantic_questions
    def run(wording):
        m=deepcopy(manifest);m['dialogues']=worded_dialogues(m['dialogues'],wording)
        r=generate_semantic_questions(facts,m,variant_id=variant_id,seed=seed)
        values={}
        for item in r['items']:
            e=item['evidence'];key=(item['qa_id'],str(e.get('target_actor_id')),e.get('event_id'))
            values[key]={'answer':item['forms']['open']['truth'],
                        'forms':{f:{k:v for k,v in form.items() if k not in {'question_zh','question_en'}}
                                 for f,form in item['forms'].items()},
                        'options':{f:request.get('options',[]) for f,request in item['model_input'].items()},
                        'stem':item['model_input']['open']['question_zh']}
        return values
    a,b=run('standard'),run('alternate')
    mismatches=[{'key':k,'standard':a.get(k),'alternate':b.get(k)} for k in set(a)|set(b)
                if k not in a or k not in b or a[k]['answer']!=b[k]['answer'] or a[k]['forms']!=b[k]['forms'] or a[k]['options']!=b[k]['options']]
    leakage=[]
    for key,row in list(a.items())+list(b.items()):
        if key[0]=='QA-26':
            scene_id=next((s['scenario_id'] for s in manifest['sounds']
                          if s.get('dialogue_role_evidence',{}).get('actor_id') is not None and str(s.get('dialogue_role_evidence',{}).get('actor_id'))==key[1]),None)
            for s in manifest['sounds']:
                if s['scenario_id']==scene_id:
                    for label in ('label_zh','label_en'):
                        if s['semantic_answer'][label].lower() in row['stem'].lower():
                            leakage.append({'key':key,'code':'QA26_stem_contains_answer_meaning'})
    return {'status':'pass' if not mismatches and not leakage else 'blocked',
            'standard_items':len(a),'alternate_items':len(b),'mismatches':mismatches,
            'stem_answer_leakage':leakage,'native_production_claim':False}
