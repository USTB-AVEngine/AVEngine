"""Truth for the proposed content-binding panel; QA-01..28 remain untouched.

Only physically capable actors or real loudspeakers emit sounds. Activity is
half open and measured, physical actor identity never comes from asset slots.
The truth reducer can be tested before assets arrive. Native production must
call admission_blockers and retain actual modality counterfactual witnesses.
"""
from __future__ import annotations
import math
import hashlib
import json
import subprocess
from pathlib import Path

PANEL_ID = 'V7-CONTENT-BIND'
KINDS = ('same_class_actor', 'mixed_class_actor', 'offscreen_origin',
         'live_or_playback', 'first_actor')
WORDINGS = {
    'same_class_actor': ('这段声音由画面中的哪一位发出？', '请指出画面里正在发出这段声音的人。'),
    'mixed_class_actor': ('现在发声的是画面中的哪一位？', '画面中谁正在发出听到的声音？'),
    'offscreen_origin': ('这段声音是画面里指定的这个人发出的吗？', '指定的画面人物是否就是这段声音的发声者？'),
    'live_or_playback': ('这段声音来自人物的现场动作，还是音箱播放？', '听到的声音是现场人物发出的，还是由音箱放出的？'),
    'first_actor': ('这段时间里，哪一位最先发声？', '请指出本段中首先发出声音的人。'),
}
CAPABILITIES = {
    'standing_human': frozenset({'speech'}),
    'phone_call': frozenset({'speech'}),
    'guitar_player': frozenset({'guitar_live', 'speech'}),
    'violin_player': frozenset({'violin_live', 'speech'}),
    'pianist': frozenset({'piano_live', 'speech'}),
    'saxophone_player': frozenset({'saxophone_live', 'speech'}),
    'flute_player': frozenset({'flute_live', 'speech'}),
    'hand_drum_player': frozenset({'hand_drum_live', 'speech'}),
    'megaphone_speaker': frozenset({'speech'}),
    'clapping_person': frozenset({'clapping', 'speech'}),
    'coughing_person': frozenset({'coughing', 'speech'}),
}
CONTENT_LABELS_ZH = {'speech':'讲话', 'guitar_live':'吉他', 'violin_live':'小提琴',
    'piano_live':'钢琴', 'saxophone_live':'萨克斯', 'flute_live':'长笛',
    'hand_drum_live':'手鼓', 'clapping':'鼓掌', 'coughing':'咳嗽'}

EMITTER_POINTS = {
    'speech': ('mouth', 'megaphone_driver', 'phone_driver'),
    'guitar_live': ('guitar_soundboard', 'guitar_amplifier'),
    'violin_live': ('violin_body',),
    'piano_live': ('piano_soundboard',),
    'saxophone_live': ('saxophone_bell',),
    'flute_live': ('flute_embouchure',),
    'hand_drum_live': ('drum_head',),
    'clapping': ('hands',),
    'coughing': ('mouth',),
}

def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite timeline coordinate required')
    return float(value)

def _scene(scene):
    lo, hi = map(_number, scene['window_s'])
    if lo < 0 or lo >= hi:
        raise ValueError('invalid half-open window')
    actors = {}
    descriptors = set()
    for row in scene['actors']:
        if row.get('actor_id') is None or isinstance(row['actor_id'],bool) or not str(row['actor_id']):
            raise ValueError('physical actor identity is missing')
        key = str(row['actor_id'])
        if key in actors:
            raise ValueError('duplicate physical actor identity')
        descriptor = row['descriptor_zh']
        if not descriptor or descriptor in descriptors:
            raise ValueError('public actor descriptors must be unique')
        descriptors.add(descriptor); actors[key] = row
    events = []
    emission_ids = set()
    for source in scene['emissions']:
        eid = source.get('emission_id')
        if not isinstance(eid,str) or not eid or eid in emission_ids:
            raise ValueError('physical emission identity is missing or duplicated')
        emission_ids.add(eid)
        mechanism = source['mechanism']
        content = source['content_class']
        if mechanism not in {'actor_live', 'loudspeaker_playback'}:
            raise ValueError('unsupported physical emission mechanism')
        if mechanism == 'actor_live':
            key = str(source['actor_id'])
            if key not in actors:
                raise ValueError('live emission has no physical actor')
            actor = actors[key]
            capability = actor.get('capability') or {}
            category = capability.get('asset_kind')
            allowed = CAPABILITIES.get(category, frozenset())
            if content not in allowed:
                raise ValueError('sound is impossible for the assigned physical asset')
            if source.get('emitter_point_kind') not in EMITTER_POINTS.get(content, ()):
                raise ValueError('sound source is not attached to its actual emitting part')
            # A hand drum or electric guitar subtype needs a matching instrument.
            visual_subtype, audio_subtype = capability.get('instrument_subtype'), source.get('instrument_subtype')
            if content in {'guitar_live', 'hand_drum_live'} and (
                    not visual_subtype or visual_subtype != audio_subtype):
                raise ValueError('instrument recording and visible subtype do not match')
            if content=='hand_drum_live' and audio_subtype not in {'tabla','bongo','conga','djembe','darbuka','doumbek','tambourine','frame_drum'}:
                raise ValueError('hand drum requires a verified concrete subtype')
            if content=='guitar_live':
                if audio_subtype not in {'acoustic','electric'}:
                    raise ValueError('guitar requires a verified concrete subtype')
                if audio_subtype=='acoustic' and source['emitter_point_kind']!='guitar_soundboard':
                    raise ValueError('acoustic guitar needs its soundboard emitter')
                if audio_subtype=='electric' and (source['emitter_point_kind']!='guitar_amplifier' or not source.get('device_id')):
                    raise ValueError('electric guitar needs its actual amplifier device')
            if content == 'speech':
                point = source['emitter_point_kind']
                if category == 'megaphone_speaker' and point != 'megaphone_driver':
                    raise ValueError('amplified megaphone source must use its physical driver')
                if category == 'phone_call' and point != 'mouth':
                    raise ValueError('onsite phone caller emits from their mouth; remote phone playback is not that actor')
        else:
            if not source.get('device_id') or source.get('emitter_point_kind') not in {'loudspeaker_driver','phone_driver'}:
                raise ValueError('playback requires its real device and emitting point')
            if source.get('actor_id') is not None:
                raise ValueError('playback cannot masquerade as a live actor')
        intervals = []
        for a, b in source['measured_activity_s']:
            a, b = _number(a), _number(b)
            if a < 0 or b <= a:
                raise ValueError('invalid measured half-open activity interval')
            intervals.append((a, b))
        if intervals != sorted(intervals) or any(a < pb for (_, pb), (a, _) in zip(intervals, intervals[1:])):
            raise ValueError('activity intervals overlap or are out of order')
        events.append((source, intervals))
    return lo, hi, actors, events

def content_truth(scene, query):
    """Return private truth, reducing distinct emitting actors rather than slots."""
    lo, hi, actors, events = _scene(scene)
    kind = query['kind']
    if kind not in KINDS:
        raise ValueError('unknown content binding question kind')
    content = query.get('content_class')
    named = query.get('names_sound_class_in_question') is True
    if named and content not in CONTENT_LABELS_ZH:
        raise ValueError('named content must have a fixed public sound label')
    # A private generation selector must never silently narrow a generic stem.
    selected_content = content if named else None
    if kind == 'first_actor':
        onsets = {}
        for source, intervals in events:
            if source['mechanism'] != 'actor_live' or (selected_content and source['content_class'] != selected_content):
                continue
            key = str(source['actor_id'])
            for a, b in intervals:
                if a < hi and b > lo:
                    onsets[key] = min(onsets.get(key, math.inf), max(a, lo))
        if len(onsets) < 2:
            raise ValueError('first-actor comparison requires at least two distinct sounding actors')
        earliest = min(onsets.values())
        first = [key for key, a in onsets.items() if a == earliest]
        if len(first) != 1:
            raise ValueError('first clipped anchor is tied')
        key = first[0]
        if actors[key].get('visible') is not True:
            raise ValueError('first actor needs an unambiguous visible answer descriptor')
        return {'answer': actors[key]['descriptor_zh'], 'actor_id': key,
                'first_anchor_s': earliest, 'distinct_actor_onsets': onsets}
    t = _number(query['time_s'])
    if not lo <= t < hi:
        raise ValueError('question time falls outside half-open window')
    public_time = _number(query.get('public_time_s',t))
    if abs(public_time-(t-lo))>1e-9:
        raise ValueError('public relative time must identify the actual native anchor')
    active = {}
    for source, intervals in events:
        if selected_content and source['content_class'] != selected_content:
            continue
        if any(a <= t < b for a, b in intervals):
            key = ('actor', str(source['actor_id'])) if source['mechanism'] == 'actor_live' else ('device', str(source['device_id']))
            active.setdefault(key, []).append(source)
    if len(active) != 1:
        raise ValueError('question requires exactly one distinct active physical emitter')
    (entity, key), sources = next(iter(active.items()))
    if kind == 'live_or_playback':
        if entity == 'device' and any(s['emitter_point_kind']!='loudspeaker_driver' for s in sources):
            raise ValueError('live-or-loudspeaker stem cannot label a phone as a loudspeaker')
        return {'answer': '现场人物' if entity == 'actor' else '音箱播放',
                'entity_kind': entity, 'entity_id': key, 'emission_ids': [s['emission_id'] for s in sources]}
    if kind == 'offscreen_origin':
        subject = str(query['subject_actor_id'])
        if (subject not in actors or actors[subject].get('visible') is not True
                or query.get('subject_descriptor_zh')!=actors[subject]['descriptor_zh']):
            raise ValueError('specified origin-test person must be visible')
        return {'answer': '是' if entity == 'actor' and key == subject else '否',
                'entity_kind': entity, 'entity_id': key, 'subject_actor_id': subject}
    if entity != 'actor' or actors[key].get('visible') is not True:
        raise ValueError('actor-choice item requires an actual visible sounding actor')
    candidates = [a for a in actors.values() if a.get('visible') is True]
    if len(candidates) < 2:
        raise ValueError('actor-choice binding requires more than one visible candidate')
    if kind == 'same_class_actor':
        if not content or any(s['content_class']!=content for s in sources):
            raise ValueError('same-class sounding content differs from the actual emitter')
        capable = [a for a in candidates if content in CAPABILITIES.get((a.get('capability') or {}).get('asset_kind'), ())]
        if len(capable) < 2:
            raise ValueError('same-class binding requires two physically capable visible actors')
    return {'answer': actors[key]['descriptor_zh'], 'actor_id': key,
            'emission_ids': [s['emission_id'] for s in sources]}

def question_text(query, wording_variant='standard'):
    if wording_variant not in {'standard', 'alternate'}:
        raise ValueError('unknown wording variant')
    text = WORDINGS[query['kind']][wording_variant == 'alternate']
    if query['kind']!='first_actor':
        text = '在本段第'+format(_number(query.get('public_time_s',query['time_s'])),'.12g')+'秒，'+text
    if query.get('names_sound_class_in_question') is True:
        label = CONTENT_LABELS_ZH[query['content_class']]
        text = '针对本段中的'+label+'声：'+text
    if query['kind'] == 'offscreen_origin':
        return text.replace('指定的这个人', query['subject_descriptor_zh']).replace('指定的画面人物', query['subject_descriptor_zh'])
    return text

def validate_modality_witness(pair, axis):
    """Recompute both answers and actual media equality for one native witness."""
    if axis not in {'same_V_different_A_answer_flip','same_A_different_V_answer_flip'}:
        raise ValueError('unknown modality witness axis')
    scenes,queries,media=pair.get('scenes',()),pair.get('queries',()),pair.get('media',())
    reasons=[]
    if len(scenes)!=2 or len(queries)!=2 or len(media)!=2:
        return {'status':'blocked','blockers':['actual_two_native_members_missing']}
    try:
        answers=[content_truth(scene,query)['answer'] for scene,query in zip(scenes,queries)]
    except (KeyError,TypeError,ValueError) as error:
        return {'status':'blocked','blockers':['actual_pair_truth_invalid:'+str(error)]}
    if answers[0]==answers[1]:reasons.append('canonical_answer_does_not_flip')
    if question_text(queries[0])!=question_text(queries[1]):
        reasons.append('different_public_question_stem')
    domains=pair.get('answer_domains') or []
    if len(domains)!=2 or not all(isinstance(d,list) and d and all(isinstance(x,str) and x for x in d) and len(d)==len(set(d)) for d in domains):
        reasons.append('actual_two_public_answer_domains_missing_or_ambiguous')
    elif domains[0]!=domains[1] or any(a not in d for a,d in zip(answers,domains)):
        reasons.append('different_or_invalid_public_answer_domain')
    hashes=[]
    for row in media:
        observed={}
        for key in ('video_path','audio_path'):
            path=Path(row.get(key,''))
            if not path.is_file():
                reasons.append('actual_native_'+key+'_missing')
                continue
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            observed[key]=digest.hexdigest()
            try:
                if key=='audio_path':
                    import soundfile as sf
                    info=sf.info(path)
                    samples,_=sf.read(path,dtype='float32',always_2d=True)
                    import numpy as np
                    expected=scenes[len(hashes)]['window_s'][1]-scenes[len(hashes)]['window_s'][0]
                    if (info.channels!=4 or len(samples)!=info.frames or not np.isfinite(samples).all()
                            or abs(info.duration-expected)>1/info.samplerate):
                        reasons.append('actual_audio_is_not_complete_four_channel_window')
                else:
                    result=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                      '-show_entries','stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration',
                      '-of','json',str(path)],capture_output=True,text=True,check=True)
                    streams=json.loads(result.stdout).get('streams') or []
                    if len(streams)!=1 or streams[0].get('width',0)<=0 or streams[0].get('height',0)<=0:
                        reasons.append('actual_video_has_no_decodable_native_frames')
                    else:
                        count=int(streams[0]['nb_read_frames'])
                        numerator,denominator=map(float,streams[0]['avg_frame_rate'].split('/'))
                        fps=numerator/denominator
                        expected=scenes[len(hashes)]['window_s'][1]-scenes[len(hashes)]['window_s'][0]
                        if count<=0 or fps<=0 or abs(count/fps-expected)>1/fps:
                            reasons.append('actual_video_does_not_decode_the_complete_native_window')
            except Exception:
                reasons.append('actual_native_'+key+'_cannot_be_decoded')
        hashes.append(observed)
    if all(set(row)=={'video_path','audio_path'} for row in hashes):
        same_video=hashes[0]['video_path']==hashes[1]['video_path']
        same_audio=hashes[0]['audio_path']==hashes[1]['audio_path']
        expected=(True,False) if axis=='same_V_different_A_answer_flip' else (False,True)
        if (same_video,same_audio)!=expected:reasons.append('actual_media_do_not_match_required_modality_axis')
    return {'status':'pass' if not reasons else 'blocked','axis':axis,
            'answers':answers,'actual_media_hashes':hashes,'blockers':sorted(set(reasons))}

def admission_blockers(scene, query, *, binding_witness=None):
    """Explicitly separate design truth from actual native AV binding admission."""
    content_truth(scene, query)
    reasons = []
    native = scene.get('native_readback') or {}
    for key in ('video', 'audio', 'pixel_actor_identity', 'measured_source_activity',
                'asset_license_and_capability', 'physical_emitter_attachment', 'audible_emitter_uniqueness'):
        proof = native.get(key) or {}
        if proof.get('status') != 'pass' or not proof.get('source'):
            reasons.append('native_' + key + '_missing')
        elif not Path(str(proof['source']).partition('#')[0]).is_file():
            reasons.append('native_' + key + '_actual_receipt_file_missing')
    if query['kind']=='first_actor':
        proof=native.get('source_order_listener_order_agree') or {}
        if proof.get('status')!='pass' or not proof.get('source'):
            reasons.append('native_source_order_listener_order_agreement_missing')
        elif not Path(str(proof['source']).partition('#')[0]).is_file():
            reasons.append('native_source_order_listener_order_actual_receipt_missing')
    witness = binding_witness or {}
    for key in ('same_V_different_A_answer_flip', 'same_A_different_V_answer_flip'):
        proof = witness.get(key) or {}
        if proof.get('status') != 'pass' or not proof.get('source'):
            reasons.append('binding_' + key + '_not_demonstrated')
        elif validate_modality_witness(proof.get('native_pair') or {},key)['status']!='pass':
            reasons.append('binding_' + key + '_actual_native_pair_failed')
        else:
            pair=proof['native_pair']
            signature=lambda a:json.dumps(a,sort_keys=True,separators=(',',':'))
            if not any(signature(scene)==signature(s) and signature(query)==signature(q)
                       for s,q in zip(pair['scenes'],pair['queries'])):
                reasons.append('binding_'+key+'_pair_does_not_include_this_candidate')
    if query.get('names_sound_class_in_question') is True and query['kind'] in {'mixed_class_actor', 'offscreen_origin'}:
        reasons.append('named_incompatible_instrument_allows_visual_only_elimination')
    return sorted(set(reasons))

def four_state_review(members):
    """Review four real states, plus native evidence; never count design mocks."""
    if len(members) != 4:
        raise ValueError('content-binding group must contain four states')
    states = {(m['active_choice'], m['appearance_permutation']) for m in members}
    if states != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError('group must cover active actor times appearance permutation')
    identities = {(m['base_world_id'], m['pairing_id']) for m in members}
    if len(identities) != 1:
        raise ValueError('group pairing identity is inconsistent')
    answers = []
    blockers = []
    for member in members:
        truth = content_truth(member['scene'], member['query'])
        answers.append(truth['answer'])
        blockers.extend(admission_blockers(member['scene'], member['query'],
                                            binding_witness=member.get('binding_witness')))
    return {'status': 'pass' if not blockers else 'blocked',
            'answers': answers, 'native_four_member_binding_eligible': not blockers,
            'blockers': sorted(set(blockers))}

def collect_content_binding(bundle, window_id, media, *, seed='v7'):
    """Publish evidence-qualified content binding as its own panel.

    No v6 family is renamed and no score/gate is relaxed. Missing producer
    readbacks yield exact deferrals, never inferred native questions.
    """
    from copy import deepcopy
    from avengine.qa.v7_roster_intake import intake_roster,standing_mouth_world
    import numpy as np
    scene=bundle['scene']
    public=[];private=[];deferred=[]
    path=bundle.get('source_registry') or scene.get('source_registry')
    try:
        if not path:raise ValueError('formal standing roster missing')
        roster=intake_roster(path)
        assets={r['asset_id']:r for r in roster['records']}
    except (KeyError,TypeError,ValueError,OSError) as error:
        return [],[],{'status':'blocked','deferred':[{'qa_id':PANEL_ID,'code':'formal_standing_roster_missing_or_invalid','detail':str(error)}]}
    actor_map={str(r['actor_id']):r for r in scene['actors']}
    standing_reasons=[]
    for actor in scene['actors']:
        asset=assets.get(actor.get('asset_id'))
        if not asset or not asset['standing_registered_qualified']:
            standing_reasons.append('content_actor_not_formally_registered')
            continue
        if actor.get('pose')!='stand':
            standing_reasons.append('content_actor_not_standing')
        if (actor.get('capability') or {}).get('asset_kind')!=asset['capability']['asset_kind']:
            standing_reasons.append('content_actor_capability_not_registered')
        pose=actor.get('standing_pose_readback') or {}
        if (pose.get('status')!='pass' or not pose.get('source')
                or not Path(str(pose['source']).partition('#')[0]).is_file()
                or str(pose.get('actor_id'))!=str(actor['actor_id'])
                or pose.get('asset_id')!=asset['asset_id']
                or pose.get('source_registry_sha256')!=roster['source_registry_sha256']):
            standing_reasons.append('content_native_standing_resource_readback_missing')
    for emission in scene['emissions']:
        if emission['mechanism']=='actor_live' and emission.get('emitter_point_kind')=='mouth':
            actor=actor_map.get(str(emission.get('actor_id'))) or {}
            asset=assets.get(actor.get('asset_id'))
            try:
                expected=np.asarray(standing_mouth_world(asset,actor['body_to_world_m']),float)
                observed=np.asarray(emission['native_emitter_world_m'],float)
                if observed.shape!=(3,) or not np.isfinite(observed).all() or np.linalg.norm(observed-expected)>.002:
                    standing_reasons.append('content_native_mouth_differs_from_formal_registration')
            except (KeyError,TypeError,ValueError):
                standing_reasons.append('content_registered_mouth_attachment_readback_missing')
    counts={k:0 for k in KINDS}
    for index,record in enumerate(bundle.get('queries',[])):
        query=record.get('query',record)
        try:
            truth=content_truth(scene,query)
            witness=record.get('binding_witness',{})
            blockers=sorted(set(standing_reasons+admission_blockers(scene,query,binding_witness=witness)))
            standard,alternate=question_text(query),question_text(query,'alternate')
            supplied_domain=record.get('answer_domain')
            if query['kind']=='offscreen_origin':domain=['是','否']
            elif query['kind']=='live_or_playback':domain=['现场人物','音箱播放']
            else:domain=sorted(a['descriptor_zh'] for a in scene['actors'] if a.get('visible') is True)
            if supplied_domain is not None and set(supplied_domain)!=set(domain):
                blockers.append('public_answer_domain_does_not_match_visible_candidates')
            if truth['answer'] not in domain or len(domain)<2 or len(set(domain))!=len(domain):
                blockers.append('public_answer_domain_invalid')
            if blockers:
                deferred.extend({'qa_id':PANEL_ID,'kind':query['kind'],'query_index':index,'code':reason} for reason in blockers)
                continue
            wording=record.get('wording_variant','standard')
            if wording not in {'standard','alternate'}:raise ValueError('unknown wording variant')
            text=standard if wording=='standard' else alternate
            pairing=bundle.get('pairing_id',bundle.get('base_world_id',''))
            order=sorted(domain,key=lambda d:hashlib.sha256(
                (str(seed)+'|'+str(pairing)+'|'+standard+'|'+d).encode()).digest())
            logical=f'{window_id}__content__q{index:05d}'
            for form in ('open','mcq'):
                qid=logical+'__'+form
                row={'question_id':qid,'logical_question_id':logical,'qa_id':PANEL_ID,
                    'protocol_panel':'v7_content_binding','modality':'audiovisual','form':form,
                    'question_zh':text,'media':deepcopy(media),'answer_type':'closed_set',
                    'question_variant':query['kind']}
                answer=truth['answer']
                if form=='mcq':
                    row['choices']=[{'id':chr(65+i),'text':d} for i,d in enumerate(order)]
                    answer=chr(65+order.index(truth['answer']))
                public.append(row)
                private.append({'question_id':qid,'logical_question_id':logical,
                    'answer':answer,'canonical_answer':truth['answer'],
                    'evidence':{'query':deepcopy(query),'scene':deepcopy(scene),
                        'truth':truth,'answer_domain':domain,'source_registry_sha256':roster['source_registry_sha256'],
                        'native_witness':deepcopy(witness),'wording_variant':wording,
                        'wording_exchange':{'standard':standard,'alternate':alternate,
                            'canonical_answer_invariant':True,'selection_depends_on_answer':False}},
                    'authority':'v7_content_binding_v1'})
            counts[query['kind']]+=1
        except (KeyError,TypeError,ValueError) as error:
            deferred.append({'qa_id':PANEL_ID,'query_index':index,'code':'content_truth_or_public_query_invalid','detail':str(error)})
    return public,private,{'status':'pass' if public else 'blocked','logical_counts_by_kind':counts,
        'deferred':deferred,'formal_dataset_count':0,'source_registry_sha256':roster['source_registry_sha256']}
