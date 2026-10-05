"""Admitted recordings emitted at observed, static Habitat scenegraph points.

An audio-only source is not a rendered person/object. Class observation uses
a separate audio-only class panel; no existing QA predicate or
source admission is changed. Live-speaker supervision excludes playback.
"""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np

# Published taxonomy, independent of any real recording's parameters.
STRICT_CLASS13={'clapping':2,'telephone':3,'phone_ring':3,'telephone_bell_ringing':3,
 'laughter':4,'footsteps':6,'door_open_close':7,'music_playback':8,
 'guitar_live':9,'violin_live':9,'saxophone_live':9,'hand_drum_live':9,
 'piano_live':9,'flute_live':9,'water_tap_faucet':10,'door_knock':12}
# The admitted families crosswalk is authoritative, including explicit None
# for cough and bird calls. Never relabel them as STARSS domestic sounds.
from avengine.qa.v7_sound_extensions import STARSS_EXTENSION_MAP
STRICT_CLASS13.update(STARSS_EXTENSION_MAP)


def recording_header(record):
    import soundfile as sf
    r=deepcopy(record);path=r.get('normalized_path') or r.get('screens',{}).get('normalization',{}).get('path')
    if not path:raise ValueError('admitted normalized waveform missing')
    info=sf.info(path)
    if info.samplerate!=16000 or info.channels!=1 or info.frames<=0:raise ValueError('admitted recording must be actual 16k mono')
    r['normalized_path']=path;r['header']={'sample_rate_hz':16000,'sample_count':info.frames,'channels':1};r['duration_s']=info.frames/16000
    return r


def attach_environmental_program(scene,program,plan,admitted,rng,starts):
    classes=plan.get('environment_sound_classes',[])
    if not classes:return
    domain=sorted({r['sound_class'] for r in admitted})
    if not classes or any(c not in domain for c in classes):raise ValueError('environment requests an unadmitted class')
    scene['admitted_sound_class_domain']=domain
    scene['production_scope']='v7-S critical coverage batch1; optional families/motion batch2'
    candidates=scene['native_placement_candidates']
    if not candidates:raise ValueError('native environmental placement candidates missing')
    for index,cls in enumerate(classes):
        pool=[r for r in admitted if r.get('sound_class')==cls and r.get('dataset')!='VCTK-Corpus 0.92']
        if not pool:raise ValueError('actual recorded environmental class pool absent: '+cls)
        record=recording_header(pool[int(rng.integers(len(pool)))]);ep='environment'+str(index+1)
        is_playback=cls in {'speech_playback','music_playback','any_audioset_class_playback'}
        mechanism='loudspeaker_playback' if is_playback else 'environment_recorded_event'
        mapped=STRICT_CLASS13.get(cls)
        point=deepcopy(candidates[index%len(candidates)])
        scene['sources'][ep]={'actor_id':1001+index,'class_id':mapped,'content_kind':cls,
            'mechanism':mechanism,'native_visual_kind':'audio_only','dry_gain':float(plan.get('environment_gain',.9)),
            'trajectory':[{'time_s':0.,'position_m':point},{'time_s':30.,'position_m':point}],
            'sound_class_provenance':{'original_id':record['original_id'],'source_uri':record['screens']['license'].get('source_uri'),'license':record['screens']['license']['license'],'label':'upstream automatic recording admission'},
            'visual_binding':'none; actual native scenegraph point has no mesh'}
        # Short observed excerpts are allowed for effects, never for authored
        # transcript supervision. Keep the actual crop in the dry receipt.
        length=min(2.,record['duration_s']);ticks=max(1,math.floor(length*10))
        activity=record['screens']['automatic_activity']['activity_intervals_samples']
        if not activity:raise ValueError('environmental clip has no measured activity')
        crop=max(0,min(int(activity[0][0]),record['header']['sample_count']-round(length*16000)))
        program['records'].append(record)
        for start in starts:
            # A late class probe and a carry-in excerpt occur in a continuous
            # world. Counts remain based on actual measured activity.
            for offset in (0.,float(plan.get('environment_probe_offset_s',7.6))):
                lo=round((start+offset)*10);hi=lo+ticks
                if hi>300:raise ValueError('environment crop exceeds the continuous world')
                eid=f'event_{len(scene["events"]):04d}'
                scene['events'].append({'event_id':eid,'endpoint_id':ep,'source_id':1001+index,'class_id':mapped,'start_tick':lo,'end_tick':hi,'mechanism':mechanism})
                program['events'].append({'event_id':eid,'sound_asset_id':record['sound_asset_id'],'crop_start_sample':crop})


def augment_environmental_native_bundle(raw,scene):
    ids={str(s['actor_id']) for s in scene['sources'].values() if s.get('native_visual_kind')=='audio_only'}
    for event in raw['audio_program']['events']:
        if event.get('actor_id') in ids:
            event['native_audio_source_id']=event['actor_id'];event['actor_id']=None
            event['visual_binding']='unbound_audio_only_native_emitter'
    return raw


def collect_admitted_class_questions(raw,window_id,media,proof,base_audit,domain):
    """Audio-only class panel, separate from v6 appearance->class QA-21.

    Activities come from the emitted dry receipt on the original 100ms grid.
    No visible entity is invented for an environmental point. Existing v6 and
    STARSS13 question/answer domains are not edited or silently expanded.
    """
    import hashlib
    from collections import Counter
    lo=round(float(proof['window_start_s'])*16000);hi=lo+160000
    tracks={}
    for eid,event in proof['events'].items():
        cls=event.get('sound_class')
        if cls not in domain or event.get('emission_mechanism','').startswith('human_live'):continue
        endpoint=event.get('native_endpoint_id')
        if not endpoint:raise ValueError('admitted class panel needs actual native source identity')
        key=(cls,endpoint);ticks=tracks.setdefault(key,set())
        for a,b in event['activity_intervals_samples_global']:
            a=max(a,lo);b=min(b,hi)
            if b>a:ticks.update(range((a-lo)//1600,math.ceil((b-lo)/1600)))
    segments=[]
    for (cls,endpoint),ticks in tracks.items():
        ordered=sorted(ticks)
        if not ordered:continue
        first=previous=ordered[0]
        for t in ordered[1:]+[ordered[-1]+2]:
            if t!=previous+1:segments.append({'sound_class':cls,'endpoint':endpoint,'start_tick':first,'end_tick':previous+1});first=t
            previous=t
    # Count competing onsets in every actually audible track, including people.
    all_tracks={}
    for event in proof['events'].values():
        key=event.get('native_endpoint_id')
        ticks=all_tracks.setdefault(key,set())
        for a,b in event['activity_intervals_samples_global']:
            a=max(a,lo);b=min(b,hi)
            if b>a:ticks.update(range((a-lo)//1600,math.ceil((b-lo)/1600)))
    starts=Counter()
    for ticks in all_tracks.values():
        starts.update(t for t in ticks if t-1 not in ticks)
    public=[];gold=[]
    for s in sorted(segments,key=lambda s:(s['start_tick'],s['endpoint'])):
        if starts[s['start_tick']]!=1:continue
        tag=f"{s['endpoint']}__{s['start_tick']}__{s['end_tick']}";logical=f'{window_id}__admitted_class34__{tag}'
        when='片段开始时已在响的那个声音' if s['start_tick']==0 else f"约第{s['start_tick']/10:.1f}秒开始的那个声音"
        # A unique onset identifies the target; labels and waveform provenance
        # remain private, like the original STARSS class question.
        q={'question_id':logical+'__open','logical_question_id':logical,'qa_id':'V7-CLASS34','protocol_panel':'v7_admitted_recording_classes','modality':'audio','form':'open','question_zh':when+'属于哪一类声音？','question_en':'Which admitted sound class is the sound at the stated onset?','media':dict(media),'answer_type':'closed_set','choices':[{'id':c,'text':c} for c in domain]}
        a={'question_id':q['question_id'],'logical_question_id':logical,'canonical_answer':s['sound_class'],'answer':s['sound_class'],'comparison_key':'admitted_class34/'+tag,'evidence':{'protocol':'continuous_sim_recorded_class_100ms_v1','window_start_s':proof['window_start_s'],'activity':s,'domain':list(domain),'source':'actual emitted dry source measured activity and upstream recording admission','native_source':'observed Habitat audio-only scenegraph point'},'counterfactual_role':'audio'}
        authority={'status':'pass','question_id':q['question_id'],'qa_id':q['qa_id'],'modality':'audio',
            'authority_kind':'v7_admitted_recording_class_100ms_v1','predicate_source':'actual admitted dry receipt activity; no v6 QA predicate modified',
            'forms':{'open':{'answer_type':'closed_set','closed_set_policy':'token_negation_v2','classes':{c:[c] for c in domain},'truth':s['sound_class']}}}
        a['authority_question_id']=q['question_id']
        a['authority_truth']={'canonical':s['sound_class'],'label':s['sound_class']}
        # v6_items is the historical transport slot used by the unchanged
        # training packer. The item explicitly declares its new authority kind.
        base_audit.setdefault('v6_items',[]).append(authority)
        public.append(q);gold.append(a)
    base_audit['admitted_class_domain_observation']={'domain':list(domain),'panel':'V7-CLASS34','v6_QA21_modified':False,'STARSS13_modified':False,'ambiguous_onsets_excluded':sum(v>1 for v in starts.values()),'logical_candidates':len(public)}
    return public,gold


def reserve_critical_audio_questions(chosen,gold,pool,private,required_class):
    """Select actual existing candidates, preserving every modality allocation."""
    by_id={r['question_id']:r for r in private};chosen=list(chosen);gold=list(gold)
    required=[('STARSS-SPEAKERS',None),('V7-CLASS34',required_class)]
    protected=set()
    for family,answer in required:
        eligible=[q for q in pool if q['qa_id']==family and q['form']=='open' and
            (answer is None or str(by_id[q['question_id']]['canonical_answer'])==answer)]
        if not eligible:raise ValueError('critical actual candidate missing: '+family+'/'+str(answer))
        keep=next((q for q in chosen if q in eligible),None) or min(eligible,key=lambda q:q['question_id'])
        if keep not in chosen:
            available=[i for i,q in enumerate(chosen) if q['modality']=='audio' and q['question_id'] not in protected]
            if not available:raise ValueError('critical audio reservation exceeds unchanged modality quota')
            index=available[-1];chosen[index]=keep;gold[index]=by_id[keep['question_id']]
        protected.add(keep['question_id'])
    return chosen,gold
