"""Select registered movable source roles without scene-specific identifiers."""
from copy import deepcopy
import numpy as np

SOURCE_TYPES=('human_speech','human_other','dog_walk','cat_walk','animal_slide','human_footsteps')
EMISSION_SPEEDS={'human_speech':(.82,1.2),'human_other':(.82,1.2),'dog_walk':(.9,1.2),'cat_walk':(.6,.8),'animal_slide':(.4,.5),'human_footsteps':(.8,1.4)}
QUIET_SPEEDS={'human_speech':.81,'human_other':.81,'dog_walk':.51,'cat_walk':.31,'animal_slide':.21,'human_footsteps':.8}

def activity(record):
    return record.get('screens',{}).get('automatic_activity',{}).get('activity_intervals_samples') or record.get('source_normalization',{}).get('measured_activity',{}).get('source_activity_intervals_samples') or []

def select_source_program(records,allowed_classes,speed_range):
    """Longest complete qualified call, with bounded outward source context.

    A long recording can supply one complete measured call. Other calls outside
    that crop remain in source evidence; activity inside the crop is never hidden.
    No waveform is repeated, stretched, or trimmed inside the selected call.
    """
    candidates=[];minimum=max(.8,.4/(float(speed_range[0])*.85))
    for record in records:
        if record.get('sound_class')not in allowed_classes or record.get('source_type')=='author_tts':continue
        sr=int(record.get('header',{}).get('sample_rate_hz',16000));n=int(record.get('header',{}).get('sample_count',round(float(record.get('duration_s',0))*sr)))
        if sr!=16000 or n<=0:continue
        spans=activity(record)
        if not spans:continue
        full_span=(spans[-1][1]-spans[0][0])/sr
        if n/sr<=2.4 and full_span>=minimum:
            candidates.append((full_span,record['sound_asset_id'],deepcopy(record),None))
        for index,(lo,hi)in enumerate(spans):
            if (hi-lo)/sr<minimum:continue
            begin=max(0,int((lo-400)//1600)*1600);end=min(n,int(np.ceil((hi+400)/1600))*1600)
            # Outward alignment and source availability are required, never pad
            # a cropped physical effect with invented samples.
            end=end//1600*1600
            if end<hi or not 0<end-begin<=38400:continue
            profile={'crop_start_sample':begin,'sample_count':end-begin,'selected_activity_index':index,
                'required_activity_intervals_samples':[[int(lo),int(hi)]],
                'reason':'complete longest measured call; outward 100ms context; all in-crop activity retained'}
            candidates.append(((hi-lo)/sr,record['sound_asset_id'],deepcopy(record),profile))
    if not candidates:raise ValueError('no complete admitted source/call with >=%.3fs activity and <=2.4s program for '%minimum+str(sorted(allowed_classes)))
    candidates.sort(key=lambda x:(-x[0],x[1],x[3]is not None))
    span,_,record,profile=candidates[0]
    return {'record':record,'program':profile,'measured_activity_span_s':span,'minimum_designed_radial_change_m':span*speed_range[0]*.85}

def select_motion_role(source_type,human,catalog,records,contract,seed,attempt=0):
    if source_type not in SOURCE_TYPES:raise ValueError('unsupported moving physical source type: '+source_type)
    if source_type=='human_speech':return {'asset':human,'record':None,'program':None}
    if source_type=='human_footsteps':return {'asset':human,'record':None,'program':None,'event_clock':'native-foot-contact-bouts'}
    band=EMISSION_SPEEDS[source_type]
    if source_type=='human_other':
        allowed=set(human['allowed_classes'])-{'speech_playback','footsteps'}
        return {'asset':human,**select_source_program(records,allowed,band)}
    if contract is None:raise ValueError('animal motion role requires the unchanged production authority contract')
    eligible=[]
    for asset in catalog:
        authority=contract.get('animal_motion',{}).get(asset['asset_id'])
        if asset.get('kind')!='animal' or not authority or not asset.get('standing_registered_qualified'):continue
        mode=authority['mode']
        if source_type=='animal_slide':ok=mode=='slide_no_animation'
        else:ok=mode=='walk_animation' and asset['species_id']==('dog'if source_type=='dog_walk'else'cat')
        if ok:eligible.append(asset)
    eligible.sort(key=lambda a:a['asset_id']);rng=np.random.default_rng(seed)
    for index in rng.permutation(len(eligible)):
        asset=eligible[(int(index)+attempt)%len(eligible)]
        try:return {'asset':asset,**select_source_program(records,set(asset['allowed_classes']),band)}
        except ValueError:continue
    raise ValueError('no complete admitted moving source/call combination for '+source_type)

def choose_motion_source_type(state):
    counts={kind:0 for kind in SOURCE_TYPES}
    for parent in state.get('parents',{}).values():
        kind=parent.get('motion_source_type')
        if kind in counts:counts[kind]+=1
    return min(SOURCE_TYPES,key=lambda kind:(counts[kind],SOURCE_TYPES.index(kind)))

def classify_source_type(spec):
    if spec.get('species_id')=='human' or spec.get('entity_class')=='articulated_human':
        cls=spec.get('sound_class')or spec.get('content_kind')
        return 'human_footsteps'if cls=='footsteps'else'human_speech'if cls in {'speech','speech_playback'}else'human_other'if cls not in {None,'silent'}else'human_silent'
    mode=spec.get('motion_metadata',{}).get('mode')
    if mode=='slide_no_animation':return 'animal_slide'
    return 'dog_walk'if spec.get('species_id')=='dog'else'cat_walk'if spec.get('species_id')=='cat'else'other_moving_animal'
