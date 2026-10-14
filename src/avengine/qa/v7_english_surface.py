"""English presentation from admitted appearance metadata; registries stay exact."""
from __future__ import annotations
import re

ENGLISH_SURFACE_REVISION='v7-natural-English-surface-20261015-v1'
OBJECT_NAMES={'floorstanding_speaker':'floor-standing speaker','bookshelf_speaker':'bookshelf speaker',
              'kitchen_timer':'kitchen timer','carbon_monoxide_alarm':'carbon monoxide alarm',
              'air_conditioner':'air conditioner','desk_telephone':'desk telephone',
              'doorbell_chime_unit':'doorbell chime unit','security_keypad':'security keypad',
              'alarm_clock':'alarm clock','wall_clock':'wall clock'}
COLOR_NAMES={'gray_silver':'silver-grey','wood_brown':'brown','burgundy':'burgundy'}


def appearance_description_en(asset,color=None):
    kind=asset.get('kind');species=asset.get('species_id')
    color=COLOR_NAMES.get(color,color or '')
    color=str(color).replace('_',' ')
    if kind in {'human','performer'} or species=='human' or asset.get('entity_class')=='articulated_human':
        gender=asset.get('visual_gender')
        noun='man' if gender=='male' else 'woman' if gender=='female' else 'person'
        return f'the {noun} in the {color} shirt' if color else f'the {noun}'
    if kind=='animal' or species in {'dog','cat','bird'}:
        noun=species if species in {'dog','cat','bird'} else 'animal'
    else:
        raw=asset.get('object_type') or asset.get('entity_class') or 'object'
        noun=OBJECT_NAMES.get(raw,str(raw).replace('_',' '))
        if noun in {'static object','loudspeaker'}:noun='speaker' if noun=='loudspeaker' else 'object'
    return 'the '+((color+' ') if color else '')+noun


def tidy_english(text):
    text=str(text)
    for token,name in OBJECT_NAMES.items():text=re.sub(r'\b'+re.escape(token)+r'\b',name,text)
    return re.sub(r'\b(the|a|an)\s+\1\b',r'\1',text,flags=re.I)


def protocol_question_en(family,evidence):
    if family=='QA-23':
        return 'How many continuous sound-activity segments occur in this 10-second clip? Count a segment already active at the start once; a missing 100-ms activity frame starts a new segment. Count segments, not people or actions.'
    if family=='STARSS-SPEAKERS':
        return 'How many different people produce live speech at any time in this 10-second clip? Count each person once, including speech already active at the start. Exclude speech played by a device.'
    anchor=float(evidence.get('anchor_s',0))
    subject='the sound already active at the start of the clip' if anchor==0 else f'the sound starting at about {anchor:g} seconds'
    if family=='QA-04':return f'Which of the eight horizontal directions does {subject} come from at its onset? Use the camera-facing direction as front.'
    if family=='QA-25':return f'What is the horizontal angle of {subject} at {evidence["query_s"]:g} seconds? Front is 0 degrees, left is positive and right is negative. Give an integer in [-180, 179].'
    if family=='QA-21':return f'Which sound category describes {subject}?'
    if family=='STARSS-CLASS13':return f'Which of the 13 STARSS sound classes describes {subject}?'
    if family=='STARSS-FRUSTUM':return f'Is the direction of {subject} inside or outside the camera frustum at its onset? Ignore occlusion.'
    raise ValueError('missing English protocol template: '+family)


def appearance_replacements(assets,facts=None):
    """Map existing presentation labels through their admitted asset metadata."""
    byid={a['asset_id']:a for a in assets};replacements={}
    colors_zh={'blue':'蓝色','yellow':'黄色','green':'绿色','white':'白色','black':'黑色',
               'gray':'灰色','gray_silver':'灰银色','wood_brown':'木棕色','beige':'米色','brown':'棕色','red':'红色'}
    for actor in (facts or {}).get('actors',{}).values():
        a=byid.get(actor.get('asset_id'))
        if a is None:continue
        appearance=actor.get('appearance') or {}
        color=appearance.get('value') or a.get('label_color') or a.get('color') or a.get('appearance',{}).get('top_color')
        description=appearance_description_en(a,color)
        for value in [appearance.get('label_zh'),appearance.get('label_en'),actor.get('display_label')]:
            if value and (not str(value).isdigit()):replacements[str(value)]=description
    for a in assets:
        color=a.get('label_color') or a.get('color') or a.get('appearance',{}).get('top_color')
        description=appearance_description_en(a,color)
        name=a.get('name_zh')
        labels=[a.get('appearance_zh')]
        if name and color:
            labels += [colors_zh.get(color,color)+'的'+name,color+' '+name]
        for value in labels:
            if value:replacements[str(value)]=description
    return replacements


def relabel_english(text,replacements):
    text=str(text)
    for old in sorted(replacements,key=len,reverse=True):text=text.replace(old,replacements[old])
    return tidy_english(text)


def reissue_item_english(item,replacements):
    """Re-render labels and accepted aliases; canonical values stay exact."""
    from copy import deepcopy
    result=deepcopy(item)
    def labels(node):
        if isinstance(node,dict):
            for key,value in list(node.items()):
                if key in {'question_en','label_en','label_zh','label'} and isinstance(value,str):
                    node[key]=relabel_english(value,replacements)
                elif key=='classes' and isinstance(value,dict):
                    for canonical,aliases in value.items():
                        if isinstance(aliases,list):
                            value[canonical]=list(dict.fromkeys(aliases+[relabel_english(a,replacements) for a in aliases]))
                else:labels(value)
        elif isinstance(node,list):
            for child in node:labels(child)
    labels(result.get('model_input',{}));labels(result.get('forms',{}))
    if isinstance(result.get('truth'),dict) and 'label' in result['truth']:
        result['truth']['label']=relabel_english(result['truth']['label'],replacements)
    if result['truth'].get('value')!=item['truth'].get('value'):raise ValueError('English presentation changed canonical truth')
    return result


def content_answer_en(value):
    """Render retained content labels without changing the private canonical."""
    value=str(value)
    direct={'是':'yes','否':'no','现场人物':'a person speaking live','音箱播放':'audio playback from a speaker'}
    if value in direct:return direct[value]
    for zh,en in [('蓝色','blue'),('黄色','yellow'),('绿色','green'),('白色','white'),('黑色','black')]:
        if value==zh+'上衣的人':return 'the person in the '+en+' shirt'
    return tidy_english(value)


def content_question_en(query):
    kind=query['kind']
    texts={'same_class_actor':'Who is speaking at this time?',
           'first_actor':'Who started speaking first in this clip?',
           'offscreen_origin':'Is the voice at this time coming from {subject}?',
           'live_or_playback':'Is the voice at this time live speech or playback from a speaker?'}
    if kind not in texts:raise ValueError('unsupported English content query: '+kind)
    text=texts[kind].format(subject=content_answer_en(query.get('subject_descriptor_zh','the visible person')))
    if kind!='first_actor':
        time=float(query.get('public_time_s',query['time_s']))
        text='At '+format(time,'.12g')+' seconds in this clip, '+text[0].lower()+text[1:]
    return text
