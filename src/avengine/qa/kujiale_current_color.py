"""Reject a colour-dependent candidate unless its current native scene agrees."""
from __future__ import annotations
from copy import deepcopy
import re
from avengine.capture.kujiale_pilot_environment import current_color_evidence

COLOR_WORDS=re.compile(r'颜色|蓝色|黄色|绿色|红色|白色|黑色|灰色|棕色|米色|酒红色|\b(?:color|colour|blue|yellow|green|red|white|black|gray|grey|brown|beige)\b',re.I)

def filter_current_scene_colors(public,private,audit,visual,start):
    """The asset stays admitted; only candidates with unsupported colour references leave."""
    raw=audit['native_bundle'];appearance=raw.get('appearance_review',{}).get('actors',{})
    physical=current_color_evidence(visual,start);answers={g['question_id']:g for g in private}
    rejected=[];checked=[]
    for q in public:
        text=q.get('question_zh','')+' '+q.get('question_en','')
        if q.get('modality')=='audio' or not COLOR_WORDS.search(text):continue
        referenced=[]
        for aid,actor in raw['actors'].items():
            label=actor.get('display_label','')
            if label and label in text and COLOR_WORDS.search(label):referenced.append(str(aid))
        # A producer's structured references take precedence where offered.
        referenced.extend(map(str,q.get('color_actor_ids',[])))
        referenced=list(dict.fromkeys(referenced))
        failures=[]
        for aid in referenced:
            if aid in physical:
                if physical[aid]['status']!='pass':failures.append({'actor_id':aid,'reason':'current_native_main_family_does_not_match_registry','measurement':physical[aid]})
            elif aid not in appearance or not appearance[aid].get('frame_refs'):
                failures.append({'actor_id':aid,'reason':'no_matching_native_colour_pixels_in_this_window'})
        if not referenced:failures.append({'reason':'colour_reference_has_no_current_native_actor_identity'})
        record={'question_id':q['question_id'],'referenced_actors':referenced,'status':'reject' if failures else 'pass','failures':failures}
        checked.append(record)
        if failures:rejected.append(record)
    ids={r['question_id'] for r in rejected}
    return [q for q in public if q['question_id'] not in ids],[g for g in private if g['question_id'] not in ids],{
        'status':'pass','checked_forms':len(checked),'discarded_forms':len(ids),
        'discarded_logical_questions':len({q['logical_question_id'] for q in public if q['question_id'] in ids}),
        'checks':checked,'rejected':rejected,'physical_current_window_colours':physical,
        'asset_labels_modified':False,'asset_admission_changed_by_colour':False}
