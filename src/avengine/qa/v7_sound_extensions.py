"""Extended real-recording classes without enlarging QA-21 or STARSS domains."""
from __future__ import annotations
import hashlib
import re
import numpy as np

STARSS_EXTENSION_MAP = {
    'clapping':2,'laughter':4,'footsteps':6,'door_open_close':7,'door_knock':12,
    'coughing':None,'guitar_live':9,'violin_live':9,'piano_live':9,
    'saxophone_live':9,'flute_live':9,'hand_drum_live':9,
    'pet_bird_parrot':None,'pet_bird_canary':None,
}
EXTENDED_CLASSES = tuple(STARSS_EXTENSION_MAP)

def recording_metadata_blockers(row):
    metadata=row.get('source_metadata') or row.get('official_metadata') or {}
    text=' '.join(str(metadata.get(k,'')) for k in ('title','description','tags')).lower()
    reasons=[]
    if re.search(r'\b(starss|dcase|tau.srir|binaural|ambisonic|surround|midi|vst|vsti|soundfont|shreddage|synth|synthesis|oscillator|synthesized|synthesised|synthesizer|synthesiser|virtual.instrument)\b',text):
        reasons.append('synthetic_spatial_or_prohibited_source_metadata')
    if re.search(r'\b(?:made|created|generated|synthesized|synthesised)\s+(?:with|using|in)\s+(?:audacity|supercollider|csound|python|software|tone.generator)\b',text):
        reasons.append('explicit_program_generated_effect_not_permitted')
    cls=row['sound_class']
    if cls in {'door_open_close','door_knock'} and not re.search(r'\bdoor(?:s)?\b',text):
        reasons.append('source_metadata_does_not_confirm_door')
    if cls=='flute_live' and re.search(r'\b(recorder|pan.flute|panflute|ocarina|tin.whistle)\b',text):
        reasons.append('flute_recording_is_different_instrument_subtype')
    if cls=='piano_live' and re.search(r'\b(rhodes|electric.piano|digital.piano|electronic.piano|epiano|synth.piano|synthesized)\b',text):
        reasons.append('piano_recording_not_confirmed_acoustic_piano')
    if cls=='hand_drum_live' and not re.search(r'\b(tabla|bongo|bongos|conga|congas|djembe|darbuka|doumbek|frame.drum|hand.drum|tambourine)\b',text):
        reasons.append('recorded_hand_drum_subtype_missing')
    if cls.startswith('pet_bird_'):
        patterns={'pet_bird_parrot':r'\b(parrot|parrots|parakeet|parakeets|budgie|budgies|cockatiel|cockatiels|macaw|macaws|cockatoo|cockatoos)\b',
                  'pet_bird_canary':r'\b(canary|canaries)\b'}
        if not re.search(patterns[cls],text):reasons.append('original_bird_species_metadata_missing')
        if cls=='pet_bird_canary' and re.search(r'\bcanary\s+(island|islands|wharf)\b',text) and not re.search(r'\b(canaries|pet.canary|canary.bird|canary.song)\b',text):
            reasons.append('canary_location_is_not_species_evidence')
    return sorted(set(reasons))

def ontology_relatives(ontology):
    children={r['id']:set(r.get('child_ids',())) for r in ontology}
    parents={}
    for node, nodes in children.items():
        for child in nodes:parents.setdefault(child,set()).add(node)
    def closure(graph, start):
        visited=set();queue=list(start)
        while queue:
            node=queue.pop()
            if node in visited:continue
            visited.add(node);queue.extend(graph.get(node,()))
        return visited
    return {node:closure(children,[node])|closure(parents,[node]) for node in children}

def score_and_rival(row, targets, relatives):
    scores=row['scores_by_mid']; targets=set(targets)
    compatible=set().union(*(relatives.get(mid,{mid}) for mid in targets)) if targets else set()
    confidence=max((scores.get(mid,0.) for mid in targets),default=0.)
    rivals={mid:value for mid,value in scores.items() if mid not in compatible}
    rival=max(rivals,key=rivals.get) if rivals else None
    return float(confidence),float(rivals.get(rival,0.)),rival

def calibrate_extended(rows, target_mids, ontology):
    """Calibrate each new acoustic class against actual source-backed negatives.

    AudioSet ancestors/descendants are compatible, not deliberately wrong
    labels. Species/subtypes still require original source metadata. Neither
    scene answers nor real evaluation media set these thresholds.
    """
    relatives=ontology_relatives(ontology);result={}
    calibration=[]
    for row in rows:
        identity=row['original_group']
        if int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)%4==0:
            calibration.append(row)
    calibration_groups={r['original_group'] for r in calibration}
    for cls,targets in target_mids.items():
        positives=[];negatives=[];negative_margins=[];positive_margins=[]
        for r in calibration:
            score,rival,_=score_and_rival(r,targets,relatives)
            if cls in r['candidate_classes'] and not r.get('calibration_source_blockers'):
                positives.append(score);positive_margins.append(score-rival)
            elif not any(set(targets)&relatives.get(mid,{mid}) for mid in specific_source_mids(r.get('trusted_original_mids',()),ontology)):
                negatives.append(score);negative_margins.append(score-rival)
        if not positives or not negatives:
            result[cls]={'status':'blocked','reason':'no_source_backed_positive_and_negative_calibration',
                         'positive_count':len(positives),'negative_count':len(negatives)}
            continue
        threshold=max(.05,float(np.quantile(negatives,.99))+1e-6)
        margin=max(.05,float(np.quantile(negative_margins,.99))+1e-6)
        passed=lambda score,delta:score>=threshold and delta>=margin
        holdout=[r for r in rows if r['original_group'] not in calibration_groups and cls in r['candidate_classes'] and not r.get('calibration_source_blockers')]
        holdout_ok=[passed(s,s-v) for r in holdout for s,v,_ in [score_and_rival(r,targets,relatives)]]
        positive_ok=[passed(s,d) for s,d in zip(positives,positive_margins)]
        result[cls]={'status':'pass' if any(positive_ok) else 'no_passing_positive_calibration','threshold':threshold,'minimum_margin':margin,
             'positive_count':len(positives),'negative_count':len(negatives),
             'observed_negative_false_accept_rate':float(np.mean([passed(s,d) for s,d in zip(negatives,negative_margins)])),
             'calibration_positive_retention':float(np.mean([passed(s,d) for s,d in zip(positives,positive_margins)])),
             'holdout_positive_count':len(holdout),'holdout_positive_retention':float(np.mean(holdout_ok)) if holdout_ok else None,
             'source_group_disjoint':True,'control_source':'official original FSD dev collection labels / metadata; deliberately wrong incompatible AudioSet heads',
             'human_listening_used':False,'model_parent_classes_are_not_wrong_labels':True}
    return result,relatives

def specific_source_mids(mids, ontology):
    """Prune broad parents when the same source has a specific child label."""
    children={r['id']:set(r.get('child_ids',())) for r in ontology}
    def descendants(node):
        seen=set();queue=list(children.get(node,()))
        while queue:
            child=queue.pop()
            if child in seen:continue
            seen.add(child);queue.extend(children.get(child,()))
        return seen
    mids=set(mids)
    return {mid for mid in mids if not (descendants(mid)&(mids-{mid}))}

def calibrate_caption_model(rows, classes, ontology):
    """Disjoint source calibration of CLAP cosine scores, not probabilities."""
    relatives=ontology_relatives(ontology)
    calibration=[r for r in rows if int(hashlib.sha256(r['original_group'].encode()).hexdigest()[:8],16)%4==0]
    calibration_groups={r['original_group'] for r in calibration}
    gates={}
    for cls,targets in classes.items():
        pos=[];negative=[]
        for row in calibration:
            score,margin=caption_score(row,cls)
            if cls in row['candidate_classes'] and not row.get('calibration_source_blockers'):
                pos.append((score,margin))
            elif not any(set(targets)&relatives.get(mid,{mid}) for mid in specific_source_mids(row.get('trusted_original_mids',()),ontology)):
                negative.append((score,margin))
        if not pos or not negative:
            gates[cls]={'status':'blocked','reason':'source_backed_caption_controls_missing'}
            continue
        threshold=max(.15,float(np.quantile([s for s,m in negative],.99))+1e-6)
        margin=max(.02,float(np.quantile([m for s,m in negative],.99))+1e-6)
        accepted=lambda s,m:s>=threshold and m>=margin
        good=[accepted(s,m) for s,m in pos]
        holdout=[r for r in rows if r['original_group'] not in calibration_groups and cls in r['candidate_classes'] and not r.get('calibration_source_blockers')]
        gates[cls]={'status':'pass' if any(good) else 'no_passing_positive_calibration',
         'threshold_cosine':threshold,'minimum_cosine_margin':margin,'positive_count':len(pos),'negative_count':len(negative),
         'calibration_positive_retention':float(np.mean(good)),
         'observed_negative_false_accept_rate':float(np.mean([accepted(s,m) for s,m in negative])),
         'holdout_positive_count':len(holdout),
         'holdout_positive_retention':float(np.mean([accepted(*caption_score(r,cls)) for r in holdout])) if holdout else None,
         'source_group_disjoint':True,'human_listening_used':False,
         'control_source':'official original FSD collection labels / source metadata or known real VCTK speech; wrong nonoverlapping specific source classes'}
    return gates

def caption_score(row, cls):
    scores=row['cosine_scores']
    # Both bird flavours share an acoustic call check; original species
    # metadata is mandatory. A generic instrument ancestor is never a rival.
    compatible={cls}
    if cls.startswith('pet_bird_'):
        compatible|={'pet_bird_parrot','pet_bird_canary'}
    confidence=float(scores.get(cls,-1.))
    rival=max((float(v) for k,v in scores.items() if k not in compatible),default=-1.)
    return confidence,confidence-rival
