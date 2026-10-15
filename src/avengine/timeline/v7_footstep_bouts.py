"""Construct owner-defined footstep bouts from actual native step evidence.

Rendering keeps its original left/right attached emitters. Question events group
same-person step onsets separated by at most 600 ms; they never add audio or
invent continuous activity in the gaps. Wet ends use measured native FIR support.
"""
from copy import deepcopy
import hashlib
import math

REVISION='native-footstep-bouts-onset-gap-600ms-v2-native-foot-geometry'

def group_footstep_bouts(scene, *, maximum_onset_gap_s=.6):
    groups={}
    for event in scene.get('events',[]):
        if event.get('sound_class')!='footsteps':continue
        endpoint=event['endpoint_id'];spec=scene['sources'][endpoint]
        owner=spec.get('owner_endpoint_id',endpoint)
        groups.setdefault(owner,[]).append(event)
    result=[]
    for owner,steps in sorted(groups.items()):
        active=[]
        for step in sorted(steps,key=lambda e:(e['start_tick'],e['event_id'])):
            if active and (step['start_tick']-active[-1]['start_tick'])/10.>maximum_onset_gap_s+1e-12:
                result.append((owner,active));active=[]
            active.append(step)
        if active:result.append((owner,active))
    return result

def footstep_bout_question_inputs(scene,evidence):
    """Return question-only scene/evidence; original media/program stay untouched.

    This also works for old admitted renders. Old gaps exceeding the explicit
    600 ms rule remain separate bouts; sparse old footsteps are never filled.
    """
    groups=group_footstep_bouts(scene)
    if not groups:return scene,evidence
    if evidence.get('footstep_bouts',{}).get('revision')==REVISION:return scene,evidence
    result=deepcopy(scene);proof=deepcopy(evidence)
    raw_ids={event['event_id']for _,steps in groups for event in steps}
    raw_endpoints={event['endpoint_id']for _,steps in groups for event in steps}
    retained=[event for event in result['events']if event['event_id']not in raw_ids]
    records=proof['events'];raw={key:deepcopy(records[key])for key in raw_ids}
    for key in raw_ids:records.pop(key)
    protocol=proof.get('protocol_scene');receipts=[]
    if protocol:
        protocol['events']=[e for e in protocol['events']if e['endpoint_id']not in raw_endpoints]
    for owner,steps in groups:
        ids=[event['event_id']for event in steps]
        missing=[key for key in ids if not raw[key].get('wet_intervals_global_s')]
        if missing:raise ValueError('footstep bout needs actual measured wet tail for every component: '+str(missing))
        activity=sorted([span for key in ids for span in raw[key]['activity_intervals_samples_global']])
        wet=sorted([span for key in ids for span in raw[key]['wet_intervals_global_s']])
        if not activity:raise ValueError('footstep bout lacks actual source samples')
        first=activity[0][0]/16000.;dry_end=max(span[1]for span in activity)/16000.;wet_end=max(span[1]for span in wet)
        identity=hashlib.sha256(('\0'.join([owner,*ids])).encode()).hexdigest()[:20]
        event={**steps[0],'event_id':'footstep_bout_'+identity,'endpoint_id':steps[0]['endpoint_id'],
               'source_id':result['sources'][owner]['actor_id'],'start_tick':math.floor(first*10),
               'end_tick':min(math.ceil(wet_end*10),round(float(scene.get('duration_s',wet_end))*10)),'emission_end_s':dry_end,'wet_end_s':wet_end,
               'event_unit':'footstep_bout','component_event_ids':ids,'mechanism':'walking_human_footstep_bout'}
        retained.append(event)
        representative=steps[0]['endpoint_id']
        spec=result['sources'][representative]
        spec.update(sound_class='footsteps',class_id=event['class_id'],content_kind='footsteps',mechanism=event['mechanism'])
        first_record=deepcopy(raw[ids[0]])
        first_record.pop('transcript',None)
        first_record['wet_support_readback']={'method':'conservative union of actual component native wet supports','component_event_ids':ids,'support_samples_global':[math.floor(min(span[0]for span in wet)*16000),math.ceil(wet_end*16000)]}
        first_record.update(event_unit='footstep_bout',native_endpoint_id=representative,
            emission_mechanism=event['mechanism'],activity_intervals_samples_global=activity,
            wet_intervals_global_s=[[min(span[0]for span in wet),wet_end]],
            event_segmentation={'method':REVISION,'maximum_adjacent_onset_gap_s':.6,
                'component_event_ids':ids,'first_step_s':first,'last_step_emission_end_s':dry_end,
                'measured_last_wet_end_s':wet_end,'raw_steps_retained':True},
            event_geometry_source='actual native attached foot chosen from the bout; same person identity; both physical render streams retained',
            physical_render_endpoints=sorted({e['endpoint_id']for e in steps}),
            component_sound_asset_ids=sorted({raw[key]['sound_asset_id']for key in ids}))
        records[event['event_id']]=first_record
        if protocol:
            protocol['events'].append({k:event[k]for k in ('event_id','endpoint_id','source_id','class_id','start_tick','end_tick','mechanism')})
            protocol['sources'][representative]=deepcopy(spec)
        receipts.append({'event_id':event['event_id'],'owner_endpoint_id':owner,'step_event_ids':ids,
            'step_onsets_s':[e['start_tick']/10. for e in steps],
            'emission_interval_s':[first,dry_end],'annotated_interval_with_measured_wet_tail_s':[first,wet_end]})
    result['events']=sorted(retained,key=lambda e:(e['start_tick'],e['event_id']))
    proof['footstep_bouts']={'revision':REVISION,'raw_step_count':len(raw_ids),'bout_count':len(groups),
        'bouts':receipts,'raw_step_evidence':raw,'waveforms_or_native_geometry_changed':False}
    result['footstep_bout_annotation_revision']=REVISION
    return result,proof
