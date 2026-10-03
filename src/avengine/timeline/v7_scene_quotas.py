"""Balanced v7 scenario requests; planned quotas never override observed truth.

The native layout/dry asset resolver must satisfy a request or reject it. These
requests are deliberately distinct from completed, media-qualified questions.
The QA-family implementation belongs to avengine.qa, not this sampler.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import math
import numpy as np
from avengine.timeline.continuous_scene import QA21_CLASSES, camera_basis, position_at

QUOTA_VERSION = 'v7-scene-quotas-20261004-v3'
QUOTA_INPUT_VERSION = 'v7-production-inputs-20261004-v4'
CELLS = ('in_clear', 'in_occluded', 'out_clear', 'out_occluded')


def starss_focus_class(index):
    # Pairwise swaps preserve balanced class totals and the fixed prefixes,
    # while zero-speaker windows cannot contain live male/female speech.
    value=(7*index)%13
    if index%9==0 and value in (0,1):return (7*(index+1))%13
    if index%9==1 and (7*(index-1))%13 in (0,1):return (7*(index-1))%13
    return value


def add_quotas(rows):
    result=[]
    for original in rows:
        row=deepcopy(original);i=row['scene_index']
        row['base_seed_schema']=row['schema'];row['schema']=QUOTA_INPUT_VERSION
        row['scene_quotas']={
            'schema':QUOTA_VERSION,
            'speakers':i%9,
            'same_actor_repeated':i%9>0,
            'multiple_people_same_direction':i%9>=2 and (i//9)%2==0,
            'starss13_focus_class':starss_focus_class(i),
            'qa21_34_focus_class':QA21_CLASSES[(13*i)%34],
            'focus_frustum':'in_frame' if (i+i//34)%2==0 else 'out_of_frame',
            'focus_occluded':bool((i//2)%2),
            'spatial_counterfactual_cells':list(CELLS),
            'spatial_member_to_focus_slot':{'f0_o0':'in_clear','f0_o1':'in_occluded','f1_o0':'out_clear','f1_o1':'out_occluded'},
            'spatial_shared_factors':['house','room','materials','camera','background','all_actor_roots','event_times'],
            'spatial_allowed_changes':['dry_bus_to_preexisting_emitter_slot'],
            'on_screen_silent_bystander_required':True,
            'spatial_counterfactual_requested':i%4==0,
            'camera_yaw_deg':float(np.random.default_rng(row['seeds']['camera']).uniform(-180,180)),
            'RIR_profile':'native_diffraction_on_qualification_pending',
            'reverb_stratum':('short','medium','long_qualification_pending')[(i+i//9)%3],
            'noise_profile':('clean','stationary','nonstationary','impulsive')[(i+i//34)%4],
            'noise_SNR_db':(None,20.,10.,0.)[(i+i//34)%4],
            'interferer_count':(i//4)%4,
            'count_supervision':'unique stable actor_id, never segments/voices/azimuth bins',
            'admission':'requested_only; measured media/answers determine realized counts',
        }
        row['native_quota_status']='awaiting_native_layout_dry_and_family_admission'
        result.append(row)
    return result


def validate_quota_row(row):
    q=row['scene_quotas']
    if row['schema']!=QUOTA_INPUT_VERSION or q['schema']!=QUOTA_VERSION:
        raise ValueError('unsupported quota schema')
    if not 0<=q['speakers']<=8 or q['focus_frustum'] not in ('in_frame','out_of_frame'):
        raise ValueError('invalid speaker/frustum quota')
    if q['qa21_34_focus_class'] not in QA21_CLASSES or not 0<=q['starss13_focus_class']<13:
        raise ValueError('invalid class quota')
    if q['spatial_counterfactual_cells']!=list(CELLS) or not -180<=q['camera_yaw_deg']<180:
        raise ValueError('invalid spatial/camera recipe')
    if q['speakers']==0 and q['starss13_focus_class'] in (0,1):
        raise ValueError('zero live speakers cannot contain native live speech focus')
    return q


def quota_summary(rows):
    for row in rows:validate_quota_row(row)
    def count(key):return dict(sorted(Counter(str(r['scene_quotas'][key]) for r in rows).items()))
    return {'parent_count':len(rows),'requested_speakers':count('speakers'),
        'requested_frustum':count('focus_frustum'),'requested_occlusion':count('focus_occluded'),
        'requested_starss13':count('starss13_focus_class'),'requested_qa21_34':count('qa21_34_focus_class'),
        'requested_noise':count('noise_profile'),'requested_reverb':count('reverb_stratum'),
        'requested_interferers':count('interferer_count'),
        'requested_spatial_groups':sum(r['scene_quotas']['spatial_counterfactual_requested'] for r in rows),
        'actual_generated_questions':0,'formal_media_qualified_questions':0,
        'status':'seed_requests_only'}


def camera_forward_for_request(row, reference_forward):
    q=validate_quota_row(row);angle=math.radians(q['camera_yaw_deg'])
    c,s=math.cos(angle),math.sin(angle)
    return (np.array([[c,0,s],[0,1,0],[-s,0,c]])@camera_basis(reference_forward)[0]).tolist()


def speech_events(speakers, duration_s=30, window_starts_s=(4.,14.)):
    """Stable identities, carry-in, repeats and overlap on half-open 100 ms ticks.

    Class/voice/position admission is supplied by the resolver. Zero speakers
    means no live speech; independent nonspeech activity may still be present.
    """
    if not 0<=speakers<=8 or duration_s<=0:raise ValueError('invalid population/duration')
    stop=round(duration_s*10);events=[]
    for actor in range(1,speakers+1):
        for block in range(0,stop,100):
            # One continuing voice plus repeated, staggered activity from each
            # person. Different people may share direction, never identity.
            spans=[(block,block+64),(block+72,block+78),(block+84,block+90),(block+96,block+99)] if actor==1 else [
                (block+32+(actor-2),block+50+(actor-2)),(block+53+(actor-2),block+61+(actor-2)),
                (block+66+(actor-2),block+78+(actor-2)),(block+82+(actor-2),block+88+(actor-2)),
                (block+92+(actor-2),block+99)]
            for lo,hi in spans:
                hi=min(hi,stop)
                if hi>lo:events.append({'event_id':f'activity{len(events):04d}',
                    'endpoint_id':f'voice{actor}','source_id':actor,'class_id':1,'start_tick':lo,'end_tick':hi})
    return events


def scene_from_native_layout(row, house, layout, speaker_specs):
    """Resolve the crowd request from native poses and explicitly admitted voices.

    A layout records native navigation and scan-ray evidence, camera and actor
    anchors. No projection/occlusion flag is substituted for native evidence.
    Formal usage must additionally qualify RGB masks, acoustics and each asset.
    """
    q=validate_quota_row(row);n=q['speakers']
    if layout.get('status')!='native_measured' or len(layout['speaker_anchors_m'])<n:
        raise ValueError('quota needs enough independently measured native anchors')
    if len(speaker_specs)!=n or len({s['actor_id'] for s in speaker_specs})!=n:
        raise ValueError('quota requires unique physical actor identities')
    sources={}
    for index,spec in enumerate(speaker_specs,1):
        if spec['actor_id']!=index or spec['class_id'] not in (0,1) or not spec.get('voice'):
            raise ValueError('live speech identity/class/voice is missing')
        position=layout['speaker_anchors_m'][index-1]
        sources[f'voice{index}']={**deepcopy(spec),'trajectory':[
            {'time_s':0.,'position_m':position},{'time_s':float(row['duration_s']),'position_m':position}]}
    events=speech_events(n,row['duration_s'])
    for event in events:event['class_id']=sources[event['endpoint_id']]['class_id']
    return {'scene_id':row['scene_id'],'house':house['house'],'split':row['split'],'seed':row['scene_seed'],
        'duration_s':row['duration_s'],'sample_rate_hz':16000,'video_fps':15,
        'listener_m':layout['listener_m'],'camera_forward':layout['camera_forward'],
        'sources':sources,'events':events,'window_starts_s':[t/10 for t in row['window_start_ticks'][:row['window_count']]],
        'independent_parent_scene_id':row['scene_id'],'paired_length_comparison':False,
        'scene_quota_request':q,'native_layout_evidence':layout,
        'visual_status':'native_layout_only_RGB_asset_and_energy_admission_pending'}


def frustum_geometry(position,listener,forward,width=1280,height=720,hfov_deg=85.):
    f,r,u=camera_basis(forward);delta=np.asarray(position)-listener;z=float(delta@f)
    if z<=0:return False
    focal=width/(2*math.tan(math.radians(hfov_deg/2)))
    px=width/2+focal*float(delta@r)/z;py=height/2-focal*float(delta@u)/z
    return 0<=px<width and 0<=py<height


def select_spatial_cells(candidates,listener,forward):
    """Select source slots by measured LOS and geometric frustum, independently.

    All four source slots/background/camera/materials are held across members;
    only dry emission association selects another premeasured physical slot.
    An on-screen clear bystander is separate from the active source slot.
    Missing cells reject a layout; occlusion never changes FRUSTUM truth.
    """
    result={}
    for point in candidates:
        if point.get('native_LOS_measured') is not True:raise ValueError('unmeasured occlusion')
        inside=frustum_geometry(point['position_m'],np.asarray(listener),forward)
        key=('in' if inside else 'out')+('_occluded' if point['blocked'] else '_clear')
        result.setdefault(key,deepcopy(point))
    if set(result)!=set(CELLS):raise ValueError('native layout lacks four spatial cells')
    return result
