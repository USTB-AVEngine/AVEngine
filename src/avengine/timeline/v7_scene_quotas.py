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

QUOTA_VERSION = 'v7-scene-quotas-20261004-v4'
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
            'reverb_stratum':'long_qualification_pending' if i%10==9 else ('short' if (i+i//10)%2==0 else 'medium'),
            'reverb_targets_s':{'short':[.2,.5],'medium':[.5,1.5],'long_qualification_pending':[1.5,2.5]},
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


DIAGNOSTIC_INPUT_VERSION = 'v7-production-inputs-20261004-v5'
DIAGNOSTIC_QUOTA_VERSION = 'v7-scene-diagnostic-quotas-20261004-v5'


def add_diagnostic_quotas(rows):
    """Extend fixed seeds without real-data priors or invented native truth.

    World azimuth and camera-relative azimuth are different axes. The 64-cell
    pool uses world azimuth: rotating the camera makes in/out possible in each
    world sector. A separate uniform camera-relative direction probe includes
    behind-camera sources; a behind-camera point cannot be in its own frustum.
    Clear in-frame live people necessarily make frame_has_person true. Native
    resolution must reject physically incompatible combinations, never relabel.
    """
    result=[];cell_counts=Counter();frustum_counts=Counter()
    for original in rows:
        validate_quota_row(original)
        row=deepcopy(original);q=row['scene_quotas'];i=row['scene_index'];n=q['speakers']
        row['base_quota_schema']=row['schema'];row['schema']=DIAGNOSTIC_INPUT_VERSION
        row['previous_window_start_ticks']=list(row['window_start_ticks'])
        seed=row['seeds']['window']
        row['window_start_ticks']=[40+2*(seed%8),160+2*((seed//8)%8)]
        raw=i%22;segments=max(n+1 if n else 0,24 if raw==21 else raw)
        if n==0:
            low_activity_cycle=(i//9)%11
            if low_activity_cycle in (0,1):segments=0
            elif low_activity_cycle in (2,3):segments=1
        sector=i%8;choices=[]
        for inside in (True,False):
            for people in (False,True):
                for occluded in (False,True):
                    if n==0 and not people:continue
                    if q['starss13_focus_class'] in (0,1) and inside and not occluded and not people:continue
                    key=(sector,inside,people,occluded)
                    choices.append((cell_counts[key],frustum_counts[inside],key))
        _,_,key=min(choices)
        _,inside,people,occluded=key
        cell_counts[key]+=1;frustum_counts[inside]+=1
        q['focus_frustum']='in_frame' if inside else 'out_of_frame'
        q['focus_occluded']=occluded
        q['diagnostic']={
            'schema':DIAGNOSTIC_QUOTA_VERSION,
            'sampling_recipe_revision':'balanced_feasible_grid_low_activity_vertical_ties_v3',
            'focus_world_azimuth_octant':i%8,
            'world_octant_centers_deg':[-157.5+45*j for j in range(8)],
            'camera_relative_direction_probe_octant':(5*i)%8,
            'frame_has_person':people,
            'visible_silent_bystander_required':people,
            'nonspeech_ambient_noise_required':True,
            'target_activity_segments_per_window':segments,
            'segment_count_scope':'protocol target activities; nonspeech room-tone noise is a separate noise bus',
            'carry_in_required':segments>0,
            'same_integer_second_multiple_onsets_required':segments>=4,
            'short_stop_ticks':1 if segments>=2 else None,
            'vertical_probe':('center','above','below','top_edge','bottom_edge','left_edge','right_edge')[(3*i)%7],
            'near_boundary_probe_boundary_px':(.25,.75,1.25,1.75)[(i//8)%4],
            'low_light_required':bool((i//7)%2),
            'independent_voice_appearance_seeds':True,
            'distance_stratum':('near','middle','far')[(i//11)%3],
            'gain_db':(-6.,0.,6.)[(i//13)%3],
            'overlap_stratum':('turn_taking','two_overlap','many_overlap')[(i//17)%3],
            'counterfactual_camera_source_pair':{
                'shared':['dry_content','voice_identity','house','materials','unaffected_actor_state'],
                'allowed_change':'camera_pose' if (i//64)%2==0 else 'focus_source_position',
                'refresh':['camera_FOA_rotation','all_audio_derived_readouts','native_pixels','pixel_projection','private_evidence'],
            },
            'e_s_model_inputs':{'source':'deployment_predictor','ground_truth_injection':False,'predictor_status':'awaiting_registered_predictor'},
            'pixel_coordinate_contract':{'source':'zero_based_pixel_center','to_boundary_offset':.5,
                'boundary_domain':'[0,W) x [0,H)','two_pixel_dead_zone':False},
            'native_resolution_status':'requested_only',
        }
        row['native_quota_status']='awaiting_native_layout_dry_motion_family_and_predictor_admission'
        result.append(row)
    return result


def validate_diagnostic_row(row):
    if row.get('schema')!=DIAGNOSTIC_INPUT_VERSION:
        raise ValueError('unsupported diagnostic seed schema')
    legacy=deepcopy(row);legacy['schema']=QUOTA_INPUT_VERSION
    q=validate_quota_row(legacy);d=q.get('diagnostic',{})
    if d.get('schema')!=DIAGNOSTIC_QUOTA_VERSION or not 0<=d.get('focus_world_azimuth_octant',-1)<8 or not 0<=d.get('camera_relative_direction_probe_octant',-1)<8:
        raise ValueError('diagnostic azimuth axes missing')
    minimum=q['speakers']+1 if q['speakers'] else 0
    if d['target_activity_segments_per_window']<minimum or row['duration_s']<30:
        raise ValueError('independent continuous world/speaker repeat schedule is impossible')
    if q['speakers']==0 and not d['frame_has_person']:
        raise ValueError('zero-speaker window needs a visible silent person')
    if d['e_s_model_inputs']['ground_truth_injection'] is not False:
        raise ValueError('E/S inputs must come from the deployment predictor')
    return q


def diagnostic_summary(rows):
    for row in rows:validate_diagnostic_row(row)
    def count(fn):return dict(sorted(Counter(str(fn(r)) for r in rows).items()))
    cells=count(lambda r:(r['scene_quotas']['diagnostic']['focus_world_azimuth_octant'],
        r['scene_quotas']['focus_frustum'],r['scene_quotas']['diagnostic']['frame_has_person'],r['scene_quotas']['focus_occluded']))
    return {'status':'requests_only','independent_parent_count':len(rows),
        'requested_world_azimuth':count(lambda r:r['scene_quotas']['diagnostic']['focus_world_azimuth_octant']),
        'requested_camera_relative_direction_probe':count(lambda r:r['scene_quotas']['diagnostic']['camera_relative_direction_probe_octant']),
        'requested_frustum':count(lambda r:r['scene_quotas']['focus_frustum']),
        'requested_frame_has_person':count(lambda r:r['scene_quotas']['diagnostic']['frame_has_person']),
        'requested_occlusion':count(lambda r:r['scene_quotas']['focus_occluded']),
        'requested_64_cell_counts':cells,
        'requested_speakers':count(lambda r:r['scene_quotas']['speakers']),
        'requested_activity_segments':count(lambda r:r['scene_quotas']['diagnostic']['target_activity_segments_per_window']),
        'requested_vertical':count(lambda r:r['scene_quotas']['diagnostic']['vertical_probe']),
        'requested_starss13':count(lambda r:r['scene_quotas']['starss13_focus_class']),
        'requested_qa21_34':count(lambda r:r['scene_quotas']['qa21_34_focus_class']),
        'actual_questions_by_request_cell':{k:0 for k in cells},
        'actual_generated_questions':0,'formal_media_qualified_questions':0,
        'simulation_model_recall_status':'not_run; deployment predictor not supplied',
        'real_dev_used_for_sampling':False}


def diagnostic_activity_events(sources, starts_s, segment_count, duration_s=30.):
    """Independent 30s world; exact activity counts in disjoint 10s windows.

    Supports are half-open 100ms ticks. Actor 1 returns after a 100ms gap;
    extra people use different identities. This function never supplies an
    actor-count answer: that supervision remains in the QA-family consumer.
    """
    if duration_s<30 or segment_count<0:
        raise ValueError('invalid continuous world/activity request')
    starts=[round(t*10) for t in starts_s]
    if any(b-a<100 for a,b in zip(starts[:-1],starts[1:])):
        raise ValueError('exact-count schedule needs disjoint crops')
    if any(t<7 or t+100>round(duration_s*10) for t in starts):
        raise ValueError('crops must have continuous history and fit the world')
    speakers=sorted((k for k,v in sources.items() if v['class_id'] in (0,1)),key=lambda k:sources[k]['actor_id'])
    nonvoices=sorted(k for k,v in sources.items() if v['class_id'] not in (0,1))
    if speakers and segment_count<len(speakers)+1:
        raise ValueError('schedule cannot realize all actors and a repeated person')
    if segment_count and not (speakers or nonvoices):
        raise ValueError('activity requires admitted physical sound sources')
    events=[]
    for window,t in enumerate(starts):
        if not segment_count:continue
        endpoint_order=([speakers[0],speakers[0],*speakers[1:]] if speakers else [nonvoices[0]]*2)
        while len(endpoint_order)<segment_count:endpoint_order.extend(speakers or nonvoices)
        lo=[t-7]
        if segment_count>=2:lo.append(t+4)
        if segment_count==3:lo.append(t+8)
        elif segment_count>=4:
            # Two onsets in one integer second for every crop offset, while
            # keeping separate half-open activities and the 100ms first stop.
            paired_first=10*math.ceil((t+7)/10)+1
            lo.extend((paired_first,paired_first+4))
            if segment_count>4:lo.extend(np.linspace(paired_first+8,t+92,segment_count-4,dtype=int).tolist())
        for j,(start,endpoint) in enumerate(zip(lo,endpoint_order[:segment_count],strict=True)):
            end=t+3 if j==0 else min(start+2,t+99)
            spec=sources[endpoint]
            events.append({'event_id':f'diagnostic_w{window}_e{j:03d}','endpoint_id':endpoint,
                'source_id':spec['actor_id'],'class_id':spec['class_id'],'start_tick':start,'end_tick':end})
    return events


def pixel_center_to_boundary(u,v,width,height):
    """Convert once; near-edge centers have no unlabelled two-pixel exclusion."""
    u,v=float(u)+.5,float(v)+.5
    return [u,v], bool(0<=u<width and 0<=v<height)
