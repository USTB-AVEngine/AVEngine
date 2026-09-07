"""Fixed-profile QA sampling over existing navigation and prepared sounds.

This module writes neutral plans only. Native execution and answerability are
separate: planned geometry never becomes an achieved-condition assertion.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.capture.neutral_readback import COORDINATE_FRAME
from avengine.qa.answerability import line_of_sight
from avengine.rooms.walkable_space import camera_grid
from avengine.routes.trajectory import resample_polyline_by_arc_length

POLICY = 'conditioned_static_v2'
RIGID = {'rigid_object', 'rigid_static_object'}


class ConditionedPlanningFailure(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__('fixed condition profile exhausted: ' + str(result['failure_histogram']))


class CandidateFailure(ValueError):
    def __init__(self, stage, reason):
        self.stage, self.reason = stage, reason
        super().__init__(reason)


def _draw(value, rng):
    if not isinstance(value, Mapping):
        return deepcopy(value)
    choices = value.get('choices', value.get('bins'))
    if not isinstance(choices, list) or not choices:
        raise ValueError('condition distribution needs nonempty choices/bins')
    weights = value.get('weights')
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(choices) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('invalid explicit condition weights')
        weights = weights / weights.sum()
    return deepcopy(choices[int(rng.choice(len(choices), p=weights))])


def resolve_condition_profile(request, registry):
    """Resolve N/S/classes and all quota choices once, before room/route retries."""
    rng = np.random.default_rng(int(request.get('seed', 0)))
    camera = request.get('camera', {})
    if camera.get('motion', request.get('camera_motion', 'static')) != 'static':
        raise ValueError('conditioned_static_v2 requires static camera motion')
    entities, profile = request.get('entities', {}), request.get('profile', {})
    explicit = request.get('source_asset_ids')
    records = {r['asset_id']: r for r in registry['assets']}
    count_spec = entities.get('total_count', len(explicit) if explicit is not None else 2)
    if explicit is not None:
        n = len(explicit)
        allowed = count_spec.get('choices') if isinstance(count_spec, Mapping) else [count_spec]
        if n not in allowed or len(set(explicit)) != n:
            raise ValueError('explicit source IDs must be distinct and agree with total_count')
        if any(a not in records for a in explicit):
            raise ValueError('explicit source asset is absent from the runtime registry')
        classes = ['rigid_static_object' if records[a]['entity_class'] in RIGID else records[a]['entity_class'] for a in explicit]
    else:
        n = int(_draw(count_spec, rng)); classes = None
    if not 2 <= n <= 4:
        raise ValueError('conditioned sampler requires 2 to 4 entities')
    silent = int(_draw(entities.get('silent_count', request.get('silent_actor_count', 0)), rng))
    if not 0 <= silent < n:
        raise ValueError('silent_count must retain a speaking entity')
    minimum = int(entities.get('min_articulated_count', 0))
    if not 0 <= minimum <= n:
        raise ValueError('invalid min_articulated_count')
    if classes is None:
        class_spec = entities.get('source_classes', 'articulated_human')
        for _ in range(10000):
            classes = [_draw(class_spec, rng) for _ in range(n)]
            if sum(c != 'rigid_static_object' for c in classes) >= minimum:
                break
        else:
            raise ValueError('source class distribution cannot meet min_articulated_count')
    if any(c not in {'articulated_human', 'articulated_animal', 'rigid_static_object'} for c in classes):
        raise ValueError('unknown source class in condition profile')
    if sum(c != 'rigid_static_object' for c in classes) < minimum:
        raise ValueError('explicit sources violate min_articulated_count')
    speakers = sorted(int(i) for i in rng.choice(n, n-silent, replace=False))
    anchor_count = int(profile.get('anchor_count', 1))
    if not 1 <= anchor_count <= len(speakers):
        raise ValueError('anchor_count must be within the speaking count')
    anchors = sorted(int(i) for i in rng.choice(speakers, anchor_count, replace=False))
    sep_spec = profile.get('separation_bin_deg', {'bins': [[15.,30.],[30.,60.],[60.,90.],[90.,180.]]})
    sep = _draw(sep_spec, rng)
    floor = float(sep_spec.get('floor_deg', 15.)) if isinstance(sep_spec, Mapping) else float(profile.get('separation_floor_deg', 15.))
    if len(sep) != 2 or not 0 <= floor <= float(sep[0]) < float(sep[1]) <= 180.:
        raise ValueError('invalid separation bin or floor; lower bounds are never relaxed')
    if profile.get('competitor_set','all_other_entities_including_offscreen')!='all_other_entities_including_offscreen':
        raise ValueError('competitor_set must include every other entity')
    if profile.get('separation_window','whole_audible_window_of_anchor')!='whole_audible_window_of_anchor':
        raise ValueError('separation must hold over the whole anchor audible window')
    result = {'total_count': n, 'speaking_count': n-silent, 'silent_count': silent,
              'source_classes': classes, 'anchor_count': anchor_count, 'separation_bin_deg': list(map(float, sep)),
              'separation_floor_deg': floor, 'speaking_indices': speakers, 'anchor_indices': anchors,
              'competitor_set': 'all_other_entities_including_offscreen',
              'native_start_hold_frames': request.get('start_hold_frames',0),
              'anchor_visibility': _draw(profile.get('anchor_visibility', 'in_fov'), rng),
              'anchor_line_of_sight': _draw(profile.get('anchor_line_of_sight', 'clear'), rng),
              'speech_motion': _draw(profile.get('speech_motion', 'all_still'), rng),
              'event_relation': _draw(profile.get('event_relation', request.get('audio_mode', 'sequential')), rng),
              'min_gap_between_audible_windows_s': float(profile.get('min_gap_between_audible_windows_s', .5)),
              'reserve_tail_s': float(profile.get('reserve_tail_s', 3.)),
              'minimum_overlap_s': float(profile.get('minimum_overlap_s', .3)),
              'retry_budget_within_profile': int(profile.get('retry_budget_within_profile', 200))}
    for key, valid in [('anchor_visibility', {'in_fov','off_screen'}), ('anchor_line_of_sight', {'clear','occluded'}),
                       ('speech_motion', {'speaker_moving','competitor_moving','all_still'}), ('event_relation', {'sequential','overlap','repeat'})]:
        if result[key] not in valid:
            raise ValueError('unsupported '+key)
    if not 1 <= result['retry_budget_within_profile'] <= 200:
        raise ValueError('retry_budget_within_profile must be within 1..200')
    if any(not math.isfinite(result[k]) or result[k]<0 for k in ('reserve_tail_s','minimum_overlap_s','min_gap_between_audible_windows_s')):
        raise ValueError('event interval parameters must be finite and nonnegative')
    return result


def neutral_source_declaration(record, actor_id):
    anchor_id = record['default_emitter_anchor_id']
    anchor = next(a for a in record['emitter_anchors'] if a['anchor_id'] == anchor_id)
    binding = {'source_slot_id': actor_id, 'asset_id': record['asset_id'], 'asset_revision': record['revision'],
               'semantic_anchor_id': anchor_id, 'emitter_offset_m': deepcopy(anchor['offset_m']),
               'offset_space': anchor['offset_space']}
    if anchor.get('local_basis'):
        binding['local_basis'] = deepcopy(anchor['local_basis'])
    return {'actor_id': actor_id, 'asset_id': record['asset_id'], 'asset_revision': record['revision'],
            'entity_class': record['entity_class'], 'identity': deepcopy(record['identity']),
            'realized_attributes': deepcopy(record['realized_attributes']), 'display_label': record['display_label'],
            'emitter_binding': binding, 'timeline': deepcopy(record.get('timeline')),
            'motion_model': 'rigid_static' if record['entity_class'] in RIGID else 'articulated',
            'native_binding_status': 'pending_executor'}


def select_entities(request, profile, registry, rng):
    records = registry['assets']; chosen = []; explicit = request.get('source_asset_ids')
    for i, cls in enumerate(profile['source_classes']):
        pool = [r for r in records if (r['entity_class'] == cls or cls == 'rigid_static_object' and r['entity_class'] in RIGID)
                and r['asset_id'] not in {x['asset_id'] for x in chosen}
                and (explicit is None or r['asset_id'] == explicit[i])]
        if not pool:
            raise CandidateFailure('assets', 'no_registered_asset_for_'+cls)
        chosen.append(neutral_source_declaration(pool[int(rng.integers(len(pool)))], f'source{i+1}'))
    return chosen


def _gender(value):
    return {'m':'male', 'f':'female', 'male':'male', 'female':'female'}.get(str(value).lower())


def sound_matches(actor, sound):
    sound_class = str(sound.get('sound_class', '')).lower()
    if actor['entity_class'] == 'articulated_human':
        if sound_class in {'speech', 'speech_playback'}:
            sex = _gender(actor['realized_attributes'].get('sex_or_gender_label'))
            gender = _gender(sound.get('gender'))
            return sex is not None and gender is not None and sex == gender
        return sound_class in {'laughter', 'cough', 'sneeze'}
    if actor['entity_class'] == 'articulated_animal':
        species = actor['identity'].get('species_id')
        declared = sound.get('species_id')
        return species is not None and ((declared == species) or sound_class in sound.get('compatible_sound_classes_by_species', {}).get(species, []))
    # Devices have no biological gender. Class/asset allowlists, when declared, still apply.
    allowed = sound.get('compatible_asset_ids')
    categories = sound.get('compatible_object_categories')
    return (not allowed or actor['asset_id'] in allowed) and (not categories or actor['identity'].get('category') in categories)


def select_sounds(actors, sounds, profile, clock, request, rng):
    sr = int(clock['sample_rate_hz']); config = request.get('sound_selection', {})
    max_samples = int(round(float(config.get('max_clip_s', 5.)) * sr))
    deadline = int(clock['sample_count']) - int(round(profile['reserve_tail_s'] * sr))
    speakers = profile['speaking_indices']; order = list(speakers); rng.shuffle(order)
    preallocated = config.get('preallocated_sound_asset_ids_by_actor')
    if preallocated is not None:
        if not isinstance(preallocated, Mapping):
            raise ValueError('preallocated sounds must be an actor-to-sound-ID mapping')
        actor_ids = {actor['actor_id'] for actor in actors}
        if set(preallocated) - actor_ids:
            raise ValueError('preallocated sounds contain an unknown actor')
        for i in speakers:
            allowed = preallocated.get(actors[i]['actor_id'])
            if not isinstance(allowed, list) or any(not isinstance(value, str) or not value for value in allowed):
                raise ValueError('preallocated sounds must explicitly cover every speaking actor')
    pools = {}
    for i in speakers:
        allowed = None if preallocated is None else preallocated[actors[i]['actor_id']]
        pool = [s for s in sounds if (allowed is None or s.get('sound_asset_id') in allowed)
                and sound_matches(actors[i], s) and 0 < int(s['sample_count']) <= max_samples
                and int(s.get('sample_rate_hz', sr)) == sr
                and (s.get('sound_class') not in {'speech','speech_playback'} or float(s.get('active_duration_s', 0)) >= float(config.get('min_audible_s', 1.5)))]
        if not pool:
            raise CandidateFailure('sounds', 'no_compatible_prepared_sound_'+actors[i]['actor_id'])
        pools[i] = pool
    repeated = speakers[int(rng.integers(len(speakers)))] if profile['event_relation']=='repeat' else None
    selected = {}; transcripts = set(); remaining = deadline
    gap = int(round(profile['min_gap_between_audible_windows_s'] * sr))
    for pos, i in enumerate(order):
        rest = order[pos+1:]
        # Retain a feasible duration suffix before drawing a clip.
        suffix = sum(min(int(s['sample_count']) for s in pools[j])*(2 if j==repeated else 1) for j in rest) + gap * (len(rest)+int(repeated in rest))
        limit = (remaining-suffix-gap*int(i==repeated))//(2 if i==repeated else 1) if profile['event_relation'] in {'sequential','repeat'} else deadline
        pool = [s for s in pools[i] if int(s['sample_count']) <= limit and
                (not config.get('unique_first_utterance_transcripts', True) or not s.get('transcript') or s['transcript'] not in transcripts)]
        if not pool:
            raise CandidateFailure('sounds', 'clip_budget_or_transcript_candidates_exhausted')
        sound = deepcopy(pool[int(rng.integers(len(pool)))]); sound['actor_id'] = actors[i]['actor_id']
        if i==repeated:sound['repeat_requested']=True
        selected[i] = sound
        if sound.get('transcript'): transcripts.add(sound['transcript'])
        remaining -= (int(sound['sample_count']) + gap)*(2 if i==repeated else 1)
    return selected


def _moving_flags(profile, actors, rng):
    flags = [False]*len(actors)
    if profile['speech_motion'] == 'speaker_moving':
        for i in profile['anchor_indices']: flags[i] = True
    elif profile['speech_motion'] == 'competitor_moving':
        candidates = [i for i,a in enumerate(actors) if a['entity_class'] not in RIGID and i not in profile['anchor_indices']]
        if not candidates:
            raise CandidateFailure('routes', 'no_articulated_competitor_for_moving_profile')
        flags[candidates[int(rng.integers(len(candidates)))]] = True
    if any(flags[i] and a['entity_class'] in RIGID for i,a in enumerate(actors)):
        raise CandidateFailure('routes', 'static_entity_cannot_satisfy_required_motion')
    if profile['speech_motion']=='speaker_moving':
        for i,a in enumerate(actors):
            if i not in profile['anchor_indices'] and a['entity_class'] not in RIGID:flags[i]=bool(rng.integers(2))
    return flags


def _native_routes(space, flags, frames, fps, rng, *, start_hold_frames=None):
    from avengine.rooms.native_qa_room import _hold_endpoint_path
    if float(space.frame_rate_hz) != fps:
        raise CandidateFailure('routes', 'native_route_clock_mismatch')
    bank = space.route_bank()
    if len(bank) < len(flags):raise CandidateFailure('routes','insufficient_native_routes')
    # Reuse the prior audit's complete native clique enumeration, now in production.
    # It keeps every unchanged native pair constraint and has no scored prefix.
    cache=getattr(space,'_qa_native_group_cache',{})
    if frames not in cache:
        full=np.asarray([_hold_endpoint_path(np.asarray(row['points_m'],dtype=float),frames) for row in bank])
        adjacency=[set() for _ in bank];pairs=[];triples=[];quads=[]
        for a in range(len(full)):
            candidates=np.arange(a+1,len(full));first=np.linalg.norm(full[candidates,0]-full[a,0],axis=1);last=np.linalg.norm(full[candidates,-1]-full[a,-1],axis=1)
            candidates=candidates[(first>=.95)&(last>=.95)&(first<=3.5)&(last<=3.5)]
            if len(candidates):
                good=np.all(np.sum((full[candidates]-full[a])**2,axis=-1)>=.95**2,axis=-1)
                for b in candidates[good]:
                    b=int(b);adjacency[a].add(b);adjacency[b].add(a);pairs.append((a,b))
        for a,b in pairs:
            common=sorted(c for c in adjacency[a]&adjacency[b] if c>b)
            for c in common:
                triples.append((a,b,c))
                for d in sorted(adjacency[c].intersection(common)):
                    if d>c:quads.append((a,b,c,d))
        cache[frames]={2:pairs,3:triples,4:quads};setattr(space,'_qa_native_group_cache',cache)
    groups=cache[frames][len(flags)]
    if not groups:raise CandidateFailure('routes','no_legal_native_route_group')
    ids=list(groups[int(rng.integers(len(groups)))]);rng.shuffle(ids)
    paths = []
    for i, selected in enumerate(ids):
        p = np.asarray(bank[int(selected)]['points_m'], dtype=float)
        if flags[i]:
            # Delay is legal only when the full captured route clock is kept.
            max_delay = max(0, frames-len(p))
            if start_hold_frames is not None and (isinstance(start_hold_frames,bool) or not isinstance(start_hold_frames,int) or not 0<=start_hold_frames<=max_delay):
                raise CandidateFailure('routes','requested_native_delay_cannot_preserve_full_route')
            delay = int(rng.integers(max_delay+1)) if start_hold_frames is None else start_hold_frames
            path = _hold_endpoint_path(p, frames, start_hold_frames=delay)
        else:
            endpoint = p[int(rng.integers(2)) * (len(p)-1)]
            path = np.repeat(endpoint[None], frames, axis=0)
        paths.append(path)
    if any(np.linalg.norm(a-b, axis=1).min() < .95 for a,b in itertools.combinations(paths, 2)):
        raise CandidateFailure('routes', 'native_group_separation_below_0.95_m')
    if any(np.linalg.norm(a[0]-b[0]) > 3.5 or np.linalg.norm(a[-1]-b[-1]) > 3.5 for a,b in itertools.combinations(paths,2)):
        raise CandidateFailure('routes', 'native_group_endpoint_separation_above_3.5_m')
    return paths, {'authority':space.metadata.get('route_authority','retained_native_route_bank'),'selected_route_ids':[bank[int(i)]['route_id'] for i in ids],
                   'selection':'uniform_over_all_legal_native_groups','legal_native_group_count':len(groups),'minimum_separation_m':.95}


def sample_routes(space, actors, profile, clock, rng, region=None, *, required_windows=None):
    frames, fps = int(clock['frame_count']), float(clock['frame_rate_hz'])
    flags = _moving_flags(profile, actors, rng)
    required_windows=required_windows or {}
    required_frames=max([1]+[int(math.ceil((b-a)*fps/clock['sample_rate_hz']))+1 for a,b in required_windows.values()])
    if space.route_bank() is not None:
        paths, metadata = _native_routes(space, flags, frames, fps, rng, start_hold_frames=profile.get('native_start_hold_frames'))
    else:
        paths=[]; records=[]; hub=space.sample_navigable(rng, region)
        bounds=space.bounds().copy() if region is None else np.asarray(region, dtype=float).copy()
        bounds[0,[0,2]]=np.maximum(bounds[0,[0,2]],hub[[0,2]]-3.1)
        bounds[1,[0,2]]=np.minimum(bounds[1,[0,2]],hub[[0,2]]+3.1)
        for i, required_motion in enumerate(flags):
            start=space.sample_navigable(rng, bounds)
            if any(np.linalg.norm(start-p[0]) < .95 for p in paths):
                raise CandidateFailure('routes','initial_source_separation_below_0.95_m')
            route=np.repeat(start[None], frames, axis=0); record={'motion':'static','route_points_m':None}
            if required_motion:
                end=space.sample_navigable(rng,bounds); poly=space.shortest_path(start,end)
                if poly is None or len(poly)<2:
                    raise CandidateFailure('routes','no_existing_navigation_path')
                length=float(np.linalg.norm(np.diff(poly,axis=0),axis=1).sum())
                if length < max(1.5,(required_frames-1)/fps*.5):
                    raise CandidateFailure('routes','path_too_short_for_moving_window')
                speed=float(rng.uniform(.5,.8)); moving_frames=max(2,int(math.ceil(length/speed*fps))+1)
                if moving_frames>=frames or moving_frames<required_frames:
                    raise CandidateFailure('routes','route_does_not_fit_clock_or_required_window')
                pause=int(rng.integers(max(1,int(fps)),max(2,int(2*fps))+1))
                modes=['walk_with_sampled_holds']
                if moving_frames>=2*required_frames and moving_frames+pause<frames:modes.append('walk_with_sampled_pause')
                mode=modes[int(rng.integers(len(modes)))];extra=pause if mode=='walk_with_sampled_pause' else 0
                start_frame=int(rng.integers(frames-moving_frames-extra+1));motion=resample_polyline_by_arc_length(poly,moving_frames)
                if extra:
                    split=int(rng.integers(required_frames,moving_frames-required_frames+1))
                    motion=np.concatenate([motion[:split],np.repeat(motion[split-1:split],pause,axis=0),motion[split:]])
                route[start_frame:start_frame+len(motion)]=motion;route[start_frame+len(motion):]=motion[-1]
                record={'motion':mode,'start_frame':start_frame,'end_frame_exclusive':start_frame+len(motion),'route_points_m':poly.tolist(),
                        'required_contiguous_motion_frames':required_frames}
            if not all(space.is_navigable(p) for p in route):
                raise CandidateFailure('routes','sampled_path_left_existing_navigation')
            if any(np.linalg.norm(route-p,axis=1).min()<.95 for p in paths):
                raise CandidateFailure('routes','all_frame_source_separation_below_0.95_m')
            paths.append(route);records.append(record)
        metadata={'authority':space.metadata['authority'],'actors':records,'minimum_separation_m':.95}
    moving=[]; rotations=[]; emitters=[]; bodies=[]
    for actor,path in zip(actors,paths):
        delta=np.diff(path,axis=0); delta=np.concatenate([delta,delta[-1:]],axis=0)
        motion=np.linalg.norm(delta,axis=1)*fps > .05; moving.append(motion)
        heading=float(rng.uniform(-math.pi,math.pi)); qs=[]; ep=[]; bp=[]
        offset=np.asarray(actor['emitter_binding']['emitter_offset_m'],dtype=float)
        anatomical=np.asarray((actor.get('timeline') or {}).get('local_anatomical_forward_axis',[1.,0.,0.]))
        anatomical_yaw=math.atan2(-float(anatomical[2]),float(anatomical[0]))
        for p,d,m in zip(path,delta,motion):
            if m: heading=math.atan2(-float(d[2]),float(d[0]))-anatomical_yaw
            c,s=math.cos(heading),math.sin(heading);rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            qs.append([0.,math.sin(heading/2),0.,math.cos(heading/2)])
            ep.append(p+rot@offset);bp.append(p+rot@np.array([0.,max(.05,float(offset[1])*.8),0.]))
        rotations.append(qs);emitters.append(ep);bodies.append(bp)
    return np.asarray(paths),np.asarray(rotations),np.asarray(moving),np.asarray(emitters),np.asarray(bodies),metadata


def _merge(ranges):
    result=[]
    for lo,hi in sorted((int(a),int(b)) for a,b in ranges if a<=b):
        if result and lo<=result[-1][1]+1: result[-1][1]=max(result[-1][1],hi)
        else: result.append([lo,hi])
    return result


def _clip_ranges(ranges,low=-math.inf,high=math.inf):
    return _merge((max(a,low),min(b,high)) for a,b in ranges if max(a,low)<=min(b,high))


def _pick_sample(ranges,rng):
    sizes=[b-a+1 for a,b in ranges]; total=sum(sizes)
    if total<=0: raise CandidateFailure('schedule','no_legal_integer_start_sample')
    draw=int(rng.integers(total))
    for (a,b),size in zip(ranges,sizes):
        if draw<size: return a+draw
        draw-=size
    raise AssertionError('unreachable integer range draw')


def legal_start_ranges(mask, sound, clock, profile):
    """Exact integer starts whose complete audible span lies in a legal frame run."""
    sr,fps=int(clock['sample_rate_hz']),float(clock['frame_rate_hz'])
    deadline=int(clock['sample_count'])-int(round(profile['reserve_tail_s']*sr))
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(int))
    runs=zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
    a,b=int(sound['audible_start_sample']),int(sound['audible_end_sample_exclusive'])
    if not 0<=a<b<=int(sound['sample_count']):
        raise ValueError('prepared activity span is outside the clip')
    return _merge((max(0,int(math.ceil(start*sr/fps))-a),
                   min(deadline-int(sound['sample_count']),int(math.floor(end*sr/fps))-b)) for start,end in runs)


def _sequential_orders(events, starts, clock, profile):
    gap=int(round(profile['min_gap_between_audible_windows_s']*clock['sample_rate_hz']))
    originals=[key for key,event in events.items() if not event.get('repeat_of')]
    repeats=[key for key,event in events.items() if event.get('repeat_of')]
    for order in itertools.permutations(originals):
        orders=[order]
        if repeats:
            repeated=repeats[0]; source=events[repeated]['repeat_of']; idx=order.index(source)
            orders=[order[:pos]+(repeated,)+order[pos:] for pos in range(idx+1,len(order)+1)]
        for candidate in orders:
            latest={}; bound=math.inf; following=None
            for key in reversed(candidate):
                if following is not None:
                    bound=latest[following]+events[following]['audible_start_sample']-events[key]['audible_end_sample_exclusive']-gap
                legal=_clip_ranges(starts[key],high=bound)
                if not legal: break
                latest[key]=legal[-1][1]; following=key
            if len(latest)==len(candidate): yield candidate,latest


def schedule_legal_events(events, starts, clock, profile, rng=None):
    """Choose only starts with a feasible suffix; feasibility calls consume no RNG."""
    relation=profile['event_relation'];sr=int(clock['sample_rate_hz'])
    if relation in {'sequential','repeat'}:
        orders=list(_sequential_orders(events,starts,clock,profile))
        if not orders:return None
        if rng is None:return True
        # First select a feasible actor order uniformly, then legal repeat insertion.
        by_original={}
        for order,latest in orders:
            original=tuple(k for k in order if not events[k].get('repeat_of'))
            by_original.setdefault(original,[]).append((order,latest))
        groups=list(by_original.values());group=groups[int(rng.integers(len(groups)))];order,latest=group[int(rng.integers(len(group)))]
        result={}; previous=None;gap=int(round(profile['min_gap_between_audible_windows_s']*sr))
        for key in order:
            low=0 if previous is None else result[previous]+events[previous]['audible_end_sample_exclusive']+gap-events[key]['audible_start_sample']
            result[key]=_pick_sample(_clip_ranges(starts[key],low,latest[key]),rng);previous=key
        return result
    overlap=int(round(profile['minimum_overlap_s']*sr));pairs=[]
    for a,b in itertools.combinations(events,2):
        ea,eb=events[a],events[b]
        if ea['actor_id']==eb['actor_id'] or min(ea['audible_end_sample_exclusive']-ea['audible_start_sample'],eb['audible_end_sample_exclusive']-eb['audible_start_sample'])<overlap:
            continue
        low_delta=ea['audible_start_sample']+overlap-eb['audible_end_sample_exclusive']
        high_delta=ea['audible_end_sample_exclusive']-overlap-eb['audible_start_sample']
        legal=_merge((max(x,u-high_delta),min(y,v-low_delta)) for x,y in starts[a] for u,v in starts[b])
        if legal:pairs.append((a,b,legal,low_delta,high_delta))
    if not pairs or any(not ranges for ranges in starts.values()):return None
    if rng is None:return True
    a,b,legal,lo,hi=pairs[int(rng.integers(len(pairs)))];result={a:_pick_sample(legal,rng)}
    result[b]=_pick_sample(_clip_ranges(starts[b],result[a]+lo,result[a]+hi),rng)
    for key in events:
        if key not in result:result[key]=_pick_sample(starts[key],rng)
    return result


def _event_bindings(sounds, profile, rng):
    result={f'event_{i+1:03d}':deepcopy(sounds[actor]) for i,actor in enumerate(sorted(sounds))}
    if profile['event_relation']=='repeat':
        keys=[k for k,e in result.items() if e.get('repeat_requested')];original=keys[0] if len(keys)==1 else list(result)[int(rng.integers(len(result)))];repeat=deepcopy(result[original]);repeat['repeat_of']=original
        result[f'event_{len(result)+1:03d}']=repeat
    return result


def select_camera_and_schedule(space, mesh, paths, moving, emitters, bodies, actors,
                                sounds, profile, clock, request, rng, region=None):
    config=request.get('camera',{});fov=float(config.get('fov_deg',request.get('camera_fov_deg',85.)))
    height=float(config.get('height_above_floor_m',1.55))
    resolution=list(config.get('resolution_hw',[720,1280]))
    if len(resolution)!=2 or any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in resolution):raise ValueError('camera resolution must be positive integer [height,width]')
    aspect=resolution[1]/resolution[0]
    if not 0<fov<180 or height<=0:raise ValueError('invalid static camera FOV/height')
    positions=camera_grid(space,step_m=.55,height_above_floor_m=height,region=region)
    yaws=np.deg2rad(np.arange(0,360,15)); forwards=np.c_[np.sin(yaws),np.zeros(24),-np.cos(yaws)]
    rights=np.c_[np.cos(yaws),np.zeros(24),np.sin(yaws)];tangent=math.tan(math.radians(fov)/2)
    events=_event_bindings(sounds,profile,rng); actor_index={a['actor_id']:i for i,a in enumerate(actors)}
    anchors=set(profile['anchor_indices']); legal=[]; stages=Counter();ray_cache={}
    distance_range=request.get('profile',{}).get('distance_range_m',[1.5,4.5])
    for pi,position in enumerate(positions):
        origin=np.asarray(position);floor=origin-np.array([0,height,0])
        if any(np.linalg.norm(path-floor,axis=1).min()<.8 for path in paths):continue
        stages['camera_positions_after_clearance']+=1
        delta=bodies-origin;depth=np.einsum('yc,nfc->ynf',forwards,delta);side=np.einsum('yc,nfc->ynf',rights,delta)
        fov_mask=(depth>.1)&(np.abs(side)<depth*tangent*.93)&(np.abs(delta[None,:,:,1])<depth*tangent/aspect*.9)
        d=emitters-origin;distance=np.linalg.norm(d,axis=-1);in_range=(distance>=distance_range[0])&(distance<=distance_range[1])
        az=np.degrees(np.arctan2(d[:,:,0],-d[:,:,2]));separation=np.abs((az[:,None]-az[None,:]+180)%360-180)
        for i in range(len(actors)):separation[i,i]=np.inf
        nearest=separation.min(axis=1);mask=np.ones_like(fov_mask,dtype=bool)
        low,high=profile['separation_bin_deg']
        for i in profile['speaking_indices']:
            visible=fov_mask[:,i] if i not in anchors or profile['anchor_visibility']=='in_fov' else ~fov_mask[:,i]
            mask[:,i]&=visible&in_range[i][None]
            if i in anchors:
                mask[:,i]&=(nearest[i]>=low)[None]&((nearest[i]<high) | ((high==180)&np.isclose(nearest[i],180)))[None]
                if profile['speech_motion']=='speaker_moving':mask[:,i]&=moving[i][None]
                elif profile['speech_motion']=='competitor_moving':mask[:,i]&=np.any(np.delete(moving,i,axis=0),axis=0)[None]
                elif profile['speech_motion']=='all_still':mask[:,i]&=~np.any(moving,axis=0)[None]
        before=[]
        for yi in range(24):
            starts={key:legal_start_ranges(mask[yi,actor_index[e['actor_id']]],e,clock,profile) for key,e in events.items()}
            if all(starts.values()) and schedule_legal_events(events,starts,clock,profile) is not None:before.append(yi)
        if not before:continue
        stages['poses_with_projection_angle_motion_schedule']+=len(before)
        needed=np.any(mask[before],axis=0);los=np.zeros((len(actors),paths.shape[1]),dtype=bool)
        for i in profile['speaking_indices']:
            desired='blocked' if i in anchors and profile['anchor_line_of_sight']=='occluded' else 'clear'
            for frame in np.flatnonzero(needed[i]):
                dest=emitters[i,frame];key=(pi,tuple(dest))
                if key not in ray_cache:ray_cache[key]=line_of_sight(mesh,origin,dest)
                los[i,frame]=ray_cache[key]==desired
                if los[i,frame] and desired=='clear':
                    # A clear emitter alone can sit above or outside an
                    # occluded body. The existing visual body proxy must
                    # also be clear for a requested clear visual source.
                    body_key=(pi,tuple(bodies[i,frame]))
                    if body_key not in ray_cache:
                        ray_cache[body_key]=line_of_sight(mesh,origin,bodies[i,frame])
                    los[i,frame]=ray_cache[body_key]=='clear'
        mask&=los[None]
        for yi in before:
            starts={key:legal_start_ranges(mask[yi,actor_index[e['actor_id']]],e,clock,profile) for key,e in events.items()}
            if all(starts.values()) and schedule_legal_events(events,starts,clock,profile) is not None:
                legal.append((pi,yi,starts,nearest.copy()))
    if not legal:
        reason='no_joint_geometry_activity_schedule' if mesh is not None else 'static_geometry_unmeasured'
        raise CandidateFailure('camera',reason)
    pi,yi,starts,nearest=legal[int(rng.integers(len(legal)))];schedule=schedule_legal_events(events,starts,clock,profile,rng)
    camera={'candidate_id':f'grid_{pi:05d}_yaw_{yi*15:03d}','position_m':positions[pi],
            'basis':{'forward':forwards[yi].tolist(),'right':rights[yi].tolist(),'up':[0.,1.,0.]},
            'horizontal_fov_deg':fov,'resolution_hw':resolution,'height_above_floor_m':height,'motion':'static','yaw_deg':int(yi*15)}
    sr=int(clock['sample_rate_hz']);tb=int(clock['time_base_hz']);render=request.get('audio_render',{});output=[]
    for key,event in events.items():
        start=int(schedule[key]);end=start+int(event['sample_count']);out=deepcopy(event)
        out.update(event_id=key,start_sample=start,end_sample=end,start_tick=int(round(start*tb/sr)),end_tick=int(round(end*tb/sr)),
                   linear_gain=float(render.get('linear_gain',event.get('linear_gain',1.))),
                   event_unit='independent_source_playback_onset',
                   planned_audible_interval_samples=[start+int(event['audible_start_sample']),start+int(event['audible_end_sample_exclusive'])])
        output.append(out)
    output.sort(key=lambda e:(e['start_sample'],e['event_id']))
    return camera,output,{'selection':'uniform_over_all_legal_static_poses','legal_candidate_count':len(legal),
                         'candidate_position_count':len(positions),'yaw_candidate_count':24,'stages':dict(stages),
                         'legal_candidate_ids':[f'grid_{p:05d}_yaw_{y*15:03d}' for p,y,_,_ in legal],
                         'legal_event_start_ranges_samples':starts,'nearest_competitor_separation_deg':nearest.tolist(),
                         'line_of_sight_source':deepcopy(getattr(mesh,'source',None)),
                         'anchor_indices':profile['anchor_indices'],'clear_los_requires':['emitter','body_proxy'],'pixel_observability':'not_run'}


def build_conditioned_plan(*, room, request, source_registry, sounds, space, mesh,
                           clock, condition_profile=None, region=None):
    profile=deepcopy(condition_profile or resolve_condition_profile(request,source_registry))
    seed=int(request.get('seed',0));failures=Counter();last_errors={}
    for attempt in range(profile['retry_budget_within_profile']):
        # Retry seeds derive from the same request; the profile is never redrawn.
        rng=np.random.default_rng(np.random.SeedSequence([seed,attempt,20260906]))
        try:
            actors=select_entities(request,profile,source_registry,rng)
            selected_sounds=select_sounds(actors,sounds,profile,clock,request,rng)
            try:
                paths,rotations,moving,emitters,bodies,route_record=sample_routes(space,actors,profile,clock,rng,region,
                    required_windows={i:[selected_sounds[i]['audible_start_sample'],selected_sounds[i]['audible_end_sample_exclusive']] for i in profile['anchor_indices']})
            except CandidateFailure:
                raise
            except ValueError as exc:
                raise CandidateFailure('routes','navigation_sampling_failed: '+str(exc)) from exc
            camera,events,conditions=select_camera_and_schedule(space,mesh,paths,moving,emitters,bodies,actors,
                                                                selected_sounds,profile,clock,request,rng,region)
        except CandidateFailure as exc:
            failures[exc.stage+':'+exc.reason]+=1;last_errors[exc.stage]=exc.reason;continue
        frames=[]; action_ticks=[int(rng.integers(int(a['timeline']['walk_phase_period_frames'])))*int(clock['ticks_per_frame']) if a.get('timeline') else 0 for a in actors]
        for f in range(int(clock['frame_count'])):
            states=[]
            for i,actor in enumerate(actors):
                timeline=actor.get('timeline');motion=bool(moving[i,f])
                action=(timeline['walking_action_id'] if motion else timeline['idle_action_id']) if timeline else 'static'
                if timeline and motion:action_ticks[i]+=int(clock['ticks_per_frame'])
                tick=action_ticks[i] if timeline and motion else 0
                period=int(timeline['walk_phase_period_frames'])*int(clock['ticks_per_frame']) if timeline else 1
                phase=(tick%period)/period if timeline and motion else 0.
                state={'actor_id':actor['actor_id'],'root_transform':{'translation_m':paths[i,f].tolist(),'rotation_xyzw':rotations[i,f].tolist(),'scale':[1.,1.,1.]},
                       'action_id':action,'action_phase':phase,'action_time_ticks':tick,'moving':motion,
                       'planned_emitter_m':emitters[i,f].tolist(),'frame_index':f}
                states.append(state)
            frames.append({'frame_index':f,'pts_ticks':f*clock['ticks_per_frame'],'actor_states':states,'camera_state':deepcopy(camera)})
        bindings=[deepcopy(selected_sounds[i]) for i in sorted(selected_sounds)]
        plan={'kind':'avengine_question_driven_episode','plan_coordinates':'renderer_neutral','coordinate_frame':dict(COORDINATE_FRAME),
              'episode_id':str(request['episode_id']),'seed':seed,'status':'research_candidate','clock':deepcopy(dict(clock)),
              'scene':{'scene_id':room.get('scene_id',room['room_id']),'room_id':room['room_id']},'request':deepcopy(dict(request)),
              'condition_profile':profile,'planned_conditions':conditions,'activity_plan':route_record,
              'camera_condition_sampling':conditions,'audio_events':events,'voice_bindings':bindings,
              'visual_plan':{'backend_role':'production_visual','camera':camera,'actors':actors,'frames':frames,
                             'render':{'frame_count':clock['frame_count'],'fps_num':clock['frame_rate_hz'],'fps_den':1,'ticks_per_frame':clock['ticks_per_frame']},
                             'authority':{'actor_state':'avengine_conditioned_static_sampler','camera_listener':'avengine_conditioned_static_sampler','backend_may_replan':False}},
              'resources':deepcopy(dict(room)),'question_condition_match':{'candidate_qa_ids':list(request.get('qa_ids',[f'QA-{i:02d}' for i in range(1,25)])),'status':'candidate','episode_validity':'not_run'},
              'room_capabilities':{'status':'potential_only','evidence_refs':{'navigation':deepcopy(space.metadata)}},
              'planning_result':{'status':'research_candidate','attempts':attempt+1,'condition_profile':profile,'failure_histogram':dict(failures)},
              'evidence_status':{'native_visual':'not_run','native_audio':'not_run','qa_validity':'not_run'},'qualification_claim':False,
              'formal_dataset_registration_authorized':False}
        return plan
    raise ConditionedPlanningFailure({'status':'failed','condition_profile':profile,'attempts':profile['retry_budget_within_profile'],
                                      'failure_histogram':dict(failures),'last_errors_by_stage':last_errors,
                                      'gap_category':'evidence_missing_or_unsampled'})


def load_conditioned_sound_pool(payload, *, source_path=None):
    """Bridge P7 crop-relative activity once; original source clocks stay unchanged."""
    from pathlib import Path
    import wave
    from types import SimpleNamespace
    prepared=isinstance(payload,Mapping) and 'prepared_set_id' in payload
    rows=payload.get('clips',[]) if prepared else payload.get('sounds',[]) if isinstance(payload,Mapping) else payload
    root=Path(source_path).resolve().parent if source_path else Path.cwd()
    result=[]
    for raw in rows:
        if prepared and raw.get('status')!='prepared':continue
        sound=deepcopy(raw)
        path=Path(raw['prepared'] if prepared else raw['path']).expanduser()
        if not path.is_absolute():path=root/path
        with wave.open(str(path), 'rb') as wav:
            info=SimpleNamespace(frames=wav.getnframes(), samplerate=wav.getframerate(), channels=wav.getnchannels())
        if info.channels!=1:raise ValueError('conditioned sound pool must be real mono source PCM')
        sound.update(path=str(path.resolve()),sample_count=int(info.frames),sample_rate_hz=int(info.samplerate))
        if prepared:
            facts=raw['facts'];source_rate=int(facts['source_rate_hz']);offset=int(raw['source_crop_start_sample'])
            intervals=[[int(round((i['start_sample']-offset)*info.samplerate/source_rate)),
                        int(round((i['end_sample_exclusive']-offset)*info.samplerate/source_rate))] for i in raw['source_activity']['intervals']]
            sound.update(sound_asset_id=raw['prepared_audio_id'],sound_class='speech',gender=raw['gender'],
                         transcript=raw.get('transcript'),source_activity_intervals_samples=intervals,
                         audible_start_sample=min(i[0] for i in intervals),audible_end_sample_exclusive=max(i[1] for i in intervals),
                         active_duration_s=float(raw['source_activity']['active_duration_s']),
                         activity_coordinate='prepared_clip_samples',prepared_review=raw.get('human_review'),
                         transcript_scope='source_transcript_for_cropped_excerpt')
            # Keep traceability without copying bulky detector/source-stage records per event.
            for key in ('facts','source_activity','activity_filter_parameters','detector_parameters','filter_parameters'):
                sound.pop(key,None)
        else:
            sound.setdefault('sound_class','speech' if sound.get('transcript') else None)
            intervals=sound.get('source_activity_intervals_samples')
            if not intervals:
                raise ValueError('conditioned sounds require measured source_activity_intervals_samples')
            sound.setdefault('audible_start_sample',min(i[0] for i in intervals))
            sound.setdefault('audible_end_sample_exclusive',max(i[1] for i in intervals))
            sound.setdefault('active_duration_s',sum(b-a for a,b in intervals)/info.samplerate)
        if not 0<=sound['audible_start_sample']<sound['audible_end_sample_exclusive']<=info.frames:
            raise ValueError('activity/crop offsets do not resolve inside prepared PCM')
        result.append(sound)
    if not result:raise ValueError('conditioned sound pool has no prepared candidates')
    return result
