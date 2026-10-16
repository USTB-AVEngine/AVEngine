"""Collect every native member before joint QA selection and window delivery.

Question-family logic stays in avengine.qa. Missing source evidence, assets or
candidate stems cannot be padded by the delivery driver. Selected questions
remain research candidates until the independent dataset admissions pass.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import os
import itertools
import math
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_production_contract import question_allocation,validate_question_counts

QUESTION_DELIVERY_REVISION='v7-native-full-pool-demand-feedback-20261015-v1'


def sample_pilot_comparison_frame(raw,scene,*,seed='v7'):
    """Sample one proven native QA14 frame; predicates and family quotas stay intact.

    The authority limits emitted frame candidates before the v7 public-stem
    deduplication. A single query frame preserves each distinct comparison,
    instead of spending the candidate budget on repeated frames of one pair.
    Missing evidence leaves the original sampling untouched and records why.
    """
    audit={'status':'not_applicable','qa_id':'QA-14','sampling_revision':QUESTION_DELIVERY_REVISION,
           'scope':'query-frame sampling only; no asset or parent admission',
           'thresholds_changed':False,'frame_readbacks_modified':False}
    if scene.get('production_scenario',{}).get('camera_policy')!='pilot_binding_coverage':
        audit['reason']='camera_policy_does_not_request_pilot_comparison_sampling'
        return audit
    audit['status']='blocked'
    truth=raw.get('pixel_visibility_truth')
    if not isinstance(truth,dict) or truth.get('status')!='computed_modal_target_only_v1':
        audit['reason']='complete_native_modal_target_visibility_missing'
        return audit
    from avengine.qa import unified_catalog as authority
    from avengine.qa.v7_families import normalize_v7_facts
    from avengine.qa.visibility_interpretation import prepare_facts_for_qa
    try:
        facts=prepare_facts_for_qa(normalize_v7_facts(raw),'QA-14')
        count=int(facts['time']['frame_count']);fps=float(facts['time']['frame_rate_hz'])
        appearances=authority._appearance_candidates(facts)
        people=[actor_id for actor_id,actor,_ in appearances if actor.get('species_id')=='human']
        if len(people)<3:
            audit['reason']='fewer_than_three_distinct_native_reviewed_people'
            return audit
        listener=facts['listener']
        tracks=[listener.get('positions_m'),*[facts['actors'][actor].get('emitter_positions_m') for actor in people]]
        if (listener.get('status')!='pass' or count<=0 or not math.isfinite(fps) or fps<=0
                or any(not isinstance(track,list) or len(track)!=count for track in tracks)
                or any(not isinstance(point,(list,tuple)) or len(point)!=3
                       or any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in point)
                       for track in tracks for point in track)):
            audit['reason']='complete_finite_native_listener_or_mouth_track_missing'
            return audit
        for actor in people:
            masks=truth.get('per_instance',{}).get(actor,{}).get('frames',[])
            if ([row.get('frame_index') for row in masks]!=list(range(count))
                    or set(facts.get('visibility',{}).get(actor,{}))!=set(range(count))):
                audit['reason']='complete_indexed_native_target_mask_stream_missing'
                audit['actor_id']=actor
                return audit
        candidates=authority._P8_CANDIDATES['QA-14'](facts)
        by_pair={};by_pair_frame={}
        for candidate in candidates:
            pair=frozenset(candidate['actor_ids']);frame=candidate['query_frame']
            by_pair.setdefault(pair,set()).add(frame)
            by_pair_frame[pair,frame]=candidate
        audit['original_frame_candidate_count']=len(candidates)
        audit['native_pair_candidate_counts']=[{'actor_ids':sorted(pair),'frames':len(frames)}
                                              for pair,frames in sorted(by_pair.items(),key=lambda x:sorted(x[0]))]
        options=[]
        for people_group in itertools.combinations(people,3):
            pairs=[frozenset(pair) for pair in itertools.combinations(people_group,2)]
            common=set.intersection(*(by_pair.get(pair,set()) for pair in pairs))
            runs=[]
            for frame in sorted(common):
                if not runs or runs[-1][1]!=frame:runs.append([frame,frame+1])
                else:runs[-1][1]=frame+1
            for run in runs:
                try:window=authority._query_window_fields(facts,run)
                except authority._Deferred:continue
                lo,hi=window['query_window_s']
                frame=min(run[1]-1,max(run[0],int((lo+hi)*fps/2)))
                options.append((-(run[1]-run[0]),run[0],people_group,pairs,run,window,frame))
        audit['publishable_common_run_count']=len(options)
        for _,_,people_group,pairs,run,window,frame in sorted(options,key=lambda row:row[:3]):
            proofs=[];stems=set()
            try:
                for pair in pairs:
                    item=authority._P8_EMITTERS['QA-14'](facts,by_pair_frame[pair,frame],str(seed))
                    proofs.append({'actor_ids':sorted(pair),'authority_status':item['status'],
                                   'distances_m':item['evidence']['distances_m'],
                                   'query_window_frames':item['evidence']['query_window_frames'],
                                   'query_window_s':item['evidence']['query_window_s']})
                    stems.add(item['question']['zh'])
            except authority._Deferred as error:
                audit.setdefault('rejected_sampling_runs',[]).append({'run_frames':run,'code':error.code})
                continue
            if len(stems)!=3:
                audit.setdefault('rejected_sampling_runs',[]).append({'run_frames':run,'code':'comparison_public_stems_not_distinct'})
                continue
            sampling=deepcopy(raw.get('sampling') or {})
            sampling.setdefault('query_frame_by_qa',{})['QA-14']=frame
            raw['sampling']=sampling
            audit.update(status='pass',reason='one_native_frame_for_three_distinct_proven_distance_pairs',
                         query_frame=frame,frame_index_scope='window',people=list(people_group),
                         common_run_frames=run,common_run_public_s=window['query_window_s'],
                         authority_comparison_proofs=proofs)
            return audit
        audit['reason']='no_three_native_distance_pairs_in_a_publishable_common_run'
        return audit
    except authority._Deferred as error:
        audit.update(reason='native_qa14_sampling_evidence_deferred',code=error.code,detail=error.detail)
        return audit
    except (KeyError,TypeError,ValueError) as error:
        audit.update(reason='native_qa14_sampling_evidence_invalid',detail=str(error))
        return audit


def collect_native_pool(scene,start,window_id,visual,media,evidence,quota_index,count):
    from avengine.qa.v7_native_adapter import native_bundle, collect_questions
    from avengine.timeline.v7_footstep_bouts import footstep_bout_question_inputs
    scene,evidence=footstep_bout_question_inputs(scene,evidence)
    proof=deepcopy(evidence)
    # The producer's observed mask stream owns this field; caller requests and
    # sparse snapshots never replace native 150-frame visibility readback.
    truth=visual.get('pixel_visibility_truth')
    if not truth or truth.get('status')!='computed_modal_target_only_v1':
        raise ValueError('full QA needs the measured native modal/target mask stream')
    proof['pixel_visibility_truth']=truth
    if 'native_appearance_review' in visual:proof['appearance_review']=deepcopy(visual['native_appearance_review'])
    raw=native_bundle(scene,start,window_id,visual,proof)
    bind_native_listener_frames(raw)
    raw['native_candidate_query_request'] = {'schema':'v7-native-fixed-query-windows-r17-v3','qa14_partition_seconds':2,'qa14_candidate_budget':12}
    from avengine.timeline.v7_environmental_sources import augment_environmental_native_bundle,collect_admitted_class_questions
    augment_environmental_native_bundle(raw,scene,visual,start)
    comparison_sampling=sample_pilot_comparison_frame(raw,scene,seed=str(proof.get('question_seed','v7')))
    public,private,audit=collect_questions(raw,window_id,media,
        seed=str(proof.get('question_seed','v7')),items_per_family=12,foa_readback=proof.get('foa_readback'),
        semantic_manifest=proof.get('semantic_manifest'),protocol_scene=proof.get('protocol_scene'),
        window_start_s=start,native_dialogue_evidence=proof.get('native_dialogue_evidence'),
        semantic_wording_variant=proof.get('semantic_wording_variant','standard'),
        content_binding_bundle=proof.get('content_binding_bundle'))
    if scene.get('admitted_sound_class_domain'):
        extra,extra_gold=collect_admitted_class_questions(raw,window_id,media,proof,audit,scene['admitted_sound_class_domain'])
        public.extend(extra);private.extend(extra_gold)
    if any(e['start_tick']<round((start+10)*10) and e['end_tick']>round(start*10) for e in proof.get('protocol_scene',{}).get('outside_class13_events',[])):
        excluded={q['question_id'] for q in public if q.get('protocol_panel')=='STARSS23_v0.2' and q['qa_id']=='QA-23'}
        public=[q for q in public if q['question_id'] not in excluded];private=[g for g in private if g['question_id'] not in excluded]
        audit['out_of_domain_activity_count_exclusions']=sorted(excluded)
    audit['candidate_budget']={'items_per_family':12,'QA14_additional_query_runs':5,'QA14_candidates_per_query_run':12,'QA14_query_partition_s':2,'modality_and_binding_gates_unchanged':True,'duplicate_stems_still_count_once':True}
    audit['pilot_native_comparison_sampling']=comparison_sampling
    demand=scene.get('scene_demand')
    if demand:
        from collections import Counter
        actual={q['logical_question_id']:q for q in public}
        available=Counter(q['qa_id'] for q in actual.values())
        from avengine.timeline.v7_demand import requested_question_families
        requested=requested_question_families(demand)
        audit['demand_realization']={'requested_conditions':demand['requested_conditions'],
            'legal_candidate_family_counts':dict(available),'missing_requested_families':sorted(f for f in requested if not available[f]),
            'native_moving_actor_ids':sorted({actor for actor,frames in raw['frame_readbacks']['actors'].items()
                if any(frame.get('moving') is True for frame in frames)}),
            'wet_support_event_ids':sorted({r['event_id'] for r in raw['audio_readback']['wet_tail_intervals']}),
            'status':'measured_candidates; unfulfilled requests remain deficits',
            'answer_and_modality_gates_changed':False}
    audit['native_bundle']=raw
    audit['native_listener_binding']='all measured native camera frames, no static pose shadow'
    return public,private,audit


def bind_native_listener_frames(raw):
    """The QA normalizer gives root.listener precedence over frame camera.

    Preserve the actual per-frame listener track in that authoritative slot.
    This is producer evidence wiring; QA predicates and scoring are unchanged.
    """
    frames=raw.get('frame_readbacks',{}).get('camera')
    count=int(raw['clock']['frame_count'])
    if not isinstance(frames,list) or len(frames)!=count or [r.get('frame_index') for r in frames]!=list(range(count)):
        raise ValueError('native listener requires every indexed camera frame')
    raw['listener']={'frames':deepcopy(frames),'source':'actual native camera per-frame readback'}



def filter_observed_nuisance_candidates(public,private,window_ids):
    """Do not reserve a V/A item whose actual nuisance partner was not emitted.

    Both original pools and every original answer stay intact. Equal answers
    are not assumed or substituted: the original final audit still checks all
    common V/A questions, including unselected ones, and all selected bindings.
    """
    from avengine.timeline.v7_candidate_generation import authority_physical_query_key
    from avengine.qa.v7_families import deduplicate_questions
    lookup={};keys={};canonical={};removed=[]
    for name in public:
        p,g,_=deduplicate_questions(public[name],private[name]);gold={r['question_id']:r for r in g}
        lookup[name]={};keys[name]={};canonical[name]={}
        for q in p:
            if q['modality']not in {'vision','audio'}:continue
            a=gold[q['question_id']];fallback='/'.join((q.get('protocol_panel','v6'),q['qa_id'],q['question_zh']))
            key=(q['qa_id'],authority_physical_query_key(q,a,window_ids[name],fallback),q['modality'])
            keys[name][q['logical_question_id']]=key;lookup[name][key]=q;canonical[name][key]=a['canonical_answer']
    result={}
    for name,pool in public.items():
        kept=[]
        for q in pool:
            if q['modality']not in {'vision','audio'}:kept.append(q);continue
            key=keys[name][q['logical_question_id']]
            other=name[:3]+('a1'if name.endswith('a0')else'a0')if q['modality']=='vision'else('v1'if name.startswith('v0')else'v0')+name[2:]
            if key not in lookup[other]:
                removed.append({'member':name,'question_id':q['question_id'],'logical_question_id':q['logical_question_id'],'qa_id':q['qa_id'],'modality':q['modality'],'comparison_key':list(key),'reason':'actual nuisance partner missing from original native pool'})
                continue
            # A disagreement is retained so the unchanged original audit can
            # reject it. Filtering a wrong truth would conceal a producer bug.
            kept.append(q)
        result[name]=kept
    return result,{'status':'measured_preselection','removed_unpaired_candidates':removed,'canonical_answers_changed':False,'final_gate_unchanged':True}



def reserve_actual_audio_family(chosen,gold,pool,private,family):
    """Retain actual content-direction candidates after critical reservations.

    The original 14-question modality allocation and all answers are retained.
    No fallback answer is created when the original candidate is absent.
    """
    byid={a['question_id']:a for a in private};chosen=list(chosen);gold=list(gold)
    eligible=[q for q in pool if q['qa_id']==family and q['form']=='open' and q['modality']=='audio']
    if not eligible:raise ValueError('requested audio family lacks an actual eligible candidate: '+family)
    if any(q in eligible for q in chosen):return chosen,gold
    protected={'STARSS-SPEAKERS','V7-CLASS34'}
    slots=[i for i,q in enumerate(chosen)if q['modality']=='audio'and q['qa_id']not in protected]
    if not slots:raise ValueError('content reservation exceeds unchanged audio allocation')
    index=slots[-1];q=min(eligible,key=lambda q:q['question_id']);chosen[index]=q;gold[index]=byid[q['question_id']]
    return chosen,gold


def acceptance_pool_allocation(public, private, requested):
    """Count only original deduplicated candidates; never fill missing truth."""
    from collections import Counter
    from avengine.qa.v7_families import deduplicate_questions
    unique,_,_=deduplicate_questions(public,private)
    logical={q['logical_question_id']:q for q in unique}
    available=Counter(q['modality'] for q in logical.values())
    actual={m:min(int(n),available[m]) for m,n in requested.items()
            if min(int(n),available[m])>0}
    if not sum(actual.values()):raise ValueError('acceptance crop has no original legal questions')
    return actual, {'requested_allocation':dict(requested),'actual_allocation':actual,
                    'available_logical_candidates':dict(available),
                    'modality_deficits':{m:int(n)-available[m] for m,n in requested.items() if available[m]<n},
                    'family_candidate_counts':dict(Counter(q['qa_id'] for q in logical.values())),
                    'owner_authorization':'20261014 acceptance: skip unavailable families; retain original native truth/scorers',
                    'answers_changed':False,'native_gates_changed':False}


def available_single_parent_allocation(public,private,requested):
    """Fill a single parent's quota from existing legal questions only."""
    from collections import Counter
    from avengine.qa.v7_families import deduplicate_questions
    questions,_,_=deduplicate_questions(public,private)
    seen=set();available=Counter()
    for q in questions:
        if q['logical_question_id']in seen:continue
        seen.add(q['logical_question_id']);available[q['modality']]+=1
    allocation={m:min(n,available[m])for m,n in requested.items()}
    missing=sum(requested.values())-sum(allocation.values())
    while missing:
        spare=[m for m in requested if allocation[m]<available[m]]
        if not spare:raise ValueError('single parent has fewer actual legal questions than requested total')
        modality=max(spare,key=lambda m:available[m]-allocation[m])
        allocation[modality]+=1;missing-=1
    allocation={m:n for m,n in allocation.items()if n}
    return allocation,{'requested':dict(requested),'actual':allocation,
        'changed':allocation!=requested,'reason':'owner: skip unavailable candidate families; keep existing legal answers',
        'actual_available':dict(available),'canonical_answers_changed':False}


def current_selection_snapshot(demand,*,next_questions):
    from avengine.timeline.v7_demand import demand_snapshot,read_demand_state,SELECTION_FEEDBACK_ENV
    if os.environ.get(SELECTION_FEEDBACK_ENV):
        state=read_demand_state(os.environ[SELECTION_FEEDBACK_ENV])
    else:
        counts=demand.get('observed_family_counts',{})
        state={'logical_questions':demand.get('observed_logical_questions',sum(counts.values())),
               'family_counts':counts}
    if demand.get('target_shares') is not None:
        state={**state,'target_shares':demand['target_shares']}
    return demand_snapshot(state,next_questions=next_questions)


def advance_independent_feedback(parent_id,chosen,answers,counts,seen):
    from collections import Counter
    from avengine.timeline.v7_question_independence import independence_key
    added=Counter()
    for member,questions in chosen.items():
        gold={g['question_id']:g for g in answers[member]}
        for q in {q['logical_question_id']:q for q in questions}.values():
            key,_=independence_key(parent_id,q,gold[q['question_id']])
            if key in seen:continue
            seen.add(key);counts[q['qa_id']]=counts.get(q['qa_id'],0)+1;added[q['qa_id']]+=1
    return dict(added)


def available_binding_reservation(public,private,allocations,requested):
    """A quota reserves existing measured keys; it is not an answer gate."""
    from avengine.qa.v7_group_selection import eligible_binding_keys
    eligible,_=eligible_binding_keys(public,private)
    capacity=min((a.get('audiovisual',0)for a in allocations.values()),default=0)
    reserved=min(int(requested),len(eligible),capacity)
    return reserved,{'requested':int(requested),'eligible_original_binding_keys':len(eligible),
        'audiovisual_capacity':capacity,'reserved':reserved,'unfilled_reservation':int(requested)-reserved,
        'policy':'owner_phase2_B_existing_candidates_only','original_group_answer_and_media_checks_unchanged':True}


def finalize_parent_question_pools(row,members,root,*,binding_items_per_window=5,family_counts=None):
    from avengine.timeline.v7_candidate_generation import stage,read_json
    from avengine.qa.v7_families import select_questions
    from avengine.qa.v7_group_selection import select_counterfactual_group
    demand=row.get('scene_demand')
    if demand:
        snapshot=current_selection_snapshot(demand,
            next_questions=row['window_count']*row['questions_per_window']*len(members))
        family_counts=snapshot['selection_counts']
    names={m['member'] for m in members}
    quartet={'v0_a0','v0_a1','v1_a0','v1_a1'}
    if names!=set(row['members']) or (names!=quartet and names!={'v0_a0'}):
        raise ValueError('complete native quartet or declared single parent required')
    if any(len(m['windows'])!=row['window_count'] for m in members):
        raise ValueError('parent has an incomplete native member crop')
    partial=row.get('acceptance_allow_partial_families') is True
    if partial and (names!={'v0_a0'} or row.get('trial_only') is not True):
        raise ValueError('partial acceptance questions require explicit trial-only single-member CPU parent')
    selected={name:[] for name in names};audits=[];feedback_seen=set()
    for index in range(row['window_count']):
        public={};private={};allocations={};window_ids={}
        for member in members:
            name=member['member'];wid=member['windows'][index]['window_id'];window_ids[name]=wid
            folder=Path(member['windows_directory'])/wid
            public[name]=read_json(folder/'questions.json');private[name]=read_json(folder/'private_answers.json')
            quota=row['question_quota_base']+row['members'].index(name)*row['window_count']+index
            allocations[name]=question_allocation(quota,row['questions_per_window'])
        ordinary={name:[q for q in public[name] if q['qa_id']!='V7-CLASS34'] for name in names}
        from avengine.timeline.v7_constructor_branch_selection import restrict_constructed_branch
        branch_audits={}
        for name in names:ordinary[name],branch_audits[name]=restrict_constructed_branch(ordinary[name],private[name],demand)
        if names==quartet:
            ordinary,pair_audit=filter_observed_nuisance_candidates(ordinary,private,window_ids)
            quartet_allocation={}
            for name in names:
                allocations[name],quartet_allocation[name]=available_single_parent_allocation(ordinary[name],private[name],allocations[name])
            actual_binding,reservation=available_binding_reservation(ordinary,private,allocations,binding_items_per_window)
            chosen,gold,audit=select_counterfactual_group(ordinary,private,allocations,
                binding_items_per_window=actual_binding,quota_index=row['question_quota_base']+index,family_counts=family_counts)
            audit['binding_reservation_availability']=reservation
            audit['observed_nuisance_preselection']=pair_audit
            audit['owner_quartet_available_candidate_allocation']=quartet_allocation
            audit['post_dedup_modality_target']={'vision':.30,'audio':.25,'audiovisual':.45}
        else:
            partial_audit=None
            if partial:
                allocations['v0_a0'],partial_audit=acceptance_pool_allocation(ordinary['v0_a0'],private['v0_a0'],allocations['v0_a0'])
                try:
                    allocations['v0_a0'],availability=available_single_parent_allocation(
                        ordinary['v0_a0'],private['v0_a0'],question_allocation(row['question_quota_base']+index,row['questions_per_window']))
                    partial_audit['actual_allocation']=dict(allocations['v0_a0'])
                    partial_audit['quota_fill_from_original_legal_candidates']=availability
                except ValueError as error:
                    if str(error)!='single parent has fewer actual legal questions than requested total':raise
                    availability={'requested':partial_audit['requested_allocation'],'actual':partial_audit['actual_allocation'],'changed':True,'reason':'fewer original legal candidates; explicit partial trial window'}
            else:
                allocations['v0_a0'],availability=available_single_parent_allocation(ordinary['v0_a0'],private['v0_a0'],allocations['v0_a0'])
                if demand:
                    availability.update(reason='owner: modality target applies after parent dedup; retain 14 original legal questions',
                        post_dedup_modality_target={'vision':.30,'audio':.25,'audiovisual':.45},
                        target_applied_after_parent_dedup=True)
            q,a=select_questions(ordinary['v0_a0'],private['v0_a0'],allocations['v0_a0'],quota_index=row['question_quota_base']+index,family_counts=family_counts)
            chosen,gold,audit={'v0_a0':q},{'v0_a0':a},{'status':'single_selected','binding_items_per_member':0,'owner_available_candidate_allocation':availability}
            if partial_audit:audit['acceptance_partial_family_policy']=partial_audit
        focus=row.get('production_scenario',{}).get('category_focus_class')
        if focus:
            from avengine.timeline.v7_environmental_sources import reserve_critical_audio_questions
            for name in names:
                try:
                    chosen[name],gold[name]=reserve_critical_audio_questions(chosen[name],gold[name],public[name],private[name],focus)
                except ValueError as error:
                    if not partial or not (str(error).startswith('critical actual candidate missing:') or str(error)=='critical audio reservation exceeds unchanged modality quota'):raise
                    audit.setdefault('skipped_unavailable_family_reservations',[]).append({'member':name,'reason':str(error)})
            audit['critical_audio_reservation']={'speaker_count_requested':True,'category_focus_class':focus,'modality_quota_changed':bool(partial and allocations['v0_a0']!=question_allocation(row['question_quota_base']+index,row['questions_per_window'])),'actually_selected_families':sorted({q['qa_id'] for qs in chosen.values() for q in qs}),'status':'partial_skipped' if audit.get('skipped_unavailable_family_reservations') else 'reserved'}
        if row.get('production_scenario',{}).get('semantic_scenario_id'):
            skipped=[]
            for name in names:
                if not any(q['qa_id']=='QA-28'and q['form']=='open'and q['modality']=='audio'for q in ordinary[name]):
                    audit.setdefault('skipped_unavailable_family_reservations',[]).append({'member':name,'family':'QA-28','reason':'no actual eligible QA-28 candidate; owner permits empty-family skip'})
                    continue
                try:
                    chosen[name],gold[name]=reserve_actual_audio_family(chosen[name],gold[name],ordinary[name],private[name],'QA-28')
                except ValueError as error:
                    if not partial or not (str(error).startswith('requested audio family lacks an actual eligible candidate:') or str(error)=='content reservation exceeds unchanged audio allocation'):raise
                    audit.setdefault('skipped_unavailable_family_reservations',[]).append({'member':name,'family':'QA-28','reason':str(error)})
            audit['actual_content_direction_reservation']={'family':'QA-28','original_candidates_only':True,'canonical_answers_changed':False,'modality_quota_changed':bool(partial and allocations['v0_a0']!=question_allocation(row['question_quota_base']+index,row['questions_per_window'])),'actually_selected':any(q['qa_id']=='QA-28' for qs in chosen.values() for q in qs),'status':'reserved' if any(q['qa_id']=='QA-28' for qs in chosen.values() for q in qs) else 'partial_skipped','skipped_empty_family':audit.get('skipped_unavailable_family_reservations',[])}
        audit['constructed_branch_selection']=branch_audits
        if demand:
            audit['demand_feedback']=deepcopy(snapshot)
            # Repeated questions in overlapping crops or quartet members do not
            # consume an independent family deficit more than once.
            audit['independent_feedback_added']=advance_independent_feedback(
                row['scene_id'],chosen,gold,family_counts,feedback_seen)
            audit['selection_feedback_unit']='independent_parent_question'
        audits.append(audit)
        for name in names:
            check=validate_question_counts(chosen[name],gold[name],expected_allocation=allocations[name],allow_partial_window=partial)
            selected[name].append((chosen[name],gold[name],check))
    # Selection of every actual member succeeds before any selected stage is
    # published. All unselected pools remain available for deficit diagnostics.
    result=[]
    for member in members:
        name=member['member'];source=Path(member['windows_directory'])
        def publish(attempt):
            required=[];windows=[]
            for index,old in enumerate(member['windows']):
                wid=old['window_id'];dest=attempt/wid;dest.mkdir()
                q,a,check=deepcopy(selected[name][index])
                portable={'video':'video.mp4','audio_foa':'audio_foa.wav','audio_mono':'audio_mono.wav'}
                for role,filename in portable.items():os.link(source/wid/filename,dest/filename)
                for item in q:item['media']={**portable,'foa_format':'ACN_WYZX_SN3D_camera'}
                qbyid={item['question_id']:item for item in q}
                binding_keys={tuple(key) for key in audits[index].get('binding_keys',[])}
                for item in a:
                    question=qbyid[item['question_id']]
                    key=(question.get('protocol_panel','v6'),question['qa_id'],question['question_zh'])
                    item.setdefault('comparison_key','/'.join(key))
                    item['counterfactual_role']='binding' if key in binding_keys else question['modality']
                write_json(dest/'questions.json',q);write_json(dest/'private_answers.json',a)
                evidence=read_json(source/wid/'evidence.json')
                evidence['full_qa_selection']={'revision':QUESTION_DELIVERY_REVISION,'audit':audits[index],
                    'question_check':check,'formal_dataset_count':0}
                write_json(dest/'evidence.json',evidence)
                windows.append({'window_id':wid,**check})
                required.extend(f'{wid}/{file}' for file in (*portable.values(),'questions.json','private_answers.json','evidence.json'))
            return {'windows':windows,'formal_dataset_count':0},required
        stage_name='selected_windows_'+name
        directory,receipt,resumed=stage(root,stage_name,{'seed_row':row,'revision':QUESTION_DELIVERY_REVISION,
            'binding_items_per_window':binding_items_per_window,'pool_directory':str(source)},publish)
        result.append({**member,'question_pool_directory':str(source),'windows_directory':str(directory),'windows':receipt['windows'],
            'window_stage_name':stage_name,'selected_full_qa':True,'full_qa_selection_resumed':resumed})
    return result,{'revision':QUESTION_DELIVERY_REVISION,'window_audits':audits,'formal_dataset_count':0}


def validate_legacy_source_evidence(evidence):
    """Refuse recording labels that the current formant producer never emitted.

    The full recorded-source compiler is still required. An admission catalog
    or QA JSON cannot change the waveform produced by parent_dry.
    """
    if evidence.get('recorded_dry_program') is not None:
        raise RuntimeError('recorded dry-program assembly is not integrated into this producer')
    for proof in evidence.get('members', {}).values():
        if proof.get('events') or proof.get('protocol_scene') or proof.get('semantic_manifest'):
            raise RuntimeError('full source QA evidence is not bound to this producer\'s actual formant dry buses; recorded dry-program assembly is required')
        profiles=proof.get('actor_profiles',{})
        if any(p.get('entity_class','articulated_human')!='articulated_human' for p in profiles.values()):
            raise RuntimeError('the current native producer renders articulated humans; external actor profiles cannot declare a loudspeaker')


def derive_native_appearance_evidence(directory,visual,assets):
    """Use the family's unchanged automatic color screen on actual snapshots."""
    import numpy as np
    from PIL import Image
    from avengine.qa.v7_visual_admission import measure_native_color,COLOR_HUE_DEGREES
    directory=Path(directory);by_asset={a['asset_id']:a for a in assets};reviews={};screens=[]
    for png in sorted(directory.glob('native_frame_*.png')):
        index=int(png.stem.rsplit('_',1)[1]);mask=directory/f'native_masks_{index:04d}.npz'
        if not mask.is_file() or not 0<=index<len(visual['frames']):continue
        rgb=np.asarray(Image.open(png));semantic=np.load(mask,allow_pickle=False)['semantic']
        for actor in visual['frames'][index]['actors']:
            asset=by_asset.get(actor['asset_id']);color=(asset or {}).get('appearance',{}).get('top_color')
            if color not in COLOR_HUE_DEGREES:continue
            try:
                result=measure_native_color(rgb,semantic,actor['semantic_id'],color)
            except IndexError as error:
                if 'size 0' not in str(error):raise
                result={'status':'blocked','reason':'insufficient_or_ambiguous_native_color','expected_color':color,'semantic_id':actor['semantic_id'],'saturated_pixels':0,'upstream_empty_roi_exception':str(error),'gate_changed':False}
            screens.append({'frame_index':index,'actor_id':actor['actor_id'],'asset_id':actor['asset_id'],**result})
            if result['status']!='pass':continue
            from avengine.qa.v7_english_surface import appearance_description_en
            key=str(actor['actor_id']);label=appearance_description_en(asset,color)
            review=reviews.setdefault(key,{'status':'reviewed','value':color,'label_zh':label,'label_en':label,'frame_refs':[],'review_method':'automatic_native_color_pixels','human_viewing_used':False})
            if review['value']!=color:raise ValueError('native appearance changed within a physical actor')
            review['frame_refs'].append(index)
    return {'revision':'automatic_native_hue_review_v1','review':{'actors':reviews},'screens':screens,
            'status':'pass' if reviews else 'not_run_or_no_eligible_color_pixels','human_viewing_used':False}
