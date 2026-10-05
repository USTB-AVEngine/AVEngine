"""Collect every native member before joint QA selection and window delivery.

Question-family logic stays in avengine.qa. Missing source evidence, assets or
candidate stems cannot be padded by the delivery driver. Selected questions
remain research candidates until the independent dataset admissions pass.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import os
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_production_contract import question_allocation,validate_question_counts

QUESTION_DELIVERY_REVISION='v7-native-full-pool-joint-selection-v2-observed-nuisance-pairs-r16-supplement'


def collect_native_pool(scene,start,window_id,visual,media,evidence,quota_index,count):
    from avengine.qa.v7_native_adapter import native_bundle, collect_questions
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
    from avengine.timeline.v7_environmental_sources import augment_environmental_native_bundle,collect_admitted_class_questions
    augment_environmental_native_bundle(raw,scene,visual,start)
    public,private,audit=collect_questions(raw,window_id,media,
        seed=str(proof.get('question_seed','v7')),foa_readback=proof.get('foa_readback'),
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


def finalize_parent_question_pools(row,members,root,*,binding_items_per_window=5,family_counts=None):
    from avengine.timeline.v7_candidate_generation import stage,read_json
    from avengine.qa.v7_families import select_questions
    from avengine.qa.v7_group_selection import select_counterfactual_group
    names={m['member'] for m in members}
    quartet={'v0_a0','v0_a1','v1_a0','v1_a1'}
    if names!=set(row['members']) or (names!=quartet and names!={'v0_a0'}):
        raise ValueError('complete native quartet or declared single parent required')
    if any(len(m['windows'])!=row['window_count'] for m in members):
        raise ValueError('parent has an incomplete native member crop')
    selected={name:[] for name in names};audits=[]
    for index in range(row['window_count']):
        public={};private={};allocations={};window_ids={}
        for member in members:
            name=member['member'];wid=member['windows'][index]['window_id'];window_ids[name]=wid
            folder=Path(member['windows_directory'])/wid
            public[name]=read_json(folder/'questions.json');private[name]=read_json(folder/'private_answers.json')
            quota=row['question_quota_base']+row['members'].index(name)*row['window_count']+index
            allocations[name]=question_allocation(quota,row['questions_per_window'])
        ordinary={name:[q for q in public[name] if q['qa_id']!='V7-CLASS34'] for name in names}
        if names==quartet:
            ordinary,pair_audit=filter_observed_nuisance_candidates(ordinary,private,window_ids)
            chosen,gold,audit=select_counterfactual_group(ordinary,private,allocations,
                binding_items_per_window=binding_items_per_window,quota_index=row['question_quota_base']+index,family_counts=family_counts)
            audit['observed_nuisance_preselection']=pair_audit
        else:
            q,a=select_questions(ordinary['v0_a0'],private['v0_a0'],allocations['v0_a0'],quota_index=row['question_quota_base']+index,family_counts=family_counts)
            chosen,gold,audit={'v0_a0':q},{'v0_a0':a},{'status':'single_selected','binding_items_per_member':0}
        focus=row.get('production_scenario',{}).get('category_focus_class')
        if focus:
            from avengine.timeline.v7_environmental_sources import reserve_critical_audio_questions
            for name in names:
                chosen[name],gold[name]=reserve_critical_audio_questions(chosen[name],gold[name],public[name],private[name],focus)
            audit['critical_audio_reservation']={'speaker_count':True,'category_focus_class':focus,'modality_quota_changed':False}
        if row.get('production_scenario',{}).get('semantic_scenario_id'):
            for name in names:
                chosen[name],gold[name]=reserve_actual_audio_family(chosen[name],gold[name],ordinary[name],private[name],'QA-28')
            audit['actual_content_direction_reservation']={'family':'QA-28','original_candidates_only':True,'modality_quota_changed':False}
        audits.append(audit)
        for name in names:
            check=validate_question_counts(chosen[name],gold[name],expected_allocation=allocations[name])
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
            key=str(actor['actor_id']);label=asset.get('appearance_zh',{'blue':'蓝色上衣的人','yellow':'黄色上衣的人','green':'绿色上衣的人','burgundy':'酒红色上衣的人'}[color])
            review=reviews.setdefault(key,{'status':'reviewed','value':color,'label_zh':label,'label_en':color,'frame_refs':[],'review_method':'automatic_native_color_pixels','human_viewing_used':False})
            if review['value']!=color:raise ValueError('native appearance changed within a physical actor')
            review['frame_refs'].append(index)
    return {'revision':'automatic_native_hue_review_v1','review':{'actors':reviews},'screens':screens,
            'status':'pass' if reviews else 'not_run_or_no_eligible_color_pixels','human_viewing_used':False}
