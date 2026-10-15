"""Native motion/readback checks with an explicit owner human quality policy."""
import numpy as np
from avengine.assets.roster_motion.quality import HUMAN_QUALITY

def motion_checks(visual):
    errors=[];warnings=[];feet=[];readbacks=[];motions=visual.get('native_motion_admissions',{})
    clearances=[];navigation=[];instances=[];quality_limits={}
    if set(motions)!=set(visual.get('phase2_requested_motion',motions)):
        errors.append('requested native motions missing or silently substituted')
    for ep,record in motions.items():
        actual=record.get('native_30Hz_readbacks',[]);source=record['source_frames']
        animal_visual_only=(record.get('species_id') in {'dog','cat'}
            and record.get('animal_visual_fallback_policy',{}).get('revision')=='owner-animal-appearance-construction-20261015')
        def contact_issue(message):
            (warnings if animal_visual_only else errors).append(ep+':'+message)
        if record.get('status')!='pass' or len(actual)!=901 or len(source)!=901:
            errors.append(ep+':incomplete original pose/native 30Hz sequence');continue
        floor_proof=record.get('original_full_skin_floor_collision',{})
        if floor_proof.get('status')!='pass' or floor_proof.get('worst_original_skin_floor_delta_m',-1)<-.00101:
            errors.append(ep+':missing or failing complete original skin/floor collision proof')
        policy=record.get('production_visual_quality_policy',{})
        owner_quality=(isinstance(policy,dict)
            and policy.get('policy_name')=='owner_visual_quality_20261014'
            and policy.get('quality_thresholds_changed') is True
            and policy.get('truth_and_answer_gates_changed') is False
            and policy.get('contact_gap_m')==.012
            and policy.get('support_lock_error_m')==.015)
        human_owner_quality=(record.get('species_id','human')not in {'dog','cat'} and record.get('quality_policy')==HUMAN_QUALITY)
        owner_quality=owner_quality or human_owner_quality
        clearance_limit=0. if animal_visual_only else HUMAN_QUALITY['minimum_nonfloor_clearance_m']if human_owner_quality else .02
        limit=.012 if owner_quality or record.get('species_id')=='cat' else .008
        support_limit=.015 if owner_quality else .003
        quality_limits[ep]={'policy_name':policy.get('policy_name',HUMAN_QUALITY['revision']) if owner_quality else 'original',
            'owner_quality':owner_quality,'contact_limit_m':limit,'support_lock_limit_m':support_limit,
            'truth_and_answer_gates_changed':False}
        is_seated=record.get('motion_kind')=='sit';moving=sum(bool(f['moving']) for f in source)
        is_static_fallback=record.get('motion_kind')=='animal_static_fallback'
        if is_static_fallback:
            roots=np.asarray(record.get('world_roots_m',[]),float)
            if (record.get('motion_mode')!='static_after_failed_route' or moving
                    or len(set(record.get('pose_glbs',[])))!=1
                    or roots.shape!=(901,3) or np.max(np.abs(roots-roots[0]))>1e-5):
                errors.append(ep+':static fallback contains actual movement or changing geometry')
        if not is_seated and not is_static_fallback and not moving:errors.append(ep+':walking request has no moving pose')
        if record['maximum_navmesh_horizontal_error_m']>.002:errors.append(ep+':navigation drift')
        navigation.append(record['maximum_navmesh_horizontal_error_m'])
        clearance=record['minimum_certified_triangle_nonfloor_clearance_m'];clearances.append(clearance)
        if clearance<=clearance_limit:errors.append(ep+':whole original skin clearance fails declared quality policy')
        if animal_visual_only and 0.<clearance<=.02:warnings.append(ep+':positive nonfloor clearance below 2cm visual preference')
        for index,(frame,row) in enumerate(zip(source,actual)):
            if row.get('root_max_error',0)>1e-5 or row.get('native_emitter_readback_error_m',1)>.0001:
                errors.append(ep+':native root/head emitter readback')
            readbacks.append(row.get('native_emitter_readback_error_m',1))
            if record.get('species_id') in {'dog','cat'}:
                matrix=np.asarray(row.get('body_to_world_m',[]),float)
                if matrix.shape!=(4,4):
                    errors.append(ep+':missing actual quadruped native body matrix')
                else:
                    up=matrix[:3,1];tilt=float(np.degrees(np.arccos(np.clip(up[1]/max(np.linalg.norm(up),1e-15),-1,1))))
                    if tilt+frame.get('upright_tilt_deg',0)>10:
                        errors.append(ep+':actual native quadruped body up not preserved')
                if not frame['moving'] and not all(v['in_contact'] for v in frame['feet'].values()):
                    contact_issue('stationary quadruped has an unplanted paw')
            contact=[v for v in frame['feet'].values() if v['in_contact']]
            if not contact:contact_issue('no actual planted support at frame '+str(index))
            for side,foot in frame['feet'].items():
                delta=float(foot['floor_delta_m'])
                minimum=-.00101 if record.get('species_id')=='cat' else -.008
                contact_passed=(abs(delta)<=limit+1e-12 if owner_quality else abs(delta)<limit+1e-12)
                passed=delta>=minimum and (not foot['in_contact'] or contact_passed)
                lock_error=foot.get('support_lock_error_m',0)
                if owner_quality:
                    if lock_error>support_limit+1e-12:passed=False
                elif lock_error>=support_limit:passed=False
                if not passed:contact_issue('foot contact/penetration/lock')
                feet.append({'endpoint_id':ep,'frame_index':index,'foot':side,'in_contact':bool(foot['in_contact']),
                    'delta_m':delta,'contact_limit_m':limit,'support_lock_limit_m':support_limit,'owner_quality':owner_quality,'passed':passed,'lock_error_m':foot.get('support_lock_error_m')})
            for child in row.get('attached_emitter_readbacks',[]):
                if child['readback_error_m']>.0001:errors.append(ep+':foot child readback')
                readbacks.append(child['readback_error_m'])
            if frame.get('upright_tilt_deg',0)>10:errors.append(ep+':animal body lost upright source orientation')
        if is_seated:
            if abs(record['seat_contact_gap_m'])>.012 or any(abs(v)>.012 for v in record['native_support_deltas_m']):
                errors.append(ep+':seated or chair floor contact')
            if record.get('chair_produces_sound'):errors.append(ep+':silent chair has sound')
        instances.append({'endpoint_id':ep,'motion_kind':record.get('motion_kind','human_walk'),'native_frames':len(actual),'moving_frames':moving,
            'species_id':record.get('species_id','human'),'maximum_contact_error_m':max((abs(v['delta_m']) for v in feet if v['endpoint_id']==ep and v['in_contact']),default=0),
            'minimum_nonfloor_clearance_m':clearance,'actual_source_registry':record['lineage']['roster']})
    return {'status':'fail' if errors else 'pass','errors':sorted(set(errors)),'appearance_warnings':sorted(set(warnings)),'instances':instances,
        'native_30Hz_frames':sum(v['native_frames'] for v in instances),'foot_rows_checked':len(feet),
        'foot_rows_passed':sum(v['passed'] for v in feet),'stance_rows':sum(v['in_contact'] for v in feet),
        'worst_stance_error_m':max((abs(v['delta_m'])for v in feet if v['in_contact']),default=0),
        'worst_support_lock_error_m':max((v['lock_error_m'] or 0 for v in feet),default=0),
        'emitter_readbacks':len(readbacks),'emitter_worst_error_m':max(readbacks,default=0),
        'minimum_full_triangle_nonfloor_clearance_m':min(clearances,default=None),
        'maximum_navmesh_horizontal_error_m':max(navigation,default=0),'foot_rows':feet,
        'thresholds_changed':any(v['owner_quality'] for v in quality_limits.values()),
        'production_visual_quality_limits_by_endpoint':quality_limits,'truth_and_answer_gates_changed':False}
