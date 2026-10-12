"""Read-only intake of the human-roster package, preserving admission limits.

A colour variant is a resource, never a physical actor. Static native evidence
does not certify a complete episode, walking collision or a non-mouth emitter.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

REQUESTED_KINDS = {
 'human_speech':('standing_human','speech',None),
 'telephone_speech':('phone_call','speech',None),
 'megaphone_speech':('megaphone_speaker','speech',None),
 'acoustic_guitar':('guitar_player','guitar_live','acoustic'),
 'violin':('violin_player','violin_live',None),
 'piano':('pianist','piano_live',None),
 'saxophone':('saxophone_player','saxophone_live',None),
 'flute':('flute_player','flute_live',None),
 'hand_drum':('hand_drum_player','hand_drum_live',None),
 'hand_clap':('clapping_person','clapping',None),
 'cough':('coughing_person','coughing',None),
}

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _file(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('roster artifact missing or escapes its declared root')
    return path

def intake_roster(manifest_path, *, authority="formal_registry"):
    if authority not in {"formal_registry", "legacy_local_rights"}:
        raise ValueError("unknown human admission authority")
    manifest_path=Path(manifest_path)
    manifest_bytes=manifest_path.read_bytes()
    registry_sha256=hashlib.sha256(manifest_bytes).hexdigest()
    manifest=json.loads(manifest_bytes)
    if manifest.get('schema') not in {'avengine_v7_human_roster_v1','avengine_v7_human_roster_v2'}:
        raise ValueError('unsupported human roster schema')
    root=Path(manifest['artifact_root'])
    if manifest_path.resolve()!=root.resolve()/'registrations/human_roster.json':
        # An exact archived registry may survive an upstream atomic update.
        # Its declared asset root and every artifact/license gate stay intact.
        snapshot=manifest_path.resolve()
        if (snapshot.name!='human_roster.json' or snapshot.parent.name!='registrations'
                or snapshot.parents[2].name!='human' or snapshot.parents[3].name!='input_snapshots'
                or snapshot.parents[1].name!=registry_sha256):
            raise ValueError('roster root and registry disagree')
    license_path=_file(root,'evidence/license_sources.json')
    licenses=json.loads(license_path.read_text())
    license_receipts={}
    for license in licenses:
        try:
            source=Path(license['source_path'])
            copy=_file(root,license['retained_copy'])
            valid=(source.is_file() and _digest(source)==license['sha256']==_digest(copy)
                   and license.get('status')=='verified_local_exact_producer')
            license_receipts[license['name']]={**license,'local_binding_valid':valid}
        except (KeyError,TypeError,ValueError,OSError):
            license_receipts[license.get('name','unknown')]={**license,'local_binding_valid':False}
    mit=license_receipts.get('rocketbox_mit') or {}
    mit_valid=mit.get('local_binding_valid') is True and mit.get('license')=='MIT'
    records=[];major_ids=set();variant_ids=set()
    for major in manifest['majors']:
        mid=major['major_id']
        if mid in major_ids:raise ValueError('duplicate body identity')
        major_ids.add(mid)
        for variant in major['minor_variants']:
            vid=variant['variant_id']
            if vid in variant_ids or variant['major_id']!=mid:
                raise ValueError('duplicate or mismatched variant lineage')
            variant_ids.add(vid)
            reasons=[];observed={};warnings=[]
            habitat=variant.get('habitat') or {};ue=variant.get('ue') or {}
            if habitat.get('status')!='pass' or ue.get('status')!='pass' or variant.get('dual_backend_static_usable') is not True:
                reasons.append('upstream_dual_backend_static_not_passed')
            if ue.get('logical_actor_count')!=1 or ue.get('blueprint_readback_status')!='pass':
                reasons.append('one_logical_actor_blueprint_not_verified')
            paths=[habitat.get('glb','')]+habitat.get('native_evidence',[])+[ue.get('native_evidence','')]
            if len(paths)<4:reasons.append('native_backend_evidence_paths_incomplete')
            for relative in paths:
                try:
                    path=_file(root,relative);observed[str(path)]=_digest(path)
                except (TypeError,ValueError,OSError):
                    reasons.append('native_artifact_missing_or_outside_root')
            gender=major.get('gender')
            if gender not in {'male','female'}:
                warnings.append('visual_gender_not_annotated_dialogue_voice_binding_blocked')
                if authority=='legacy_local_rights':
                    reasons.append('visual_gender_not_annotated')
            rocketbox=str(major.get('source','')).startswith('Rocketbox')
            if rocketbox:
                if not mit_valid:reasons.append('Rocketbox_original_copy_MIT_binding_failed')
            else:
                warnings.append('upstream_formal_static_registration_is_owner_authority_for_lx')
                if authority=='legacy_local_rights':
                    reasons.append('lx_ISNet_checkpoint_rights_and_source_commit_unverified')
            point=variant.get('mouth_local_glb_m')
            ue_point=variant.get('mouth_local_ue_cm')
            try:
                a,b=np.asarray(point,float),np.asarray(ue_point,float)
                valid=(a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all()
                       and np.max(np.abs(b-100*np.array([-a[2],a[0],a[1]])))<=1e-6)
                if not valid:reasons.append('mouth_coordinate_transform_failed')
            except (ValueError,TypeError):reasons.append('mouth_coordinate_transform_failed')
            request=variant.get('sound_class_requested')
            kind,content,subtype=REQUESTED_KINDS.get(request,(None,None,None))
            if kind is None:reasons.append('physical_action_capability_mapping_missing')
            records.append({
                'asset_id':vid,'body_identity_id':mid,'source':major['source'],
                'source_registry':str(manifest_path),'source_registry_sha256':registry_sha256,
                'habitat_glb':str(root/habitat.get('glb','')),'actual_artifact_sha256':observed,
                'visual_gender':gender,'color':variant['color'],
                'mouth_local_glb_m':point,'mouth_local_ue_cm':ue_point,
                'capability':{'asset_kind':kind,'instrument_subtype':subtype,
                    'content_class':content,'source':'roster requested class; no extra geometry inference'},
                'static_research_qualified':not reasons,'static_blockers':sorted(set(reasons)),
                'admission_authority':authority,'standing_registered_qualified':not reasons,
                'dialogue_voice_binding_eligible':not reasons and gender in {'male','female'},
                'upstream_warnings':warnings,'allowed_pose_for_this_task':'stand',
                'ue_generated_class':ue.get('generated_class_path'),
                'habitat_stand':habitat.get('stand'),
                'native_episode_admitted':False,'episode_blockers':['complete_episode_not_run',
                    'multiactor_collision_and_room_placement_not_verified'],
                'walk_status':habitat.get('walk'),'sit_status':habitat.get('sit'),
                'non_mouth_emitter_status':'not_provided' if content not in {'speech','coughing'} or kind=='megaphone_speaker' else 'mouth_only',
                'limits':'upstream static receipts are read-only; no native job is run by this intake',
            })
    summary=manifest['summary']
    if summary['major_count']!=len(major_ids) or summary['minor_count']!=len(variant_ids):
        raise ValueError('declared roster counts differ from physical lineage')
    if _digest(manifest_path)!=registry_sha256:
        raise ValueError('roster changed during read-only intake')
    return {'records':records,'source_registry':str(manifest_path),'source_registry_sha256':registry_sha256,
        'admission_authority':authority,'license_receipts':license_receipts,'major_count':len(major_ids),'minor_count':len(variant_ids),
        'static_research_qualified_minors':sum(r['static_research_qualified'] for r in records),
        'static_research_qualified_majors':len({r['body_identity_id'] for r in records if r['static_research_qualified']}),
        'dialogue_voice_eligible_minors':sum(r['dialogue_voice_binding_eligible'] for r in records),
        'full_episode_admission_count':0,'new_native_jobs_run_by_intake':0,
        'no_human_viewing_or_listening':True}

def bind_physical_actors(instances, intake):
    """Instances may reuse a variant, but must never reuse physical actor IDs."""
    by_asset={r['asset_id']:r for r in intake['records']}
    actors=[];uids=set();native_ids=set()
    for instance in instances:
        uid=instance.get('actor_uid');native=instance.get('native_actor_id')
        if (not isinstance(uid,str) or not uid or uid in uids
                or isinstance(native,bool) or not isinstance(native,int) or native<=0 or native in native_ids):
            raise ValueError('physical actor identities must be unique')
        asset=by_asset[instance['asset_id']]
        if not asset['static_research_qualified']:
            raise ValueError('asset has not passed formal static registration')
        if instance.get('pose','stand')!='stand':
            raise ValueError('this task permits registered standing pose only')
        uids.add(uid);native_ids.add(native)
        actors.append({**instance,'body_identity_id':asset['body_identity_id'],
            'visual_gender':asset['visual_gender'],'capability':asset['capability'],
            'native_episode_admitted':False})
    return actors

def standing_mouth_world(asset, body_to_world_m):
    """Transform the registered GLB mouth, without guessing or changing it.

    This computes the attachment expected by QA; the producer must compare it
    to its actual native/RIR readback. No room contact is certified here.
    """
    matrix=np.asarray(body_to_world_m,dtype=float)
    point=np.asarray(asset['mouth_local_glb_m'],dtype=float)
    if (matrix.shape!=(4,4) or point.shape!=(3,) or not np.isfinite(matrix).all()
            or not np.isfinite(point).all()
            or not np.allclose(matrix[3],[0,0,0,1],atol=1e-8)):
        raise ValueError('invalid native metre body transform')
    rotation=matrix[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<0:
        raise ValueError('registered standing mouth needs a rigid metre transform')
    return (matrix@np.r_[point,1])[:3].tolist()
