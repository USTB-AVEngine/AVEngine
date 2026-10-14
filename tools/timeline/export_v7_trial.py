"""Export admitted parents to the v7 row format for an isolated trial only.

This does not change producer admission, private canonicals, or alpha export.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from avengine.timeline.v7_alpha_delivery import readable_training_answer
from avengine.timeline.v7_production_contract import question_allocation, validate_question_counts
from avengine.timeline.v7_s_batch_checks import open_delivered_media
from avengine.qa.unified_scoring import score_unified_item


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def retain(source, target, owned_root):
    source=Path(source);target=Path(target)
    if not source.is_file() or not source.resolve().is_relative_to(owned_root):
        raise ValueError('trial payload must come from this task-owned sealed output')
    if not target.resolve().is_relative_to(owned_root):
        raise ValueError('trial export must remain in this task output')
    target.parent.mkdir(parents=True,exist_ok=True)
    sha=digest(source)
    if target.exists():
        if digest(target)!=sha: raise ValueError('retained trial payload differs; use a fresh export')
    else: os.link(source,target)
    return {'path':str(target),'bytes':target.stat().st_size,'sha256':sha}


def parent_payload(summary, output, owned_root):
    pointer_path=Path(summary['seal_path']);pointer=load(pointer_path)
    parent=pointer_path.parent;seal=parent/pointer['attempt'];result=pointer['result']
    kind=result['status']
    if kind not in {'integration_pilot_admitted','formal_alpha_admitted'}:
        raise ValueError('original producer did not admit this parent')
    checks=result.get('machine_checks')
    if checks is not None and checks.get('status')!='pass': raise ValueError('original checks failed')
    original=pointer['recipe']['input_recipe']
    row=original.get('row',original.get('seed_row'))
    if not row or row['split']!='train': raise ValueError('only the authorized non-test trial houses are exported')
    scene=row['scene_id'];delivery=output/'parents'/scene
    files=[]
    for name in pointer['required_files']:
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('invalid seal payload path')
        files.append(retain(seal/name,delivery/name,owned_root))
    files.append(retain(pointer_path,delivery/'private/source_producer_seal.json',owned_root))
    training=[];logical=set();family=Counter();forms=Counter();negative=Counter()
    for win in result['windows']:
        wid=win['window_id'];member=win.get('member',wid.rsplit('__',2)[-2])
        questions=load(seal/'windows'/wid/'questions.json')
        answers=load(seal/'private'/(wid+'_answers.json'));by_id={g['question_id']:g for g in answers}
        quota=row['question_quota_base']+row['members'].index(member)*row['window_count']+int(wid.rsplit('w',1)[1])
        partial=row.get('acceptance_allow_partial_families') is True
        evidence=load(seal/'private'/(wid+'_evidence.json'))
        allocation=question_allocation(quota,14)
        if partial:
            audit_policy=evidence.get('full_qa_selection',{}).get('audit',{}).get('acceptance_partial_family_policy')
            if kind!='integration_pilot_admitted' or row.get('trial_only') is not True or not audit_policy:
                raise ValueError('partial trial export lacks original explicit selection policy')
            allocation=audit_policy['actual_allocation']
        # Retain the original producer's explicitly audited available-candidate
        # allocation. This is export metadata, not a change to any answer gate.
        owner=evidence.get('full_qa_selection',{}).get('audit',{}).get('owner_available_candidate_allocation')
        if owner and owner.get('changed') is True:
            check=evidence['full_qa_selection'].get('question_check',{})
            if (kind!='integration_pilot_admitted' or owner.get('canonical_answers_changed') is not False
                    or owner.get('reason')!='owner: skip unavailable candidate families; keep existing legal answers'
                    or check.get('status')!='pass' or check.get('logical_questions')!=14
                    or evidence['full_qa_selection'].get('formal_dataset_count')!=0):
                raise ValueError('original available-candidate allocation lacks explicit unchanged-answer proof')
            authorized_helper=(row.get('owner_authorized_helper_seated_results') is True
                               or summary.get('owner_authorized_helper_seated_results') is True
                               or summary.get('origin')=='owner_authorized_helper_seated_results')
            if not authorized_helper and row.get('trial_only') is not True:
                raise ValueError('available-candidate allocation requires owner-authorized isolated trial')
            allocation=owner['actual']
        validate_question_counts(questions,answers,expected_allocation=allocation,allow_partial_window=partial)
        selected=load(parent/('selected_windows_'+member+'_completed.json'))
        raw=Path(selected['recipe']['pool_directory'])/wid
        audit_path=raw/'candidate_audit.json'
        if not audit_path.resolve().is_relative_to(owned_root):
            if not (row.get('owner_authorized_helper_seated_results') is True
                    or summary.get('owner_authorized_helper_seated_results') is True
                    or summary.get('origin')=='owner_authorized_helper_seated_results'):
                raise ValueError('external candidate pool is not an authorized helper import')
            imported=load(owned_root/'receipts/HELPER_SEATED_IMPORTED_20261014.json')
            match=next((item for item in imported['files'] if item.get('source')==str(audit_path)),None)
            if not match:raise ValueError('helper candidate audit missing from byte-identical import receipt')
            audit_path=Path(match.get('destination',match.get('dest','')))
            if not audit_path.resolve().is_relative_to(owned_root) or digest(audit_path)!=match['sha256']:
                raise ValueError('helper candidate audit copy differs from original import receipt')
        audit=load(audit_path)
        authority={item['question_id']:item for item in audit['v6_items']}
        files.append(retain(audit_path,delivery/'private'/(wid+'_candidate_audit.json'),owned_root))
        for question in questions:
            gold=by_id[question['question_id']]
            answer,check=readable_training_answer(question,gold,authority)
            if gold.get('authority_question_id'):
                bad=score_unified_item(authority[gold['authority_question_id']],
                    '__invalid_trial_control_answer__',form=question['form'])
                if bad.get('status')=='scored' and bad.get('score')==1.:
                    raise ValueError('unchanged scorer accepted the invalid-answer control')
                negative[bad.get('status','unknown')]+=1
            folder=Path('parents')/scene/'windows'/wid
            training.append({'question_id':question['question_id'],'logical_question_id':question['logical_question_id'],
                'independent_parent_scene_id':scene,'window_id':wid,'split':row['split'],'house':row['house'],
                'qa_id':question['qa_id'],'protocol_panel':question['protocol_panel'],'modality':question['modality'],
                'question_zh':question['question_zh'],'question_en':question.get('question_en'),
                'video':str(folder/'video.mp4'),'audio_foa':str(folder/'audio_foa.wav'),'audio_mono':str(folder/'audio_mono.wav'),
                'foa_format':'ACN_WYZX_SN3D_camera','model_answer':answer,'canonical_answer':gold['canonical_answer'],
                'private_evidence':str(Path('parents')/scene/'private'/(wid+'_evidence.json')),
                'target_verification':check,'E_S_input_source':'deployment_prediction_not_supplied',
                'dataset_role':'trial_only','producer_admission_status':kind})
            logical.add(question['logical_question_id']);forms[question['form']]+=1
    unique_questions={r['logical_question_id']:r for r in training}
    family.update(r['qa_id'] for r in unique_questions.values())
    manifest={'schema':'v7-isolated-production-format-trial-parent-v1','dataset_role':'trial_only',
        'source_seal':str(pointer_path),'source_seal_sha256':digest(pointer_path),'scene_id':scene,
        'producer_admission_status':kind,'formal_release_promoted':False,'windows':result['windows'],
        'logical_questions':len(logical),'rows':len(training),'question_family_counts':dict(family),
        'form_counts':dict(forms),'invalid_answer_control_results':dict(negative),
        'required_payload_files':files,'payload_logical_bytes':sum(f['bytes'] for f in files),
        'training_dataset_merge':False,'CCI_transfer':False}
    write(delivery/'trial_delivery_manifest.json',manifest)
    return training,manifest


def export(registry_path, summary_path, output):
    registry_path=Path(registry_path).resolve();registry=load(registry_path)
    pilot=registry_path.parent;owned_root=pilot.parent.resolve();output=Path(output).resolve()
    if registry.get('schema')!='v7-isolated-trial-30-parent-registry-v1':
        raise ValueError('explicit authorized isolated trial registry required')
    requested_count=registry.get('requested_parents')
    if requested_count!=len(registry.get('parents',[])):
        raise ValueError('trial registry parent count does not match registered slots')
    if requested_count!=30 and registry.get('owner_authorized_1500_question_expansion') is not True:
        raise ValueError('expanded trial requires the owner-authorized 1500-question task')
    if output.parent!=pilot or output.name!='export': raise ValueError('export must be pilot_20261014/export')
    if registry.get('training_dataset_merge') is not False or registry.get('CCI_transfer') is not False:
        raise ValueError('isolated trial scope required')
    summary=load(summary_path);requested={r['slot']:r for r in registry['parents']}
    output.mkdir(exist_ok=True);training=[];parents=[];errors=[]
    tests=set(load(registry['contract']['path'])['test_houses'])
    for parent in summary['parents']:
        if parent['slot'] not in requested: raise ValueError('unregistered trial parent')
        if parent['status']!='passed': continue
        if parent['house'] in tests: raise ValueError('heldout physical test house')
        try:
            rows,manifest=parent_payload(parent,output,owned_root)
            training.extend(rows);parents.append({'slot':parent['slot'],'family':parent['family'],**manifest})
        except Exception as error:
            errors.append({'slot':parent['slot'],'scene_id':parent.get('scene_id'),'error':str(error)})
    ids=[r['question_id'] for r in training]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate trial question rows')
    # Updating only this task-owned trial table allows newly sealed parents to
    # join a later readback. Original producer files and labels stay unchanged.
    temporary=output/'train.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in training))
    temporary.replace(output/'train.jsonl')
    if training:
        temporary=output/'train.csv.tmp'
        with temporary.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(training[0]));writer.writeheader();writer.writerows(training)
        temporary.replace(output/'train.csv')
    readback=open_delivered_media(output)
    # Read the written rows rather than relying on the in-memory exporter.
    read_rows=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    for row in read_rows:
        gold=load(output/'parents'/row['independent_parent_scene_id']/'private'/(row['window_id']+'_answers.json'))
        actual=next(g for g in gold if g['question_id']==row['question_id'])
        if actual['canonical_answer']!=row['canonical_answer']: raise ValueError('canonical changed in the production-format row')
    logical={r['logical_question_id']:r for r in read_rows}
    receipt={'schema':'v7-isolated-production-format-trial-export-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'pass' if parents and not errors and readback['status']=='pass' else 'blocked',
        'dataset_role':'trial_only','registry':str(registry_path),'registry_sha256':digest(registry_path),
        'source_summary':str(Path(summary_path).resolve()),'export_root':str(output),
        'exported_parents':len(parents),'logical_questions':len(logical),'question_rows':len(read_rows),
        'question_family_counts':dict(Counter(r['qa_id'] for r in logical.values())),
        'row_family_counts':dict(Counter(r['qa_id'] for r in read_rows)),
        'payload_logical_bytes':sum(p['payload_logical_bytes'] for p in parents),
        'existing_reader':'avengine.timeline.v7_s_batch_checks.open_delivered_media',
        'media_reader_check':readback,'canonical_rows_readback_checked':len(read_rows),'parent_errors':errors,
        'parents':parents,'training_dataset_merge':False,'CCI_transfer':False,'formal_release_promoted':False}
    write(output/'TRIAL_ONLY.json',{'dataset_role':'trial_only','never_merge_into_training':True,'CCI_transfer':False})
    write(pilot/'TRIAL_EXPORT_READER_CHECK.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','exported_parents','logical_questions','question_rows','payload_logical_bytes','parent_errors']}))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['registry','summary','output']: parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args();receipt=export(args.registry,args.summary,args.output)
    raise SystemExit(0 if receipt['status']=='pass' else 2)


if __name__=='__main__': main()
