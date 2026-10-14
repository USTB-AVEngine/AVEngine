from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from PIL import Image
import pytest
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_full_question_delivery import (
    finalize_parent_question_pools,derive_native_appearance_evidence,validate_legacy_source_evidence,
)
pytestmark=pytest.mark.fast_unit
MEMBERS=['v0_a0','v0_a1','v1_a0','v1_a1']


def fixture_parent(tmp_path, *, deficient_second=False):
    root=tmp_path/'parents'/'unit_parent';root.mkdir(parents=True)
    row={'scene_id':root.name,'members':MEMBERS,'window_count':2,'questions_per_window':14,'question_quota_base':0}
    members=[]
    for member in MEMBERS:
        pool=root/('pool_'+member);pool.mkdir();windows=[]
        for index in range(2):
            wid=member+'_w'+str(index);dest=pool/wid;dest.mkdir()
            (dest/'video.mp4').write_bytes(member[:2].encode())
            import soundfile as sf
            amplitude=.1 if member.endswith('a1') else 0.
            sf.write(dest/'audio_foa.wav',np.full((160,4),amplitude),16000,subtype='PCM_16')
            sf.write(dest/'audio_mono.wav',np.full(160,amplitude),16000,subtype='PCM_16')
            media={k:str(dest/v) for k,v in {'video':'video.mp4','audio_foa':'audio_foa.wav','audio_mono':'audio_mono.wav'}.items()}
            vision=1 if deficient_second and index==1 else 6
            public=[];private=[]
            for i,mod in enumerate(['audiovisual']*8+['audio']*6+['vision']*vision):
                qid=wid+'_'+str(i);answer=(str(member in ('v0_a0','v1_a1')) if i<6 else 'invariant AV control') if mod=='audiovisual' else str(i)
                public.append({'question_id':qid,'logical_question_id':qid,'qa_id':'QA-02' if mod=='audiovisual' else 'QA-14' if mod=='vision' else 'QA-03',
                    'protocol_panel':'v6','question_zh':'unit control '+str(i),'form':'open','modality':mod,'media':media})
                private.append({'question_id':qid,'logical_question_id':qid,'canonical_answer':answer,'answer':answer,
                    'evidence':{'kind':'controlled_unit_fixture_not_native_data'}})
            write_json(dest/'questions.json',public);write_json(dest/'private_answers.json',private);write_json(dest/'evidence.json',{'formal_dataset_count':0})
            windows.append({'window_id':wid})
        members.append({'member':member,'windows_directory':str(pool),'windows':windows})
    return row,members,root


def test_all_four_members_selected_before_publication_and_resume(tmp_path):
    row,members,root=fixture_parent(tmp_path)
    selected,audit=finalize_parent_question_pools(row,members,root)
    assert len(selected)==4 and all(a['binding_items_per_member']==5 for a in audit['window_audits'])
    for member in selected:
        for window in member['windows']:
            folder=Path(member['windows_directory'])/window['window_id']
            q=json.loads((folder/'questions.json').read_text());a=json.loads((folder/'private_answers.json').read_text())
            assert len(q)==len(a)==14
            assert all(item['media']['foa_format']=='ACN_WYZX_SN3D_camera' for item in q)
            assert all(item['comparison_key'] for item in a)
            assert (folder/'video.mp4').read_bytes()==member['member'][:2].encode()
            assert json.loads((folder/'evidence.json').read_text())['full_qa_selection']['formal_dataset_count']==0
    resumed,_=finalize_parent_question_pools(row,members,root)
    assert all(m['full_qa_selection_resumed'] for m in resumed)


def test_late_window_deficit_publishes_no_partial_selected_member(tmp_path):
    row,members,root=fixture_parent(tmp_path,deficient_second=True)
    with pytest.raises(ValueError,match='vision: 1 legal'):
        finalize_parent_question_pools(row,members,root)
    assert not list(root.glob('selected_windows_*_completed.json'))
    assert not list(root.glob('selected_windows_*_attempt_*'))
    with pytest.raises(ValueError,match='complete native'):
        finalize_parent_question_pools(row,members[:-1],root)


def test_recording_labels_cannot_replace_actual_formant_dry_program():
    validate_legacy_source_evidence({'members':{'v0_a0':{}}})
    with pytest.raises(RuntimeError,match='not bound'):
        validate_legacy_source_evidence({'members':{'v0_a0':{'events':{'e':{'sound_class':'water_tap_faucet'}}}}})
    with pytest.raises(RuntimeError,match='loudspeaker'):
        validate_legacy_source_evidence({'members':{'v0_a0':{'actor_profiles':{'1':{'entity_class':'loudspeaker'}}}}})


def test_native_color_review_blocks_wrong_asset_label(tmp_path):
    rgb=np.zeros((30,30,3),np.uint8);rgb[:]=[25,85,220]
    Image.fromarray(rgb).save(tmp_path/'native_frame_0000.png')
    np.savez_compressed(tmp_path/'native_masks_0000.npz',semantic=np.full((30,30),1001))
    visual={'frames':[{'actors':[{'actor_id':1,'asset_id':'body','semantic_id':1001}]}]}
    asset={'asset_id':'body','appearance':{'top_color':'blue'}}
    good=derive_native_appearance_evidence(tmp_path,visual,[asset])
    assert good['review']['actors']['1']['value']=='blue'
    assert good['review']['actors']['1']['frame_refs']==[0]
    wrong=deepcopy(asset);wrong['appearance']['top_color']='yellow'
    bad=derive_native_appearance_evidence(tmp_path,visual,[wrong])
    assert bad['review']['actors']=={} and bad['screens'][0]['status']=='blocked'
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/'native_frame_0000.png')),rgb)


def test_unreserved_av_items_do_not_claim_binding_and_all_nuisance_answers_are_checked(tmp_path):
    from avengine.timeline.v7_candidate_generation import audit_parent_group
    row,members,root=fixture_parent(tmp_path)
    row.update(house='unit_house',room_id='unit_room',split='train')
    selected,_=finalize_parent_question_pools(row,members,root)
    base={'listener_m':[0,1,0],'camera_forward':[0,0,-1],'sources':{},'events':[]}
    audit=audit_parent_group(row,base,selected)
    assert audit['status']=='pass'
    assert len(audit['verified_binding_logical_question_ids'])==40
    assert any(a['canonical_answer']=='invariant AV control' for a in audit['observed_private_answers'])
    # Alter an unselected pure-vision counterpart too: pool evidence remains
    # part of the invariant check instead of being hidden by quota selection.
    path=Path(members[1]['windows_directory'])/members[1]['windows'][0]['window_id']/'private_answers.json'
    rows=json.loads(path.read_text());rows[-1]['canonical_answer']='wrong';path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='counterfactual answers failed'):audit_parent_group(row,base,selected)


def test_single_parent_skips_empty_family_and_fills_only_from_actual_legal_pool(tmp_path):
    row,members,root=fixture_parent(tmp_path,deficient_second=True)
    row['members']=['v0_a0'];row['production_scenario']={'semantic_scenario_id':'unit_actual_pool_has_no_QA28'}
    source=Path(members[0]['windows_directory'])
    original={g['question_id']:g['canonical_answer']for win in members[0]['windows']
              for g in json.loads((source/win['window_id']/'private_answers.json').read_text())}
    selected,audit=finalize_parent_question_pools(row,members[:1],root)
    assert audit['window_audits'][1]['owner_available_candidate_allocation']['changed']
    assert all(a['actual_content_direction_reservation']['skipped_empty_family']for a in audit['window_audits'])
    for win in selected[0]['windows']:
        folder=Path(selected[0]['windows_directory'])/win['window_id']
        questions=json.loads((folder/'questions.json').read_text())
        gold=json.loads((folder/'private_answers.json').read_text())
        assert len(questions)==14 and all(g['canonical_answer']==original[g['question_id']]for g in gold)
