"""Training-only policy must not waive another room's unresolved checks."""
import copy
import math
import pytest

from tools.acoustics.report_train_room_bands import (
    LEAK_REVIEW_REASON, counts, index_rooms, patch_candidates, policy_row, training_decision,
)

pytestmark = pytest.mark.fast_unit


def room(leak=.10):
    band=leak>.05
    return dict(house='hm3d_train_fixture', family='hm3d', room_label='R0', room_type='bedroom',
                stage1_status='pass', visual_status='pass', acoustic_status='review' if band else 'pass',
                status='review' if band else 'pass', tier='leak_5_to_15_review' if band else 'training',
                reasons=[LEAK_REVIEW_REASON] if band else [], max_escape_fraction=leak,
                max_foa_angle_deg=15., distance_exponents=[.7,1.3], valid_rirs=6,
                distinct_listeners=2, max_direct_delay_ms=3., training_eligible=not band,
                smy_label='skip', so_seen=True, m1_seen=True)


@pytest.mark.parametrize('leak,eligible,band',[(.05,True,False),(.0500001,True,True),(.15,True,True),(.1500001,False,False)])
def test_exact_leakage_boundaries_and_train_only(leak,eligible,band):
    result=training_decision(room(leak))
    assert result['training_eligible'] is eligible
    assert result['acoustic_review_band_5_15'] is band
    assert result['allowed_splits']==['train']
    assert result['development_test_eligible'] is False


@pytest.mark.parametrize('field,value',[
    ('max_escape_fraction',None),('max_escape_fraction',float('nan')),('max_escape_fraction',-.001),
    ('max_foa_angle_deg',None),('max_foa_angle_deg',15.0001),('max_foa_angle_deg',float('inf')),
    ('distance_exponents',[]),('distance_exponents',[None]),('distance_exponents',[.69999]),
    ('distance_exponents',[1.30001]),('distance_exponents',[float('nan')]),
    ('valid_rirs',5),('valid_rirs',True),('distinct_listeners',1),
    ('max_direct_delay_ms',None),('max_direct_delay_ms',3.001),('max_direct_delay_ms',float('nan')),
    ('visual_status','review'),('stage1_status','review'),('acoustic_status','fail'),
])
def test_band_cannot_hide_missing_or_failed_other_metric(field,value):
    row=room();row[field]=value
    assert not training_decision(row)['training_eligible']


@pytest.mark.parametrize('reason',[
    'FOA_THREE_POSITIONS_MISSING','MULTIPLE_LISTENER_POSITIONS_MISSING','RT60_OUTSIDE_STARTING_RANGE',
    'ACOUSTIC_GEOMETRY_BEFORE_WAVEFORM_UNAVAILABLE','RAW_WITNESS_VS_ACOUSTIC_DIRECT_PATH_DISAGREEMENT',
])
def test_another_frozen_review_reason_is_never_waived(reason):
    row=room();row['reasons'].append(reason)
    result=training_decision(row)
    assert not result['training_eligible'] and reason in result['training_exclusion_reasons']


def test_training_overlay_preserves_native_review_and_input():
    original=room();before=copy.deepcopy(original)
    result=policy_row(original)
    assert original==before
    assert result['training_eligible'] and result['status']=='qualified'
    assert result['acoustic_status']=='review' and result['reasons']==[LEAK_REVIEW_REASON]
    assert result['training_tags']==['acoustic_review_band_5_15']
    assert result['smy_label']=='skip' and result['so_seen'] and result['m1_seen']


def test_semantic_geometry_needs_real_before_waveform_even_without_explicit_reason():
    row=room();row['acoustic_geometry_derivation']={'removed_faces':2}
    row['acoustic_before_waveform_available']=False
    assert not training_decision(row)['training_eligible']


def test_derived_visual_requires_comparison():
    row=room();row['geometry_derivation_label']='派生几何：清退化三角形'
    row['geometry_comparison_status']='review'
    assert not training_decision(row)['training_eligible']


def test_patch_tag_is_independent_of_leak_tag_and_train_only():
    strict=room(.04);strict['acoustic_patched']=True
    assert training_decision(strict)['training_tags']==['acoustic_patched']
    band=room();band['acoustic_patched']=True
    assert training_decision(band)['training_tags']==['acoustic_review_band_5_15','acoustic_patched']


def test_counts_preserve_unknown_types_and_unique_house_count():
    a=room();b=dict(a,room_label='R1',room_type='unknown')
    assert counts([a,b])=={'houses':1,'rooms':2,'room_types':{'bedroom':1,'unknown':1}}
    assert len(index_rooms([a,b]))==2
    with pytest.raises(ValueError,match='Duplicate'):
        index_rooms([a,a])


def test_unfinished_patch_task_never_adds_unverified_rooms(tmp_path):
    additions,audit=patch_candidates(tmp_path,{}, {}, set())
    assert additions=={} and audit['status']=='pending_final_room_delivery'
    assert 'summary.json' in audit['missing_final_files']


def test_completed_report_is_published_without_overwriting_prior_results(tmp_path):
    from tools.acoustics.report_train_room_bands import publish_directory
    staging=tmp_path/'staging';staging.mkdir();(staging/'room.csv').write_text('complete\n')
    target=tmp_path/'report_final';publish_directory(staging,target)
    assert not staging.exists() and (target/'room.csv').read_text()=='complete\n'
    retry=tmp_path/'retry';retry.mkdir();(retry/'room.csv').write_text('different\n')
    with pytest.raises(FileExistsError):publish_directory(retry,target)
    assert (target/'room.csv').read_text()=='complete\n' and retry.is_dir()


def test_completed_no_op_patch_never_supplies_rescue(tmp_path):
    import json
    (tmp_path/'evidence').mkdir()
    (tmp_path/'summary.json').write_text(json.dumps({'missing_houses':[]}))
    (tmp_path/'all_house_results.json').write_text(json.dumps([
        {'house':'mp3d_noop','group':'blocked','patch_count':0}]))
    (tmp_path/'evidence/room_verification_execution.json').write_text(json.dumps({'missing':[]}))
    additions,audit=patch_candidates(tmp_path,{}, {}, set())
    assert not additions and audit['status']=='completed_delivery_checked'
    assert audit['admitted_rooms']==[]


def test_reused_waveform_requires_its_explicit_retained_root(tmp_path):
    from tools.acoustics.report_train_room_bands import checked_path
    current=tmp_path/'current';current.mkdir()
    retained=tmp_path/'retained';retained.mkdir();wave=retained/'rirs.npz';wave.write_bytes(b'retained')
    with pytest.raises(ValueError,match='escaped'):checked_path(wave,[current])
    assert checked_path(wave,[current,retained])==wave.resolve()
