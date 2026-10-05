"""Physics/sign/identity checks for the bounded formal static alpha profile."""
from copy import deepcopy
import math
import numpy as np
import pytest
from avengine.capture.v7_alpha_native import rotate_world_foa
from avengine.timeline.continuous_scene import raw_world_to_camera_sn3d
from avengine.timeline.v7_alpha_sampling import camera_knots,alpha_row

pytestmark=pytest.mark.fast_unit


def test_actual_camera_rotation_matches_existing_fixed_camera_foa():
    a=np.random.default_rng(701).normal(size=(4,16000))
    for yaw in [-175.,-90.,-10.,0.,85.,179.]:
        frames=[{'frame_index':0,'yaw_deg':yaw},{'frame_index':15,'yaw_deg':yaw}]
        f=np.array([math.sin(math.radians(yaw)),0.,-math.cos(math.radians(yaw))])
        np.testing.assert_allclose(rotate_world_foa(a,frames),raw_world_to_camera_sn3d(a,f),rtol=1e-12,atol=1e-12)


def test_observed_camera_pan_preserves_foa_vector_energy_at_wrap():
    a=np.random.default_rng(702).normal(size=(4,32000))
    frames=[{'frame_index':0,'yaw_deg':179.},{'frame_index':30,'yaw_deg':-179.}]
    result=rotate_world_foa(a,frames)
    np.testing.assert_array_equal(result[0],a[0])
    np.testing.assert_allclose(np.sum(result[1:]**2,axis=0),np.sum(a[1:]**2,axis=0)/3,rtol=1e-12,atol=1e-12)


def test_alpha_projection_preserves_parent_seed_and_split_without_claiming_v5_quotas():
    original={'assignment_status':'admitted','house':'h','schema':'v5-diagnostic','room_id':'old',
        'members':['v0_a0'],'window_start_ticks':[54,198],'scene_seed':314,'split':'train','scene_id':'s'}
    layout={'house':'h','status':'pass','room_id':'observed'}
    row=alpha_row(original,layout,7)
    assert row['scene_seed']==314 and row['window_start_ticks']==[54,198] and row['split']=='train'
    assert row['upstream_members']==['v0_a0'] and row['upstream_room_id']=='old'
    assert row['alpha_scope']['full_v5_quotas_realized'] is False
    assert row['alpha_scope']['excluded_families']==['QA-26','QA-27','QA-28']


def test_protocol_rigid_basis_change_matches_measured_camera_geometry():
    from avengine.qa.v7_panel_coverage import boundary_projection
    delta=np.array([1.2,.3,-3.8]);reference=-42.
    for actual in [-155.,-80.,-42.,0.,45.,131.]:
        def forward(yaw):return [math.sin(math.radians(yaw)),0.,-math.cos(math.radians(yaw))]
        angle=math.radians(actual-reference);c,s=math.cos(angle),math.sin(angle)
        rotated=np.array([[c,0,s],[0,1,0],[-s,0,c]])@delta
        base={'resolution_wh':[1280,720],'hfov_deg':85.}
        direct=boundary_projection({**base,'camera_forward':forward(actual)},delta)
        converted=boundary_projection({**base,'camera_forward':forward(reference)},rotated)
        assert direct['frustum']==converted['frustum']
        if direct['pixel_boundary_xy'] is not None:
            np.testing.assert_allclose(direct['pixel_boundary_xy'],converted['pixel_boundary_xy'],atol=1e-10)


def test_whole_window_transcript_is_required_when_recording_is_cropped():
    # A 600ms carry-in utterance is never claimed as an audible full sentence.
    from avengine.timeline.v7_alpha_audio import source_qa_evidence
    scene={'events':[{'event_id':'e','endpoint_id':'voice1','source_id':1,'class_id':1}],
        'sources':{'voice1':{'actor_id':1,'mechanism':'human_live_recorded','trajectory':[{'position_m':[0,1,-2]}]}},
        'listener_m':[0,1,0],'camera_forward':[0,0,-1],'seed':1}
    program={'records':[{'sound_asset_id':'a','original_id':'p225_001','transcript':'whole phrase',
        'path':'dry.wav','screens':{'automatic_activity':{'activity_intervals_samples':[[0,16000]]}}}]}
    receipt={'events':[{'event_id':'e','sound_asset_id':'a','output_start_sample':0,
        'output_end_sample_exclusive':16000,'semantic_content_complete':True}]}
    visual={'protocol_camera_readbacks':[{'time_s':0.,'yaw_deg':0.}],
        'camera_frame_readbacks':[],'renderer':{'resolution_hw':[720,1280],'hfov_deg':85.},
        'frames':[{'actors':[{'actor_id':1}]}]}
    proof=source_qa_evidence(scene,program,receipt,visual,'foa.wav',{'directions':[]},
        question_episode_id='w',window_start_s=.6)
    assert 'transcript' not in proof['events']['e']
    assert proof['events']['e']['source_qc']['whole_utterance_in_window'] is False
    assert proof['events']['e']['activity_status']=='observed'


def _authority_fixture():
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    spec=spec_from_file_location('alpha_authority_fixture',Path(__file__).parent/'unit/test_qa_unified_catalog.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    return module._fixture()


def test_multi_actor_distance_question_names_the_compared_pair_without_changing_truth():
    from avengine.qa import unified_catalog as u
    raw=_authority_fixture();raw['sampling']={'time_display_precision':2}
    facts=u.normalize_episode_bundle(raw)
    result=u.generate_unified_questions(raw,qa_ids=['QA-14'],seed='alpha-qa14-pairs',items_per_type=5)
    assert len(result['items'])>=2
    for item in result['items']:
        distances=item['evidence']['distances_m'];ids=list(distances)
        assert len(ids)==2 and abs(distances[ids[0]]-distances[ids[1]])>=.5
        assert item['truth']['value']==min(ids,key=lambda actor:distances[actor])
        stem=item['model_input']['open']['question_zh']
        for actor in ids:
            label=u._appearance_phrases(facts['actors'][actor]['appearance'])[1]
            assert label in stem
        assert '这两个个体中' in stem


def test_original_two_actor_distance_stem_and_truth_are_preserved():
    from avengine.qa import unified_catalog as u
    raw=_authority_fixture();keep={'a0','a1'};raw['actors']={k:v for k,v in raw['actors'].items() if k in keep}
    for field in ['actors','emitters']:
        raw['frame_readbacks'][field]={k:v for k,v in raw['frame_readbacks'][field].items() if k in keep}
    raw['pixel_visibility_truth']['per_instance']={k:v for k,v in raw['pixel_visibility_truth']['per_instance'].items() if k in keep}
    raw['audio_program']['events']=[e for e in raw['audio_program']['events'] if e['actor_id'] in keep]
    raw['sampling']={'time_display_precision':2}
    result=u.generate_unified_questions(raw,qa_ids=['QA-14'],seed='alpha-qa14-two',items_per_type=1)
    assert result['items']
    item=result['items'][0]
    assert '这两个个体中' not in item['model_input']['open']['question_zh']
    distances=item['evidence']['distances_m']
    assert item['truth']['value']==min(distances,key=distances.get)


def test_full_recorded_utterance_keeps_every_sample_and_only_pads_one_tick(tmp_path):
    import soundfile as sf
    from avengine.timeline.v7_audio_sources import assemble_recorded_buses
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    spec=spec_from_file_location('alpha_recorded_fixture',Path(__file__).parent/'unit/test_v7_audio_sources.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    scene,program=module.fixture(tmp_path);r=program['records'][0]
    original=np.linspace(-.2,.2,16111);sf.write(r['path'],original,16000,subtype='FLOAT')
    r.update(speaker_id='hermetic-speaker',gender='M',transcript='hermetic complete sentence')
    scene['events'][0]['end_tick']=13
    program['events'][0].update(crop_start_sample=0,preserve_full_recorded_clip=True)
    dry,receipt=assemble_recorded_buses(scene,program)
    wave,_=sf.read(r['path']);np.testing.assert_array_equal(dry['voice1'][3200:3200+len(wave)],wave*.8)
    assert not np.any(dry['voice1'][3200+len(wave):20800])
    assert receipt['events'][0]['semantic_content_complete'] is True
    assert receipt['events'][0]['trailing_padding_samples']==1489
    scene['events'][0]['end_tick']=14
    with pytest.raises(ValueError,match='at most one tick'):assemble_recorded_buses(scene,program)


def test_alpha_production_preserves_original_single_member_and_question_prefix():
    original={'assignment_status':'admitted','house':'h','schema':'v5-diagnostic','room_id':'old',
        'members':['v0_a0'],'window_start_ticks':[54,198],'scene_seed':314,'split':'train','scene_id':'s',
        'question_quota_base':138,'counterfactual_plan':'single','allowed_member_changes':[]}
    saved=deepcopy(original);layout={'house':'h','status':'pass','room_id':'observed'}
    row=alpha_row(original,layout,7,force_quartet=False)
    assert original==saved and row['members']==saved['members']
    assert row['question_quota_base']==138 and row['counterfactual_plan']=='single'
    assert row['scene_seed']==314 and row['window_start_ticks']==saved['window_start_ticks']


def test_training_actor_label_passes_original_scorer_and_preserves_private_canonical():
    from avengine.qa import unified_catalog as u
    from avengine.timeline.v7_alpha_delivery import readable_training_answer
    raw=_authority_fixture();raw['sampling']={'time_display_precision':2}
    item=u.generate_unified_questions(raw,qa_ids=['QA-14'],seed='alpha-readable',items_per_type=1)['items'][0]
    gold={'authority_question_id':item['question_id'],'authority_truth':deepcopy(item['truth']),
        'canonical_answer':item['truth']['value']};saved=deepcopy(gold)
    answer,verification=readable_training_answer({'form':'open'},gold,{item['question_id']:item})
    assert gold==saved and answer==str(item['truth']['label'])
    assert verification['result']['score']==1 and verification['method']=='unchanged_v6_scorer'


def test_training_protocol_answer_retains_original_direction_canonical():
    from avengine.timeline.v7_alpha_delivery import readable_training_answer
    gold={'canonical_answer':'back-left'}
    answer,verification=readable_training_answer({'form':'open'},gold,{})
    assert answer=='back-left' and gold=={'canonical_answer':'back-left'}
    assert verification['method']=='original_v0.2_protocol_canonical'


def test_appearance_seed_is_independent_of_audio_and_keeps_physical_endpoint_slots():
    from avengine.timeline.v7_alpha_sampling import parent_appearance_assets
    assets=[{'asset_id':color,'endpoint_id':'voice'+str(i+1)} for i,color in enumerate(['blue','yellow','green','burgundy'])]
    saved=deepcopy(assets);row={'seeds':{'appearance':123,'audio':456}}
    first=parent_appearance_assets(assets,row)
    assert assets==saved and [a['endpoint_id'] for a in first]==[a['endpoint_id'] for a in assets]
    changed=deepcopy(row);changed['seeds']['audio']=999
    assert parent_appearance_assets(assets,changed)==first
    seen=set()
    for seed in range(20):
        row['seeds']['appearance']=seed
        seen.update(a['asset_id'] for a in parent_appearance_assets(assets,row)[:2])
    assert seen=={'blue','yellow','green','burgundy'}
