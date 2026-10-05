from copy import deepcopy
import numpy as np
import pytest
from avengine.timeline.v7_scene_quotas import add_quotas,quota_summary,speech_events,select_spatial_cells,scene_from_native_layout
from avengine.timeline.continuous_scene import activity_segments,window_segments,generate_v02_questions,raw_world_to_camera_sn3d
pytestmark=pytest.mark.fast_unit

def rows(n):
    return [{'schema':'v7-production-inputs-20261003-v3','scene_index':i,'seeds':{'camera':i+7},'scene_id':f'p{i}',
        'scene_seed':i,'house':'h','split':'train','duration_s':30,'window_start_ticks':[40,140],'window_count':2} for i in range(n)]

def test_small_full_prefix_and_complete_balanced_quotas():
    small=add_quotas(rows(400));full=add_quotas(rows(7000))
    assert full[:400]==small
    c=quota_summary(small)
    for field,size in [('requested_speakers',9),('requested_starss13',13),('requested_qa21_34',34)]:
        assert len(c[field])==size and max(c[field].values())-min(c[field].values())<=1
    assert c['requested_frustum']=={'in_frame':200,'out_of_frame':200}
    assert c['actual_generated_questions']==0
    assert c['requested_reverb']=={'long_qualification_pending':40,'medium':180,'short':180}

def test_zero_to_eight_count_is_identity_not_activity_count():
    for n in range(9):
        row=add_quotas(rows(n+1))[-1]
        layout={'status':'native_measured','listener_m':[0,1.3,0],'camera_forward':[0,0,-1],
            'speaker_anchors_m':[[.1*i,1.3,-2-i*.2] for i in range(n)]}
        specs=[{'actor_id':i,'class_id':1,'voice':'test'} for i in range(1,n+1)]
        scene=scene_from_native_layout(row,{'house':'h'},layout,specs)
        _,answers,_=generate_v02_questions(scene,4.,f'w{n}',{})
        speaker=[a['canonical_answer'] for a in answers if '__STARSS-SPEAKERS__' in a['question_id']]
        assert speaker==[str(n),str(n)]
        segments=window_segments(activity_segments(scene['events']),4.)
        assert len({e['source_id'] for e in segments})==n
        if n:assert any(e['carry_in'] for e in segments)
        if n>=2:assert len(segments)>6

def test_occlusion_does_not_change_geometric_frustum():
    candidates=[{'position_m':[x,1.3,z],'blocked':blocked,'native_LOS_measured':True}
                for x,z in [(0,-2),(0,2)] for blocked in [False,True]]
    cells=select_spatial_cells(candidates,[0,1.3,0],[0,0,-1])
    assert len(cells)==4
    with pytest.raises(ValueError,match='lacks four'):select_spatial_cells(candidates[:3],[0,1.3,0],[0,0,-1])
    candidates[0]['native_LOS_measured']=False
    with pytest.raises(ValueError,match='unmeasured'):select_spatial_cells(candidates,[0,1.3,0],[0,0,-1])

def test_camera_foa_rotation_changes_all_directional_channels_together():
    raw=np.array([[1.,2.],[.1,.2],[.3,.4],[.5,.6]])
    a=raw_world_to_camera_sn3d(raw,[0,0,-1]);b=raw_world_to_camera_sn3d(raw,[1,0,0])
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_allclose(np.square(a[1:]).sum(axis=0),np.square(b[1:]).sum(axis=0))


def test_music_tap_and_speaker_populations_do_not_encode_nuisance_strata():
    full=add_quotas(rows(7000))
    for cls in ('music_playback','water_tap_faucet'):
        selected=[r['scene_quotas'] for r in full if r['scene_quotas']['qa21_34_focus_class']==cls]
        assert len({q['noise_profile'] for q in selected})==4
        assert len({q['focus_frustum'] for q in selected})==2
    for n in range(9):
        selected=[r['scene_quotas'] for r in full if r['scene_quotas']['speakers']==n]
        assert len({q['reverb_stratum'] for q in selected})==3


def test_zero_speaker_recipe_never_requests_live_speech_focus():
    full=add_quotas(rows(7000))
    for row in full:
        q=row['scene_quotas']
        assert q['speakers']>0 or q['starss13_focus_class'] not in (0,1)
        assert set(q['spatial_member_to_focus_slot'].values())=={'in_clear','in_occluded','out_clear','out_occluded'}
    counts=quota_summary(full)['requested_starss13']
    assert max(counts.values())-min(counts.values())<=1
