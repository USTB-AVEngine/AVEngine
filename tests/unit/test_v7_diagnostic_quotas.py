import numpy as np
import pytest
from avengine.timeline.v7_scene_quotas import (
    add_quotas,add_diagnostic_quotas,diagnostic_summary,diagnostic_activity_events,pixel_center_to_boundary,
)
from avengine.timeline.continuous_scene import activity_segments,window_segments
pytestmark=pytest.mark.fast_unit


def rows(n):
    return [{'schema':'v7-production-inputs-20261003-v3','scene_index':i,'seeds':{'camera':i+7,'window':i+17},
        'scene_id':f'p{i}','scene_seed':i,'house':'h','split':'train','duration_s':30,
        'window_start_ticks':[40,140],'window_count':2} for i in range(n)]


def test_prefix_uniform_azimuth_and_multiple_worlds_per_activity_stratum():
    small=add_diagnostic_quotas(add_quotas(rows(400)));full=add_diagnostic_quotas(add_quotas(rows(7000)))
    assert full[:400]==small
    summary=diagnostic_summary(small)
    assert len(summary['requested_64_cell_counts'])==64
    assert len(summary['requested_vertical'])==7
    assert set(summary['requested_world_azimuth'].values())=={50}
    assert set(summary['requested_camera_relative_direction_probe'].values())=={50}
    assert abs(summary['requested_frustum']['in_frame']-summary['requested_frustum']['out_of_frame'])<=2
    assert set(map(int,summary['requested_activity_segments']))==set(range(21))|{24}
    assert min(summary['requested_activity_segments'].values())>=2
    assert summary['actual_generated_questions']==0
    assert all(r['scene_quotas']['diagnostic']['e_s_model_inputs']['ground_truth_injection'] is False for r in full)


def test_continuous_event_schedule_has_exact_counts_carry_in_and_100ms_stop():
    for speakers in range(9):
        sources={f'voice{i}':{'actor_id':i,'class_id':i%2} for i in range(1,speakers+1)}
        if not sources:sources={'environment':{'actor_id':9,'class_id':10}}
        for n in (0,1,8,20,24):
            if speakers and n<speakers+1:continue
            events=diagnostic_activity_events(sources,[4.,16.],n)
            segments=activity_segments(events)
            for start in (4.,16.):
                observed=window_segments(segments,start)
                assert len(observed)==n
                if n:assert observed[0]['carry_in']
            if n>=2:
                same=[e for e in events if e['endpoint_id']==events[0]['endpoint_id']]
                assert same[1]['start_tick']-same[0]['end_tick']==1
            if speakers:
                observed=window_segments(segments,4.)
                assert len({e['source_id'] for e in observed if e['class_id'] in (0,1)})==speakers


def test_half_open_pixel_boundary_includes_near_edge_samples_without_dead_zone():
    assert pixel_center_to_boundary(-.25,0,1280,720)==([.25,.5],True)
    assert pixel_center_to_boundary(1279.25,719.25,1280,720)==([1279.75,719.75],True)
    assert pixel_center_to_boundary(-.5,-.5,1280,720)==([0.,0.],True)
    assert pixel_center_to_boundary(1279.5,0,1280,720)[1] is False
    assert pixel_center_to_boundary(0,719.5,1280,720)[1] is False


def test_development_prefix_covers_all_cells_and_two_worlds_in_low_activity_strata():
    dev=add_diagnostic_quotas(add_quotas(rows(100)))
    summary=diagnostic_summary(dev)
    assert len(summary['requested_64_cell_counts'])==64
    assert len(summary['requested_vertical'])==7
    assert min(summary['requested_activity_segments'].values())>=2
    for row in dev:
        q=row['scene_quotas'];d=q['diagnostic']
        if q['speakers']==0:assert d['frame_has_person']
        if q['starss13_focus_class'] in (0,1) and q['focus_frustum']=='in_frame' and not q['focus_occluded']:
            assert d['frame_has_person']


def test_integer_onset_ties_hold_for_every_crop_offset_and_short_activity_world():
    from collections import Counter
    sources={'ambient':{'actor_id':9,'class_id':10}}
    for tick in range(40,57,2):
        for n in (4,5,8,20,24):
            events=diagnostic_activity_events(sources,[tick/10,(tick+120)/10],n)
            for start in (tick/10,(tick+120)/10):
                segments=window_segments(activity_segments(events),start)
                assert len(segments)==n
                starts=Counter(e['start_tick']//10 for e in events if start*10<=e['start_tick']<(start+10)*10)
                assert max(starts.values())>=2
