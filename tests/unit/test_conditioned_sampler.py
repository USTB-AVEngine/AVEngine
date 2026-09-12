from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from avengine.qa.answerability import (MeshHandle, line_of_sight, listener_azimuth_deg,
    max_concurrent_entities, separation_stats, structural_baselines)
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms.furniture_layout import clock_config
from avengine.rooms.walkable_space import RasterWalkableSpace, NativeRouteWalkableSpace
from avengine.routes.raster_pathfinder import RasterPathfinder


def registry():
    return {'assets':[{'asset_id':f'human_{i}','revision':'v1','entity_class':'articulated_human',
        'identity':{'species_id':'human'},'display_label':f'person {i}',
        'realized_attributes':{'sex_or_gender_label':'male','top_color':color},
        'timeline':{'idle_action_id':'idle','walking_action_id':'walk','walk_phase_period_frames':30,
                    'local_anatomical_forward_axis':[1.,0.,0.]},
        'default_emitter_anchor_id':'mouth','emitter_anchors':[{'anchor_id':'mouth','offset_m':[0.,1.6,0.],
              'offset_space':'final_scaled_asset_root'}]} for i,color in enumerate(['blue','green','red','yellow'])]}


def request(**extra):
    return {'episode_id':'fixture','seed':111,'sampling_policy':cs.POLICY,'source_asset_ids':['human_0','human_1'],
        'profile':{'anchor_count':1,'separation_bin_deg':[15,60],'speech_motion':'all_still','event_relation':'sequential',
                   'reserve_tail_s':1.,'retry_budget_within_profile':30},**extra}


def sounds():
    return [{'sound_asset_id':f'speech_{i}','sound_class':'speech','gender':'M','transcript':f'utterance {i}',
        'sample_count':32000,'sample_rate_hz':16000,'audible_start_sample':800,'audible_end_sample_exclusive':31200,
        'active_duration_s':1.9,'source_activity_intervals_samples':[[800,31200]],
        'path':f'/prepared/{i}.wav'} for i in range(4)]


def space():
    pf=RasterPathfinder(np.ones((32,32),dtype=bool),bounds_m=[[0,-1,0],[8,1,8]],floor_height_m=0.)
    return RasterWalkableSpace(pf,{'floor_height_m':0.,'resolution_m':.25,'authority':'fixture_retained_grid'})


def clock():
    return clock_config(frame_count=120,frame_rate_hz=15,sample_rate_hz=16000)


def test_fixed_profile_failure_keeps_denominator_and_histogram(monkeypatch):
    r=request();r['profile']['retry_budget_within_profile']=3;p=cs.resolve_condition_profile(r,registry());seen=[]
    def fail(*args,**kwargs):
        seen.append(deepcopy(args[2]));raise cs.CandidateFailure('routes','fixture_no_path')
    monkeypatch.setattr(cs,'sample_routes',fail)
    with pytest.raises(cs.ConditionedPlanningFailure) as error:
        cs.build_conditioned_plan(room={'room_id':'r'},request=r,source_registry=registry(),sounds=sounds(),space=space(),mesh=None,clock=clock(),condition_profile=p)
    assert error.value.result['attempts']==3
    assert error.value.result['failure_histogram']=={'routes:fixture_no_path':3}
    assert seen==[p,p,p]


def test_explicit_source_count_conflict_is_rejected():
    with pytest.raises(ValueError,match='agree with total_count'):
        cs.resolve_condition_profile(request(entities={'total_count':3}),registry())


def test_rigid_pair_not_excluded_by_unrequested_min_articulated_rule():
    reg=registry()
    for r in reg['assets']:r['entity_class']='rigid_object';r.pop('timeline')
    profile=cs.resolve_condition_profile(request(),reg)
    assert profile['source_classes']==['rigid_static_object']*2


def test_unknown_gender_never_pairs_biological_human():
    actor=cs.neutral_source_declaration(registry()['assets'][0],'source1')
    assert cs.sound_matches(actor,{'sound_class':'speech','gender':'M'})
    assert not cs.sound_matches(actor,{'sound_class':'speech','gender':'F'})
    assert not cs.sound_matches(actor,{'sound_class':'speech'})
    actor['entity_class']='rigid_object'
    assert cs.sound_matches(actor,{'sound_class':'speech','gender':'F'})


def test_half_open_legal_activity_start_ranges_include_sample_zero():
    c={'sample_rate_hz':10,'frame_rate_hz':2,'sample_count':30}
    s={'sample_count':10,'audible_start_sample':2,'audible_end_sample_exclusive':8}
    ranges=cs.legal_start_ranges([True,True,False,True,True,False],s,c,{'reserve_tail_s':0})
    assert ranges==[[0,2],[13,17]]


def test_sequential_uniform_starts_keep_feasible_suffix():
    events={'a':{'actor_id':'source1','audible_start_sample':0,'audible_end_sample_exclusive':10},
            'b':{'actor_id':'source2','audible_start_sample':0,'audible_end_sample_exclusive':10}}
    starts={'a':[[0,20]],'b':[[15,15]]};p={'event_relation':'sequential','min_gap_between_audible_windows_s':0}
    values=[cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p,np.random.default_rng(i)) for i in range(60)]
    assert {r['a'] for r in values}==set(range(6))
    assert all(r['b']==15 and r['a']+10<=r['b'] for r in values)


def test_overlap_uses_declared_duration_not_merely_order():
    events={'a':{'actor_id':'source1','audible_start_sample':1,'audible_end_sample_exclusive':8},
            'b':{'actor_id':'source2','audible_start_sample':2,'audible_end_sample_exclusive':7}}
    p={'event_relation':'overlap','minimum_overlap_s':3};starts={'a':[[0,20]],'b':[[2,30]]}
    for seed in range(30):
        r=cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p,np.random.default_rng(seed))
        assert min(r['a']+8,r['b']+7)-max(r['a']+1,r['b']+2)>=3
    p['minimum_overlap_s']=6
    assert cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p) is None


def test_repeat_keeps_original_playback_and_unique_event_times():
    events={'a':{'actor_id':'source1','audible_start_sample':0,'audible_end_sample_exclusive':3},
            'b':{'actor_id':'source2','audible_start_sample':0,'audible_end_sample_exclusive':3},
            'r':{'actor_id':'source1','audible_start_sample':0,'audible_end_sample_exclusive':3,'repeat_of':'a'}}
    p={'event_relation':'repeat','min_gap_between_audible_windows_s':1};starts={k:[[0,20]] for k in events}
    for seed in range(20):
        r=cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p,np.random.default_rng(seed))
        assert r['r']>=r['a']+4
        order=sorted(r,key=r.get)
        assert all(r[b]>=r[a]+4 for a,b in zip(order,order[1:]))


def test_native_routes_preserve_points_and_never_call_raster_solver():
    raster=space();bank=[{'route_id':'a','points_m':[[0,0,0],[0,0,1],[0,0,2]]},
                        {'route_id':'b','points_m':[[2,0,0],[2,0,1],[2,0,2]]}]
    native=NativeRouteWalkableSpace(raster.pathfinder,raster.metadata,bank,15.)
    paths,record=cs._native_routes(native,[True,True],5,15.,np.random.default_rng(3))
    assert set(record['selected_route_ids'])=={'a','b'}
    for route,rid in zip(paths,record['selected_route_ids']):
        original=np.asarray(next(b['points_m'] for b in bank if b['route_id']==rid))
        assert all(any(np.array_equal(p,x) for x in original) for p in route)
        assert any(np.array_equal(route[start:start+3],original) for start in range(3))
    with pytest.raises(ValueError,match='only permits retained'):
        native.shortest_path([0,0,0],[2,0,2])


def test_full_plan_same_seed_bytes_and_camera_membership(monkeypatch):
    kwargs={'room':{'room_id':'fixture'},'request':request(),'source_registry':registry(),'sounds':sounds(),
            'space':space(),'mesh':MeshHandle(np.zeros((0,3)),np.zeros((0,3),dtype=int)),'clock':clock()}
    a=cs.build_conditioned_plan(**kwargs);b=cs.build_conditioned_plan(**kwargs)
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    assert a['visual_plan']['camera']['candidate_id'] in a['planned_conditions']['legal_candidate_ids']
    assert a['condition_profile']['anchor_count']==1
    assert 'achieved_conditions' not in a
    assert all(f['camera_state']==a['visual_plan']['camera'] for f in a['visual_plan']['frames'])
    assert all('translation_ue_cm' not in s for f in a['visual_plan']['frames'] for s in f['actor_states'])
    p=dict(kwargs);p['request']=request(seed=112);c=cs.build_conditioned_plan(**p)
    assert a['audio_events']!=c['audio_events']
    assert a['visual_plan']['camera']!=c['visual_plan']['camera']
    from avengine.capture.qa_plan_adapters import materialize_ue_episode_plan
    from avengine.rooms import qa_episode
    monkeypatch.setattr(qa_episode, 'source_declaration',
                        lambda _registry, asset_id, actor_id: {'asset_id': asset_id, 'actor_id': actor_id})
    from avengine import runtime_profiles
    monkeypatch.setattr(runtime_profiles, 'resolve_source_asset_runtime_profile',
                        lambda reg, asset_id: next(row for row in reg['assets'] if row['asset_id']==asset_id))
    native = materialize_ue_episode_plan(a, registry())
    assert [f['camera_state']['frame_index'] for f in native['visual_plan']['frames']] == list(range(clock()['frame_count']))
    assert all('frame_index' not in f['camera_state'] for f in a['visual_plan']['frames'])
    assert native['visual_plan']['camera']['resolution_hw'] == a['visual_plan']['camera']['resolution_hw']


def test_azimuth_matches_catalog_horizontal_projection():
    from avengine.qa.unified_catalog import _listener_azimuth
    for yaw in np.linspace(-180,180,17):
        angle=np.deg2rad(yaw);basis={'forward':[np.sin(angle),.1,-np.cos(angle)],'right':[np.cos(angle),0.,np.sin(angle)],'up':[0.,1.,0.]}
        pose={'position_m':[1.,2.,3.],'basis':basis};listener={'positions_m':[[1.,2.,3.]],'basis_m3':[basis]}
        for p in [[2,1,4],[0,0,0],[-2,8,4]]:
            assert listener_azimuth_deg(pose,p)==_listener_azimuth(p,listener,0)


def test_concurrency_sweep_deduplicates_entities_and_touching_boundaries():
    assert max_concurrent_entities([(0,10,'a'),(1,2,'b'),(8,9,'c')])==2
    assert max_concurrent_entities([(0,3,'a'),(1,5,'a'),(5,6,'b')])==1


def test_separation_tracks_nearest_switch_and_sustained_seconds():
    r=separation_stats([0]*4,{'offscreen':[30,30,60,60],'visible':[50,50,20,20]},[0,4],frame_rate_hz=2,thresholds_deg=[25])
    assert r['min']==20 and r['max']==30 and r['nearest_competitor_changed']
    assert r['sustained_s_above']['25.0']==1.


def test_los_keeps_missing_geometry_unmeasured_and_exact_triangle_block():
    mesh=MeshHandle([[0,0,0],[0,2,0],[0,0,2]],[[0,1,2]])
    assert line_of_sight(None,[-1,.5,.5],[1,.5,.5])=='unmeasured'
    assert line_of_sight(mesh,[-1,.5,.5],[1,.5,.5])=='blocked'
    assert line_of_sight(mesh,[-1,3,3],[1,3,3])=='clear'


def test_structure_reports_majority_shortcut_without_refusing():
    r=structural_baselines({'a':'blue','b':'blue','c':'red'},'c')
    assert r['unique_minority_hits']==1 and r['majority_hits']==0 and r['random_hits']==pytest.approx(1/3)
    assert structural_baselines({'a':None,'b':1},'b')['status']=='unmeasured'


def test_clear_camera_rejects_occluded_body_even_with_clear_emitter(monkeypatch):
    r=request();profile=cs.resolve_condition_profile(r,registry());n=clock()['frame_count']
    actors=[cs.neutral_source_declaration(record,f'source{i+1}') for i,record in enumerate(registry()['assets'][:2])]
    paths=np.repeat(np.array([[[-1.,0.,-3.]],[[1.,0.,-3.]]]),n,axis=1)
    emitters=paths+np.array([0.,1.6,0.]);bodies=paths+np.array([0.,1.28,0.])
    selected={i:{**sounds()[i],'actor_id':actor['actor_id']} for i,actor in enumerate(actors)}
    monkeypatch.setattr(cs,'camera_grid',lambda *args,**kwargs:[[0.,1.55,0.]])
    def trace(_mesh,_origin,target):
        return 'clear' if target[1]>1.5 else 'blocked'
    monkeypatch.setattr(cs,'line_of_sight',trace)
    with pytest.raises(cs.CandidateFailure,match='no_joint_geometry_activity_schedule'):
        cs.select_camera_and_schedule(space(),object(),paths,np.zeros((2,n),dtype=bool),emitters,bodies,
                                      actors,selected,profile,clock(),r,np.random.default_rng(0))
    monkeypatch.setattr(cs,'line_of_sight',lambda *args:'clear')
    camera,events,conditions=cs.select_camera_and_schedule(space(),object(),paths,np.zeros((2,n),dtype=bool),
        emitters,bodies,actors,selected,profile,clock(),r,np.random.default_rng(0))
    assert camera['candidate_id'] in conditions['legal_candidate_ids']
    assert len(events)==2


class TwoFloorWalkableSpace:
    """Constructed two-floor nav mesh; cells exist at y=0 and y=3 only."""

    def __init__(self, floors=(0.0, 3.0), size=8.0, step=0.5):
        self.floors = [float(v) for v in floors]
        self.size = float(size)
        self.step = float(step)
        self.metadata = {
            "floor_height_m": self.floors[0],
            "authority": "fixture_two_floor_navmesh",
            "floor_heights_m": list(self.floors),
            "resolution_m": self.step,
        }
        xs = np.arange(self.step / 2, self.size, self.step)
        zs = np.arange(self.step / 2, self.size, self.step)
        pts = []
        for y in self.floors:
            for x in xs:
                for z in zs:
                    pts.append([float(x), float(y), float(z)])
        self._points = np.asarray(pts, dtype=float)

    def bounds(self):
        return np.array([[0.0, min(self.floors) - 0.5, 0.0],
                         [self.size, max(self.floors) + 0.5, self.size]], dtype=float)

    def route_bank(self):
        return None

    def is_navigable(self, point):
        p = np.asarray(point, dtype=float)
        if not (0.0 <= p[0] <= self.size and 0.0 <= p[2] <= self.size):
            return False
        return min(abs(p[1] - y) for y in self.floors) <= cs.SAME_FLOOR_Y_TOLERANCE_M

    def floor_height(self, point):
        p = np.asarray(point, dtype=float)
        return float(min(self.floors, key=lambda y: abs(p[1] - y)))

    def points(self, region=None):
        pts = self._points
        if region is not None:
            bounds = np.asarray(region, dtype=float)
            pts = pts[np.all((pts >= bounds[0]) & (pts <= bounds[1]), axis=1)]
        return pts

    def sample_navigable(self, rng, region=None):
        pts = self.points(region)
        if not len(pts):
            raise ValueError("requested region has no navigable cells")
        return pts[int(rng.integers(len(pts)))].copy()

    def shortest_path(self, start, end):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        if abs(start[1] - end[1]) > cs.SAME_FLOOR_Y_TOLERANCE_M:
            return None
        return np.stack([start, end])


def _actors():
    return [cs.neutral_source_declaration(record, f"source{i+1}")
            for i, record in enumerate(registry()["assets"][:2])]


def test_declared_floor_tokens_and_same_floor_lock():
    room = {"subrooms": [{"subroom_id": "R3_floor_0.1634"}, {"subroom_id": "R3_floor_3.1634"}]}
    assert cs.declared_floor_heights_m(room) == [0.1634, 3.1634]
    space = TwoFloorWalkableSpace()
    space.metadata.pop("floor_heights_m", None)
    seen = set()
    for seed in range(20):
        bounds, floor_y = cs.lock_same_floor_region(space, np.random.default_rng(seed), room=room)
        seen.add(round(floor_y, 4))
        assert abs(bounds[0, 1] - (floor_y - cs.SAME_FLOOR_Y_TOLERANCE_M)) < 1e-9
        assert abs(bounds[1, 1] - (floor_y + cs.SAME_FLOOR_Y_TOLERANCE_M)) < 1e-9
    assert seen == {0.1634, 3.1634}


def test_two_floor_navmesh_keeps_sources_within_0_3m():
    space = TwoFloorWalkableSpace()
    room = {"room_id": "two_floor", "subrooms": ["L_floor_0.0", "L_floor_3.0"]}
    profile = cs.resolve_condition_profile(request(), registry())
    actors = _actors()
    clk = clock()
    kept = 0
    floors_seen = set()
    for seed in range(40):
        rng = np.random.default_rng(seed)
        try:
            paths, _rot, _moving, _emit, _bodies, meta = cs.sample_routes(
                space, actors, profile, clk, rng, room=room)
        except cs.CandidateFailure:
            continue
        ys = np.asarray(paths)[:, :, 1]
        floor_y = float(meta["selected_floor_height_m"])
        assert floor_y in space.floors
        assert np.all(np.abs(ys - floor_y) <= cs.SAME_FLOOR_Y_TOLERANCE_M)
        assert float(np.max(ys) - np.min(ys)) <= cs.SAME_FLOOR_Y_TOLERANCE_M
        floors_seen.add(floor_y)
        kept += 1
        if kept >= 8:
            break
    assert kept >= 8


def test_cross_floor_pair_is_rejected():
    ok, _ = cs._points_same_floor([[0.0, 0.16, 0.0], [1.0, 2.02, 1.0]])
    assert ok is False
    ok, floor_y = cs._points_same_floor([[0.0, 0.16, 0.0], [1.0, 0.40, 1.0]])
    assert ok is True
    assert abs(floor_y - 0.28) < 1e-9 or abs(floor_y - 0.16) <= 0.3


def test_histogram_reports_achieved_5deg_bins_not_requested_box():
    report = cs.histogram_separation_5deg([60.5, 61.9, 89.0, 180.0])
    assert report["requested_bin_is_not_coverage"] is True
    assert report["bin_width_deg"] == 5
    by_lo = {row["lo_deg"]: row["count"] for row in report["bins"]}
    assert by_lo[60] == 2
    assert by_lo[85] == 1
    assert by_lo[175] == 1
    assert by_lo[90] == 0
    occupied = {row["lo_deg"] for row in report["occupied_bins"]}
    assert occupied == {60, 85, 175}


def test_clip_span_fit_policy_is_wired_and_rejects_unknown():
    r = request()
    r["sound_selection"] = {"clip_span_fit_policy": "not_a_policy"}
    profile = cs.resolve_condition_profile(r, registry())
    actors = _actors()
    with pytest.raises(ValueError, match="unsupported clip_span_fit_policy"):
        cs.select_sounds(actors, sounds(), profile, clock(), r, np.random.default_rng(0))
    r["sound_selection"] = {"clip_span_fit_policy": cs.CLIP_SPAN_FIT_POLICY, "max_clip_s": 5.0}
    selected = cs.select_sounds(actors, sounds(), profile, clock(), r, np.random.default_rng(0))
    assert len(selected) == 2


def test_speaker_moving_does_not_require_competitors_still():
    r = request()
    r["profile"]["speech_motion"] = "speaker_moving"
    profile = cs.resolve_condition_profile(r, registry())
    actors = _actors()
    flags = [cs._moving_flags(profile, actors, np.random.default_rng(seed)) for seed in range(40)]
    assert all(row[profile["anchor_indices"][0]] for row in flags)
    other = [i for i in range(len(actors)) if i not in profile["anchor_indices"]]
    assert other
    seen = {bool(row[other[0]]) for row in flags}
    assert seen == {False, True}


def test_off_screen_anchor_portrait_is_legal(monkeypatch):
    r = request()
    r["profile"].update(anchor_visibility="off_screen", competitor_visibility="in_fov",
                        separation_bin_deg=[90, 180], distance_range_m=[1.5, 6.0])
    profile = cs.resolve_condition_profile(r, registry())
    assert profile["anchor_visibility"] == "off_screen"
    n = clock()["frame_count"]
    actors = _actors()
    paths = np.repeat(np.array([[[0.0, 0.0, 2.8]], [[0.0, 0.0, -3.0]]]), n, axis=1)
    emitters = paths + np.array([0.0, 1.6, 0.0])
    bodies = paths + np.array([0.0, 1.28, 0.0])
    selected = {i: {**sounds()[i], "actor_id": actor["actor_id"]} for i, actor in enumerate(actors)}
    monkeypatch.setattr(cs, "camera_grid", lambda *args, **kwargs: [[0.0, 1.55, 0.0]])
    monkeypatch.setattr(cs, "line_of_sight", lambda *args: "clear")
    camera, events, conditions = cs.select_camera_and_schedule(
        space(), object(), paths, np.zeros((2, n), dtype=bool), emitters, bodies,
        actors, selected, profile, clock(), r, np.random.default_rng(0))
    assert conditions["anchor_visibility"] == "off_screen"
    assert camera["candidate_id"] in conditions["legal_candidate_ids"]
    assert len(events) == 2
    origin = np.asarray(camera["position_m"], dtype=float)
    forward = np.asarray(camera["basis"]["forward"], dtype=float)
    anchor_i = profile["anchor_indices"][0]
    depth = float(np.dot(bodies[anchor_i, 0] - origin, forward))
    assert depth <= 0.1


def test_off_screen_competitor_portrait_is_legal(monkeypatch):
    r = request()
    r["profile"].update(anchor_visibility="in_fov", competitor_visibility="off_screen",
                        separation_bin_deg=[90, 180], distance_range_m=[1.5, 6.0])
    profile = cs.resolve_condition_profile(r, registry())
    n = clock()["frame_count"]
    actors = _actors()
    paths = np.repeat(np.array([[[-0.2, 0.0, -3.0]], [[0.0, 0.0, 2.8]]]), n, axis=1)
    emitters = paths + np.array([0.0, 1.6, 0.0])
    bodies = paths + np.array([0.0, 1.28, 0.0])
    selected = {i: {**sounds()[i], "actor_id": actor["actor_id"]} for i, actor in enumerate(actors)}
    monkeypatch.setattr(cs, "camera_grid", lambda *args, **kwargs: [[0.0, 1.55, 0.0]])
    monkeypatch.setattr(cs, "line_of_sight", lambda *args: "clear")
    camera, events, conditions = cs.select_camera_and_schedule(
        space(), object(), paths, np.zeros((2, n), dtype=bool), emitters, bodies,
        actors, selected, profile, clock(), r, np.random.default_rng(0))
    assert conditions["competitor_visibility"] == "off_screen"
    assert len(events) == 2
    origin = np.asarray(camera["position_m"], dtype=float)
    forward = np.asarray(camera["basis"]["forward"], dtype=float)
    competitor = [i for i in range(2) if i not in profile["anchor_indices"]][0]
    depth = float(np.dot(bodies[competitor, 0] - origin, forward))
    assert depth <= 0.1


def test_two_floor_full_plan_stays_on_one_floor():
    room = {"room_id": "two_floor", "subrooms": ["L_floor_0.0", "L_floor_3.0"]}
    kwargs = {
        "room": room, "request": request(), "source_registry": registry(), "sounds": sounds(),
        "space": TwoFloorWalkableSpace(),
        "mesh": MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        "clock": clock(),
    }
    plan = cs.build_conditioned_plan(**kwargs)
    translations = [np.asarray(state["root_transform"]["translation_m"])
                    for frame in plan["visual_plan"]["frames"] for state in frame["actor_states"]]
    ys = np.asarray(translations)[:, 1]
    floor_y = plan["planned_conditions"]["selected_floor_height_m"]
    assert float(np.max(ys) - np.min(ys)) <= cs.SAME_FLOOR_Y_TOLERANCE_M
    assert np.all(np.abs(ys - floor_y) <= cs.SAME_FLOOR_Y_TOLERANCE_M)
    cam_y = float(plan["visual_plan"]["camera"]["position_m"][1])
    assert abs(cam_y - 1.55 - floor_y) <= cs.SAME_FLOOR_Y_TOLERANCE_M
    hist = plan["planned_conditions"]["planned_separation_histogram_5deg"]
    assert hist["requested_bin_is_not_coverage"] is True
    assert hist["count"] == 1


# --------------------------------------------------------------------- P05 solving

FULL_TIMELINE = {"idle_action_id": "idle", "walking_action_id": "walk",
                 "walk_phase_period_frames": 30, "body_plan_id": "biped_v1",
                 "template_id": "human_v1", "local_anatomical_forward_axis": [1., 0., 0.]}


def full_registry():
    """The fixture registry plus the Timeline fields locomotion capability reads."""
    reg = registry()
    for record in reg["assets"]:
        record["timeline"] = dict(FULL_TIMELINE)
    reg["assets"].append({
        "asset_id": "desk_phone_0", "revision": "v1", "entity_class": "rigid_object",
        "identity": {"object_type": "desk_telephone", "category": "appliance"},
        "display_label": "desk telephone", "realized_attributes": {},
        "default_emitter_anchor_id": "body",
        "emitter_anchors": [{"anchor_id": "body", "offset_m": [0., .75, 0.],
                             "offset_space": "final_scaled_asset_root"}]})
    return reg


def wide_space():
    pf = RasterPathfinder(np.ones((40, 40), dtype=bool), bounds_m=[[0, -1, 0], [10, 1, 10]],
                          floor_height_m=0.)
    return RasterWalkableSpace(pf, {"floor_height_m": 0., "resolution_m": .25,
                                    "authority": "fixture_retained_grid"})


def plan_for(request_body, sound_pool=None, frame_count=150, registry_value=None):
    from avengine.qa.answerability import MeshHandle
    return cs.build_conditioned_plan(
        room={"room_id": "fixture"}, request=request_body,
        source_registry=registry_value or full_registry(), sounds=sound_pool or sounds(),
        space=wide_space(), mesh=MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        clock=clock_config(frame_count=frame_count, frame_rate_hz=15, sample_rate_hz=16000))


def moving_by_actor(plan):
    result = {}
    for frame in plan["visual_plan"]["frames"]:
        for state in frame["actor_states"]:
            result.setdefault(state["actor_id"], []).append(bool(state["moving"]))
    return result


def anchor_window_frames(plan, actor_id):
    event = next(e for e in plan["audio_events"] if e["actor_id"] == actor_id)
    low, high = event["planned_audible_interval_samples"]
    fps, sr = plan["clock"]["frame_rate_hz"], plan["clock"]["sample_rate_hz"]
    return range(int(low * fps / sr), int(math.ceil(high * fps / sr)))


def test_declared_knobs_are_the_knobs_the_profile_actually_carries():
    from avengine.qa.generation_conditions import resolve_generator_capabilities

    declaration = cs.describe_generator_capabilities()
    profile = cs.resolve_condition_profile(request(), registry())
    assert set(declaration["knobs"]) == set(cs.DECLARED_SAMPLER_KNOBS)
    # Every advertised knob is a key the resolved profile really carries.
    assert set(cs.DECLARED_SAMPLER_KNOBS) <= set(profile)
    assert "competitor_motion" in declaration["knobs"]
    assert "visibility_transition" in declaration["knobs"]
    # A pixel occlusion state used to be unclaimed. It is claimed now, and the
    # claim is checked rather than trusted: conditioned_visibility turns the knob
    # into a VisibilityRequirement, and measure_sampler_capabilities honours it.
    assert "pixel_occlusion_transition" in declaration["knobs"]
    measured = resolve_generator_capabilities(cs)
    assert measured.supports("pixel_occlusion_transition")
    capabilities = resolve_generator_capabilities(cs)
    assert capabilities.supports("competitor_motion")
    assert capabilities.supports("visibility_transition")
    assert capabilities.supports("pixel_occlusion_transition")


def test_sound_matching_delegates_to_the_shared_semantics():
    actor = cs.neutral_source_declaration(registry()["assets"][0], "source1")
    verdict = cs.sound_match_verdict(actor, {"sound_class": "speech", "gender": "F"})
    assert verdict["decided_by"] == "avengine.dataset.source_capabilities.sound_compatibility"
    assert verdict["reason"] == "speech_gender_does_not_match_registered_appearance"
    assert cs.sound_matches(actor, {"sound_class": "speech", "gender": "M"})
    device = cs.neutral_source_declaration(full_registry()["assets"][-1], "source2")
    ring = {"sound_class": "telephone_bell_ringing", "compatible_asset_ids": ["desk_phone_0"]}
    # Without a declared mapping the legacy allowlist rule still decides.
    assert cs.sound_matches(device, ring)
    assert cs.sound_match_verdict(device, ring)["declared_mapping"] == cs.LEGACY_SOUND_CLASS_SHIM
    declared = {"object_sound_classes": {"desk_telephone": ["telephone_bell_ringing"]}}
    assert cs.sound_matches(device, ring, declared)
    speech_on_a_phone = {"sound_class": "speech", "gender": "F"}
    assert not cs.sound_matches(device, speech_on_a_phone, declared)
    assert cs.sound_match_verdict(device, speech_on_a_phone, declared)["reason"] == (
        "sound_class_is_not_declared_for_this_object_type")


def test_competitor_motion_is_stated_rather_than_drawn():
    actors = _actors()
    for stated, expected in (("still", {False}), ("moving", {True})):
        r = request()
        r["profile"].update(speech_motion="speaker_moving", competitor_motion=stated)
        profile = cs.resolve_condition_profile(r, registry())
        others = [i for i in range(len(actors)) if i not in profile["anchor_indices"]]
        flags = [cs._moving_flags(profile, actors, np.random.default_rng(seed)) for seed in range(25)]
        assert all(row[profile["anchor_indices"][0]] for row in flags)
        assert {bool(row[others[0]]) for row in flags} == expected
    contradiction = request()
    contradiction["profile"].update(speech_motion="competitor_moving", competitor_motion="still")
    with pytest.raises(ValueError, match="contradicts competitor_motion"):
        cs.resolve_condition_profile(contradiction, registry())


def test_two_instances_of_one_asset_stay_two_entities():
    plan = plan_for({"episode_id": "same_asset", "seed": 3, "sampling_policy": cs.POLICY,
                     "source_asset_ids": ["human_0", "human_0"],
                     "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.,
                                 "retry_budget_within_profile": 60}})
    actors = plan["visual_plan"]["actors"]
    assert [a["asset_id"] for a in actors] == ["human_0", "human_0"]
    instances = [a["entity_instance_id"] for a in actors]
    assert instances == ["human_0#instance01", "human_0#instance02"]
    # The slot names the backend endpoint convention depends on are unchanged.
    assert [a["actor_id"] for a in actors] == ["source1", "source2"]
    assert [a["source_endpoint_id"] for a in actors] == ["source1_mouth", "source2_mouth"]
    assert sorted(plan["instance_role_map"]) == instances
    assert {e["entity_instance_id"] for e in plan["audio_events"]} == set(instances)
    assert len({e["instance_event_id"] for e in plan["audio_events"]}) == len(plan["audio_events"])
    assert sorted({s["entity_instance_id"] for f in plan["visual_plan"]["frames"]
                   for s in f["actor_states"]}) == instances
    assert [row["entity_instance_id"] for row in plan["entity_instances"]] == instances


def test_target_and_competitor_can_answer_the_motion_question_differently():
    plan = plan_for({"episode_id": "opposite", "seed": 11, "sampling_policy": cs.POLICY,
                     "source_asset_ids": ["human_0", "human_1"],
                     "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.,
                                 "speech_motion": "all_still", "competitor_motion": "moving",
                                 "retry_budget_within_profile": 120}})
    actors = plan["visual_plan"]["actors"]
    anchor = actors[plan["condition_profile"]["anchor_indices"][0]]["actor_id"]
    competitor = next(a["actor_id"] for a in actors if a["actor_id"] != anchor)
    states = moving_by_actor(plan)
    window = anchor_window_frames(plan, anchor)
    assert not any(states[anchor][f] for f in window)
    assert all(states[competitor][f] for f in window)


def test_a_cropped_segment_is_not_capped_at_the_old_five_second_filter():
    long_segment = {"sound_asset_id": "segment_long", "sound_class": "speech", "gender": "M",
                    "transcript": "a cropped excerpt", "sample_count": 96000,
                    "sample_rate_hz": 16000, "audible_start_sample": 1600,
                    "audible_end_sample_exclusive": 94000, "active_duration_s": 5.02,
                    "source_activity_intervals_samples": [[1600, 40000], [52000, 94000]],
                    "path": "/prepared/segment_long.wav",
                    "source_origin": "/library/original_16min.wav",
                    "source_crop_start_sample": 1234567,
                    "source_crop_end_sample_exclusive": 1330567, "source_rate_hz": 48000,
                    "activity_interval_coordinates": "prepared_segment_samples"}
    partner = {**sounds()[1], "sample_count": 32000, "audible_start_sample": 400,
               "audible_end_sample_exclusive": 31600}
    body = {"episode_id": "cropped", "seed": 4, "sampling_policy": cs.POLICY,
            "source_asset_ids": ["human_0", "human_1"],
            "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 3.,
                        "retry_budget_within_profile": 60}}
    plan = plan_for(body, [long_segment, partner], frame_count=225)
    chosen = {e["sound_asset_id"]: e for e in plan["audio_events"]}
    assert "segment_long" in chosen
    assert chosen["segment_long"]["clip_length_bound_source"] == "episode_remaining_budget_after_reserved_tail"
    # The crop's provenance travels with the event rather than being summarised away.
    assert chosen["segment_long"]["source_crop_start_sample"] == 1234567
    assert chosen["segment_long"]["source_origin"] == "/library/original_16min.wav"
    assert chosen["segment_long"]["activity_interval_coordinates"] == "prepared_segment_samples"
    # A caller that states the old bound still gets it.
    capped = {**body, "episode_id": "capped", "sound_selection": {"max_clip_s": 5.}}
    with pytest.raises(cs.ConditionedPlanningFailure) as error:
        plan_for(capped, [long_segment], frame_count=225)
    assert "sounds:" in "".join(error.value.result["failure_histogram"])


def test_measured_activity_intervals_let_a_pause_fall_outside_the_legal_run():
    clock = {"sample_rate_hz": 10, "frame_rate_hz": 2, "sample_count": 60}
    paused = {"sample_count": 20, "audible_start_sample": 0, "audible_end_sample_exclusive": 20,
              "source_activity_intervals_samples": [[0, 5], [15, 20]]}
    spanned = {"sample_count": 20, "audible_start_sample": 0, "audible_end_sample_exclusive": 20}
    mask = [True, True, False, True, True, True, True, True, True, True, True, True]
    profile = {"reserve_tail_s": 0}
    assert cs.activity_intervals(paused)[1] == "measured_activity_intervals"
    assert cs.activity_intervals(spanned)[1] == "audible_span"
    # Start 0 is legal for the paused clip because only the pause covers the
    # illegal frame; the whole span version has to wait for the later run.
    assert cs.legal_start_ranges(mask, paused, clock, profile) == [[0, 5], [15, 40]]
    assert cs.legal_start_ranges(mask, spanned, clock, profile) == [[15, 40]]


def test_visibility_transition_needs_a_real_frustum_crossing():
    assert cs.visibility_transition_ok([False, False, True, True], "out_of_view_to_visible")
    assert not cs.visibility_transition_ok([True, True, True], "out_of_view_to_visible")
    assert not cs.visibility_transition_ok([True, True, False], "out_of_view_to_visible")
    assert cs.visibility_transition_ok([True, True, False], "visible_then_hidden")
    assert cs.visibility_transition_ok([True] * 4, "none")
    with pytest.raises(ValueError, match="unsupported visibility_transition"):
        cs.visibility_transition_ok([True, False], "teleport")


def test_static_sources_cannot_satisfy_a_requested_visibility_crossing(monkeypatch):
    r = request()
    r["profile"].update(visibility_transition="out_of_view_to_visible")
    profile = cs.resolve_condition_profile(r, registry())
    n = clock()["frame_count"]
    actors = _actors()
    paths = np.repeat(np.array([[[-1., 0., -3.]], [[1., 0., -3.]]]), n, axis=1)
    emitters = paths + np.array([0., 1.6, 0.])
    bodies = paths + np.array([0., 1.28, 0.])
    selected = {i: {**sounds()[i], "actor_id": actor["actor_id"],
                    "entity_instance_id": actor["entity_instance_id"],
                    "source_endpoint_id": actor["source_endpoint_id"]}
                for i, actor in enumerate(actors)}
    monkeypatch.setattr(cs, "camera_grid", lambda *a, **k: [[0., 1.55, 0.]])
    monkeypatch.setattr(cs, "line_of_sight", lambda *a: "clear")
    with pytest.raises(cs.CandidateFailure, match="no_joint_geometry_activity_schedule"):
        cs.select_camera_and_schedule(wide_space(), object(), paths, np.zeros((2, n), dtype=bool),
                                      emitters, bodies, actors, selected, profile, clock(), r,
                                      np.random.default_rng(0))


def test_post_sound_window_gate_refuses_and_keeps_every_failure_class():
    body = {"episode_id": "qa17", "seed": 21, "sampling_policy": cs.POLICY,
            "source_asset_ids": ["human_0", "human_1"],
            "qa_ids": ["QA-17"], "question_branches": {"QA-17": "yes"},
            "public_time_precision": 0,
            "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.,
                        "minimum_motion_s": 1.0,
                        "retry_budget_within_profile": 40}}
    plan = plan_for(body)
    window = plan["planned_conditions"]["planned_query_window"]
    assert window["requires_public_window"] is True
    low, high = window["public_query_window_s"]
    assert low == int(low) and high == int(high) and high > low
    anchor_row = next(row for row in window["windows"] if row["is_anchor_event"])
    assert anchor_row["public_query_window_s"] == [low, high]

    starved = {**body, "episode_id": "qa17_starved",
               "profile": {**body["profile"], "reserve_tail_s": 5.}}
    with pytest.raises(cs.ConditionedPlanningFailure) as error:
        plan_for(starved)
    result = error.value.result
    question_errors = result["errors_by_stage"]["question"]
    expected_codes = {
        "motion_solver_rejected:no_legal_query_frame_after_reserved_tail",
        "motion_solver_rejected:no_displayable_integer_query_window",
    }
    assert set(question_errors) <= expected_codes
    assert set(question_errors)
    for code, count in question_errors.items():
        assert count == result["failure_histogram"]["question:" + code]
    # A later stage's reason never overwrites an earlier real error.
    assert set(result["first_errors_by_stage"]) >= {"question"}
    assert len(result["failure_histogram"]) >= 1


def test_explicit_camera_and_walking_speed_survive_the_conditioned_route():
    body = {"episode_id": "explicit", "seed": 9, "sampling_policy": cs.POLICY,
            "source_asset_ids": ["human_0", "human_1"],
            "camera": {"fov_deg": 62., "height_above_floor_m": 1.35, "resolution_hw": [480, 640]},
            "motion": {"speed_range_mps": [1.2, 1.25]},
            "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.,
                        "speech_motion": "speaker_moving", "competitor_motion": "still",
                        "retry_budget_within_profile": 80}}
    plan = plan_for(body)
    camera = plan["visual_plan"]["camera"]
    assert camera["horizontal_fov_deg"] == 62.
    assert camera["height_above_floor_m"] == 1.35
    assert camera["resolution_hw"] == [480, 640]
    assert plan["condition_profile"]["walk_speed_range_mps"] == [1.2, 1.25]
    passthrough = plan["request_passthrough"]
    assert passthrough["camera"]["fov_deg"] == {"requested": 62., "applied": 62., "source": "request"}
    assert passthrough["motion"]["speed_range_mps"]["requested"] == [1.2, 1.25]
    speeds = [record["route_points_m"] for record in plan["activity_plan"]["actors"]
              if record["motion"] != "static"]
    assert speeds


def test_solve_entry_returns_the_plan_with_its_instance_map():
    from avengine.qa.answerability import MeshHandle

    result = cs.solve_conditioned_episode(
        request={"episode_id": "entry", "seed": 5, "sampling_policy": cs.POLICY,
                 "source_asset_ids": ["human_0", "human_1"], "frame_count": 150,
                 "frame_rate_hz": 15, "sample_rate_hz": 16000,
                 "qa_ids": ["QA-06"], "question_branches": {"QA-06": "still"},
                 "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.,
                             "retry_budget_within_profile": 60}},
        source_registry=full_registry(), sounds=sounds(), room={"room_id": "fixture"},
        space=wide_space(), mesh=MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)))
    assert result["plan"]["kind"] == "avengine_question_driven_episode"
    assert result["condition_profile"]["speech_motion"] == "all_still"
    assert result["question_conditions"]["status"] == "candidate"
    assert set(result["instance_role_map"]) == {row["entity_instance_id"]
                                                for row in result["entity_instances"]}
    assert result["plan"]["evidence_status"] == {"native_visual": "not_run",
                                                 "native_audio": "not_run", "qa_validity": "not_run"}
    assert result["plan"]["qualification_claim"] is False


def test_an_instance_may_be_named_anything_but_the_backend_slot_may_not():
    plan = plan_for({"episode_id": "named", "seed": 3, "sampling_policy": cs.POLICY,
                     "source_asset_ids": ["human_0", "human_1"],
                     "entities": {"instances": [
                         {"instance_id": "talker", "asset_id": "human_0", "speaking": True},
                         {"instance_id": "listener", "asset_id": "human_1", "speaking": True}]},
                     "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.,
                                 "retry_budget_within_profile": 60}})
    assert [a["entity_instance_id"] for a in plan["visual_plan"]["actors"]] == ["talker", "listener"]
    assert [a["actor_id"] for a in plan["visual_plan"]["actors"]] == ["source1", "source2"]
    assert [a["source_endpoint_id"] for a in plan["visual_plan"]["actors"]] == [
        "source1_mouth", "source2_mouth"]
    with pytest.raises(ValueError, match="source endpoint convention"):
        cs.instance_requests({"entities": {"instances": [
            {"instance_id": "talker", "source_slot_id": "kitchen"}]}})



# --------------------------------------------- a request that cannot be one Episode


REPOSITORY = Path(__file__).resolve().parents[2]


def shipped_registry():
    """The registry the ordinary entry uses; the fixture one has no animal or device."""
    from avengine.runtime_profiles import load_source_asset_runtime_registry
    return load_source_asset_runtime_registry(
        REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")


_REAL_HUMAN = "rocketbox_human_male_adult_01_top_blue_research_v1"
_REAL_DOG = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
_REAL_DEVICE = "generated_desk_telephone_corded_desk_unit_black_research_v1"


def _rows(*specs):
    return [{"instance_id": name, "asset_id": asset, "role": role, "speaking": True}
            for name, asset, role in specs]


_TWO_HUMANS = _rows(("human_target", _REAL_HUMAN, "anchor"),
                    ("dog_competitor", _REAL_DOG, "competitor"))
_HUMAN_AND_DEVICE = _rows(("human_target", _REAL_HUMAN, "anchor"),
                          ("device_competitor", _REAL_DEVICE, "competitor"))


def _conflict_request(qa_targets, *, instances, profile=None):
    request = {
        "seed": 7,
        "camera": {"motion": "static"},
        "entities": {"instances": instances, "total_count": len(instances)},
        "qa_ids": sorted({row["qa_id"] for row in qa_targets}),
        "qa_targets": [dict(row, target_source="config", items=1) for row in qa_targets],
    }
    if profile is not None:
        request["profile"] = profile
    return request


def test_a_stated_profile_that_contradicts_the_question_is_refused():
    """The Episode used to be planned for the recipe that cannot answer the question."""

    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "moving", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS, profile={"speech_motion": "all_still"})
    with pytest.raises(cs.ConditionedRequestConflict) as raised:
        cs.resolve_conditioned_request(request, shipped_registry(), generator=cs)
    assert {"speech_motion"} == {row["knob"] for row in raised.value.conflicts}
    message = str(raised.value)
    assert "all_still" in message and "speaker_moving" in message and "QA-06" in message


def test_a_stated_profile_that_agrees_with_the_question_is_kept():
    """Agreement is not a conflict, and the stated value still wins."""

    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "moving", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS, profile={"speech_motion": "speaker_moving"})
    profile, questions = cs.resolve_conditioned_request(request, shipped_registry(), generator=cs)
    assert profile["speech_motion"] == "speaker_moving"
    assert profile["knob_sources"]["speech_motion"] == "request_profile"
    assert questions["drives_sampler"] is True


def test_two_questions_that_need_opposite_competitors_are_refused():
    """QA-06 still wants a moving competitor; QA-17 yes wants a still one."""

    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "still", "target_instance_ids": ["human_target"]},
         {"qa_id": "QA-17", "branch": "yes", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    with pytest.raises(cs.ConditionedRequestConflict) as raised:
        cs.resolve_conditioned_request(request, shipped_registry(), generator=cs)
    assert {"competitor_motion"} == {row["knob"] for row in raised.value.conflicts}
    assert "QA-06:still" in str(raised.value) and "QA-17:yes" in str(raised.value)


def test_a_static_device_cannot_be_the_moving_competitor():
    """_moving_flags refused this only after routes had been drawn."""

    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "still", "target_instance_ids": ["human_target"]}],
        instances=_HUMAN_AND_DEVICE)
    with pytest.raises(cs.ConditionedRequestConflict) as raised:
        cs.resolve_conditioned_request(request, shipped_registry(), generator=cs)
    assert {"competitor_motion"} == {row["knob"] for row in raised.value.conflicts}
    assert "static objects" in str(raised.value)


def test_the_same_branch_with_an_articulated_competitor_is_accepted():
    """The negative control above is about the device, not about the branch."""

    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "still", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    profile, _ = cs.resolve_conditioned_request(request, shipped_registry(), generator=cs)
    assert profile["competitor_motion"] == "moving"
    assert profile["speech_motion"] == "all_still"


def test_qa06_still_now_states_a_competitor_that_answers_differently():
    """The still branch used to state nothing, so every candidate shared its answer."""

    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "still", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    _, questions = cs.resolve_conditioned_request(request, shipped_registry(), generator=cs)
    assert questions["sampler_profile"]["competitor_motion"] == "moving"

    moving = _conflict_request(
        [{"qa_id": "QA-06", "branch": "moving", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    _, other = cs.resolve_conditioned_request(moving, shipped_registry(), generator=cs)
    assert other["sampler_profile"]["competitor_motion"] == "still"


def test_a_preallocation_may_name_its_actors_by_instance():
    """prepare keys the map by instance_id while the actor carries the sourceN slot.

    Reading only the slot made every descriptively named request die here, which
    is what stopped the ordinary prepare/plan entry for this batch.
    """
    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "moving", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    request["sound_selection"] = {"preallocated_sound_asset_ids_by_actor": {
        "human_target": ["speech_0"], "dog_competitor": ["speech_1"]}}
    profile = cs.resolve_condition_profile(request, shipped_registry())
    actors = cs.instance_rows(profile, request)
    assert [row["actor_id"] for row in actors] == ["source1", "source2"]
    assert [row["entity_instance_id"] for row in actors] == ["human_target", "dog_competitor"]
    # The instance-named map is accepted: whatever happens next, it is no longer
    # rejected as naming an actor this request does not have.
    try:
        cs.select_sounds(actors, sounds(), profile, clock(), request,
                         np.random.default_rng(3))
    except Exception as error:  # a later stage may still refuse these fixture sounds
        assert "unknown actor" not in str(error)


def test_a_preallocation_naming_nobody_still_fails_and_says_who_exists():
    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "moving", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    request["sound_selection"] = {"preallocated_sound_asset_ids_by_actor": {
        "nobody_at_all": ["speech_0"]}}
    profile = cs.resolve_condition_profile(request, shipped_registry())
    actors = cs.instance_rows(profile, request)
    with pytest.raises(ValueError) as raised:
        cs.select_sounds(actors, sounds(), profile, clock(), request, np.random.default_rng(3))
    assert "nobody_at_all" in str(raised.value)
    assert "human_target" in str(raised.value)



# ------------------------------------------------- identity start-point pins


def pinned_bank():
    """Four retained routes whose endpoints are far enough apart to form a group."""
    return [
        {"route_id": "r00596", "points_m": [[-1.4833, 0.28, 0.1295],
                                            [-2.05, 0.28, -0.4],
                                            [-2.6, 0.28, -0.9]]},
        {"route_id": "r01752", "points_m": [[0.7308, 0.28, 1.2741],
                                            [0.15, 0.28, 0.73],
                                            [-0.4, 0.28, 0.2]]},
        {"route_id": "r09999", "points_m": [[2.0, 0.28, 2.5],
                                            [1.4, 0.28, 1.9],
                                            [0.8, 0.28, 1.3]]},
    ]


def pinned_space():
    raster = space()
    return NativeRouteWalkableSpace(raster.pathfinder, raster.metadata, pinned_bank(), 15.)


# The two starts the real v5 apartment pool publishes as fixed_starts_m.
V5_FIXED_STARTS = [[-1.4833, 0.28, 0.1295], [0.7308, 0.28, 1.2741]]


def test_pinned_static_positions_seat_each_actor_on_its_own_route():
    """A pin selects among real route endpoints instead of inventing a position."""

    paths, record = cs._native_routes(
        pinned_space(), [False, False], 5, 15., np.random.default_rng(3),
        pinned_static_positions_m=V5_FIXED_STARTS)

    assert record["selection"] == "uniform_over_legal_native_groups_at_pinned_static_positions"
    assert record["pinned_static_actor_indices"] == [0, 1]
    # Each actor stands exactly on the position the pool declared, for every frame.
    for index, wanted in enumerate(V5_FIXED_STARTS):
        assert np.allclose(paths[index], np.asarray(wanted), atol=1e-9)
    # And they came from two different retained routes.
    assert len(set(record["selected_route_ids"])) == 2


def test_a_pin_that_matches_no_retained_endpoint_is_refused():
    """The refusal names the pins, rather than quietly drawing a random endpoint."""

    with pytest.raises(cs.CandidateFailure) as raised:
        cs._native_routes(pinned_space(), [False, False], 5, 15., np.random.default_rng(3),
                          pinned_static_positions_m=[[9.9, 0.28, 9.9], V5_FIXED_STARTS[1]])
    assert raised.value.reason == "no_legal_native_route_group_at_pinned_static_positions"


def test_two_actors_cannot_be_pinned_to_one_route():
    """Both pins sit on r00596's own endpoints, so no group can seat them apart."""

    both_on_one_route = [[-1.4833, 0.28, 0.1295], [-2.6, 0.28, -0.9]]
    with pytest.raises(cs.CandidateFailure) as raised:
        cs._native_routes(pinned_space(), [False, False], 5, 15., np.random.default_rng(3),
                          pinned_static_positions_m=both_on_one_route)
    assert raised.value.reason == "no_legal_native_route_group_at_pinned_static_positions"


def test_a_pinned_actor_may_not_also_be_asked_to_walk():
    """A static initial position and a walking body are two different statements."""

    with pytest.raises(cs.CandidateFailure) as raised:
        cs._native_routes(pinned_space(), [True, False], 5, 15., np.random.default_rng(3),
                          pinned_static_positions_m=[V5_FIXED_STARTS[0], None])
    assert raised.value.reason == "pinned_static_position_requested_for_a_moving_actor"


def test_without_pins_the_old_native_behaviour_is_unchanged():
    """An unspecified pin must not change what the sampler already did."""

    plain = cs._native_routes(pinned_space(), [False, False], 5, 15.,
                              np.random.default_rng(11))
    explicit_none = cs._native_routes(pinned_space(), [False, False], 5, 15.,
                                      np.random.default_rng(11),
                                      pinned_static_positions_m=None)
    assert plain[1]["selection"] == "uniform_over_all_legal_native_groups"
    assert plain[1]["selected_route_ids"] == explicit_none[1]["selected_route_ids"]
    assert plain[1]["pinned_static_actor_indices"] == []
    for left, right in zip(plain[0], explicit_none[0]):
        assert np.array_equal(left, right)


def test_pins_are_aligned_by_the_instance_the_request_named():
    """The pin follows the declared instance, not the order the routes were drawn."""

    actors = [{"actor_id": "source1", "entity_instance_id": "human_target",
               "source_slot_id": "source1"},
              {"actor_id": "source2", "entity_instance_id": "dog_competitor",
               "source_slot_id": "source2"}]
    profile = {"pinned_static_positions_m": {"dog_competitor": V5_FIXED_STARTS[1]}}
    assert cs._pins_for_actors(profile, actors) == [None, V5_FIXED_STARTS[1]]

    by_slot = {"pinned_static_positions_m": {"source1": V5_FIXED_STARTS[0]}}
    assert cs._pins_for_actors(by_slot, actors) == [V5_FIXED_STARTS[0], None]

    assert cs._pins_for_actors({}, actors) is None


def test_a_pin_naming_an_absent_instance_is_refused():
    """A pin that quietly does nothing is worse than one that fails."""

    actors = [{"actor_id": "source1", "entity_instance_id": "human_target",
               "source_slot_id": "source1"}]
    with pytest.raises(ValueError) as raised:
        cs._pins_for_actors(
            {"pinned_static_positions_m": {"nobody": V5_FIXED_STARTS[0]}}, actors)
    assert "nobody" in str(raised.value) and "human_target" in str(raised.value)


def test_a_start_pin_survives_profile_resolution():
    """resolve_condition_profile builds a fixed key set, so an unlisted knob vanishes.

    The pin was dropped there in silence at first: the request stated it, the
    route sampler read profile['pinned_static_positions_m'], and the two never
    met because the resolved profile never carried the key.
    """
    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "moving", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    request["profile"] = {"pinned_static_positions_m": {"dog_competitor": V5_FIXED_STARTS[1]}}

    profile = cs.resolve_condition_profile(request, shipped_registry())
    assert profile["pinned_static_positions_m"] == {"dog_competitor": V5_FIXED_STARTS[1]}
    assert profile["knob_sources"]["pinned_static_positions_m"] == "request_profile"

    actors = cs.instance_rows(profile, request)
    assert cs._pins_for_actors(profile, actors) == [None, V5_FIXED_STARTS[1]]


def test_no_pin_resolves_to_no_constraint():
    request = _conflict_request(
        [{"qa_id": "QA-06", "branch": "moving", "target_instance_ids": ["human_target"]}],
        instances=_TWO_HUMANS)
    profile = cs.resolve_condition_profile(request, shipped_registry())
    assert profile["pinned_static_positions_m"] is None
    assert cs._pins_for_actors(profile, cs.instance_rows(profile, request)) is None


# ------------------------------- the interfaces added in C01-R2


def test_the_occlusion_vocabulary_is_the_solver_s_own():
    """Two lists drifted: the solver implemented five states and this table named two."""
    from avengine.rooms.conditioned_visibility import _TRANSITION_TO_REQUIREMENT

    cs._fill_pixel_occlusion_values()
    assert set(cs.SOLVER_KNOB_VALUES['pixel_occlusion_transition']) == set(
        _TRANSITION_TO_REQUIREMENT)
    # The three that used to be refused at profile resolution.
    for value in ('fully_occluded', 'registered_occluder_visible',
                  'visible_occluded_to_visible_clear'):
        assert value in cs.SOLVER_KNOB_VALUES['pixel_occlusion_transition']


def test_a_fully_occluded_request_reaches_a_real_pixel_requirement():
    """Accepting the string is not the same as building the requirement behind it."""
    from avengine.rooms import conditioned_visibility as cv
    from avengine.rooms.qa_episode import compile_question_conditions

    instances = [
        {'entity_instance_id': 'human_target', 'asset_id': _REAL_HUMAN,
         'source_class': 'articulated_human', 'role': 'anchor'},
        {'entity_instance_id': 'dog_competitor', 'asset_id': _REAL_DOG,
         'source_class': 'articulated_animal', 'role': 'competitor'},
    ]
    result = compile_question_conditions(
        ['QA-24'], instances,
        targets=[{'qa_id': 'QA-24', 'branch': 'fully_occluded',
                  'target_instance_ids': ['human_target'], 'items': 1,
                  'target_source': 'config'}],
        generator=cs, include_compiled=True)
    compiled = result.pop('_compiled_conditions', ())
    assert result['sampler_profile']['pixel_occlusion_transition'] == 'fully_occluded'

    actors = [{'entity_instance_id': 'human_target'}, {'entity_instance_id': 'dog_competitor'}]
    requirements = cs._visibility_requirements_for_selection(
        compiled, actors, {'anchor_indices': [0], 'public_time_precision': 0}, 150)
    states = {(r.kind, r.subject, r.state) for r in requirements}
    assert ('visibility_state', 'human_target', 'fully_occluded') in states
    # The dataclass refuses anything that is not a real pixel state, so a
    # requirement carrying this state cannot be a mislabelled capability word.
    with pytest.raises(cv.ConditionedVisibilityError):
        cv.VisibilityRequirement(kind='visibility_state', subject='human_target',
                                 state='available')


def test_the_earliest_speaker_is_scheduled_not_drawn():
    """A 'who spoke first' question cannot be served by a uniform draw over orders."""
    events = {'e1': {'entity_instance_id': 'dog_competitor', 'actor_id': 'source2'},
              'e2': {'entity_instance_id': 'human_target', 'actor_id': 'source1'}}
    orders = [(('e1', 'e2'), {'e1': 0, 'e2': 0}), (('e2', 'e1'), {'e1': 0, 'e2': 0})]

    kept = cs._orders_with_first_speaker(orders, events, 'human_target')
    assert [order for order, _ in kept] == [('e2', 'e1')]

    none_left = cs._orders_with_first_speaker(orders, events, 'nobody')
    assert none_left == []


def test_the_first_speaker_knob_is_compiled_and_carried():
    request = _conflict_request(
        [{'qa_id': 'QA-03', 'branch': None, 'target_instance_ids': ['human_target']}],
        instances=_TWO_HUMANS)
    profile, questions = cs.resolve_conditioned_request(
        request, shipped_registry(), generator=cs)
    assert profile['first_speaker_instance_id'] == 'human_target'
    assert profile['knob_sources']['first_speaker_instance_id'] == 'compiled_question_condition'
    assert profile['event_relation'] in {'sequential', 'repeat'}


def test_an_overlapping_schedule_cannot_promise_an_earliest_speaker():
    with pytest.raises(ValueError) as raised:
        cs.schedule_legal_events(
            {'e1': {'entity_instance_id': 'a', 'actor_id': 'source1',
                    'audible_start_sample': 0, 'audible_end_sample_exclusive': 10}},
            {'e1': [[0, 10]]}, {'sample_rate_hz': 16000},
            {'event_relation': 'overlap', 'minimum_overlap_s': 0.3,
             'first_speaker_instance_id': 'a'})
    assert 'requires event_relation=sequential' in str(raised.value)


def test_a_caller_supplied_profile_may_not_silently_drop_the_question():
    """This is how the CLI bypassed every compiled knob and still read as applied."""
    request = _conflict_request(
        [{'qa_id': 'QA-06', 'branch': 'moving', 'target_instance_ids': ['human_target']}],
        instances=_TWO_HUMANS)
    base = cs.resolve_condition_profile(request, shipped_registry())
    assert base['speech_motion'] == 'all_still'

    with pytest.raises(cs.ConditionedRequestConflict) as raised:
        cs.resolve_conditioned_request(request, shipped_registry(),
                                       condition_profile=base, generator=cs)
    assert {'speech_motion', 'competitor_motion'} <= {
        row['knob'] for row in raised.value.conflicts}


def test_a_caller_supplied_profile_that_agrees_is_reported_honestly():
    request = _conflict_request(
        [{'qa_id': 'QA-06', 'branch': 'moving', 'target_instance_ids': ['human_target']}],
        instances=_TWO_HUMANS)
    solved, _ = cs.resolve_conditioned_request(request, shipped_registry(), generator=cs)
    profile, questions = cs.resolve_conditioned_request(
        request, shipped_registry(), condition_profile=solved, generator=cs)
    assert questions['knob_application'] == (
        'caller_supplied_condition_profile_verified_against_compiled_knobs')
    assert 'speech_motion' in questions['caller_profile_agrees_with_knobs']
    assert questions['caller_profile_unverified_knobs'] == []


def test_a_preallocation_key_resolves_by_primary_identity_not_dictionary_order():
    """One actor's instance id equalling another's slot used to overwrite silently."""
    actors = [{'actor_id': 'source1', 'entity_instance_id': 'source2',
               'source_slot_id': 'source1'},
              {'actor_id': 'source2', 'entity_instance_id': 'talker',
               'source_slot_id': 'source2'}]
    resolved = cs._resolve_preallocation_keys({'source2': ['a'], 'talker': ['b']}, actors)
    # 'source2' is actor source1's declared instance, and the instance wins.
    assert resolved == {'source1': ['a'], 'source2': ['b']}


def test_two_preallocation_keys_claiming_one_actor_are_refused():
    actors = [{'actor_id': 'source1', 'entity_instance_id': 'talker',
               'source_slot_id': 'source1'},
              {'actor_id': 'source2', 'entity_instance_id': 'dog',
               'source_slot_id': 'source2'}]
    with pytest.raises(ValueError) as raised:
        cs._resolve_preallocation_keys({'talker': ['a'], 'source1': ['b']}, actors)
    assert 'ambiguously' in str(raised.value)


# ------------------------------- QA-25 AV: one event, visible then hidden


def _av_sound(sample_count=48000, a=0, b=None):
    b = sample_count if b is None else b
    return {'sound_asset_id': 'av_0', 'sample_rate_hz': 16000, 'sample_count': sample_count,
            'audible_start_sample': a, 'audible_end_sample_exclusive': b,
            'source_activity_intervals_samples': [[a, b]], 'active_duration_s': (b - a) / 16000}


def _av_clock():
    return clock_config(frame_count=150, frame_rate_hz=15, sample_rate_hz=16000)


def test_the_anchor_event_splits_into_its_visible_and_hidden_frames():
    fov = np.zeros(150, dtype=bool)
    fov[0:40] = True                       # visible for the first 40 frames
    sound = _av_sound(sample_count=48000)  # 3 s, so frames 0..44 at a zero start
    split = cs.anchor_event_subwindows(fov, sound, _av_clock(), 0)

    assert split['window_frames'][0] == 0
    assert min(split['visible_frames']) == 0
    assert min(split['hidden_frames']) == 40
    assert max(split['visible_frames']) < min(split['hidden_frames'])


def test_a_placement_needs_a_visible_anchor_a_hidden_tail_and_a_whole_second():
    clock = _av_clock()
    sound = _av_sound(sample_count=48000)

    good = np.zeros(150, dtype=bool); good[0:20] = True
    assert cs._visible_then_hidden_ok(good, sound, clock, 0) is True

    # Hidden first and visible afterwards is the wrong order for this question:
    # the emitter has to be identified before it disappears.
    reversed_series = np.zeros(150, dtype=bool); reversed_series[20:] = True
    assert cs._visible_then_hidden_ok(reversed_series, sound, clock, 0) is False

    # Visible throughout: nothing is hidden, so there is nothing to ask about.
    assert cs._visible_then_hidden_ok(np.ones(150, dtype=bool), sound, clock, 0) is False

    # A hidden tail too short to contain a whole second cannot be published.
    late = np.zeros(150, dtype=bool); late[0:44] = True
    assert cs._visible_then_hidden_ok(late, sound, clock, 0) is False


def test_visible_then_hidden_start_ranges_keep_only_placements_that_work():
    clock = _av_clock()
    sound = _av_sound(sample_count=48000)
    fov = np.zeros(150, dtype=bool); fov[0:30] = True
    kept = cs.visible_then_hidden_start_ranges([(0, 16000)], fov, sound, clock)
    assert kept, 'a legal placement exists and was dropped'
    for low, high in kept:
        assert cs._visible_then_hidden_ok(fov, sound, clock, low)

    never = np.ones(150, dtype=bool)
    assert cs.visible_then_hidden_start_ranges([(0, 16000)], never, sound, clock) == []


def test_qa25_av_compiles_the_event_shape_and_not_a_whole_clip_crossing():
    request = _conflict_request(
        [{'qa_id': 'QA-25', 'branch': 'AV', 'target_instance_ids': ['human_target']}],
        instances=_TWO_HUMANS)
    profile, questions = cs.resolve_conditioned_request(
        request, shipped_registry(), generator=cs)
    assert profile['anchor_visibility'] == 'visible_then_hidden'
    assert profile['knob_sources']['anchor_visibility'] == 'compiled_question_condition'
    # in_fov was the default, and it actively contradicted the hidden query frame.
    assert profile['visibility_transition'] == 'none'


def test_qa25_av_asks_about_a_moment_inside_its_own_sound():
    """It reads the wet tail, which used to make the solver call it post-sound."""
    from avengine.rooms import conditioned_motion as cm
    from avengine.rooms.qa_episode import compile_question_conditions

    instances = [
        {'entity_instance_id': 'human_target', 'asset_id': _REAL_HUMAN,
         'source_class': 'articulated_human', 'role': 'anchor'},
        {'entity_instance_id': 'dog_competitor', 'asset_id': _REAL_DOG,
         'source_class': 'articulated_animal', 'role': 'competitor'},
    ]
    result = compile_question_conditions(
        ['QA-25'], instances,
        targets=[{'qa_id': 'QA-25', 'branch': 'AV',
                  'target_instance_ids': ['human_target'], 'items': 1,
                  'target_source': 'config', 'forms': ['open']}],
        generator=cs, include_compiled=True)
    compiled = result.pop('_compiled_conditions')[0]
    meaning = cm.motion_semantics(compiled)
    assert meaning['semantics'] == 'during_audible_window'
    assert meaning['target_moves'] is True
    assert meaning['requires_measured_wet_tail'] is True


# ------------------------------- C05 contact correction at the canonical plan stage


def _contact_control(actor, control, support, root_above=0.001965,
                     evidence='native_executed_pose'):
    return {'actor_id': actor, 'asset_id': 'asset_' + actor, 'control': control,
            'contact_offset': {'evidence_kind': evidence, 'root_above_contact_m': root_above,
                               'measurement': 'measured'},
            'correction': {'support_level_height_m': support}}


def _contact_actors(*names):
    return [{'entity_instance_id': n, 'actor_id': n, 'source_slot_id': n,
             'asset_id': 'asset_' + n} for n in names]


def _contact_paths(count, y=9.0):
    return [np.tile(np.array([1.0, y, 2.0]), (5, 1)) for _ in range(count)]


def test_a_measured_support_moves_the_whole_trajectory_before_anything_derives_from_it():
    """C05's contract: a height applied to a finished plan leaves the rest stale."""
    controls = [_contact_control('human_target', 'native_measured_sole', 3.1030396264849385)]
    paths = _contact_paths(1)
    report = cs.apply_contact_correction(
        _contact_actors('human_target'), paths,
        {'contact_correction': {'controls': controls}})

    expected = 3.1030396264849385 + 0.001965
    assert np.allclose(paths[0][:, 1], expected), 'only some frames were corrected'
    assert np.allclose(paths[0][:, 0], 1.0) and np.allclose(paths[0][:, 2], 2.0), \
        'the correction moved the body sideways'
    assert report['stage'] == 'canonical_plan_before_rotation_emitter_and_body_derivation'
    row = report['applied'][0]
    assert row['corrected_root_height_m'] == pytest.approx(expected)
    assert row['root_delta_m'] == pytest.approx(expected - 9.0)


def test_a_body_without_a_measured_support_is_refused_not_dropped_by_a_constant():
    controls = [_contact_control('human_target', 'no_measured_visual_floor', 3.10)]
    paths = _contact_paths(1)
    with pytest.raises(cs.CandidateFailure) as raised:
        cs.apply_contact_correction(_contact_actors('human_target'), paths,
                                    {'contact_correction': {'controls': controls}})
    assert 'contact_correction_without_measured_support' in raised.value.reason
    assert np.allclose(paths[0][:, 1], 9.0), 'a refused body was moved anyway'


def test_an_unmeasured_evidence_kind_is_refused_even_under_the_right_control_word():
    controls = [_contact_control('human_target', 'native_measured_sole', 3.10,
                                 evidence='assumed_from_registry')]
    paths = _contact_paths(1)
    with pytest.raises(cs.CandidateFailure):
        cs.apply_contact_correction(_contact_actors('human_target'), paths,
                                    {'contact_correction': {'controls': controls}})
    assert np.allclose(paths[0][:, 1], 9.0)


def test_no_contact_correction_configured_changes_nothing():
    """The running chain must not move because this interface exists."""
    paths = _contact_paths(2)
    assert cs.apply_contact_correction(_contact_actors('a', 'b'), paths, {}) is None
    for path in paths:
        assert np.allclose(path[:, 1], 9.0)


def test_a_body_the_document_does_not_mention_is_left_alone():
    controls = [_contact_control('someone_else', 'native_measured_sole', 3.10)]
    paths = _contact_paths(1)
    report = cs.apply_contact_correction(_contact_actors('human_target'), paths,
                                         {'contact_correction': {'controls': controls}})
    assert report['applied'] == []
    assert np.allclose(paths[0][:, 1], 9.0)
