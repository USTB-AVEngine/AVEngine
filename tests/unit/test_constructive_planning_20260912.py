"""Constructive planning (2026-09-12).

Routes are grown to the length a question needs instead of being drawn between
two random points and refused when short; floors are drawn by navigable area;
a visibility question chooses its camera first and walks the subject through
the states it needs; an unresolved screen is refused before a render; and the
judges gain the two conditions the review found missing (entry depth, and a
negative reappearance presupposing the target was seen).
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.qa import unified_catalog as catalog
from avengine.qa.answerability import MeshHandle
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms import conditioned_visibility as cv
from avengine.rooms.furniture_layout import clock_config
from avengine.rooms.walkable_space import RasterWalkableSpace
from avengine.routes.raster_pathfinder import RasterPathfinder

REPOSITORY = Path(__file__).resolve().parents[2]


def registry():
    return {'assets': [{
        'asset_id': f'human_{i}', 'revision': 'v1', 'entity_class': 'articulated_human',
        'identity': {'species_id': 'human'}, 'display_label': f'person {i}',
        'realized_attributes': {'sex_or_gender_label': 'male', 'top_color': color},
        'timeline': {'idle_action_id': 'idle', 'walking_action_id': 'walk',
                     'walk_phase_period_frames': 30, 'body_plan_id': 'test_body',
                     'template_id': 'test_template',
                     'local_anatomical_forward_axis': [1., 0., 0.]},
        'default_emitter_anchor_id': 'mouth',
        'emitter_anchors': [{'anchor_id': 'mouth', 'offset_m': [0., 1.6, 0.],
                             'offset_space': 'final_scaled_asset_root'}],
    } for i, color in enumerate(['blue', 'green', 'red'])]}


def sounds():
    return [{'sound_asset_id': f'speech_{i}', 'sound_class': 'speech', 'gender': 'M',
             'transcript': f'utterance {i}', 'sample_count': 32000, 'sample_rate_hz': 16000,
             'audible_start_sample': 800, 'audible_end_sample_exclusive': 31200,
             'active_duration_s': 1.9, 'source_activity_intervals_samples': [[800, 31200]],
             'path': f'/prepared/{i}.wav'} for i in range(4)]


def space(size_m=10.0, cells=40):
    pathfinder = RasterPathfinder(np.ones((cells, cells), dtype=bool),
                                  bounds_m=[[0, -1, 0], [size_m, 1, size_m]], floor_height_m=0.)
    return RasterWalkableSpace(pathfinder, {'floor_height_m': 0., 'resolution_m': size_m / cells,
                                            'authority': 'fixture_grid'})


def clock():
    return clock_config(frame_count=150, frame_rate_hz=15, sample_rate_hz=16000)


def empty_mesh():
    return MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))


def wall_mesh(x0=3.0, x1=7.0, z=5.0, height=2.6):
    """A vertical wall across the middle of the fixture floor (+Y up)."""
    vertices = np.array([[x0, 0., z], [x1, 0., z], [x1, height, z], [x0, height, z]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return MeshHandle(vertices, triangles)


def plain_request(seed=111, ordinary=False, **profile):
    request = {'episode_id': 'fixture', 'seed': seed, 'sampling_policy': cs.POLICY,
               'source_asset_ids': ['human_0', 'human_1'],
               'profile': {'anchor_count': 1, 'event_relation': 'sequential', 'reserve_tail_s': 1.,
                           'retry_budget_within_profile': 40, **profile}}
    if ordinary:
        request['qa_sampling'] = {'acceptance_policy': {'question_mode': 'ordinary_observation'}}
    return request


def visibility_request(qa_id, branch, seed=7, camera_budget=160, **profile):
    return {
        'episode_id': f'cp_{qa_id}_{branch}', 'seed': seed, 'sampling_policy': cs.POLICY,
        'source_asset_ids': ['human_0', 'human_1'],
        'entities': {'total_count': 2, 'silent_count': 0, 'instances': [
            {'instance_id': 'human_target', 'asset_id': 'human_0',
             'source_class': 'articulated_human', 'role': 'target', 'speaking': True},
            {'instance_id': 'human_other', 'asset_id': 'human_1',
             'source_class': 'articulated_human', 'role': 'competitor', 'speaking': True}]},
        'qa_ids': [qa_id], 'question_branches': {qa_id: branch},
        'camera': {'fov_deg': 85, 'height_above_floor_m': 1.55, 'resolution_hw': [720, 1280],
                   'visibility_solver': {'body_geometry': 'emitter_proxy', 'max_candidates': 8,
                                         'max_ray_poses': 16, 'candidate_pool_budget': 64,
                                         'constructive_camera_budget': camera_budget}},
        'qa_sampling': {'acceptance_policy': {'question_mode': 'ordinary_observation'}},
        'profile': {'anchor_count': 1, 'separation_bin_deg': [15, 180], 'reserve_tail_s': 1.0,
                    'retry_budget_within_profile': 30, 'distance_range_m': [1.0, 6.0], **profile}}


def build(request, mesh):
    return cs.build_conditioned_plan(room={'room_id': 'fixture'}, request=request,
                                     source_registry=registry(), sounds=sounds(), space=space(),
                                     mesh=mesh, clock=clock())


def selected_screen(plan):
    solver = plan['planned_conditions']['visibility_solver']
    chosen = plan['visual_plan']['camera']['candidate_id']
    return next(c for c in solver['candidates'] if c['candidate_id'] == chosen)


# --------------------------------------------------------------------------- routes

def test_walk_grows_to_the_requested_length_and_stays_navigable():
    grid = space()
    poly = cs._walk_to_length(grid, np.array([5., 0., 5.]), grid.bounds().copy(),
                              np.random.default_rng(3), 0., 4.0)
    assert poly is not None
    assert cs._polyline_length(poly) >= 4.0 - 1e-6
    assert all(grid.is_navigable(point) for point in poly)


def test_random_route_draws_no_longer_die_on_short_paths():
    histogram = Counter()
    for seed in range(12):
        plan = build(plain_request(seed=seed, speech_motion='speaker_moving',
                                   separation_bin_deg=[15, 60]), empty_mesh())
        histogram.update(plan['planning_result']['failure_histogram'])
        walkers = [a for a in plan['activity_plan']['actors'] if a.get('motion') != 'static']
        assert walkers
        assert all(cs._polyline_length(a['route_points_m']) >= 1.5 for a in walkers)
    assert histogram.get('routes:path_too_short_for_moving_window', 0) == 0


class TwoFloorSpace:
    """A fake navigation with a big floor at y=0.16 and a landing at y=3.96."""

    metadata = {'authority': 'fixture', 'floor_height_m': 0.16}

    def __init__(self):
        big = np.array([[x, 0.16, z] for x in np.linspace(0.5, 9.5, 19)
                        for z in np.linspace(0.5, 9.5, 19)])
        small = np.array([[x, 3.96, z] for x in np.linspace(4.0, 5.0, 3)
                          for z in np.linspace(4.0, 5.0, 3)])
        self._points = np.concatenate([big, small])

    def points(self, region=None):
        points = self._points
        if region is not None:
            bounds = np.asarray(region, dtype=float)
            points = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
        return points

    def bounds(self):
        return np.array([[0., -1., 0.], [10., 5., 10.]])

    def sample_navigable(self, rng, region=None):
        points = self.points(region)
        if not len(points):
            raise ValueError('requested region has no navigable cells')
        return points[int(rng.integers(len(points)))].copy()


def test_floor_draw_is_weighted_by_navigable_area():
    fake = TwoFloorSpace()
    room = {'room_package': {'floor_heights_m': [0.16, 3.96]}}
    weights = cs._floor_navigable_weights(fake, [0.16, 3.96], np.random.default_rng(0))
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == 0.0
    chosen = {cs.lock_same_floor_region(fake, np.random.default_rng(seed), None, room)[1]
              for seed in range(40)}
    assert chosen == {0.16}


# --------------------------------------------------------------------------- ordinary defaults

def test_ordinary_defaults_drop_the_research_knobs_and_keep_stated_ones():
    ordinary = cs.resolve_condition_profile(plain_request(ordinary=True), registry())
    assert ordinary['competitor_visibility'] == 'any'
    assert ordinary['separation_bin_deg'] == [15.0, 180.0]
    assert 'ordinary_defaults' in ordinary
    research = cs.resolve_condition_profile(plain_request(), registry())
    assert research['competitor_visibility'] == 'in_fov'
    assert 'ordinary_defaults' not in research
    stated = cs.resolve_condition_profile(
        plain_request(ordinary=True, competitor_visibility='in_fov', separation_bin_deg=[15, 60]),
        registry())
    assert stated['competitor_visibility'] == 'in_fov'
    assert stated['separation_bin_deg'] == [15.0, 60.0]


def test_ordinary_requests_draw_a_walking_speaker_half_the_time():
    seen = Counter(cs.resolve_condition_profile(plain_request(seed=seed, ordinary=True),
                                                registry())['speech_motion']
                   for seed in range(40))
    assert set(seen) == {'all_still', 'speaker_moving'}
    assert 8 <= seen['speaker_moving'] <= 32
    still = cs.resolve_condition_profile(plain_request(seed=1), registry())
    assert still['speech_motion'] == 'all_still'


def test_any_visibility_is_a_legal_knob_that_plans():
    plan = build(plain_request(ordinary=True, competitor_visibility='any',
                               anchor_visibility='any', speech_motion='all_still'), empty_mesh())
    assert plan['condition_profile']['competitor_visibility'] == 'any'
    assert plan['planned_conditions']['competitor_visibility'] == 'any'


# --------------------------------------------------------------------------- screens and gates

def camera():
    return cv.CameraPose(candidate_id='grid_00000_yaw_000', position_m=(0., 1.55, 0.),
                         forward=(0., 0., -1.), right=(1., 0., 0.), up=(0., 1., 0.),
                         horizontal_fov_deg=85., resolution_hw=(720, 1280))


def entry_series(columns_after_entry, out_frames=10):
    frames = []
    for index in range(out_frames):
        frames.append({'frame_index': index, 'state': 'out_of_view', 'in_view': False, 'side': None,
                       'side_beyond_dead_zone': None, 'predicted_column_px': None, 'refutes': []})
    for offset, column in enumerate(columns_after_entry):
        frames.append({'frame_index': out_frames + offset, 'state': 'visible_clear', 'in_view': True,
                       'side': 'left', 'side_beyond_dead_zone': True,
                       'predicted_column_px': column, 'refutes': ['out_of_view']})
    return {'frames': frames, 'tier': 'frustum', 'camera': camera().as_report(),
            'policy': cv.screen_policy(None).as_report()}


def test_entry_screen_needs_the_body_inside_the_edge_not_a_sliver():
    view = cv._facts_view(states_by_instance={}, frame_count=150, frame_rate_hz=15.0, precision=0)
    requirement = cv.VisibilityRequirement(kind='out_of_view_to_visible', subject='s', side='left')
    sliver = cv._evaluate_requirement(requirement, series=entry_series([6.0] * 40),
                                      facts_view=view, frame_count=150)
    assert sliver['verdict'] == 'refuted'
    assert 'inside the entry edge' in sliver['reason']
    deep = cv._evaluate_requirement(requirement, series=entry_series([300.0] * 40),
                                    facts_view=view, frame_count=150)
    assert deep['verdict'] == 'consistent'
    assert deep['selected']['deep_sustain_frames'] == 40
    assert deep['selected']['entry_depth_px_required'] == pytest.approx(64.0)


def test_an_unresolved_screen_is_refused_before_a_render(monkeypatch):
    def unresolved(requirements, *, camera_poses, **kwargs):
        poses = [p if isinstance(p, cv.CameraPose) else cv.CameraPose.from_mapping(p)
                 for p in camera_poses]
        return {'status': 'screened', 'tier': 'frustum', 'camera_pose_count': len(poses),
                'stages': {}, 'budgets': {}, 'refuted': [],
                'candidates': [{'candidate_id': p.candidate_id, 'verdict': 'undetermined',
                                'requirement_screens': []} for p in poses]}
    monkeypatch.setattr(cv, 'solve_visibility_candidates', unresolved)
    request = visibility_request('QA-09', 'no')
    request['camera']['visibility_solver']['constructive'] = False
    with pytest.raises(cs.ConditionedPlanningFailure) as error:
        build(request, wall_mesh())
    assert 'camera:visibility_screen_has_no_consistent_candidate' in (
        error.value.result['failure_histogram'])
    request['camera']['visibility_solver']['allow_unresolved_candidates'] = True
    plan = build(request, wall_mesh())
    assert plan['planned_conditions']['visibility_solver']['selection_preference'] == 'screen_unresolved'


# --------------------------------------------------------------------------- camera-first construction

def test_camera_first_construction_walks_the_target_behind_the_wall_and_out_again():
    plan = build(visibility_request('QA-09', 'yes'), wall_mesh())
    assert plan['activity_plan']['motion_construction'] == 'constructive_visibility_camera_first'
    chosen = selected_screen(plan)
    assert chosen['verdict'] == 'consistent'
    assert chosen['predicted_state_counts']['human_target']['fully_occluded'] > 0
    screen = chosen['requirement_screens'][0]
    assert screen['kind'] == 'fully_occluded_then_visible'
    assert screen['returning_runs']
    walker = next(a for a in plan['activity_plan']['actors'] if a.get('motion') != 'static')
    assert walker['motion'] == 'constructive_visibility_walk'
    assert plan['visual_plan']['camera']['candidate_id'] == (
        plan['activity_plan']['visibility_construction']['camera_candidate_id'])


def test_camera_first_construction_hides_the_target_until_the_end_for_no():
    plan = build(visibility_request('QA-09', 'no'), wall_mesh())
    chosen = selected_screen(plan)
    assert chosen['verdict'] == 'consistent'
    screen = chosen['requirement_screens'][0]
    assert screen['kind'] == 'fully_occluded_without_return'
    assert screen['terminal_runs'] and screen['terminal_runs'][0]['end'] == 150
    # The target speaks while it is still visible; the audible window lies before the hide.
    hidden_from = min(run['start'] for run in screen['terminal_runs'])
    target_events = [e for e in plan['audio_events'] if e['entity_instance_id'] == 'human_target']
    assert target_events
    assert all(e['planned_audible_interval_samples'][1] / 16000 * 15 <= hidden_from + 1 for e in target_events)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_camera_first_construction_enters_from_the_requested_side(side):
    plan = build(visibility_request('QA-07', side, seed=11), wall_mesh())
    chosen = selected_screen(plan)
    assert chosen['verdict'] == 'consistent'
    selected = chosen['requirement_screens'][0]['selected']
    assert selected['side'] == side
    assert selected['deep_sustain_frames'] >= 2
    construction = plan['activity_plan']['visibility_construction']
    assert construction['kind'] == 'out_of_view_to_visible'


def test_construction_is_opt_out_and_the_legacy_draw_still_runs():
    request = visibility_request('QA-07', 'left', seed=11)
    request['camera']['visibility_solver']['constructive'] = False
    request['camera']['visibility_solver']['allow_unresolved_candidates'] = True
    try:
        plan = build(request, wall_mesh())
    except cs.ConditionedPlanningFailure as error:
        histogram = error.value.result['failure_histogram'] if hasattr(error, 'value') else error.result['failure_histogram']
        assert histogram
        return
    assert plan['activity_plan']['motion_construction'] == 'legacy_random_route_then_solver'


# --------------------------------------------------------------------------- judges

def judge_reappearance(states, allow=False):
    series = [{'frame_index': i, 'state': s} for i, s in enumerate(states)]
    policy = {'accept_observed_branches': {'QA-09': ['yes', 'no']}} if allow else {}
    facts = {'time': {'frame_count': len(states), 'frame_rate_hz': 15},
             'visibility': {'source1': {r['frame_index']: r for r in series}},
             'sampling': {'acceptance_policy': policy}}
    requirement = cv.VisibilityRequirement(kind='fully_occluded_without_return', subject='target',
                                           qa_id='QA-09', require_complete_coverage=False)
    return cv._judge_requirement(requirement, instance='source1', series=series, facts_view=facts,
                                 frame_count=len(states), occluder_registry=None,
                                 observed=sorted(set(states)))


def test_a_negative_reappearance_needs_the_target_seen_before_it_hides():
    row = judge_reappearance(['fully_occluded'] * 5)
    assert row['status'] == 'fail'
    assert 'never visible before' in row['reason']
    row = judge_reappearance(['visible_clear', 'fully_occluded', 'fully_occluded'])
    assert row['status'] == 'pass'
    assert row['measured']['visible_before_full_occlusion'] == [0]


def test_the_question_generator_skips_a_hidden_target_nobody_saw():
    frames = 30
    rows = {i: {'frame_index': i, 'state': 'fully_occluded', 'target_pixels': 0,
                'visible_pixels': 0} for i in range(frames)}
    facts = {
        'time': {'frame_count': frames, 'frame_rate_hz': 15},
        'visibility': {'source1': rows},
        'visibility_meta': {'resolution_hw': [720, 1280]},
        'actors': {'source1': {'appearance': {'value': 'blue', 'label': 'blue-shirt person',
                                              'field': 'top_color'}}},
        'appearance_review': {'source1': {'status': 'pass', 'value': 'blue', 'frame_refs': [0]}},
        'events': [], 'audio': {},
    }
    with pytest.raises(catalog._Deferred) as error:
        catalog._generate_qa_09(facts, 'seed')
    assert error.value.code == 'target_never_visible_before_occlusion'
    assert catalog._p8_reappearance_candidates(facts) == []


def test_policy_numbers_are_read_from_the_acceptance_policy():
    facts = {'sampling': {'acceptance_policy': {'qa15_reversal_tolerance_m': 0.15}}}
    assert catalog._policy_number(facts, ('qa15_reversal_tolerance_m',), default=0.05,
                                  name='reversal tolerance') == 0.15
    assert catalog._policy_number({'sampling': {}}, ('qa15_reversal_tolerance_m',), default=0.05,
                                  name='reversal tolerance') == 0.05


# --------------------------------------------------------------------------- backfill and bank

def load_tool(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_candidate_variation_alternates_branches_and_rotates_rooms():
    module = load_tool('sparse_backfill_cp', 'tools/dataset/backfill_sparse_qa.py')
    request = {'qa_targets': [{'qa_id': 'QA-07', 'branch': 'left'}], 'room_id': 'a'}
    config = {'alternate_branches': True, 'room_ids': ['a', 'b', 'c']}
    assert module.apply_candidate_variation(deepcopy(request), 4, config)['qa_targets'][0]['branch'] == 'left'
    varied = module.apply_candidate_variation(deepcopy(request), 5, config)
    assert varied['qa_targets'][0]['branch'] == 'right'
    assert varied['room_id'] == 'c'
    assert module.apply_candidate_variation(deepcopy(request), 5, {})['qa_targets'][0]['branch'] == 'left'


def test_observed_branch_recovery_renders_audio_from_the_kept_capture(tmp_path):
    module = load_tool('sparse_backfill_cp2', 'tools/dataset/backfill_sparse_qa.py')
    wave = tmp_path / 'waves/0001'
    episode = 'ordinary_x_0001'
    attempt = wave / 'run/work' / episode / 'capture/attempt_01/episode'
    (attempt / 'capture').mkdir(parents=True)
    (attempt / 'capture/research_receipt.json').write_text('{}')
    module.write(wave / 'run/state.json', {'scopes': [{'scope_key': episode, 'results': [{
        'stage': 'capture', 'status': 'fail', 'work_item_id': f'{episode}:capture:01',
        'reason': 'RequestedVisibilityError: the measured reappearance answer is yes; this branch needs no'}]}]})
    options = {episode: {'preplanned_episode_root': '/x'}}
    reopened = module.observed_branch_recovery(wave, options, {'accept_observed_branches': {'QA-09': ['yes', 'no']}})
    assert reopened == [episode + '/capture']
    assert options[episode]['render_audio_from_retained_capture'] is True
    assert 'preplanned_episode_root' not in options[episode]


def test_screen_verdict_reads_the_selected_camera():
    module = load_tool('sparse_backfill_cp3', 'tools/dataset/backfill_sparse_qa.py')
    plan = {'visual_plan': {'camera': {'candidate_id': 'grid_00001_yaw_015'}},
            'camera_condition_sampling': {'visibility_solver': {'candidates': [
                {'candidate_id': 'grid_00000_yaw_000', 'verdict': 'consistent'},
                {'candidate_id': 'grid_00001_yaw_015', 'verdict': 'undetermined'}]}}}
    assert module.screen_verdict(plan) == 'undetermined'
    assert module.screen_verdict({}) is None


def test_bank_caps_questions_per_scene_and_type():
    module = load_tool('retained_bank_cp', 'tools/dataset/generate_retained_qa_bank.py')
    def candidate(scene, qa, index):
        return {'item': {'qa_id': qa, 'question_id': f'{scene}_{qa}_{index}'},
                'source': {'facts_path': scene, 'room_family': 'hm3d', 'asset_ids': [], 'sound_asset_ids': [], 'world_id': scene},
                'media': {}}
    candidates = [candidate('scene_a', 'QA-18', i) for i in range(6)] + [candidate('scene_b', 'QA-18', i) for i in range(2)]
    selected = module.select_balanced(candidates, 10, 0, max_per_scene_type=2)
    per_scene = Counter(c['source']['facts_path'] for c in selected)
    assert per_scene == {'scene_a': 2, 'scene_b': 2}
    assert len(module.select_balanced(candidates, 10, 0)) == 8


def test_a_missed_target_keeps_the_scene_when_the_policy_says_so(tmp_path):
    fixture = load_tool('requested_visibility_fixture', 'tests/unit/test_requested_visibility_before_audio.py')
    from avengine.dataset import binding_group_native as native
    plan, request, capture = fixture.case(tmp_path, enters=False)
    with pytest.raises(native.RequestedVisibilityError):
        native.check_requested_visibility(plan, request, capture)
    request['qa_sampling'] = {'acceptance_policy': {'keep_scene_when_target_unmet': True}}
    report_path = tmp_path / 'kept.json'
    report = native.check_requested_visibility(plan, request, capture, report_path=report_path)
    assert report['status'] == 'fail'
    assert report['requested_target_unmet'] is True
    assert json.loads(report_path.read_text())['salvage'] == 'scene_kept_for_other_question_types'


# --------------------------------------------------------------------------- route-bank rooms

def bank_space(routes):
    from avengine.rooms.walkable_space import NativeRouteWalkableSpace
    base = space()
    return NativeRouteWalkableSpace(base.pathfinder, {**base.metadata, 'route_authority': 'fixture_bank'},
                                    routes, 15.0)


def straight_route(route_id, start, end, frames=150):
    points = np.linspace(np.asarray(start, dtype=float), np.asarray(end, dtype=float), frames)
    return {'route_id': route_id, 'points_m': points.tolist()}


def bank_routes():
    """Whole-clock walks north of the fixture wall (x 3-7, z 5), 0.6-0.9 m/s."""
    return [straight_route('r_cross', [0.5, 0., 7.5], [9.5, 0., 7.5]),
            straight_route('r_end_hidden', [0.5, 0., 7.5], [6.5, 0., 7.5]),
            straight_route('r_far', [0.5, 0., 9.0], [9.5, 0., 9.0]),
            straight_route('r_back', [9.5, 0., 8.0], [0.5, 0., 8.0]),
            straight_route('r_near_a', [0.5, 0., 6.0], [8.5, 0., 6.0]),
            straight_route('r_near_b', [1.0, 0., 6.0], [8.0, 0., 6.0]),
            straight_route('r_south', [2.0, 0., 3.0], [8.0, 0., 3.0])]


def build_bank(request, mesh, routes):
    return cs.build_conditioned_plan(room={'room_id': 'fixture_bank'}, request=request,
                                     source_registry=registry(), sounds=sounds(),
                                     space=bank_space(routes), mesh=mesh, clock=clock())


@pytest.mark.parametrize('branch', ['yes', 'no'])
def test_a_route_bank_room_takes_the_same_camera_first_construction(branch):
    # The 10 m fixture floor holds whole-clock routes only up to about 8 m from a
    # camera, so the fixture range is wider than the production 1-6 m default.
    request = visibility_request('QA-09', branch, seed=5, speech_motion='speaker_moving',
                                 distance_range_m=[1.0, 8.0], retry_budget_within_profile=12)
    request['camera']['visibility_solver']['constructive_camera_budget'] = 240
    # A 2 m wall: the returning branch needs the shadow to end well inside the frustum.
    plan = build_bank(request, wall_mesh(4.0, 6.0), bank_routes())
    assert plan['activity_plan']['motion_construction'] == 'constructive_visibility_camera_first'
    construction = plan['activity_plan']['visibility_construction']
    assert construction['route_source'] == 'retained_native_route_bank'
    chosen = selected_screen(plan)
    assert chosen['verdict'] == 'consistent'
    assert chosen['predicted_state_counts']['human_target']['fully_occluded'] > 0
    walker = next(a for a in plan['activity_plan']['actors'] if a.get('motion') != 'static')
    assert walker['native_route_id'] in {r['route_id'] for r in bank_routes()}
    assert walker['end_frame_exclusive'] - walker['start_frame'] == 150  # the retained route is kept whole


@pytest.mark.parametrize('side', ['left', 'right'])
def test_a_route_bank_room_enters_from_the_requested_side(side):
    request = visibility_request('QA-07', side, seed=3, speech_motion='speaker_moving')
    plan = build_bank(request, wall_mesh(), bank_routes())
    chosen = selected_screen(plan)
    assert chosen['verdict'] == 'consistent'
    assert chosen['requirement_screens'][0]['selected']['side'] == side
    assert plan['activity_plan']['visibility_construction']['route_source'] == 'retained_native_route_bank'

