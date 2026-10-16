"""QA21 conditions tested against the original authority and real projection math."""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from avengine.construction.joint.families.qa21 import (
    REVISION, target_options, emission_mechanism, native_geometry_candidate_check as native_candidate_check,
    certify_qa21_observation, MINIMUM_TARGET_PIXELS)
from avengine.qa.unified_catalog import normalize_episode_bundle
from test_qa_unified_catalog import _fixture

pytestmark = pytest.mark.fast_unit


def animal(aid, species, cls):
    return {'asset_id': aid, 'kind': 'animal', 'species_id': species,
        'allowed_classes': [cls], 'attachment_surface': 'floor', 'silent_decor': False,
        'label_color': 'brown' if species == 'cat' else 'black', 'standing_registered_qualified': True}


def test_choose_requested_actual_class_before_geometry_and_keep_inputs_unchanged():
    cat = animal('cat', 'cat', 'cat_meow'); dog = animal('dog', 'dog', 'dog_bark')
    catalog = [cat, dog]; original = deepcopy(catalog)
    room = {'family': 'hm3d', 'categories': ['floor_cat', 'floor_dog']}
    records = [{'sound_class': 'cat_meow'}, {'sound_class': 'dog_bark'}]
    options = target_options(room, catalog, records, {}, {'qa21_positive_counts': {'cat_meow': 5}})
    assert options[0][0]['asset_id'] == 'dog'
    assert options[0][1] == 'dog_bark'
    requested = target_options(room, catalog, records, {}, {}, requested_class='cat_meow')
    assert len(requested) == 1 and requested[0][1] == 'cat_meow'
    assert catalog == original


def test_no_unadmitted_class_no_illegal_category_no_wrong_species():
    cat = animal('cat', 'cat', 'cat_meow')
    room = {'family': 'hm3d', 'categories': ['floor_dog']}
    assert target_options(room, [cat], [{'sound_class': 'cat_meow'}], {}, {}) == []
    room['categories'] = ['floor_cat']
    assert target_options(room, [cat], [{'sound_class': 'dog_bark'}], {}, {}) == []
    bad = animal('wrong', 'cat', 'dog_bark')
    assert target_options(room, [bad], [{'sound_class': 'dog_bark'}], {}, {}) == []
    excluded = {'policy': {'temporarily_excluded_asset_ids': {'cat': 'not ready'}}}
    assert target_options(room, [cat], [{'sound_class': 'cat_meow'}], excluded, {}) == []


def test_actual_emission_mechanism_cannot_be_assigned_to_silent_decor_or_person():
    assert emission_mechanism(animal('cat', 'cat', 'cat_meow'), 'cat_meow') == 'animal_live_recorded'
    speaker = {'kind': 'object', 'entity_class': 'loudspeaker', 'allowed_classes': ['music_playback', 'cat_meow']}
    assert emission_mechanism(speaker, 'music_playback') == 'loudspeaker_playback'
    with pytest.raises(ValueError): emission_mechanism(speaker, 'cat_meow')
    with pytest.raises(ValueError): emission_mechanism({'kind': 'human', 'allowed_classes': ['cat_meow']}, 'cat_meow')


def test_original_qa21_selects_expected_target_and_class():
    facts = normalize_episode_bundle(_fixture())
    assert certify_qa21_observation(facts, 'a0', 'bark')['status'] == 'pass'
    assert certify_qa21_observation(facts, 'a1', 'bark')['status'] == 'fail'


def test_original_qa21_rejects_unreviewed_duplicate_and_multiclass_appearance():
    facts = normalize_episode_bundle(_fixture())
    facts['appearance_review'].clear()
    assert certify_qa21_observation(facts, 'a0', 'bark')['status'] == 'fail'
    facts = normalize_episode_bundle(_fixture())
    facts['actors']['a1']['appearance'] = deepcopy(facts['actors']['a0']['appearance'])
    facts['appearance_review']['a1'] = deepcopy(facts['appearance_review']['a0'])
    assert certify_qa21_observation(facts, 'a0', 'bark')['status'] == 'fail'
    facts = normalize_episode_bundle(_fixture())
    event = {**facts['events'][0], 'event_id': 'second_class', 'sound_class': 'cat_meow'}
    facts['events'].append(event)
    assert certify_qa21_observation(facts, 'a0', 'bark')['status'] == 'fail'


def native_scene():
    sensor = NS(render_camera=NS(projection_matrix=np.eye(4)),
        node=NS(absolute_transformation=lambda: np.eye(4)))
    video = NS(sim=NS(_sensors={'rgb': NS(_sensor_object=sensor)}))
    scene = {'automatic_construction': {'family': 'QA-21', 'revision': REVISION,
        'qa21_target_asset_id': 'target'}, 'listener_m': [0., 0., 0.]}
    entry = {'declaration': {'asset_id': 'target', 'kind': 'animal'},
        'applied_root_input': np.eye(4),
        'original_mesh': NS(vertices=np.array([[-.25, -.25, 0.], [.25, .25, 0.]]))}
    return video, scene, entry


def test_native_preflight_uses_measured_pixels_and_complete_mesh_projection():
    video, scene, entry = native_scene()
    assert native_candidate_check(video, scene, [entry], {'target': MINIMUM_TARGET_PIXELS})['status'] == 'pass'
    assert native_candidate_check(video, scene, [entry], {'target': 299})['status'] == 'fail'
    entry['original_mesh'].vertices[1, 0] = 1.05
    result = native_candidate_check(video, scene, [entry], {'target': 1000})
    assert result['status'] == 'fail' and result['edge_margin_px'] < 0.


def test_later_actual_occlusion_cannot_hide_target_and_unrelated_scenes_are_untouched():
    video, scene, entry = native_scene()
    other = deepcopy(entry);other['declaration']['asset_id'] = 'other'
    other['applied_root_input'][0, 3] = 2.
    assert native_candidate_check(video, scene, [entry, other], {'target': 350, 'other': 0})['status'] == 'pass'
    assert native_candidate_check(video, scene, [entry, other], {'target': 15, 'other': 0})['status'] == 'fail'
    scene['automatic_construction']['family'] = 'QA-10'
    assert native_candidate_check(video, scene, [], {})['status'] == 'not_applicable'


def test_multi_target_public_authority_preserves_each_independent_target():
    from avengine.qa.unified_catalog import generate_unified_questions
    facts = normalize_episode_bundle(_fixture())
    expected = {'a0': 'cat_meow', 'a1': 'dog_bark', 'a2': 'music_playback', 'a3': 'speech_playback'}
    for event in facts['events']:event['sound_class'] = expected[event['actor_id']]
    result = generate_unified_questions(facts, qa_ids=['QA-21'], items_per_type=5, seed='multi')
    observed = {i['truth']['evidence']['target_actor_id']: i['truth']['value'] for i in result['items']}
    assert observed == expected and not result['deferred']
    for actor, cls in expected.items():assert certify_qa21_observation(facts, actor, cls)['status'] == 'pass'


def test_source_selection_backtracks_for_three_distinct_appearances():
    from avengine.construction.joint.families.qa21 import select_target_set
    options = [
        ({**animal('cat', 'cat', 'cat_meow'), 'label_color': 'brown'}, 'cat_meow', {}),
        ({**animal('dog0', 'dog', 'dog_bark'), 'label_color': 'black_white'}, 'dog_bark', {}),
        ({**animal('dog1', 'dog', 'dog_bark'), 'label_color': 'black'}, 'dog_bark', {}),
        ({'asset_id': 'speaker', 'kind': 'object', 'entity_class': 'loudspeaker', 'label_color': 'white'}, 'music_playback', {})]
    chosen = select_target_set(options, 3)
    assert [o[0]['asset_id'] for o in chosen] == ['cat', 'dog1', 'speaker']


def source_schedule_fixture():
    row = {'qa21_taxonomy_schedule': {'target_asset_ids': ['cat', 'dog', 'speaker']},
        'source_classes': {'cat': ['cat_meow'], 'dog': ['dog_bark'], 'speaker': ['music_playback']}}
    records = [{'sound_asset_id': 'tts', 'sound_class': 'speech_playback', 'source_type': 'author_tts',
        'header': {'sample_count': 24000, 'sample_rate_hz': 16000}}]
    records += [{'sound_asset_id': aid, 'sound_class': cls, 'source_type': 'recording'}
        for aid, cls in [('cat', 'cat_meow'), ('dog', 'dog_bark'), ('speaker', 'music_playback')]]
    events = [{'event_id': aid, 'endpoint_id': aid, 'start_tick': 0, 'end_tick': 20,
        'sound_class': cls, 'mechanism': 'actual_registered'}
        for aid, cls in [('cat', 'cat_meow'), ('dog', 'dog_bark'), ('speaker', 'music_playback')]]
    bindings = [{'event_id': aid, 'sound_asset_id': aid, 'crop_start_sample': 4000} for aid in ['cat', 'dog', 'speaker']]
    return row, {'events': events}, {'records': records, 'events': bindings}, [
        {'asset_id': aid, 'endpoint_id': aid} for aid in ['cat', 'dog', 'speaker']]


def test_schedule_uses_real_duration_and_binding_without_changing_source_inputs():
    from avengine.construction.joint.families.qa21 import apply_emission_schedule
    row, scene, program, assets = source_schedule_fixture();original = deepcopy((scene, program))
    realized, audio = apply_emission_schedule(row, scene, program, assets)
    assert (scene, program) == original
    assert audio['records'] == program['records']
    assert [e['start_tick'] for e in realized['events']] == [20, 42, 64, 120, 142, 164, 220, 242, 264]
    assert [e['end_tick'] for e in realized['events']] == [40, 62, 84, 140, 162, 184, 240, 262, 284]
    assert {b['crop_start_sample'] for b in audio['events']} == {4000}
    assert apply_emission_schedule(row, realized, audio, assets) == (realized, audio)
    program['records'][0]['header']['sample_count'] = 160000
    with pytest.raises(ValueError, match='do not fit'):apply_emission_schedule(row, scene, program, assets)


def test_geometry_preflight_accounts_for_other_actual_body_plans():
    from avengine.construction.joint.families.qa21 import Qa21Plugin
    calls = []
    class Context:
        def predict(self, camera, point, body, *, occluders=()):
            calls.append(occluders)
            return NS(state='fully_occluded' if occluders else 'visible_clear',
                visible_pixels_lower=0 if occluders else 1000, edge_margin_px=20.)
    body = NS(radius_m=.3)
    actors = {'cat': NS(positions_m=np.tile([0., 0., -4.], (901, 1)), body=body),
              'dog': NS(positions_m=np.tile([0., 0., -2.8], (901, 1)), body=body)}
    plugin = Qa21Plugin({'cat': body, 'dog': body}, {}, 'cat', 'cat_meow',
        sounds=[NS(entity_id='cat', sound_class='cat_meow')])
    checks = list(plugin.verify(Context(), NS(branch='cat_meow', target_ids=('cat',)),
        NS(camera=None, actors=actors, sounds=plugin.sounds)))
    assert calls and calls[0][0][0] == 'dog'
    assert any(c.name == 'qa21_visible_cat' and c.status == 'fail' for c in checks)


def test_unlabelled_or_unqualified_mesh_cannot_be_a_qa21_target():
    cat = animal('cat', 'cat', 'cat_meow');room = {'family': 'hm3d', 'categories': ['floor_cat']}
    records = [{'sound_class': 'cat_meow'}]
    assert target_options(room, [{**cat, 'label_color': None}], records, {}, {}) == []
    assert target_options(room, [{**cat, 'standing_registered_qualified': False}], records, {}, {}) == []


def test_native_appearance_preflight_uses_original_reviewer_and_restores_video_state():
    from avengine.construction.joint.families.qa21 import native_appearance_check
    asset = {'asset_id': 'cat', 'native_instance_id': 1, 'kind': 'animal',
        'label_color': 'brown', 'registered_label_color': 'brown', 'species_id': 'cat'}
    entry = {'declaration': asset}
    old_colors = [{'existing': True}];old_materials = [{'existing': True}];old_actors = ['existing']
    video = NS(actors=old_actors, color_checks=old_colors, material_checks=old_materials,
        sim=NS(get_sensor_observations=lambda: {'rgb': np.zeros((2, 2, 4)), 'semantic': np.ones((2, 2))}),
        animal_color_thresholds={'min_pixels': 200})
    def measure(rgb, semantic, frame):
        assert video.actors == [entry]
        video.color_checks.append({'asset_id': 'cat', 'matched': True, 'frame_index': 0, 'roi_pixels': 300})
    video._check_source_frame = measure
    scene = {'automatic_construction': {'qa21_target_asset_ids': ['cat']}}
    result = native_appearance_check(video, scene, [entry])
    assert result['status'] == 'pass' and result['reviewed_target_ids'] == ['cat']
    assert video.actors is old_actors and video.color_checks is old_colors and video.material_checks is old_materials
    def wrong(rgb, semantic, frame):
        video.color_checks.append({'asset_id': 'cat', 'matched': False, 'frame_index': 0, 'roi_pixels': 300})
    video._check_source_frame = wrong
    assert native_appearance_check(video, scene, [entry])['status'] == 'fail'
    assert video.color_checks is old_colors


def test_actual_core_proxy_is_not_an_actual_pixel_lower_bound_and_exact_mesh_is_used(tmp_path):
    import trimesh
    from avengine.construction.joint import CameraPose, BodySpec, ActorPlan, ScenePlan, SoundEvent, FamilyRequest
    from avengine.construction.joint.geometry import DepthBuffer, predict_visibility
    from avengine.construction.joint.families.qa21 import Qa21Plugin
    camera = CameraPose((0., 1.25, 0.), 0., width=320, height=180)
    depth = DepthBuffer(camera, np.full((180, 320), np.inf), source='visual_mesh')
    root = np.array([0., 0., -3.8]);body = BodySpec('cat', .4, .6, .04)
    proxy = predict_visibility(camera, root, body, depth=depth)
    assert proxy.visible_pixels_lower == 0
    assert proxy.evidence['inner_proxy_visible_pixels'] > 100
    mesh = trimesh.creation.box(extents=[.65, .65, .35]);mesh.apply_translation([0., .325, 0.])
    path = tmp_path/'cat.glb';mesh.export(path)
    actor = ActorPlan('cat', np.tile(root, (901, 1)), body)
    event = SoundEvent('cat', 'actual_record', 'cat_meow', 1., 2., 2.5)
    plan = ScenePlan(camera, {'cat': actor}, (event,))
    plugin = Qa21Plugin({'cat': body}, {'cat': 'inside'}, 'cat', 'cat_meow', sounds=(event,),
        assets={'cat': {'habitat_glb': str(path), 'kind': 'animal', 'species_id': 'cat'}})
    context = NS(depth=lambda pose: depth)
    request = FamilyRequest('QA-21', 'cat_meow', ('cat',))
    checks = list(plugin.verify(context, request, plan))
    visible = next(c for c in checks if c.name == 'qa21_visible_cat')
    assert visible.status == 'pass'
    assert visible.evidence['prediction_confidence'] == 'exact_mesh_prediction'
    assert visible.evidence['mesh_prepared']
    assert visible.evidence['visible_pixels_lower'] >= 300


def test_static_domain_precheck_delegates_to_core_and_does_not_mutate_shared_grid(monkeypatch):
    from avengine.construction.joint import CameraPose, BodySpec, Certificate, CheckResult
    from avengine.construction.joint.context import CandidateGrid
    from avengine.construction.joint.families.qa21 import _StaticDomainContext
    import avengine.construction.joint.certification as core
    points = np.array([[-1., 0., -3.], [1., 0., -3.]])
    camera = CameraPose((0., 1.25, 0.), 0.);body = BodySpec('cat', .2, .5)
    grid = CandidateGrid(points, np.ones(2), np.zeros(2), np.array(['inside', 'outside']),
        np.ones(2), np.zeros(2, bool), np.array(['', '']), {'shared': True})
    calls = []
    def check(context, plan):
        actor = plan.actors['qa21_static_point'];calls.append(actor.positions_m.copy())
        return Certificate((CheckResult('core_clearance', 'pass' if actor.positions_m[0, 0] > 0 else 'fail'),))
    monkeypatch.setattr(core, 'certify_plan', check)
    base = NS(points_m=points, grid=lambda camera, body: grid)
    wrapped = _StaticDomainContext(base)
    filtered = wrapped.grid(camera, body)
    assert filtered.frustum_state.tolist() == ['edge', 'outside']
    assert grid.frustum_state.tolist() == ['inside', 'outside'] and grid.evidence == {'shared': True}
    assert all(p.shape == (2, 3) and np.array_equal(p[0], p[1]) for p in calls)
    assert wrapped.grid(camera, body) is filtered and len(calls) == 2


def test_exact_target_visibility_is_part_of_the_single_source_domain(tmp_path):
    import trimesh
    from avengine.construction.joint import CameraPose, BodySpec
    from avengine.construction.joint.geometry import DepthBuffer
    from avengine.construction.joint.families.qa21 import Qa21Plugin
    camera = CameraPose((0., 1.25, 0.), 0., width=320, height=180)
    mesh = trimesh.creation.box(extents=[.65, .65, .35]);mesh.apply_translation([0., .325, 0.])
    path=tmp_path/'cat.glb';mesh.export(path);body=BodySpec('cat',.4,.65,.04)
    plugin=Qa21Plugin({'cat':body},{'cat':'inside'},'cat','cat_meow',
        assets={'cat':{'habitat_glb':str(path),'kind':'animal','species_id':'cat'}})
    clear=NS(depth=lambda _:DepthBuffer(camera,np.full((180,320),np.inf),source='visual_mesh'))
    blocked=NS(depth=lambda _:DepthBuffer(camera,np.ones((180,320)),source='visual_mesh'))
    assert plugin.point_visible(clear,camera,body,np.array([0.,0.,-3.8]))
    assert not plugin.point_visible(blocked,camera,body,np.array([0.,0.,-3.8]))


def test_proven_outside_target_points_do_not_launch_near_plane_full_image_rays():
    from avengine.construction.joint import CameraPose, BodySpec
    from avengine.construction.joint.families.qa21 import Qa21Plugin
    body=BodySpec('cat',.3,.5,.04);camera=CameraPose((0.,1.25,0.),0.,width=320,height=180)
    plugin=Qa21Plugin({'cat':body},{'cat':'inside'},'cat','cat_meow',
        assets={'cat':{'habitat_glb':'not_opened.glb','kind':'animal','species_id':'cat'}})
    assert not plugin.point_visible(None,camera,body,np.array([.6,0.,0.]))


def test_native_support_contract_replays_the_solved_mesh_root(tmp_path):
    import trimesh
    from avengine.construction.joint import CameraPose,BodySpec,ActorPlan,ScenePlan
    from avengine.construction.joint.families.qa21 import solved_native_supports
    from avengine.capture.v7_visible_native import attachment_template
    from avengine.capture.v7_visible_attachment import support_points
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    mesh=trimesh.creation.box(extents=[.4,.8,.4]);mesh.apply_translation([.1,.4,-.15])
    path=tmp_path/'speaker.glb';mesh.export(path)
    asset={'asset_id':'speaker','kind':'object','attachment_surface':'floor','habitat_glb':str(path)}
    root=np.array([2.,0.,-4.]);actor=ActorPlan('speaker',np.tile(root,(901,1)),BodySpec('slide',.5,.8))
    plan=ScenePlan(CameraPose((0.,1.25,0.),0.),{'speaker':actor})
    room={'floor_polygon':{'type':'Polygon','coordinates':[[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]},'measured_floor_y_m':0.}
    supports=solved_native_supports(plan,{'speaker':asset},42,room);pin=supports['speaker']
    original=indexed_static_mesh(str(path));meta={**asset,'_attachment_template':attachment_template(np.asarray(original.vertices),'floor','object',mesh=original)}
    observed,yaw,_,_=support_points(meta,pin)
    assert pin['joint_solved_pose'] and np.allclose(observed,root,atol=1.e-8)
    assert pin['qa21_expected_root_m']==root.tolist()


def test_real_stage_ray_rejects_a_hidden_emission_part():
    import trimesh
    from avengine.construction.joint import CameraPose,BodySpec,ActorPlan,ScenePlan
    from avengine.construction.joint.geometry import MeshRaycaster
    from avengine.construction.joint.families.qa21 import Qa21Plugin
    camera=CameraPose((0.,1.25,0.),0.);body=BodySpec('cat',.2,.5)
    actor=ActorPlan('cat',np.tile([0.,0.,-3.8],(901,1)),body)
    plan=ScenePlan(camera,{'cat':actor})
    plugin=Qa21Plugin({'cat':body},{'cat':'inside'},'cat','cat_meow',assets={
        'cat':{'kind':'animal','emitter_local_m':[0.,.25,0.]}})
    wall=trimesh.creation.box(extents=[2.,2.,.1]);wall.apply_translation([0.,1.,-2.])
    assert plugin.emitter_path_clear(NS(raycaster=MeshRaycaster(wall)),plan,'cat') is False
    wall.apply_translation([5.,0.,0.])
    assert plugin.emitter_path_clear(NS(raycaster=MeshRaycaster(wall)),plan,'cat') is True


def test_explicit_playback_class_keeps_three_actual_capable_distinct_bodies():
    from avengine.construction.joint.families.qa21 import select_target_set
    cat = animal('cat', 'cat', 'cat_meow'); dog = animal('dog', 'dog', 'dog_bark')
    speaker = {'asset_id': 'speaker', 'kind': 'object', 'entity_class': 'loudspeaker',
        'allowed_classes': ['music_playback', 'speech_playback'], 'attachment_surface': 'floor',
        'label_color': 'white', 'standing_registered_qualified': True}
    catalog = [cat, dog, speaker]; original = deepcopy(catalog)
    records = [{'sound_class': c} for c in ('cat_meow', 'dog_bark', 'music_playback', 'speech_playback')]
    room = {'family': 'hm3d', 'categories': ['floor_cat', 'floor_dog', 'floor_object']}
    options = target_options(room, catalog, records, {}, {},
        preferred_asset_ids=['cat', 'dog', 'speaker'], asset_class_choices={'speaker': 'speech_playback'})
    assert {a['asset_id']: cls for a, cls, _ in select_target_set(options)} == {
        'cat': 'cat_meow', 'dog': 'dog_bark', 'speaker': 'speech_playback'}
    assert catalog == original
    with pytest.raises(ValueError, match='capability'):
        target_options(room, catalog, records, {}, {}, asset_class_choices={'speaker': 'cat_meow'})
    with pytest.raises(ValueError, match='not an admitted source'):
        target_options(room, catalog, records[:-1], {}, {}, asset_class_choices={'speaker': 'speech_playback'})


def test_family_preflight_keeps_only_appearance_while_core_owns_acoustic_gate(monkeypatch):
    import avengine.construction.joint.families.qa21 as module
    monkeypatch.setattr(module, 'native_candidate_check', lambda *args: {'status': 'pass', 'actual_pixels': 500})
    video = NS(_native_pair_pixels=lambda scene, entries: {'target': 500})
    checks = list(module.native_family_obligations(video, {}, []))
    assert len(checks) == 1 and checks[0].name == 'native:qa21:appearance_and_pixels'
    assert checks[0].status == 'pass' and checks[0].evidence['actual_pixels'] == 500


def test_delivered_class_feedback_cannot_replace_explicitly_preferred_reviewed_bodies():
    from avengine.construction.joint.families.qa21 import select_target_set
    cat = animal('cat', 'cat', 'cat_meow'); dog = animal('dog', 'dog', 'dog_bark')
    speaker = {'asset_id': 'speaker', 'kind': 'object', 'entity_class': 'loudspeaker',
        'allowed_classes': ['music_playback', 'speech_playback'], 'attachment_surface': 'floor',
        'label_color': 'white', 'standing_registered_qualified': True}
    distractor = {**speaker, 'asset_id': 'other_speaker'}
    catalog = [cat, dog, speaker, distractor];original = deepcopy(catalog)
    records = [{'sound_class': c} for c in ('cat_meow', 'dog_bark', 'music_playback', 'speech_playback')]
    room = {'family': 'hm3d', 'categories': ['floor_cat', 'floor_dog', 'floor_object']}
    feedback = {'qa21_positive_counts': {'cat_meow': 10, 'dog_bark': 10, 'music_playback': 10}}
    selected = select_target_set(target_options(room, catalog, records, {}, feedback,
        preferred_asset_ids=['cat', 'dog', 'speaker'], asset_class_choices={'speaker': 'music_playback'}))
    assert {a['asset_id']: cls for a, cls, _ in selected} == {
        'cat': 'cat_meow', 'dog': 'dog_bark', 'speaker': 'music_playback'}
    assert catalog == original


def test_dog_height_is_solved_from_actual_scan_before_any_native_execution(monkeypatch):
    import trimesh
    from avengine.construction.joint import CameraPose, BodySpec, ScenePlan, SoundEvent
    from avengine.construction.joint.routes import static_actor
    from avengine.construction.joint import MeshRaycaster
    from avengine.construction.joint.families.qa21 import Qa21Plugin
    import avengine.construction.joint as core
    import avengine.capture.v7_visible_mesh as meshes
    floor = trimesh.creation.box((10., .1, 10.));floor.apply_translation([0., -.038, 0.])
    mesh = trimesh.creation.box((.6, .6, .4));mesh.apply_translation([0., .3, 0.])
    actor = static_actor('dog', [0., .002, -4.], BodySpec('dog', .35, .6), facing_rad=0.)
    sound = SoundEvent('dog', 'bark_record', 'dog_bark', 2., 3., wet_end_s=3.5)
    base = ScenePlan(CameraPose((0., 1.25, 0.), 0.), {'dog': actor}, sounds=(sound,))
    monkeypatch.setattr(core, 'enumerate_static_layouts', lambda *a, **kw: iter([base]))
    monkeypatch.setattr(meshes, 'indexed_static_mesh', lambda path: mesh)
    asset = {**animal('dog', 'dog', 'dog_bark'), 'habitat_glb': 'actual-admitted-dog.glb'}
    plugin = Qa21Plugin({'dog': actor.body}, {'dog': 'inside'}, 'dog', 'dog_bark', sounds=(sound,), assets={'dog': asset})
    result = list(plugin.candidates(NS(floor_raycaster=MeshRaycaster(floor)), None))[0]
    # Construct at the measured scan floor, recording residuals instead of
    # selecting the former cosmetic 2mm floating variant; input stays immutable.
    assert result.actors['dog'].positions_m[0, 1] == pytest.approx(.012, abs=1e-6)
    assert result.actors['dog'].metadata['contact_quality_observation']['root_support_verified']
    np.testing.assert_allclose(actor.positions_m[:, 1], .002)
    np.testing.assert_array_equal(result.actors['dog'].positions_m[:, [0, 2]], actor.positions_m[:, [0, 2]])
    np.testing.assert_array_equal(result.actors['dog'].headings_rad, actor.headings_rad)
    assert result.sounds == base.sounds
    assert result.actors['dog'].metadata['floor_contact_refinement']['native_contact_checks_required']
    fallback = list(plugin.candidates(NS(), None))[0]
    np.testing.assert_array_equal(fallback.actors['dog'].positions_m, actor.positions_m)
    assert 'dog' in fallback.metadata['qa21_grounding_proposal_skipped']
