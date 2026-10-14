import pytest
from avengine.timeline.v7_production_inputs import normalize_rooms, validate_recipe, sample_unique_shirts, SCHEMA

pytestmark = pytest.mark.fast_unit


def test_mp_review_budget_and_aggregate_cannot_admit_assets():
    room = {'house': 'mp3d_train', 'room_id': 'r', 'split': 'train',
            'first_layer_available': True, 'production_sampling_available': True,
            'categories': {'floor_cat': {'available': True, 'budget_limited': True},
                           'floor_dog': {'available': False}, 'floor_animals': {'available': True},
                           'chair_seated_human': {'available': True}}}
    assert normalize_rooms({'rooms': {'r': room}}, 'mp3d', set())[0]['categories'] == ['chair_seated']
    room['production_sampling_available'] = False
    assert normalize_rooms({'rooms': [room]}, 'mp3d', set()) == []


def test_hm_house_gate_and_test_house_override_placement_pass():
    r = {'house': 'hm', 'room_id': 'r', 'task_split': 'train', 'categories': {
        'floor_object': {'usable': True, 'sampler_usable_with_original_house_acoustic_gate': False},
        'standing_human': {'usable': True, 'sampler_usable_with_original_house_acoustic_gate': True}}}
    assert normalize_rooms({'rooms': [r]}, 'hm3d', {'hm'}) == []
    assert normalize_rooms({'rooms': [r]}, 'hm3d', set())[0]['categories'] == ['standing_human']


def test_ue_r0_is_scoped_by_house_and_detailed_object_category():
    make = lambda house: {'house': house, 'room_id': 'R0', 'split': 'train',
                         'native_room_render_status': 'pass', 'floor_support_check': {'status': 'pass'},
                         'categories': [{'category_key': 'object.cellphone.table', 'status': 'usable_for_tested_representative'}]}
    rooms = normalize_rooms({'rooms': [make('a'), make('b')]}, 'kujiale', set())
    assert len({r['key'] for r in rooms}) == 2
    manifest = {'schema': SCHEMA, 'test_houses': [], 'rooms': rooms, 'animal_motion': {}}
    asset = {'asset_id': 'phone', 'kind': 'object', 'object_type': 'landline_phone', 'attachment_surface': 'wall'}
    with pytest.raises(ValueError, match='does not admit'):
        validate_recipe({'house': 'a', 'room_id': 'R0', 'split': 'train', 'asset_ids': ['phone']}, [asset], manifest)


def test_five_colors_deterministic_and_duplicate_recipe_rejected():
    catalog = [{'asset_id': c + str(i), 'controlled_shirt': True, 'color': c, 'kind': 'human'}
               for c in ['black', 'blue', 'green', 'white', 'yellow'] for i in range(2)]
    assert sample_unique_shirts(catalog, 5, 31) == sample_unique_shirts(catalog, 5, 31)
    assert len({a['color'] for a in sample_unique_shirts(catalog, 5, 31)}) == 5
    with pytest.raises(ValueError):
        sample_unique_shirts(catalog, 6, 31)
    manifest = {'schema': SCHEMA, 'test_houses': [], 'animal_motion': {}, 'rooms': [
        {'key': 'h::r', 'family': 'hm3d', 'categories': ['standing_human']}]}
    row = {'house': 'h', 'room_id': 'r', 'split': 'train', 'asset_ids': ['blue0', 'blue1']}
    with pytest.raises(ValueError, match='distinct'):
        validate_recipe(row, catalog, manifest)


def test_v13_diagnostic_controls_do_not_fabricate_a_legacy_positive_pass():
    from avengine.timeline.v7_audio_sources import source_blockers
    # Only inspect the compatibility gate here; unrelated absent waveform and
    # model screens still reject this intentionally incomplete record.
    record = {'sound_asset_id': 'diagnostic', 'source_type': 'generated_audio',
              'reference_dedup': {'status': 'pass', 'coverage_complete_for_listed_roots': True,
                                  'positive_controls_are_not_an_admission_gate': True},
              'automatic_label': []}
    assert 'generated_audio_complete_reference_dedup_missing' not in source_blockers(record)
    assert 'positive_controls_pass' not in record['reference_dedup']
    record['reference_dedup'].pop('positive_controls_are_not_an_admission_gate')
    assert 'generated_audio_complete_reference_dedup_missing' in source_blockers(record)
    record['reference_dedup']['positive_controls_pass'] = True
    record['reference_dedup']['coverage_complete_for_listed_roots'] = False
    assert 'generated_audio_complete_reference_dedup_missing' in source_blockers(record)


def test_runtime_contract_cannot_be_silently_omitted_or_replaced(tmp_path):
    import json
    from avengine.timeline.v7_production_inputs import attach_runtime_contract
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps({'schema': SCHEMA}))
    row = {'scene_id': 'candidate'}
    result = attach_runtime_contract(row, {'production_input_contract': str(path)})
    assert 'production_sampling' in result and 'production_sampling' not in row
    result['production_sampling']['manifest_sha256'] = 'changed'
    with pytest.raises(ValueError, match='differ'):
        attach_runtime_contract(result, {'production_input_contract': str(path)})


def test_production_five_shirts_stay_unique_with_a_sixth_professional():
    from avengine.timeline.v7_registered_sampling import choose_assets
    catalog = [{'asset_id': gender + c, 'body_identity_id': gender,
                'controlled_shirt': c != 'original', 'color': c, 'dialogue_voice_binding_eligible': True,
                'visual_gender': gender} for gender in ('male', 'female')
               for c in ('blue', 'yellow', 'green', 'white', 'black', 'original')]
    row = {'production_scenario': {'speaker_count': 5, 'gender': 'F'},
           'seeds': {'appearance': 4}, 'production_sampling': {'manifest_path': 'explicit'}}
    assets = choose_assets(catalog, row)
    colors = [a['color'] for a in assets if a['controlled_shirt']]
    assert len(assets) == 6 and len(colors) == len(set(colors)) == 5
    assert assets[-1]['color'] == 'original'


def test_dialogue_header_uses_actual_registry_without_editing_sound_records():
    from copy import deepcopy
    from avengine.timeline.v7_production_inputs import bind_semantic_registry
    manifest = {'sounds': [{'sound_asset_id': 'retained', 'sha256': 'unchanged'}]}
    original = deepcopy(manifest)
    human = {'kind': 'human',
             'source_registry': '/retained/registrations/human_roster.json'}
    adapted = bind_semantic_registry(manifest, [human])
    assert manifest == original and adapted['sounds'] == original['sounds']
    assert adapted['source_registry'] == human['source_registry']
    other = dict(human, source_registry='/different/registry.json')
    with pytest.raises(ValueError, match='one actual'):
        bind_semantic_registry(manifest, [human, other])
    with pytest.raises(ValueError, match='one actual'):
        bind_semantic_registry(manifest, [])
