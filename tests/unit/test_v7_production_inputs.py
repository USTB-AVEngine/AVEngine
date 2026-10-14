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
