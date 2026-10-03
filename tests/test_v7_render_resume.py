"""Completed old candidates cannot bypass newly required native readbacks."""
import json
import pytest
from avengine.timeline import v7_candidate_generation as generator
from avengine.assets.source_materials import MATERIAL_REVISION
from avengine.capture.continuous_native_visibility import VISIBILITY_REVISION

pytestmark = pytest.mark.fast_unit


def fixture_parent(tmp_path, monkeypatch, *, current):
    row = {'scene_id': 'neutral_parent', 'house': 'neutral_house'}
    house, inputs, assets = {}, {}, []
    root = tmp_path / 'parents' / row['scene_id']
    root.mkdir(parents=True)
    recipe = {'seed_row': row, 'house': house, 'runtime_inputs': inputs, 'assets': assets}
    if current:
        recipe['visual_contract'] = {'source_material_revision': MATERIAL_REVISION,
                                     'pixel_visibility_revision': VISIBILITY_REVISION}
    (root / 'input_recipe.json').write_text(json.dumps(recipe))
    attempt = root / 'delivery_attempt_retained'; attempt.mkdir()
    (attempt / 'evidence.json').write_text('{}')
    (root / 'delivery_completed.json').write_text(json.dumps({
        'recipe': recipe, 'attempt': attempt.name, 'required_files': ['evidence.json'],
        'result': {'status': 'candidate_only', 'formal_dataset_count': 0}}))
    monkeypatch.setattr(generator, 'seeded_scene', lambda *_: (house, {}))
    import avengine.capture.continuous_native_video as native
    def forbidden_renderer(*args, **kwargs):
        raise AssertionError('completed-resume must not execute a native renderer')
    monkeypatch.setattr(native, 'NativeContinuousVideo', forbidden_renderer)
    return row, house, inputs, assets


def test_old_seal_cannot_skip_source_material_or_pixel_visibility_acceptance(tmp_path, monkeypatch):
    args = fixture_parent(tmp_path, monkeypatch, current=False)
    with pytest.raises(ValueError, match='parent input recipe changed'):
        generator.generate_parent(*args, tmp_path)
    assert (tmp_path / 'parents' / 'neutral_parent' / 'delivery_completed.json').is_file()


def test_current_complete_seal_resumes_without_rerendering(tmp_path, monkeypatch):
    args = fixture_parent(tmp_path, monkeypatch, current=True)
    result = generator.generate_parent(*args, tmp_path)
    assert result['resumed_sealed_delivery'] is True
    assert result['formal_dataset_count'] == 0
