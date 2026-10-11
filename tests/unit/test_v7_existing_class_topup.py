import hashlib

import numpy as np
import pytest
import soundfile as sf

from avengine.qa.v7_extended_admission import build_recorded_admission

pytestmark = pytest.mark.fast_unit

CC0 = 'http://creativecommons.org/publicdomain/zero/1.0/'
ONTOLOGY = [dict(id=m, child_ids=[]) for m in ('/m/02pjr4', '/m/0c3f7m', '/m/05tny_', '/m/04rlf')]
TARGETS = {'blender': {'target_mids': ['/m/02pjr4']}, 'fire_alarm': {'target_mids': ['/m/0c3f7m']},
           'dog_bark': {'target_mids': ['/m/05tny_']}}
GROUPS = {'fire_alarm': [['fire'], ['alarm']]}


def clip(tmp_path, n):
    x = np.zeros(16000)
    x[4000:12000] = 0.1 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)
    path = tmp_path / f'fsd_{n}.wav'
    sf.write(path, x, 16000, subtype='FLOAT')
    return str(path)


def rows(tmp_path):
    specs = [  # id, class, metadata title, model score on the class target
        ('1', 'blender', 'kitchen blender', .30), ('2', 'blender', 'kitchen blender', .15),
        ('3', 'fire_alarm', 'office alarm', .40), ('4', 'fire_alarm', 'fire alarm test', .40),
        ('5', 'dog_bark', 'dog barking', .30)]
    fsd, beats = [], []
    for n, cls, title, score in specs:
        path = clip(tmp_path, n)
        fsd.append(dict(sound_asset_id=f'fsd_{n}', original_id=n, path=path, candidate_classes=[cls],
                        official_zip_crc_verified=True, source_license=CC0, source_dataset='FSD50K',
                        source_uri=f'https://freesound.org/s/{n}/', split='train', official_collection_labels=[],
                        target_mids=TARGETS[cls]['target_mids'],
                        source_metadata=dict(title=title, description='', tags=[], uploader=f'u{n}', license=CC0)))
        scores = {m['id']: 0. for m in ONTOLOGY}
        scores[TARGETS[cls]['target_mids'][0]] = score
        beats.append(dict(sound_asset_id=f'fsd_{n}', sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest(),
                          scores_by_mid=scores, top8=[]))
    dedup = dict(complete_for_listed_roots=True, records={r['sound_asset_id']: {'status': 'pass'} for r in fsd},
                 positive_controls=[], reference_file_count=1, reference_waveform_count=1, seconds=0., scope_limit='test')
    return fsd, beats, dedup


def run(tmp_path, name, **topup):
    fsd, beats, dedup = rows(tmp_path)
    result = build_recorded_admission(pool=[], old_inventory=[], old_admission={'records': []}, vctk_lineages=[],
                                      fsd_rows=fsd, beats_rows=beats, clap_rows=[], ontology=ONTOLOGY,
                                      collection_mids={}, class_targets=TARGETS, deduplications=[dedup],
                                      output=tmp_path / name, **topup)
    return {r['sound_asset_id']: r for r in result['records']}


def test_existing_class_topup_keeps_old_gate_and_subtype_rule(tmp_path):
    default = run(tmp_path, 'default')
    topup = run(tmp_path, 'topup', existing_gate_classes=('dog_bark', 'blender', 'fire_alarm'),
                existing_subtype_metadata_groups=GROUPS)
    # dog_bark is unchanged by the generalisation.
    for got in (default, topup):
        dog = got['fsd_5__dog_bark']
        assert dog['strict_v7_eligible'] and dog['automatic_label'][0]['gate']['threshold'] == .1917129977
        assert 'Bark recordings' in dog['automatic_label'][0]['gate']['source']
    # Without the opt-in, an old class has no calibrated gate on this tiny pool.
    assert default['fsd_1__blender']['current_policy_status'] == 'blocked'
    # With it, the unchanged existing threshold and margin decide.
    assert topup['fsd_1__blender']['strict_v7_eligible']
    assert topup['fsd_1__blender']['automatic_label'][0]['gate']['threshold'] == .1917129977
    assert 'no_source_calibrated_automatic_label_model_pass' in topup['fsd_2__blender']['current_policy_blockers']
    # The original subtype metadata rule still applies to the new recordings.
    assert ('original_metadata_does_not_confirm_specific_subtype'
            in topup['fsd_3__fire_alarm']['current_policy_blockers'])
    assert topup['fsd_4__fire_alarm']['strict_v7_eligible']
    assert all(m['model'] == 'BEATs' for r in topup.values() for m in r['automatic_label'])
