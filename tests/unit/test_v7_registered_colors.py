from collections import Counter

import pytest

from avengine.timeline.v7_registered_sampling import choose_assets

pytestmark = pytest.mark.fast_unit

COLORS = ('blue', 'yellow', 'green', 'white', 'black')


def catalog():
    rows = []
    for gender, prefix in (('female', 'f'), ('male', 'm')):
        for b in range(8):
            for c in COLORS:
                rows.append(dict(asset_id=f'{prefix}{b}_{c}', body_identity_id=f'{prefix}{b}', color=c,
                                 visual_gender=gender, dialogue_voice_binding_eligible=True))
    return rows


def row(seed, n, gender):
    return dict(seeds=dict(appearance=seed), production_scenario=dict(speaker_count=n, gender=gender))


def test_colours_do_not_encode_speaking_order_or_gender():
    cat = catalog()
    first, silent_third = Counter(), Counter()
    for seed in range(3000):
        n = (0, 1, 2)[seed % 3]
        assets = choose_assets(cat, row(seed, n, 'FM'[seed % 2]))
        assert assets == choose_assets(cat, row(seed, n, 'FM'[seed % 2]))  # deterministic per seed
        assert {a['color'] for a in assets[:3]} == {'blue', 'yellow', 'green'}
        assert all(a['color'] in ('white', 'black') for a in assets[3:])
        assert assets[0]['body_identity_id'] == assets[1]['body_identity_id']  # the colour-only twin stays
        if n >= 1:
            first[assets[0]['color']] += 1
        if n <= 2:
            silent_third[(assets[2]['color'], assets[2]['visual_gender'])] += 1
    for c in ('blue', 'yellow', 'green'):
        assert 0.28 < first[c] / sum(first.values()) < 0.39
        share = silent_third[(c, 'female')] / (silent_third[(c, 'female')] + silent_third[(c, 'male')])
        assert 0.4 < share < 0.6
    assert all(a['appearance']['top_color'] == a['color'] for a in choose_assets(cat, row(7, 2, 'F'))[:3])
