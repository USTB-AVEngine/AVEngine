import math
import numpy as np
from avengine.assets.roster_motion.reflect_walk import reflect_walk, schedule_emissions
from avengine.assets.roster_motion.reflect_navmesh import _short_window_walk, SHORT_WALK_LEAD_S


def _free(p):
    return 0.35 <= p[0] <= 5.65 and 0.35 <= p[1] <= 4.65 and np.hypot(p[0] - 0.4, p[1] - 0.4) >= 1.5


def test_short_window_walk_stands_outside_and_keeps_species_speed():
    w = reflect_walk(_free, (3.0, 2.5), 0.3, 0.85, min_speed=0.8, rng=np.random.default_rng(1))
    slots = [s for s in schedule_emissions(w, (0.4, 0.4), utterance_s=(1.5, 1.5), post_s=1.5, wet_tail_s=.5,
                                           margin_s=SHORT_WALK_LEAD_S) if s['qa17']]
    assert slots
    pos, yaws, win = _short_window_walk(w, slots[0])
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 30
    t = np.arange(len(pos)) / 30
    assert np.all(v[t[1:] < win['walk_start_s'] - 1e-6] < 1e-9)          # standing before
    assert np.all(v[t[:-1] > win['walk_end_s'] + 1e-6] < 1e-9)           # standing after
    moving = v[(t[1:] > win['walk_start_s'] + .35) & (t[1:] < win['walk_end_s'] - .35)]
    assert len(moving) and abs(float(np.median(moving)) - 0.85) < 0.02 and moving.max() <= 0.85 + 1e-6
    assert win['walk_start_s'] <= slots[0]['start_s'] - SHORT_WALK_LEAD_S + 1e-6
    assert len(set(np.round(yaws, 9))) == 1
    assert all(_free(p) for p in pos)
