"""Regressions for shortcuts observed in retained bank questions."""
from avengine.qa.unified_catalog import _named_alternatives


def test_reused_event_ids_do_not_fix_wording_across_episodes():
    choices=(("nearer","nearer","更近"),("farther","farther","更远"))
    seed="claude-constructive-render-20260912"
    orders=[_named_alternatives(seed,"QA-15","event_001",choices,episode_id=f"episode_{i}")[2] for i in range(32)]
    assert {tuple(x) for x in orders} == {("nearer","farther"),("farther","nearer")}
    assert orders[0] == _named_alternatives(seed,"QA-15","event_001",choices,episode_id="episode_0")[2]
