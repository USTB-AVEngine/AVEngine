import pytest
from tools.rooms.benchmark_kujiale_native_concurrency import choose_topology

pytestmark=pytest.mark.fast_unit


def snapshot(free):
    return [{'index':i,'free_mib':value} for i,value in enumerate(free)]


def test_eight_uses_one_card_when_aggregate_reserve_fits():
    assert choose_topology(snapshot([45000,17000,13000,29000]),8,3900)==[0]*8


def test_eight_selects_actual_feasible_cards_in_shared_server():
    assert choose_topology(snapshot([15000,31000,13000,29000]),8,3900)==[1]*4+[3]*4


def test_insufficient_aggregate_space_stops_before_launch():
    with pytest.raises(RuntimeError):choose_topology(snapshot([15000,17000,13000,18000]),8,3900)


def test_measured_peak_replaces_initial_peak():
    with pytest.raises(RuntimeError):choose_topology(snapshot([20000,None,None,None]),4,4500)


def test_unknown_instance_count_rejected():
    with pytest.raises(ValueError):choose_topology(snapshot([45000]*4),3,3900)
