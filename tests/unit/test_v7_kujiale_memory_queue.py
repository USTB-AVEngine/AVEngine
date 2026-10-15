import pytest
from avengine.timeline.v7_kujiale_cpu_phases import remaining_peak_reserve,oldest_fitting_waiter
pytestmark=pytest.mark.fast_unit


def request(pid,order,peak,retained=0):return {'pid':pid,'time_utc':str(order),'estimate_bytes':peak,'already_counted_bytes':retained}


def test_retained_worker_memory_does_not_get_charged_twice():
    rows=[request(1,1,4000000000,3000000000)]
    assert remaining_peak_reserve(4000000000,3000000000)==1000000000
    assert oldest_fitting_waiter(rows,38000000000,0,1,0,40000000000)==1
    assert oldest_fitting_waiter(rows,39500000000,0,1,0,40000000000)is None


def test_oldest_request_that_fits_runs_without_blocking_a_lighter_legal_job():
    rows=[request(2,2,1000000000),request(1,1,6000000000)]
    assert oldest_fitting_waiter(rows,33000000000,0,2,0,40000000000)==1
    assert oldest_fitting_waiter(rows,38000000000,0,2,0,40000000000)==2


def test_cpu_and_available_memory_limits_still_pause_new_work():
    rows=[request(1,1,1000000000)]
    assert oldest_fitting_waiter(rows,10000000000,0,16,0,40000000000)is None
    assert oldest_fitting_waiter(rows,10000000000,0,1,0,29000000000)is None
