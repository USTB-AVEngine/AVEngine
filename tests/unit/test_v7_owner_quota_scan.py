import threading,time
import pytest
from tools.timeline.v7_s_production import OwnerQuotaScan


def finish(scan):
    end=time.monotonic()+2
    while scan.check_in_progress and time.monotonic()<end:
        state=scan.poll()
        if state['status']!='checking':return state
        time.sleep(.005)
    return scan.poll()


def test_initial_scan_blocks_dispatch_and_exact_reserve_is_enforced():
    gate=threading.Event()
    def read(root):gate.wait(2);return 90
    scan=OwnerQuotaScan('/own',100,10,reader=read)
    try:
        assert scan.poll()['new_dispatch_allowed'] is False
        gate.set();state=finish(scan)
        assert state['status']=='exhausted' and not state['new_dispatch_allowed']
    finally:gate.set();scan.close()


def test_rescan_blocks_stale_fallback_but_does_not_duplicate_scans():
    gate=threading.Event();clock=[0];calls=[]
    def read(root):
        calls.append(root)
        if len(calls)>1:gate.wait(2)
        return 89
    scan=OwnerQuotaScan('/own',100,10,reader=read,clock=lambda:clock[0])
    try:
        assert finish(scan)['new_dispatch_allowed']
        clock[0]=901
        assert not scan.poll()['new_dispatch_allowed']
        for _ in range(5):assert scan.poll()['status']=='checking'
        gate.set();assert finish(scan)['new_dispatch_allowed'];assert len(calls)==2
    finally:gate.set();scan.close()


@pytest.mark.parametrize('result',[-1,None,True])
def test_malformed_reading_never_grants_dispatch(result):
    scan=OwnerQuotaScan('/own',100,10,reader=lambda root:result)
    try:
        state=finish(scan)
        assert state['status']=='error' and not state['new_dispatch_allowed']
    finally:scan.close()


def test_scan_exception_is_durable_refusal_without_stale_fallback():
    def read(root):raise TimeoutError('known slow inventory')
    scan=OwnerQuotaScan('/own',100,10,reader=read)
    try:
        state=finish(scan);assert state['status']=='error'
        assert 'TimeoutError' in state['error'] and not state['new_dispatch_allowed']
    finally:scan.close()
