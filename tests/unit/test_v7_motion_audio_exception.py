import pytest
from avengine.timeline.v7_motion_audio_exception import recordable_energy_decay,event_exception_budget
pytestmark=pytest.mark.fast_unit
@pytest.mark.parametrize('failure',['direction','DRR','arrival_delay','tail_truncation','invalid_native_metric'])
def test_other_native_failures_never_receive_an_exception(failure):
 assert not recordable_energy_decay({'failures':['0:energy','1:'+failure]})
def test_only_energy_and_decay_are_recordable():
 assert recordable_energy_decay({'failures':['0:energy','1:decay']})
 assert not recordable_energy_decay({'failures':[]})
def test_union_is_counted_once_and_exact_100ms_is_allowed():
 scene={'events':[{'event_id':'one','endpoint_id':'target','start_tick':0,'end_tick':10}]}
 assert event_exception_budget(scene,'target',[(0,1000),(800,1600)])['status']=='pass'
 assert event_exception_budget(scene,'target',[(0,1601)])['status']=='fail'
def test_event_budgets_are_separate_and_unscoped_errors_are_refused():
 scene={'events':[{'event_id':str(i),'endpoint_id':'target','start_tick':i*10,'end_tick':i*10+5}for i in [0,1]]}
 assert event_exception_budget(scene,'target',[(0,1600),(16000,17600)])['status']=='pass'
 assert event_exception_budget(scene,'target',[(9000,9100)])['status']=='fail'
