import json,os
import pytest
from avengine.timeline.v7_cpu_visual_runtime import configure_cpu_visual,ENV
pytestmark=pytest.mark.fast_unit


@pytest.fixture(autouse=True)
def isolate_visual_environment(monkeypatch):
    monkeypatch.delenv(ENV,raising=False)
    monkeypatch.setenv('LP_NUM_THREADS','1')


def test_visual_threads_can_use_only_the_explicit_admitted_cpu_group(monkeypatch):
    calls=[];monkeypatch.setattr(os,'sched_setaffinity',lambda pid,cpus:calls.append(set(cpus)))
    result=configure_cpu_visual({'cpu_visual_threads':4},cpu_ids=[16,17,18,19,20,21,22,23])
    assert result['LP_NUM_THREADS']==4 and calls==[set(range(16,24))]
    assert json.loads(os.environ[ENV])['owner_pid']==os.getpid()


def test_single_cpu_visual_keeps_existing_parent_affinity(monkeypatch):
    monkeypatch.delenv(ENV,raising=False)
    monkeypatch.setattr(os,'sched_getaffinity',lambda pid:{32})
    monkeypatch.setattr(os,'sched_setaffinity',lambda *args:pytest.fail('must not widen a single-core task'))
    assert configure_cpu_visual({'cpu_visual_threads':1})['LP_NUM_THREADS']==1


def test_visual_thread_count_never_exceeds_approved_four_or_the_cpu_group(monkeypatch):
    monkeypatch.setattr(os,'sched_setaffinity',lambda *args:None)
    assert configure_cpu_visual({'cpu_visual_threads':8},cpu_ids=[44,45])['LP_NUM_THREADS']==2
