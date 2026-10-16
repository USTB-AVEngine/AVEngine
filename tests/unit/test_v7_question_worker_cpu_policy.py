import pytest
from tools.timeline.run_v7_acceptance import question_worker_cpu_policy
pytestmark=pytest.mark.fast_unit


def test_qa_processes_are_disjoint_across_admitted_parent_groups():
    m={'registered_question_workers':4}
    a=question_worker_cpu_policy(m,{},range(16,24));b=question_worker_cpu_policy(m,{},range(24,32))
    assert a['registered_question_workers']==b['registered_question_workers']==4
    assert set(a['registered_question_cpu_ids'])<=set(range(16,24))
    assert set(b['registered_question_cpu_ids'])<=set(range(24,32))
    assert not(set(a['registered_question_cpu_ids'])&set(b['registered_question_cpu_ids']))
    assert question_worker_cpu_policy(m,{},[19])=={'registered_question_workers':1,'registered_question_cpu_ids':[19]}
