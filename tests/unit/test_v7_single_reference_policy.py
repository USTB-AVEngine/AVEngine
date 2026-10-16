import numpy as np
import pytest
from avengine.timeline.v7_rir_reference_policy import ENV_KEY,reference_count,checkpoint_references
from avengine.timeline.v7_phase2_motion_audio import exact_native_endpoint_checks,median_reference_check
from avengine.acoustics.moving_rir_calibration import CorrelatedRIRTimeline
from avengine.timeline.v7_rir_submission import RIRSubmissionQueue
from avengine.timeline.v7_independent_reference_memo import ExactIndependentReferenceMemo

pytestmark=pytest.mark.fast_unit


def wave():
    rng=np.random.default_rng(902);t=np.arange(16000)/16000
    w=rng.normal(size=len(t))*np.exp(-9*t)*.01;w[:140]=0;w[140]=.08
    return np.stack((w,np.zeros(len(t)),np.zeros(len(t)),np.sqrt(3)*w))


def test_new_parents_default_single_and_opt_out_is_explicit(monkeypatch):
    monkeypatch.delenv(ENV_KEY,raising=False);assert reference_count()==1
    monkeypatch.setenv(ENV_KEY,'1');assert reference_count()==1
    monkeypatch.setenv(ENV_KEY,'0');assert reference_count()==3
    monkeypatch.setenv(ENV_KEY,'2')
    with pytest.raises(ValueError):reference_count()


def test_single_endpoints_compare_and_qualify_once_each(monkeypatch):
    from avengine.timeline import v7_phase2_motion_audio as audio
    kernels=[wave(),wave()*.99];timeline=CorrelatedRIRTimeline(kernels,[0,1])
    trajectory=[{'time_s':0.,'position_m':[3.,0.,0.]},{'time_s':1.,'position_m':[3.,0.,0.]}]
    real_compare=audio.compare_rirs;real_qualify=audio.qualify_energy;compares=[];qualifies=[]
    def compare(*args,**kw):compares.append(1);return real_compare(*args,**kw)
    def qualify(checks):qualifies.append(len(checks));return real_qualify(checks)
    monkeypatch.setattr(audio,'compare_rirs',compare);monkeypatch.setattr(audio,'qualify_energy',qualify)
    rows=exact_native_endpoint_checks(timeline,0,1,2,trajectory,lambda s:[kernels[s]],
        listener=np.zeros(3),forward=np.array([1.,0.,0.]),reference_count=1)
    assert len(compares)==2 and qualifies==[1,1]
    assert all(len(r['raw_checks'])==1 and r['qualification']['sample_count']==1 and r['check']['independent_references']==1 for r in rows)
    for row in rows:
        old=median_reference_check(row['raw_checks']*3)
        new=row['check']
        assert {k:v for k,v in old.items()if k not in('independent_references','reference_rule')}=={k:v for k,v in new.items()if k not in('independent_references','reference_rule')}
        assert row['qualification']['status']=='pass'


@pytest.mark.parametrize('error',['gain','delay'])
def test_single_reference_keeps_the_same_absolute_rejections(error):
    kernels=[wave(),wave()];timeline=CorrelatedRIRTimeline(kernels,[0,1])
    trajectory=[{'time_s':0.,'position_m':[3.,0.,0.]},{'time_s':1.,'position_m':[3.,0.,0.]}]
    def independent(sample):
        w=kernels[sample]
        if sample==1:w=w*2 if error=='gain'else np.pad(w,((0,0),(8,0)))
        return[w]
    rows=exact_native_endpoint_checks(timeline,0,1,2,trajectory,independent,
        listener=np.zeros(3),forward=np.array([1.,0.,0.]),reference_count=1)
    assert rows[0]['qualification']['status']=='pass' and rows[1]['qualification']['status']=='fail'
    assert all(r['qualification']['criteria']['direct_total_octave_energy_max_abs_db']==1. for r in rows)
    assert all(r['qualification']['criteria']['arrival_delay_max_abs_ms']==.25 for r in rows)


def test_single_submission_never_starts_trial1_or_trial2():
    calls=[]
    class Renderer:
        def _render_one(self,p,role,trial):calls.append((tuple(p),role,trial));return np.ones((4,8)),'native',1.
    q=RIRSubmissionQueue(Renderer(),3)
    try:
        q.references([[1,2,3],[2,2,3]],reference_count=1)
        q.get([1,2,3],'independent_midpoint_reference',0);q.get([2,2,3],'independent_midpoint_reference',0)
    finally:q.executor.shutdown()
    assert len(calls)==2 and all(c[2]==0 for c in calls)


def test_legacy_checkpoint_selects_only_trial0_without_opening_other_files():
    items=[{'path':'trial0'},{'path':'unused_missing_trial1'},{'path':'unused_missing_trial2'}]
    assert checkpoint_references(items,1)==[items[0]]
    assert checkpoint_references(items,3)==items
    with pytest.raises(ValueError):checkpoint_references(items[:1],3)


def test_single_pose_memo_reuses_one_reference_with_original_evidence():
    memo=ExactIndependentReferenceMemo({'scene':'s'},reference_count=1);calls=[]
    def measure():calls.append(1);return {'paths':['trial0'],'processes':[123]}
    assert memo.get([1,2,3],measure)[1]is False
    assert memo.get([1,2,3],measure)[1]is True
    assert calls==[1]


@pytest.mark.parametrize('enabled,count',[('1',1),('0',3)])
def test_optional_prefetch_uses_the_same_declared_reference_count(tmp_path,monkeypatch,enabled,count):
    import json
    from types import SimpleNamespace
    from avengine.timeline.v7_phase2_rir_prefetch import run_prefetch_plan
    from avengine.timeline.v7_phase2_fresh_rir import REVISION
    monkeypatch.setenv(ENV_KEY,enabled)
    parent=tmp_path/'parent';root=parent/'audio/fresh';request_path=root/'token/request.json';request_path.parent.mkdir(parents=True)
    identity={'scene':'s','phase2_context_policy':REVISION}
    plan={'schema':'v7-fresh-rir-prefetch-plan-v1','plan_id':'single','parent_directory':str(parent),
          'cache_directory':str(parent/'cache'),'cache_identity':identity,'workers':2,'cpu_ids':[],
          'primary_positions_m':[],'references':[{'endpoint_id':'voice1','sample':123,'position_m':[1,2,3]}]}
    (root/'prefetch_plan.json').write_text(json.dumps(plan));calls=[]
    class Renderer:
        def __init__(self,*args):self.process_evidence=[]
        def render_independent_reference(self,positions,trial):
            calls.append(trial)
            return SimpleNamespace(samples=np.ones((1,4,8)),native_process_evidence=[])
    monkeypatch.setattr('avengine.timeline.v7_phase2_fresh_rir.FreshNativeContextRenderer',Renderer)
    result=run_prefetch_plan(request_path,{'inputs':{},'house':{},'expected_native_identity':{'scene':'s'},
          'source_position_m':[9,9,9],'evidence_role':'primary'}, {})
    assert json.loads(open(result).read())['status']=='pass'
    checkpoint=json.loads((parent/'independent_native_reference_checkpoint/voice1/00000123.json').read_text())
    assert calls==list(range(count)) and len(checkpoint['references'])==count and checkpoint['reference_count']==count
