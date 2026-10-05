"""Contracts from concrete native identity and shared-disk failure cases."""
from copy import deepcopy
import os,json
from pathlib import Path
from types import SimpleNamespace
import pytest
pytestmark=pytest.mark.fast_unit


def _native():
    base={"sources":{"voice1":{"actor_id":1,"trajectory":[{"time_s":0,"position_m":[1,0,0]}]},
                     "voice2":{"actor_id":2,"trajectory":[{"time_s":0,"position_m":[2,0,0]}]}},
          "events":[{"endpoint_id":"voice1","source_id":1}]}
    rows=[{"endpoint_id":"voice1","actor_id":2},{"endpoint_id":"voice2","actor_id":1}]
    visual={"frames":[{"actors":rows},{"actors":deepcopy(rows)}],
            "observed_mouth_trajectories":{k:v["trajectory"] for k,v in base["sources"].items()}}
    return base,visual


def test_native_instance_follows_loaded_actor_at_actual_emitter():
    from avengine.timeline.v7_candidate_generation import member_scene
    base,visual=_native();normal=member_scene(base,visual,"v1_a0");swapped=member_scene(base,visual,"v1_a1")
    assert normal["events"][0]["source_id"]==2
    assert swapped["events"][0]["endpoint_id"]=="voice2"
    assert swapped["events"][0]["source_id"]==1
    assert base["events"][0]["source_id"]==1
    assert normal["sources"]["voice1"]["trajectory"]==swapped["sources"]["voice1"]["trajectory"]


def test_native_instance_cannot_change_mid_world():
    from avengine.timeline.v7_candidate_generation import member_scene
    base,visual=_native();visual["frames"][1]["actors"][0]["actor_id"]=99
    with pytest.raises(ValueError,match="identity changed"):member_scene(base,visual,"v0_a0")


def test_atomic_success_leaves_no_duplicate_sibling_and_never_replaces(tmp_path,monkeypatch):
    monkeypatch.delenv("AVENGINE_V7_STORAGE_BUDGET",raising=False)
    from avengine.timeline.continuous_pilot import write_json
    p=tmp_path/"receipt.json";write_json(p,{"observed":True})
    assert list(tmp_path.glob("*.partial"))==[]
    with pytest.raises(FileExistsError):write_json(p,{"observed":False})
    assert json.loads(p.read_text())=={"observed":True}


def test_budget_counts_hardlinks_once(tmp_path):
    from avengine.timeline.v7_storage_budget import allocated_bytes
    p=tmp_path/"cache";p.write_bytes(b"x"*8192);os.link(p,tmp_path/"duplicate")
    assert allocated_bytes(tmp_path)==p.stat().st_blocks*512


def test_disk_floor_and_task_budget_reject_before_output(tmp_path,monkeypatch):
    import avengine.timeline.v7_storage_budget as b
    cfg={"root":str(tmp_path),"filesystem":str(tmp_path),"minimum_free_bytes":50_000_000_000,
         "maximum_task_bytes":10_000_000_000,"concurrent_write_margin_bytes":64_000_000}
    monkeypatch.setenv(b.ENV_KEY,json.dumps(cfg))
    monkeypatch.setattr(b.shutil,"disk_usage",lambda p:SimpleNamespace(free=49_999_999_999))
    with pytest.raises(b.StorageBudgetExceeded,match="headroom stop"):b.check_write(tmp_path/"blocked.mp4",expected_bytes=1)
    monkeypatch.setattr(b.shutil,"disk_usage",lambda p:SimpleNamespace(free=60_000_000_000))
    monkeypatch.setattr(b,"allocated_bytes",lambda p:9_999_999_999)
    with pytest.raises(b.StorageBudgetExceeded,match="output budget stop"):b.check_write(tmp_path/"blocked.mp4",expected_bytes=1)
    assert not (tmp_path/"blocked.mp4").exists()


def test_48g_cannot_start_ninth_review_parent():
    from avengine.timeline.v7_storage_budget import configure_local_review,StorageBudgetExceeded
    with pytest.raises(StorageBudgetExceeded,match="eight parents"):
        configure_local_review(Path("/data/own_v7/test"),Path("/data/own_v7"),parents=9,workers=1)


def _retained_registry_fixture(tmp_path):
    import hashlib,json
    root=tmp_path/'upstream'
    (root/'evidence').mkdir(parents=True)
    (root/'registrations').mkdir()
    (root/'evidence/license_sources.json').write_text('[]')
    raw=json.dumps({'schema':'avengine_v7_human_roster_v1','artifact_root':str(root),
                    'majors':[],'summary':{'major_count':0,'minor_count':0}}).encode()
    digest=hashlib.sha256(raw).hexdigest()
    live=root/'registrations/human_roster.json';live.write_bytes(raw)
    snapshot=tmp_path/'input_snapshots/human'/digest/'registrations/human_roster.json'
    snapshot.parent.mkdir(parents=True);snapshot.write_bytes(raw)
    live.write_text('{"schema":"avengine_v7_human_roster_v2"}')
    return snapshot,digest


def test_exact_registry_archive_survives_upstream_update(tmp_path):
    from avengine.qa.v7_roster_intake import intake_roster
    snapshot,digest=_retained_registry_fixture(tmp_path)
    result=intake_roster(snapshot)
    assert result['source_registry_sha256']==digest
    assert result['records']==[]


def test_modified_registry_archive_is_not_admitted(tmp_path):
    from avengine.qa.v7_roster_intake import intake_roster
    snapshot,_=_retained_registry_fixture(tmp_path)
    snapshot.write_bytes(snapshot.read_bytes()+b' ')
    with pytest.raises(ValueError,match='roster root and registry disagree'):
        intake_roster(snapshot)


def _actual_query_key_fixture(tmp_path,wrong_association=False):
    import numpy as np
    import soundfile as sf
    from avengine.timeline.continuous_pilot import write_json
    members=[]
    for name in ['v0_a0','v0_a1','v1_a0','v1_a1']:
        v,a=name.split('_');folder=tmp_path/name/'w00';folder.mkdir(parents=True)
        meanings=['meaning_1','meaning_3'] if (v=='v0')==(a=='a0') else ['meaning_3','meaning_1']
        questions=[];gold=[]
        for index,color in enumerate(['蓝','黄']):
            qid=name+str(index)
            questions.append({'question_id':qid,'logical_question_id':qid,'protocol_panel':'v6',
                'qa_id':'QA-26','question_zh':color+'色上衣的人说了什么？','modality':'audiovisual','form':'open'})
            # These authority serials identify utterances; pairing by them
            # incorrectly compares different public people with equal meanings.
            gold.append({'question_id':qid,'logical_question_id':qid,'canonical_answer':meanings[index],
                'comparison_key':'utterance_'+meanings[index],'counterfactual_role':'binding'})
        for modality,qa,stem,answer in [
                ('vision','QA-14','画面中哪位更近？','blue' if v=='v0' else 'yellow'),
                ('audio','STARSS-SPEAKERS','有多少个不同讲话者？',2)]:
            qid=name+modality
            questions.append({'question_id':qid,'logical_question_id':qid,'protocol_panel':'simulation_fixture',
                'qa_id':qa,'question_zh':stem,'modality':modality,'form':'open'})
            gold.append({'question_id':qid,'logical_question_id':qid,'canonical_answer':answer,
                'comparison_key':qa,'counterfactual_role':modality})
        write_json(folder/'questions.json',questions);write_json(folder/'private_answers.json',gold)
        (folder/'video.mp4').write_bytes(v.encode())  # Algorithm-only fixture, never native video admission.
        audio=.1 if a=='a0' or wrong_association else .2
        sf.write(folder/'audio_foa.wav',np.full((160,4),audio),16000,subtype='FLOAT')
        members.append({'member':name,'selected_full_qa':True,'windows':[{'window_id':'w00'}],
            'windows_directory':str(folder.parent),'question_pool_directory':str(folder.parent)})
    row={'house':'simulation_fixture','room_id':'room','split':'train','window_count':1}
    base={'listener_m':[0,0,0],'camera_forward':[0,0,-1],'sources':{},'events':[]}
    return row,base,members


def test_binding_compares_public_question_not_hidden_utterance_serial(tmp_path,monkeypatch):
    monkeypatch.delenv('AVENGINE_V7_STORAGE_BUDGET',raising=False)
    from avengine.timeline.v7_candidate_generation import audit_parent_group
    result=audit_parent_group(*_actual_query_key_fixture(tmp_path))
    assert result['status']=='pass'
    assert len(result['verified_binding_logical_question_ids'])==8


def test_public_query_fix_keeps_wrong_audio_association_rejected(tmp_path,monkeypatch):
    monkeypatch.delenv('AVENGINE_V7_STORAGE_BUDGET',raising=False)
    from avengine.timeline.v7_candidate_generation import audit_parent_group
    with pytest.raises(ValueError,match='binding association did not change FOA'):
        audit_parent_group(*_actual_query_key_fixture(tmp_path,wrong_association=True))


def test_export_does_not_hardlink_external_readonly_sources(tmp_path,monkeypatch):
    import avengine.timeline.v7_storage_budget as budget
    from avengine.timeline.v7_alpha_delivery import _copy
    owned=tmp_path/'own';owned.mkdir();source=tmp_path/'readonly.wav';source.write_bytes(b'original')
    links=source.stat().st_nlink
    monkeypatch.setenv(budget.ENV_KEY,json.dumps({'root':str(owned),'filesystem':str(tmp_path),
        'minimum_free_bytes':50_000_000_000,'maximum_task_bytes':10_000_000_000,'concurrent_write_margin_bytes':0}))
    monkeypatch.setattr(budget.shutil,'disk_usage',lambda p:SimpleNamespace(free=60_000_000_000))
    _copy(source,owned/'packaged.wav')
    assert source.stat().st_nlink==links
    assert source.stat().st_ino!=(owned/'packaged.wav').stat().st_ino
    assert (owned/'packaged.wav').read_bytes()==source.read_bytes()


def test_native_listener_track_does_not_freeze_camera_rotation():
    from avengine.timeline.v7_full_question_delivery import bind_native_listener_frames
    from avengine.qa import unified_catalog as u
    frames=[{'frame_index':i,'position_m':[0.,0.,0.],'yaw_deg':yaw}
            for i,yaw in enumerate([0.,60.,-60.])]
    raw={'clock':{'frame_count':3},'frame_readbacks':{'camera':frames},
         'listener':{'position_m':[0.,0.,0.],'yaw_deg':0.}}
    bad=u._listener_block(raw,raw['frame_readbacks'],None,{},frame_count=3)
    assert [u._listener_azimuth([0.,0.,-1.],bad,i) for i in range(3)]==[0.,0.,0.]
    bind_native_listener_frames(raw)
    good=u._listener_block(raw,raw['frame_readbacks'],None,{},frame_count=3)
    assert good['yaw_deg']==[0.,60.,-60.]
    assert [u._listener_azimuth([0.,0.,-1.],good,i) for i in range(3)]==pytest.approx([0.,-60.,60.])
    raw['frame_readbacks']['camera'][1]['yaw_deg']=99.
    assert raw['listener']['frames'][1]['yaw_deg']==60.


def test_native_listener_track_rejects_incomplete_native_frames():
    from avengine.timeline.v7_full_question_delivery import bind_native_listener_frames
    with pytest.raises(ValueError,match='every indexed camera frame'):
        bind_native_listener_frames({'clock':{'frame_count':3},'frame_readbacks':{'camera':[
            {'frame_index':0,'position_m':[0.,0.,0.],'yaw_deg':0.}]}})


@pytest.mark.parametrize('row', [{'split':'test','members':['v0_a0']}, {'split':'train','members':['v0_a0','v0_a1','v1_a0','v1_a1']}])
def test_requalification_cannot_admit_test_or_unverified_counterfactuals(row):
    from avengine.timeline.v7_native_requalification import require_single_training_or_dev
    with pytest.raises(ValueError,match='single train/dev worlds'):
        require_single_training_or_dev(row)


def test_authority_key_ignores_candidate_enumeration_only():
    from avengine.timeline.v7_candidate_generation import authority_physical_query_key
    item={'qa_id':'QA-14'}
    a={'comparison_key':'QA-14____3____16','authority_question_id':'qa_14__left__frame_3__time_0.2__1_2_3'}
    b={'comparison_key':'QA-14____3____17','authority_question_id':'qa_14__right__frame_3__time_0.2__1_2_3'}
    assert authority_physical_query_key(item,a,'left','fallback')==authority_physical_query_key(item,b,'right','fallback')
    bad={**b,'authority_question_id':'qa_14__right__frame_3__time_0.2__1_4_3'}
    assert authority_physical_query_key(item,a,'left','fallback')!=authority_physical_query_key(item,bad,'right','fallback')
    # Unknown/unstructured authority must retain the strict original key.
    assert authority_physical_query_key(item,{'comparison_key':'different actual target'},'right','fallback')=='different actual target'
