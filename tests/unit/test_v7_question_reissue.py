import pytest
from tools.timeline.reissue_v7_footstep_questions import prefix_media
pytestmark=pytest.mark.fast_unit

def test_revision_keeps_physical_parent_identity_and_versions_all_model_media_paths():
    before=[{'independent_parent_scene_id':'same-parent','question_id':'same-question','video':'parents/p/windows/w/video.mp4','audio_foa':'parents/p/windows/w/audio_foa.wav','audio_mono':'parents/p/windows/w/audio_mono.wav','private_evidence':'parents/p/private/w.json'}]
    after=prefix_media(before,'question_reissues/new-revision')
    assert after[0]['independent_parent_scene_id']=='same-parent'
    assert after[0]['question_id']=='same-question'
    for field in ('video','audio_foa','audio_mono','private_evidence'):
        assert after[0][field]=='question_reissues/new-revision/'+before[0][field]
    assert before[0]['video']=='parents/p/windows/w/video.mp4'


def test_finalization_dispatch_changes_only_after_admitted_input_or_frozen_policy_changes(tmp_path):
    import json
    from tools.timeline.finalize_v7_acceptance import delivery_fingerprint
    folder=tmp_path/'evidence/jobs';folder.mkdir(parents=True)
    m={'output_root':str(tmp_path),'code_commit':'a'*40,'parents':[{'slot':0},{'slot':1}]}
    (folder/'000.json').write_text(json.dumps({'status':'passed','questions':5}))
    (folder/'001.json').write_text(json.dumps({'status':'running','cpu_s':1}))
    initial=delivery_fingerprint(m)
    (folder/'001.json').write_text(json.dumps({'status':'running','cpu_s':100}))
    assert delivery_fingerprint(m)==initial
    (folder/'001.json').write_text(json.dumps({'status':'passed','questions':3}))
    assert delivery_fingerprint(m)!=initial
    old=delivery_fingerprint(m);m['code_commit']='b'*40
    assert delivery_fingerprint(m)!=old
