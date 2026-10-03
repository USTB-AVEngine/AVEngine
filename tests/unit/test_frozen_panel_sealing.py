"""Hermetic scoring-private seal tests; no model predictions/evaluation are run."""
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark=pytest.mark.fast_unit
path=Path(__file__).resolve().parents[2]/'tools/qa/score_frozen_panel.py'
spec=importlib.util.spec_from_file_location('frozen_scoring_script',path)
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)

def make_shard(root,group):
    (root/'inputs').mkdir(parents=True)
    scorer.write(root/'inputs/shard_config.json',{'output':str(root),'split':'test','panel_id':'fixture','seed':42,'groups_per_house':1,'houses':[]})
    for directory in ('scoring_private/answers_parts','scoring_private/certificates','public/groups'):(root/directory).mkdir(parents=True)
    members=[]
    for vi in range(2):
        for ai in range(2):
            sid=f'{group}_{vi}{ai}';folder=root/'public/groups'/sid;folder.mkdir()
            row={'sample_id':sid,'question':'说目标句的是哪个人？','qa_id':'QA-27','form':'open','split':'test',
                 'video':f'groups/{sid}/video.mp4','foa':f'groups/{sid}/foa.wav','binaural':f'groups/{sid}/binaural.wav',
                 'mono':f'groups/{sid}/mono.wav','frame_indices':list(range(40)),'frame_times_s':[i/15 for i in range(40)]}
            scorer.write(folder/'request.json',row)
            members.append({'sample_id':sid,'group':group,'video_index':vi,'audio_index':ai,'question':row['question'],
                            'physical_person_id':f'{group}:source{ai+1}','answer_label':'blue' if vi==ai else 'green'})
    scorer.write(root/'scoring_private/answers_parts'/f'{group}.json',{'members':members})
    scorer.write(root/'scoring_private/certificates'/f'{group}.json',{'group':group,'fixture':'not_native_certificate',
      'joint_solvability':{'appearance':[{'actors':{'source1':{'value':'blue'},'source2':{'value':'green'}}},
                                      {'actors':{'source1':{'value':'green'},'source2':{'value':'blue'}}}]}})
    return root

def test_seal_checkerboard_commitment_and_no_overwrite(tmp_path):
    root=make_shard(tmp_path/'a','parent_a');result=scorer.seal(root,1)
    assert result['checkerboard_groups']==1 and result['members']==4
    assert scorer.audit(root)['status']=='pass'
    assert (root/'scoring_private').stat().st_mode&0o777==0o700
    assert (root/'scoring_private/answers.json').stat().st_mode&0o777==0o400
    with pytest.raises(FileExistsError):scorer.seal(root,1)
    path=root/'scoring_private/certificates/parent_a.json';path.chmod(0o600)
    changed=json.loads(path.read_text());changed['fixture']='modified fixture';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='commitment'):scorer.audit(root)

def test_wrong_checkerboard_is_rejected_as_a_whole(tmp_path):
    root=make_shard(tmp_path/'bad','parent_bad')
    path=root/'scoring_private/answers_parts/parent_bad.json';data=json.loads(path.read_text());data['members'][0]['answer_label']='green';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checkerboard'):scorer.seal(root,1)
    assert not (root/'public/COMMITMENTS.sha256').exists()

def test_sealed_shards_assemble_with_one_private_reader(tmp_path):
    a=make_shard(tmp_path/'a','parent_a');b=make_shard(tmp_path/'b','parent_b')
    scorer.seal(a,1);scorer.seal(b,1)
    result=scorer.assemble(tmp_path/'assembled',[a,b],2)
    assert result['groups']==2 and result['members']==8
    assert scorer.audit(tmp_path/'assembled')['status']=='pass'

def test_duplicate_parent_shards_refused(tmp_path):
    a=make_shard(tmp_path/'a','parent');b=make_shard(tmp_path/'b','parent')
    scorer.seal(a,1);scorer.seal(b,1)
    with pytest.raises(ValueError,match='overlaps shards'):scorer.assemble(tmp_path/'assembled',[a,b],2)

def test_checkerboard_with_wrong_clothing_grounding_is_rejected(tmp_path):
    root=make_shard(tmp_path/'bad','parent_bad')
    path=root/'scoring_private/answers_parts/parent_bad.json';data=json.loads(path.read_text())
    for member in data['members']:member['answer_label']='green' if member['answer_label']=='blue' else 'blue'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='pixel-reviewed clothing'):scorer.seal(root,1)
