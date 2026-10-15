import json
import numpy as np
import pytest
from avengine.timeline.v7_s_batch_checks import open_delivered_media
pytestmark=pytest.mark.fast_unit


def test_reader_reuses_only_equal_video_bytes_and_rechecks_every_audio(tmp_path,monkeypatch):
    rows=[]
    for i,video in enumerate([b'same-video',b'same-video',b'changed-video']):
        for filename,data in [(f'v{i}.mp4',video),(f'f{i}.wav',b'foa'),(f'm{i}.wav',b'mono')]:
            (tmp_path/filename).write_bytes(data)
        rows.append({'window_id':str(i),'video':f'v{i}.mp4','audio_foa':f'f{i}.wav','audio_mono':f'm{i}.wav'})
    (tmp_path/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows));commands=[];reads=[]
    def check(command):
        commands.append(command)
        if command[0]=='ffprobe':return json.dumps({'streams':[{'width':1,'height':1,'nb_read_frames':150,'duration':'10'}]}).encode()
        return b'RGB'
    def read(path,**kwargs):
        reads.append(str(path));return np.zeros((160000,4 if path.name.startswith('f')else 1),dtype='float32'),16000
    monkeypatch.setattr('avengine.timeline.v7_s_batch_checks.subprocess.check_output',check)
    monkeypatch.setattr('avengine.timeline.v7_s_batch_checks.sf.read',read)
    result=open_delivered_media(tmp_path)
    assert result['status']=='pass'and result['actual_decoded_windows']==3
    assert result['actual_unique_video_decodes']==2 and result['same_video_bytes_reuse_windows']==1
    assert len(reads)==6 and len(commands)==4
    assert all(command[command.index('-threads')+1]=='1'for command in commands)
