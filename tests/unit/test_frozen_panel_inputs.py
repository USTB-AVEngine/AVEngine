from copy import deepcopy
import numpy as np
import pytest
import soundfile as sf
from avengine.qa import frozen_panel_inputs as inputs

pytestmark=pytest.mark.fast_unit

@pytest.fixture
def sample(tmp_path,monkeypatch):
    public=tmp_path/'public';folder=public/'groups'/'transport';folder.mkdir(parents=True)
    n=160000;wave=np.zeros((n,2),np.float32);wave[159900:,0]=.25;wave[159900:,1]=.75
    foa=np.zeros((n,4),np.float32);foa[-100:,0]=.5;foa[-100:,3]=.25
    for name,a in [('binaural',wave),('mono',wave.mean(axis=1)),('foa',foa)]:sf.write(folder/f'{name}.wav',a,16000,subtype='FLOAT')
    (folder/'video.mp4').write_bytes(b'hermetic-video-placeholder')
    frames=np.arange(150*4*4*3,dtype=np.uint8).reshape(150,4,4,3);pts=(np.arange(150)/15).tolist()
    monkeypatch.setattr(inputs,'video_pixels',lambda path:(frames,pts,10.))
    indices=np.rint(np.linspace(0,149,40)).astype(int).tolist()
    row={'sample_id':'transport','question':'说目标句的是哪个人？','qa_id':'QA-27','form':'open','split':'pilot',
      **{k:f'groups/transport/{k}.wav' for k in ('foa','binaural','mono')},'video':'groups/transport/video.mp4',
      'frame_indices':indices,'frame_times_s':[pts[i] for i in indices]}
    return public,row

def test_entire_audio_including_last_100_samples_is_kept(sample):
    public,row=sample;payload=inputs.build_model_payload(public,row,'level2')
    assert payload['audio'].shape==(160000,) and np.all(payload['audio'][-100:]==.5)
    assert payload['foa'].shape==(160000,4) and payload['foa'][-1,3]==.25
    assert set(payload)=={'text','frames','audio','foa','frame_times_s','fps'}
    assert 'transport' not in payload['text']

@pytest.mark.parametrize('field',['group_id','member_id','correct_answer','physical_person_id','cache_key'])
def test_private_fields_are_rejected_before_media_loading(sample,field):
    public,row=sample;row={**row,field:'leak'}
    with pytest.raises(ValueError,match='whitelist'):inputs.build_model_payload(public,row,'raw_av')

def test_evidence_cache_follows_media_and_never_transport_or_question(sample):
    public,row=sample;changed=deepcopy(row);changed.update(sample_id='a-different-id',question='另一个公开问题')
    first=inputs.evidence_keys(public,row,'producer',{'revision':1})
    assert first==inputs.evidence_keys(public,changed,'producer',{'revision':1})
    wave,rate=sf.read(public/row['foa'],dtype='float32');wave[-1,1]=.125;sf.write(public/row['foa'],wave,rate,subtype='FLOAT')
    second=inputs.evidence_keys(public,row,'producer',{'revision':1})
    assert first['M1']!=second['M1'] and first['cross_modal']!=second['cross_modal'] and first['M2']==second['M2']

def test_level1_refuses_missing_or_wrong_evidence_provenance(sample):
    public,row=sample
    with pytest.raises(ValueError,match='no oracle'):inputs.build_model_payload(public,row,'level1')
    with pytest.raises(ValueError,match='provenance'):
        inputs.build_model_payload(public,row,'level1',producer='p',settings={},evidence={'keys':{'M1':'parent-world'},'heard':'prediction'})

def test_mono_conversion_and_path_escape_are_detected(sample,tmp_path):
    public,row=sample;outside=tmp_path/'private.wav';sf.write(outside,np.zeros(160000,np.float32),16000,subtype='FLOAT')
    changed={**row,'mono':'../private.wav'}
    with pytest.raises(ValueError,match='escape'):inputs.build_model_payload(public,changed,'raw_av')
    sf.write(public/row['mono'],np.zeros(160000,np.float32),16000,subtype='FLOAT')
    with pytest.raises(ValueError,match='conversion'):inputs.build_model_payload(public,row,'raw_av')

def test_frame_receipt_and_production_split(sample):
    public,row=sample;row={**row,'split':'test'}
    assert inputs.build_model_payload(public,row,'av_mono')['frames'].shape[0]==40
    row=deepcopy(row);row['frame_times_s'][-1]+=.01
    with pytest.raises(ValueError,match='timestamps'):inputs.build_model_payload(public,row,'av_mono')
