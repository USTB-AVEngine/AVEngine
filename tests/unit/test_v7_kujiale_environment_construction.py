import json
from pathlib import Path
import pytest
from avengine.timeline.v7_kujiale_environment_construction import native_class_candidates,bind_inventory_project,separate_environment_schedule,_clear_segment
import numpy as np
pytestmark=pytest.mark.fast_unit

def test_required_native_class_cannot_be_replaced_by_unimported_or_silent_asset(tmp_path):
    project=tmp_path/'P.uproject';package=tmp_path/'Content/Dog/D.uasset';package.parent.mkdir(parents=True);package.write_bytes(b'UE original dog')
    valid={'asset_id':'dog','ue_native_physical_status':'pass','ue_import_status':'pass','silent_decor':False,'allowed_v7_classes_candidate':['dog_bark'],'ue_asset_paths':['/Game/Dog/D.D'],'dimensions_m':[.3,.4,.5]}
    result=native_class_candidates({'assets':[valid]},['dog_bark'],project,0);assert result[0][1][0]['asset_id']=='dog'
    valid['silent_decor']=True
    with pytest.raises(ValueError,match='no admitted'):native_class_candidates({'assets':[valid]},['dog_bark'],project,0)
    valid['silent_decor']=False;package.unlink()
    with pytest.raises(ValueError,match='no admitted'):native_class_candidates({'assets':[valid]},['dog_bark'],project,0)

def test_runtime_project_view_preserves_source_registration(tmp_path):
    source=tmp_path/'source.json';source.write_text(json.dumps({'assets':[{'asset_id':'dog','ue_project':'original.uproject'}]}));before=source.read_bytes();inputs={'kujiale_ue_asset_admission':str(source)};house={'input':{'ue_config':{'uproject':'runtime.uproject'}}}
    result=bind_inventory_project(inputs,house,tmp_path);bound=json.loads(Path(result['kujiale_ue_asset_admission']).read_text());assert source.read_bytes()==before and bound['assets'][0]['original_registered_ue_project']=='original.uproject'and bound['assets'][0]['ue_project']=='runtime.uproject'

def test_required_animal_starts_after_whole_human_speech_and_stays_in_each_cycle():
    sounds=[{'scenario_id':'s','sound_asset_id':str(i),'role_id':i,'voice_reference':{'gender':'female','speaker_id':'v'+str(i)},'synthesis_readback':{'duration_s':d}}for i,d in enumerate([1.72,1.76])];row={'scene_seed':0,'production_scenario':{'gender':'F'}}
    end,effect=separate_environment_schedule(row,{'sounds':sounds});assert end<=4.8 and effect>=end+.499 and effect+2<8.35
    assert row['production_scenario']['environment_start_offsets_s']==[effect]

def test_camera_to_animal_segment_checks_registered_human_envelope():
    assert not _clear_segment(np.array([0,1.,0]),np.array([0,.3,4]),[(np.array([-.4,0,1.5]),np.array([.4,1.9,2.3]))])
    assert _clear_segment(np.array([0,1.,0]),np.array([2,.3,4]),[(np.array([-.4,0,1.5]),np.array([.4,1.9,2.3]))])
