from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from avengine.assets.animal_motion.joint_execution import load_prepared_geometry

pytestmark=pytest.mark.fast_unit


def prepared(tmp_path):
    pose=tmp_path/'original_pose.glb';pose.write_bytes(b'unit fixture original pose')
    roots=np.column_stack((np.zeros(901),np.zeros(901),np.linspace(-6.,-3.,901)))
    record={'motion_mode':'slide_no_animation','lineage':{'admitted_native_motion':'native.json','master':'master.glb'},
            'world_roots_m':roots.tolist(),'pose_glbs':[str(pose)]*901}
    payload={'asset_id':'animal','house':'house','room_id':'room','listener_m':[0.,1.1,0.],'camera_yaw_deg':0.,'record':record}
    path=tmp_path/'prepared.json';path.write_text(json.dumps(payload))
    request={'joint_animal_prepared_geometry':str(path),
             'animal_motion_admission':{'mode':'slide_no_animation','native_motion':'native.json','source_master':'master.glb'},
             'motion_quartet_route':{'positions_xz_m':roots[:,[0,2]].tolist()}}
    entry={'declaration':{'asset_id':'animal'}}
    scene={'listener_m':[0.,1.1,0.],'camera_yaw_knots':[{'time_s':0,'yaw_deg':0.},{'time_s':30,'yaw_deg':0.}]}
    house={'house':'house','room':{'room_label':'room'}}
    return entry,scene,house,request,path,roots


def test_execution_consumes_all_original_pose_buffers_and_roots_without_replanning(tmp_path):
    entry,scene,house,request,path,roots=prepared(tmp_path)
    before=path.read_bytes();result=load_prepared_geometry(entry,scene,house,request)
    np.testing.assert_array_equal(result['world_roots_m'],roots)
    assert result['joint_geometry_executed_without_replanning']
    result['world_roots_m'][0][0]=99.
    assert path.read_bytes()==before


@pytest.mark.parametrize('change', ['route','camera_position','camera_yaw','source_mode','room'])
def test_changed_solved_inputs_are_rejected_before_any_native_pose_execution(tmp_path,change):
    entry,scene,house,request,path,roots=prepared(tmp_path)
    if change=='route':request['motion_quartet_route']['positions_xz_m'][100][0]+=.01
    elif change=='camera_position':scene['listener_m'][0]+=.01
    elif change=='camera_yaw':scene['camera_yaw_knots'][0]['yaw_deg']=1.
    elif change=='source_mode':request['animal_motion_admission']['mode']='walk_animation'
    else:house['room']['room_label']='another_room'
    with pytest.raises(ValueError):load_prepared_geometry(entry,scene,house,request)
