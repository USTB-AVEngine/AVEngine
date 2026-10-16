"""Rule B keeps real animal questions when an optional human cannot be placed."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import box,mapping
from avengine.rooms.walkable_map import WalkableMap,SPECIES_RADII_M
from avengine.construction.joint import FamilyRequest
from avengine.construction.joint.context import RoomContext
from avengine.construction.joint.certification import certify_plan
from avengine.construction.joint.families.motion_animal import MotionAnimalPlugin
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('kind,species,mode',[('dog_walk','dog','walk_animation'),('cat_walk','cat','walk_animation'),('animal_slide','cat','slide_no_animation')])
@pytest.mark.parametrize('bouts',[2,3])
def test_joint_camera_source_windows_keep_all_original_motion_families(tmp_path,kind,species,mode,bouts):
    layers={key:np.ones((300,320),bool) for key in SPECIES_RADII_M}
    metadata={'version':'walkable-map-v1','room_id':'fixture','species_radii_m':SPECIES_RADII_M,
        'footprint_policy':'independent-radius-coverage-v2','source_identity':[]}
    walkable=WalkableMap(np.array([-8.,-12.]),.05,0.,layers,metadata,np.zeros((300,320)))
    walkable.path=str(tmp_path/'map_fixture.npz')
    room={'room_label':'fixture','floor_polygon':mapping(box(-8,-12,8,3)),'measured_floor_y_m':0.,'placement':{'camera_m':[0.,1.1,0.]}}
    context=RoomContext(room,walkable_map=walkable,pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p,float)),camera_limit=12,camera_heights_m=(1.1,),width=1280,height=720)
    animal={'asset_id':'animal','kind':'animal','species_id':species,'walk_allowed':mode=='walk_animation','dimensions_m':[.24,.6,.7],'mouth_local_glb_m':[0.,.3,0.]}
    human={'asset_id':'speaker','kind':'human','species_id':'human','walk_allowed':True,'visual_gender':'male'}
    duration=2.3
    def recording(key,cls,seconds):
        return {'sound_asset_id':key,'sound_class':cls,'header':{'sample_rate_hz':16000,'sample_count':round(seconds*16000)},'normalized_path':key+'.wav','screens':{'automatic_activity':{'activity_intervals_samples':[[0,round(seconds*16000)]]}}}
    record=recording('call','dog_bark' if species=='dog' else 'cat_meow',duration)
    speech=recording('tts','speech_playback',1.5);speech['source_type']='author_tts'
    data={'animal':animal,'companion':human,'record':record,'program':None,'animal_activity_s':((.1,duration-.07),),'animal_duration_s':duration,
        'speech':speech,'speech_activity_s':((0.,1.5),),'speech_duration_s':1.5,'scenario_id':'fixture','kind':kind}
    def geometry(ctx,house,runtime,asset,authority,actor,route,directory):
        directory.mkdir(parents=True);filename=directory/'fixture.json'
        filename.write_text(json.dumps({'record':{'world_roots_m':actor.positions_m.tolist(),'source_frames':[{'mouth_m':[0.,.3,0.]} for _ in range(901)],'yaw_rad':0.,'yaw_rad_per_frame':[0.]*901}}))
        actor.metadata.update(joint_animal_prepared_geometry=str(filename),native_yaw_rad=0.)
    plugin=MotionAnimalPlugin(data,{'scene_id':'fixture','scene_seed':1},[animal,human],[record,speech],{'animal_motion':{'animal':{'mode':mode}}},{'house':'fixture'},{'joint_max_geometry_preparations':1,'motion_animal_endpoint_solver':True,'motion_animal_joint_camera_solver':True,'joint_budget_s':30.},tmp_path/'geometry',geometry_preparer=geometry)
    request=FamilyRequest('motion_animal','nearer',('animal',),{'movement_bouts':bouts})
    plan=next(plugin.candidates(context,request))
    assert 'animal' in plan.actors
    assert certify_plan(context,plan).passed
    checks=list(plugin.verify(context,request,plan))
    assert checks and all(c.status=='pass' for c in checks),[(c.name,c.reason) for c in checks if c.status!='pass']
    row=plan.metadata['row']
    assert all(row['motion_quartet_planned_verification'].values())
    assert len(row['motion_during_condition_plan']['plans'])==4*bouts
    assert plan.metadata['retained_movement_bouts']==bouts
    assert plan.metadata['animal_endpoint_solution']['solver']=='physical endpoints -> route_between -> construct_route -> camera_solve'

    assert plan.metadata['source_file_starts_s']['animal'][0]<1.3
    assert plan.metadata['animal_endpoint_solution']['sound_window_role_mode'] in {'preferred_existing_call_margin','original_call_margin'}
