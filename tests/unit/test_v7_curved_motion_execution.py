from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.roster_motion.navmesh_execution import REVISION,SPEED_RANGES,prepare_scene_motion,curved_route_proposals
pytestmark=pytest.mark.fast_unit

def test_scene_speed_is_selected_in_each_species_band_without_changing_sound_roles():
 assets=[{'asset_id':k,'kind':'human'if k=='human'else'animal','species_id':k if k!='slide'else'cat'}for k in SPEED_RANGES]
 row={'scene_seed':71,'motion_execution_revision':REVISION,'source_classes':{'human':['speech_playback']},'motion_requests':{k:{'action':'walk',**({'animal_motion_admission':{'mode':'slide_no_animation'}}if k=='slide'else{})}for k in SPEED_RANGES}}
 result=prepare_scene_motion(row,assets)
 for k,bounds in SPEED_RANGES.items():assert bounds[0]<=result['motion_requests'][k]['speed_m_s']<=bounds[1]
 assert result['source_classes']==row['source_classes'];assert 'speed_m_s'not in row['motion_requests']['human']

def test_curved_route_follows_native_navmesh_at_scene_speed_and_has_smooth_tangent():
 class Path:
  pass
 class Pathfinder:
  def snap_point(self,p):return np.asarray(p,float)
  def find_path(self,p):p.geodesic_distance=float(np.linalg.norm(p.requested_end-p.requested_start));return True
 entry={'applied_root_input':np.eye(4),'declaration':{'kind':'human'}}
 video=SimpleNamespace(actors=[entry],pathfinder=Pathfinder(),hs=SimpleNamespace(ShortestPath=Path))
 scene={'seed':8,'listener_m':[6,1.3,0],'production_scenario':{'motion_floor_polygon':{'type':'Polygon','coordinates':[[[-4,-4],[4,-4],[4,4],[-4,4],[-4,-4]]]}}}
 route=next(curved_route_proposals(video,entry,scene,np.arange(901)/30.,np.array([0,0,-1]),{'speed_m_s':1.}))
 speed=np.linalg.norm(np.diff(route.positions,axis=0),axis=1)*30
 assert np.max(np.abs(speed-1.))<.002
 assert np.ptp(route.yaws)>2*np.pi
 assert route.receipt['maximum_horizontal_snap_error_m']==0 and len(route.receipt['native_waypoint_connections'])==8
 assert np.max(np.abs(np.diff(route.yaws)))<.1
