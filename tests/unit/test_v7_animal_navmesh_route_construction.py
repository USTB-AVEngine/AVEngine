from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import box
from avengine.assets.animal_motion.production_episode import route_navmesh_preview,native_walkable_starts
pytestmark=pytest.mark.fast_unit

def test_complete_segment_filter_catches_a_hole_between_coarse_samples():
 class Pathfinder:
  def snap_point(self,p):
   p=np.array(p).copy()
   if .329<=p[0]<=.331:p[0]+=.01
   return p
 assert not route_navmesh_preview(Pathfinder(),[0,0,0],[1,0,0])

def test_visible_floor_search_constructs_at_least_200_distinct_native_starts():
 video=SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)))
 scene={'listener_m':[0,1.3,0],'camera_yaw_knots':[{'time_s':0,'yaw_deg':0}]}
 starts,receipt=native_walkable_starts(video,scene,box(-5,-8,5,-1),np.array([0.,0.,-3.]),[],[0,.3,0],0.)
 assert len(starts)>=200
 assert receipt['visible_walkable_starts']==len(starts)
 assert receipt['headings_per_start']==16 and receipt['complete_segment_samples']==901
 assert len({tuple(p[[0,2]])for p in starts})==len(starts)
