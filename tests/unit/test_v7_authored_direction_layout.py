from copy import deepcopy
import math
import pytest
from avengine.timeline.v7_authored_direction_construction import construct_authored_direction_scene
pytestmark=pytest.mark.fast_unit


def test_visible_authored_speakers_stay_in_front_sector_and_keep_separate_depths(monkeypatch):
    original={'semantic_human_asset_ids':['a','b','c'],'offscreen_construction':{'offscreen_asset_ids':['c']},'measured_static_layout':{'valid':[{}, {}, {'position_m':[3.,1.,0.]}]},'listener_m':[0.,1.,0.],'camera_yaw_deg':0.,'native_view_roles':{'c':{'view':'outside'}},'automatic_construction':{},'production_scenario':{}}
    monkeypatch.setattr('avengine.timeline.v7_family_constructors.construct_family_scene',lambda *args,**kwargs:{'row':deepcopy(original)})
    monkeypatch.setattr('avengine.timeline.v7_family_constructors._request',lambda row,*args:row.update(scene_demand={}))
    monkeypatch.setattr('avengine.timeline.v7_production_inputs.validate_recipe',lambda *args:None)
    row=construct_authored_direction_scene({}, {}, {}, [], [], {}, {}, {},scene_id='new',seed=42)['row']
    angles=row['semantic_azimuth_targets_deg'];radii=row['native_attachment_distance_targets_m']
    assert all(22.5-abs(angles[a])>=15. for a in ['a','b'])
    assert all(abs(angles[a])<42.5 for a in ['a','b'])
    points=[(radii[a]*math.sin(math.radians(angles[a])),radii[a]*math.cos(math.radians(angles[a])))for a in ['a','b']]
    assert math.dist(*points)>=1.
    assert angles['c']==90. and row['native_view_roles']['c']['view']=='outside'
    assert row['phase2_authored_speech_schedule_s']['a']==[3.,13.,23.]
