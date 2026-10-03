"""Regressions for historical evidence shapes and qualified review plots."""
import json
import numpy as np
import pytest
from PIL import Image
from tools.rooms.v7_room_viz.common import display_levels
from tools.rooms.v7_room_viz.render import view_checks, mark_interior, top_view_scope
from tools.rooms.v7_room_viz.charts import band_stats, leakage, distance_series
from tools.rooms.v7_room_viz.page import verdict

pytestmark = pytest.mark.fast_unit


def polygon(x=0):
    return {'type':'Polygon','coordinates':[[[x,0],[x+2,0],[x+2,2],[x,2],[x,0]]]}


def room(name,y=0):
    return dict(room_label=name,room_type='living',floor_polygon=polygon(),
                placement=dict(camera_m=[0,y+1.5,0],source_1_m=[0,y+1.2,-3],source_2_m=[0,y+1.2,3]))


def test_historical_cpu_checks_keep_original_listener_and_get_room_label():
    check={'probe':'R10_source_1','listener_m':[6.875,.292859,-4.375],
           'source_m':[8.125,.122896,-5.625],'camera_forward':[.4226,0,-.9063]}
    job={'visual':{'checks':[check]},'result':{},'house':{'acoustic_rooms':[room('R10')]}}
    result=view_checks(job)
    assert result[0]['room_label']=='R10'
    assert result[0]['listener_m']==check['listener_m']
    assert result[0]['camera_forward']==check['camera_forward']
    assert len(result)>=2


def test_missing_floor_metrics_and_split_level_witness_remain_displayable():
    rows=[{'room_label':'R0','metrics':None,'legacy_room':{'bbox_xz_m':[[0,0],[8,8]],'floor_y_m':0}},
          {'room_label':'R1','metrics':None,'legacy_room':{'bbox_xz_m':[[0,0],[2,2]],'floor_y_m':1.4}}]
    house={'family':'hm3d','acoustic_rooms':[room('R0'),room('R1',1.4)]}
    levels=display_levels(rows,house)
    assert sorted(r for l in levels for r in l['acoustic_room_labels'])==['R0','R1']
    extra=[l for l in levels if l['kind']=='witness_height']
    assert len(extra)==1
    assert extra[0]['floor_y_m']==pytest.approx(1.4)
    assert extra[0]['acoustic_room_labels']==['R1']


def test_native_depth_annotation_has_json_safe_booleans_and_keeps_occlusion():
    check={'room_label':'R0'}
    job={'house':{'acoustic_rooms':[{'room_label':'R0','placement':{
          'source_1_m':[0,0,-3],'source_2_m':[0,0,3]}}]}}
    projection=[[1.1,0,0,0],[0,1.94,0,0],[0,0,-1,-.1],[0,0,-1,0]]
    cal={'view':np.eye(4).tolist(),'projection':projection}
    _,markers=mark_interior(Image.new('RGB',(1280,720)),np.full((720,1280),2.),cal,job,check)
    json.dumps(markers)
    assert markers[0]['inside_frustum'] is True
    assert markers[0]['center_depth_visible'] is False
    assert markers[1]['inside_frustum'] is False
    assert markers[0]['pixel']==[640.,360.]


def test_invalid_t30_falls_back_to_valid_t20_without_zero_filling():
    result={'rirs':[{'octaves':[
        {'center_hz':500,'t30':{'status':'invalid'},'t20':{'status':'valid','rt60_s':.7}},
        {'center_hz':1000,'t30':{'status':'invalid'},'t20':{'status':'invalid'}}]}]}
    assert band_stats(result)=={'500':{'median_s':.7,'min_s':.7,'max_s':.7,'n':1}}


def test_distance_plot_keeps_the_primary_far_source_and_uses_its_distances():
    rirs=[dict(room_label='R4',probe='R4_'+suffix,distance_m=d,direct_energy=e)
          for suffix,d,e in [('range_near',.8,1),('range_middle',1.7,.2),('source_1',2.6,.3),
                             ('source_2',2.2,.8),('alternate_listener_source_1_m',2.4,.1)]]
    result={'status':'review','reasons':['DIRECT_DISTANCE_ATTENUATION_OUTLIER'],'rirs':rirs,
            'distance_attenuation':[{'room_label':'R4','pressure_exponent':.5}]}
    assert [r['distance_m'] for r in distance_series(result)['R4']]==[.8,1.7,2.6]
    assert '2.60 米处比 1.70 米处更响' in verdict(result)


def test_unmeasured_leakage_is_not_rendered_as_zero_percent():
    plot=leakage({'status':'review','max_escape_fraction':None})
    assert '尚未完成这项测量' in plot
    assert '0.00%' not in plot


def test_overhead_framing_keeps_boundary_candidates_outside_the_room_polygon():
    level={'geometry':polygon(),'acoustic_room_labels':['R0']}
    job={'house':{'acoustic_rooms':[room('R0')]},'result':{},'boundary':{
        'escaped_ray_boundary_candidates':[{'room_label':'R0','candidate_boundary_midpoint_m':[8,-1,1]}]}}
    scope=top_view_scope(job,level)
    assert scope.bounds==(0.,-3.,8.,3.)
    assert level['geometry']==polygon()
