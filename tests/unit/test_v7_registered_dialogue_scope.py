from copy import deepcopy
import pytest
from avengine.timeline.v7_registered_dialogue_scope import registered_dialogue_view
pytestmark=pytest.mark.fast_unit

def case():
 human={'actor_id':1,'endpoint_id':'voice1','asset_id':'registered_human'}
 dog={'actor_id':1001,'endpoint_id':'environment1','asset_id':'dog','entity_class':'static_animal','moving':False,'action_id':'registered_static_support','root_max_error':0.,'emitter_position_m':[1,0,0],'native_foot_contact':{'status':'pass','gap_m':.002,'tolerance_m':.012}}
 frames=[{'frame_index':i,'actors':[deepcopy(human),deepcopy(dog)]}for i in range(450)]
 physical={'status':'pass','frame_count':450,'all_sources_have_drawable_geometry':True,'source_labels_modified':False,'current_joint_admission':{'environment1':{'status':'pass','asset_id':'dog','emitter':{'actual_m':[1,0,0]}}},'frames':[{'frame_index':i,'entities':[{'endpoint_id':'environment1','asset_id':'dog','source_id':1001,'frame_index':i,'source_geometry_present':True,'root_error_m':0.,'emitter_position_m':[1,0,0],'visible_pixels':80}]}for i in range(450)]}
 return {'require_current_native_environment':True,'sources':{'environment1':{'native_asset_id':'dog','native_visual_kind':'registered_physical'}}},{'frames':frames,'native_physical_environment':physical}

def test_human_certificate_keeps_all_original_dog_pixels_and_readbacks():
 scene,visual=case();before=deepcopy(visual);view,proof=registered_dialogue_view(scene,visual)
 assert visual==before and all(len(f['actors'])==2 for f in visual['frames'])
 assert all(len(f['actors'])==1 for f in view['frames'])
 assert 'body_to_world_m'not in view['frames'][0]['actors'][0] # missing human evidence is never fabricated
 assert proof['complete_native_frames_checked']==450 and proof['other_physical_entities'][0]['asset_id']=='dog'

@pytest.mark.parametrize('broken',['identity','emitter','clock'])
def test_other_entity_must_keep_original_native_identity_clock_and_attachment(broken):
 scene,visual=case()
 if broken=='identity':scene['sources']['environment1']['native_asset_id']='unregistered'
 elif broken=='emitter':visual['frames'][200]['actors'][1]['emitter_position_m']=[1.01,0,0]
 else:visual['native_physical_environment']['frames'][200]['frame_index']=199
 with pytest.raises(ValueError):registered_dialogue_view(scene,visual)
