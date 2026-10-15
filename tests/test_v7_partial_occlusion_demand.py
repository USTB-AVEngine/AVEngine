import numpy as np
import pytest
from avengine.qa.pixel_visibility import _frame_truth
from avengine.capture.v7_occlusion_demand import matches_partial_layout_request
pytestmark=pytest.mark.fast_unit

def test_original_partial_pixel_truth_can_drive_an_obvious_registered_layout():
    target=np.ones((4,4),dtype=np.int32)
    modal=target.copy();modal[0,:]=2
    truth=_frame_truth(modal_mask=modal,target_only_mask=target,target_semantic_id=1,frame_index=0)
    assert truth['state']=='visible_occluded' and truth['occlusion_fraction']==.25
    frame={'occluder_instance_ids':['2'],'unknown_occluder_pixels':0}
    assert matches_partial_layout_request(truth,frame,'2',.25)
    assert not matches_partial_layout_request(truth,frame,'3',.25)
    frame['unknown_occluder_pixels']=1
    assert not matches_partial_layout_request(truth,frame,'2',.25)


def test_reviewed_occluder_keeps_an_english_label_for_actual_person_and_device():
    from avengine.timeline.v7_registered_audio import reviewed_occluder_display_label
    human=reviewed_occluder_display_label({'value':'blue','label_zh':'蓝色上衣的人'},{'entity_class':'articulated_human'})
    device=reviewed_occluder_display_label({'value':'black','label_zh':'黑色音箱'},{'entity_class':'loudspeaker'})
    assert 'blue'in human and 'person'in human
    assert device=='the black loudspeaker'
    assert reviewed_occluder_display_label({'value':'black','label_en':'the reviewed tower speaker'},{'entity_class':'loudspeaker'})=='the reviewed tower speaker'
