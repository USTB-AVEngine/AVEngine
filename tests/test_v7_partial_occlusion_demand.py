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
