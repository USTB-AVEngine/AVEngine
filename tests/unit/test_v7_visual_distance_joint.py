from types import SimpleNamespace
import pytest
from avengine.timeline.v7_visual_distance_joint import native_distance_witness,visual_distance_row
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('gap,passed',[(.49,False),(.5,True),(.6,True)])
def test_actual_three_dimensional_native_distance_uses_original_answer_margin(gap,passed):
    entries=[{'declaration':{'asset_id':s}}for s in ('near','far')]
    points={'near':[0.,0.,-2.],'far':[0.,0.,-2.-gap]}
    video=SimpleNamespace(_apply_entry=lambda e,*a,**k:{'emitter_position_m':points[e['declaration']['asset_id']]})
    checks=native_distance_witness(video,{'listener_m':[0.,0.,0.]},entries,('near','far'))
    assert (checks[0].status=='pass')is passed


def test_row_keeps_two_visible_quiet_targets_and_one_real_complete_tts_source(monkeypatch):
    from avengine.construction.joint import CameraPose
    from avengine.timeline.v7_speech_construction import schedule_epochs
    people=[{'asset_id':s}for s in ('near','far','speaker')]
    epochs=[[{'scenario_id':f's{i}','role_id':'role','sound_asset_id':f'sound{i}',
        'synthesis_readback':{'duration_s':2.,'sample_count':32000,'sample_rate_hz':16000}}]for i in range(3)]
    schedule=schedule_epochs(epochs,['speaker']);plan=SimpleNamespace(camera=CameraPose((0.,1.25,0.),35.))
    validated=[];monkeypatch.setattr('avengine.timeline.v7_production_inputs.validate_recipe',lambda row,*args:validated.append(row))
    row=visual_distance_row({'house':'fresh_house','room_id':'fresh_room','family':'hm3d'},plan,people,epochs,schedule,{}, {},people,scene_id='fresh',seed=4)
    assert row['asset_ids']==['near','far','speaker']and row['silent_asset_ids']==['near','far']
    assert row['source_classes']=={'speaker':['speech_playback']}and row['semantic_human_asset_ids']==['speaker']
    assert row['native_view_roles']['speaker']['view']=='outside'and row['native_view_roles']['near']['view']=='inside'
    assert row['speech_schedule']==schedule and row['joint_prepared_scene_required']
    assert row['automatic_construction']['scene_template_files_read']==[] and row['automatic_construction']['old_absolute_distance_bands_removed']
    assert 'native_distance_bands_m'not in row and 'native_attachment_distance_targets_m'not in row
    assert all(k['yaw_deg']==35. for k in row['camera_yaw_knots'])and validated
