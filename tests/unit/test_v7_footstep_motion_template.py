import pytest
from avengine.timeline.v7_footstep_motion_template import footstep_motion_template
pytestmark=pytest.mark.fast_unit

def test_unseen_room_constructs_footsteps_without_speech_on_the_walker():
    def sound(name,role,duration,gender,voice):
        return {'sound_asset_id':name,'scenario_id':'unseen','role_id':role,'voice_gender':gender,
                'synthesis_readback':{'duration_s':duration},'voice_reference':{'speaker_id':voice}}
    semantic={'sounds':[sound('a','one',2.2,'female','f'),sound('b','two',1.3,'male','m')]}
    catalog=[{'asset_id':str(i),'kind':'human','controlled_shirt':True,'standing_registered_qualified':True,
              'walk_allowed':True,'visual_gender':'female' if i%2 else 'male','color':str(i)}for i in range(8)]
    row=footstep_motion_template({'scene_id':'new_unseen','scene_seed':17},
        {'placement':{'camera_m':[0,1.3,0],'source_1_m':[1,1.3,-2],'source_2_m':[-1,1.3,-2]}},catalog,semantic)
    walker=next(iter(row['motion_requests']));speaker=row['semantic_human_asset_ids'][0]
    assert walker!=speaker
    assert row['source_classes'][walker]==['footsteps']
    assert walker not in row['semantic_human_asset_ids']
    assert row['motion_requests'][walker]['intervals_s']==[[0.,30.]]
    assert .8<=row['motion_requests'][walker]['speed_m_s']<=1.4
    assert len(row['asset_ids'])==4 and len(row['silent_asset_ids'])==2
    assert speaker not in row['motion_requests']
