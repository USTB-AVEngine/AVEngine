from avengine.rooms.conditioned_sampler import _anchor_pre_silence_start_ranges


def test_post_sound_anchor_excludes_only_the_catalog_forbidden_band():
    events={'other':{'sample_count':1710}}
    profile={'target_moved_after_sound':False,'first_speaker_instance_id':'other1','anchor_indices':[0],'instances':[{'entity_instance_id':'human_target','source_slot_id':'source1'}]}
    ranges=_anchor_pre_silence_start_ranges([[1500,2200]],{'actor_id':'source1','entity_instance_id':'human_target'},{'other':30},events,{'sample_rate_hz':1000},profile)
    assert ranges==[[1500,1739],[1840,2200]]
    # Clip padding may overlap while the separate audible-gap constraint holds.
    assert ranges[0][0]==1500


def test_other_questions_and_competitors_keep_their_original_ranges():
    original=[[0,2200]]
    assert _anchor_pre_silence_start_ranges(original,{'actor_id':'source1','entity_instance_id':'human_target'},{},{},{'sample_rate_hz':1000},{}) is original
    profile={'target_moved_after_sound':True,'first_speaker_instance_id':'other1','anchor_indices':[0],'instances':[{'entity_instance_id':'human_target','source_slot_id':'source1'}]}
    assert _anchor_pre_silence_start_ranges(original,{'actor_id':'source2','entity_instance_id':'other1'},{},{},{'sample_rate_hz':1000},profile) is original
