"""Whole authored directional cues, including a real side/rear speaker."""
import math


def construct_authored_direction_scene(room,house,state,catalog,records,semantic,contract,runtime,
        *,scene_id,seed,attempt=0):
    from avengine.timeline.v7_family_constructors import construct_family_scene,_request
    from avengine.timeline.v7_production_inputs import validate_recipe
    plan=construct_family_scene(room,house,state,catalog,records,semantic,contract,runtime,
        scene_id=scene_id,seed=seed,family='V7-CONTENT-BIND',attempt=attempt,_content_silent_bystander=False)
    row=plan['row'];speakers=row['semantic_human_asset_ids'];outside=row['offscreen_construction']['offscreen_asset_ids'][0]
    point=row['measured_static_layout']['valid'][2]['position_m'];listener=row['listener_m'];yaw=row['camera_yaw_deg']
    angle=(math.degrees(math.atan2(point[0]-listener[0],-(point[2]-listener[2])))-yaw+180)%360-180
    center=((round(angle/45)*45+180)%360)-180
    row.update(schema='v7-visible-registered-pilot-phase2-v1',members=['v0_a0'],trial_only=True,acceptance_allow_partial_families=True,
        # The second visible speaker must not be centered beyond the 42.5deg
        # half field of view. Both visible voices stay inside the front sector;
        # a real offscreen voice supplies a different direction answer.
        semantic_azimuth_targets_deg={speakers[0]:-5.,speakers[1]:5.,outside:center},
        native_attachment_distance_targets_m={speakers[0]:3.2,speakers[1]:4.4},
        native_placement_preferences_m={},
        phase2_authored_speech_schedule_s={aid:[epoch+offset for epoch in (0.,10.,20.)]
            for aid,offset in zip(speakers,(3.,6.5,0.),strict=True)})
    row.pop('production_scenario',None)
    row['native_view_roles'][outside]['preferred_azimuth_deg']=center
    row['automatic_construction'].update(family='QA-28',constructor='construct_authored_direction_scene')
    row['offscreen_construction']['all_waveforms_author_tts']=True
    _request(row,state,'QA-28')
    row['scene_demand']['required_sector_boundary_margin_deg']=15.
    validate_recipe(row,catalog,contract)
    return {'profile':'phase2','row':row}
