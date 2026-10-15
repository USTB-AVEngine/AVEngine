"""Human mouth certificates and independently admitted physical scene entities.

All original actors remain in RGB, masks, audio and general QA. A static dog
uses its actual physical UE authority rather than a human standing roster.
"""
import numpy as np


def registered_dialogue_view(scene,visual):
    frames=visual['frames'];others=[a for a in frames[0]['actors']if a.get('entity_class')in {'static_animal','static_object'}]
    if not others:return visual,{'status':'not_applicable_no_other_physical_entities'}
    physical=visual.get('native_physical_environment',{})
    if (not scene.get('require_current_native_environment') or physical.get('status')!='pass'
            or physical.get('frame_count')!=450 or len(physical.get('frames',[]))!=450
            or physical.get('all_sources_have_drawable_geometry')is not True or physical.get('source_labels_modified')is not False):
        raise ValueError('other dialogue-scene entities need complete original native physical authority')
    wanted={a['endpoint_id']:a for a in others};certificates=[]
    for ep,actor in wanted.items():
        source=scene['sources'][ep];admission=physical['current_joint_admission'][ep]
        if (source.get('native_asset_id')!=actor['asset_id'] or source.get('native_visual_kind')!='registered_physical'
                or admission['status']!='pass' or admission['asset_id']!=actor['asset_id']):
            raise ValueError('physical source is not its original admitted native asset')
        certificates.append({'endpoint_id':ep,'actor_id':actor['actor_id'],'asset_id':actor['asset_id'],
            'authority':'original current joint UE physical admission, all 450 native depth/mask/root/emitter records',
            'registered_human_mouth_certificate':False})
    scoped=[];seen={ep:False for ep in wanted}
    for i,(frame,observed)in enumerate(zip(frames,physical['frames'],strict=True)):
        ids=[a['actor_id']for a in frame['actors']]
        if len(ids)!=len(set(ids)):raise ValueError('duplicate original physical actor IDs')
        if frame['frame_index']!=i or observed['frame_index']!=i:raise ValueError('physical dialogue scene clock changed')
        for ep,expected in wanted.items():
            actors=[a for a in frame['actors']if a['endpoint_id']==ep];records=[a for a in observed['entities']if a['endpoint_id']==ep]
            if len(actors)!=1 or len(records)!=1:raise ValueError('physical entity absent from its native frame')
            a,r=actors[0],records[0];admission=physical['current_joint_admission'][ep];foot=a['native_foot_contact']
            if (a['asset_id']!=expected['asset_id'] or a['actor_id']!=expected['actor_id'] or r['asset_id']!=a['asset_id']
                    or r['source_id']!=a['actor_id'] or r['frame_index']!=i or r.get('source_geometry_present')is not True
                    or a.get('moving')is not False or a['action_id']!='registered_static_support'
                    or a['root_max_error']>.0005 or r['root_error_m']>.0005
                    or foot['status']!='pass' or abs(foot['gap_m'])>foot['tolerance_m']):
                raise ValueError('other physical entity failed original native identity/contact/pose evidence')
            if (np.linalg.norm(np.asarray(a['emitter_position_m'])-r['emitter_position_m'])>.0005
                    or np.linalg.norm(np.asarray(a['emitter_position_m'])-admission['emitter']['actual_m'])>.0005):
                raise ValueError('other physical emitter differs from actual original UE attachment')
            seen[ep]|=r['visible_pixels']>=80
        scoped.append({**frame,'actors':[a for a in frame['actors']if a['endpoint_id']not in wanted]})
    if not all(seen.values()):raise ValueError('other physical entity never appeared in actual native RGB/masks')
    return {**visual,'frames':scoped},{'status':'pass','registered_human_scope_only':True,
        'all_original_entities_retained_in_raw_media_and_general_QA':True,'complete_native_frames_checked':450,
        'other_physical_entities':certificates}
