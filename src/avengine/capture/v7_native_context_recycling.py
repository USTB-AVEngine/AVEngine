"""Bound native posed-asset residency while retaining real per-frame readbacks."""
import gc,ctypes
import numpy as np

def release_unused_native_memory():
    gc.collect()
    try:ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError,OSError):pass


def recycle_pose_context(video,scene,frame_index):
    """Replace only native handles, preserving actor dictionaries and evidence."""
    config=video._pose_context_configuration;actors=video.actors
    if getattr(video,'seat_material_entries',[])or getattr(video,'environment_nodes',{}):
        return False  # Separate inserted props need their own reconstruction.
    roots={e['endpoint_id']:np.asarray(e['object'].root_scene_node.absolute_transformation(),float).copy()for e in actors}
    identities={e['endpoint_id']:e['declaration']['asset_id']for e in actors}
    previous_count=len(getattr(video,'_pose_resources_in_context',set()))
    for entry in actors:
        entry.pop('object',None);entry.pop('emitter_node',None)
        entry.pop('motion_pose_path',None);entry.pop('motion_pose_index',None)
        entry.pop('phase2_foot_surface_mesh_cache',None)
    old=video.sim;video.sim=None;old.close();del old;release_unused_native_memory()
    replacement=type(video)(*config['arguments'],**config['keywords'])
    fresh={e['endpoint_id']:e for e in replacement.actors}
    if {key:value['declaration']['asset_id']for key,value in fresh.items()}!=identities:
        replacement.close();raise ValueError('native context reconstruction changed a required scene entity')
    for key in ('sim','hs','mn','qt','pathfinder','material_rendering'):
        setattr(video,key,getattr(replacement,key))
    replacement.sim=None
    from avengine.assets.habitat_capture import _apply_root_with_habitat
    for entry in actors:
        ep=entry['endpoint_id'];entry['object']=fresh[ep]['object']
        _apply_root_with_habitat(entry['object'],roots[ep],qt=video.qt,mn=video.mn)
        actual=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
        if np.max(np.abs(actual-roots[ep]))>1e-5:
            raise ValueError('reconstructed native root differs from the observed source root')
    replacement.actors.clear();replacement.__dict__.clear();del replacement
    stream=getattr(video,'_current_visibility_stream',None)
    if stream is not None:stream.last.clear()  # Measure fresh target masks in the new context.
    video._pose_resources_in_context=set()
    video.receipt.setdefault('native_context_restarts',[]).append({'frame_index':int(frame_index),
        'previous_unique_pose_resources':previous_count,'actual_actor_ids_preserved':identities,
        'whole_frame_clock_preserved':True,'original_pixels_and_native_gates_remeasured':True})
    release_unused_native_memory()
    return True
