"""Restore physical objects without prematurely admitting parked participants."""
import numpy as np


def restore_actor_state(video,entry,scene,index,moving,saved_root,apply_root):
    if entry.get('phase2_layout_qualified') is False:
        apply_root(entry['object'],saved_root,qt=video.qt,mn=video.mn)
        observed=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
        if np.max(np.abs(observed-saved_root))>1e-5:
            raise ValueError('parked participant changed root during native context refresh')
        entry['applied_root_input']=observed.copy()
        return
    if moving:video._apply_motion_entry(entry,index)
    else:video._apply_entry(entry,scene)
