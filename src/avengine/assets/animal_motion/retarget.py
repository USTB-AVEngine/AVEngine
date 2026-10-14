"""Convert a target's desired world rotation into its local bone basis."""
import numpy as np

def local_from_world_delta(rest_world, parent_rest_world, parent_pose_world,
                           world_delta):
    rest = np.asarray(rest_world, float)
    parent_rest = np.asarray(parent_rest_world, float)
    parent_pose = np.asarray(parent_pose_world, float)
    target = np.asarray(world_delta, float) @ rest
    inherited = parent_pose @ np.linalg.inv(parent_rest) @ rest
    return np.linalg.solve(inherited, target)
