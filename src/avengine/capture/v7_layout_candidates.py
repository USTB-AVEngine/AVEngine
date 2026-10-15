"""Bounded deterministic position/yaw retries, never native admissions."""
import math


def candidate_index(scene):
    return int(scene.get('automatic_construction',{}).get('candidate_index',0))


def order_support_candidates(scene,candidates):
    if not scene.get('automatic_construction')or not candidates:return list(candidates)
    attempt=candidate_index(scene)
    # Inner native placement still tries support points and clearance variants.
    # A later full-scene retry starts in the next part of the same legal cell.
    shift=((attempt//4)*max(1,len(candidates)//8))%len(candidates)
    return list(candidates[shift:])+list(candidates[:shift])


def facing_yaw(scene,asset,listener,point):
    dx=float(listener[0])-float(point[0]);dz=float(listener[2])-float(point[2])
    base=math.atan2(-dx,-dz)if asset['kind']in {'human','performer','animal'}else math.atan2(-dz,dx)
    if not scene.get('automatic_construction'):return base
    offset=(0.,15.,-15.,30.)[candidate_index(scene)%4]
    return base+math.radians(offset)
