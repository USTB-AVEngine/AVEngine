"""Backtrack complete required human groups before abandoning a native camera."""
from itertools import chain
import math
import numpy as np


class NativeStandingPlacementExhausted(ValueError):
    pass


def standing_group_proposals(scene,assets,*,maximum_groups=24,minimum_camera_m=1.5):
    """Propose complete groups; native contact/collision/acoustics remain authority.

    Original angular/distance margins rank proposals rather than vetoing an
    otherwise legal room. Every full group is requalified on the live UE.
    """
    from avengine.timeline.v7_fixed_camera_layout import binding_candidate_in_fixed_view
    if len(assets)!=3:return
    listener=np.asarray(scene['listener_m'],float);yaw=scene['camera_yaw_knots'][0]['yaw_deg']
    original=[scene['sources'][a['endpoint_id']]['trajectory'][0]['position_m'] for a in assets]
    pool=[];seen=set()
    for point in chain(original,scene.get('native_placement_candidates',[])):
        p=np.asarray(point,float);key=tuple(np.round(p,5))
        if key in seen or not np.isfinite(p).all():continue
        seen.add(key)
        if np.linalg.norm((p-listener)[[0,2]])<max(1.2,minimum_camera_m)-1e-9:continue
        if not binding_candidate_in_fixed_view(listener,p,yaw):continue
        pool.append(p)
    if len(pool)<3:return
    points=np.asarray(pool);rays=points-listener;radii=np.linalg.norm(rays,axis=1)
    bearings=np.degrees(np.arctan2(rays[:,0],-rays[:,2]));delta=np.abs((bearings[:,None]-bearings[None,:]+180)%360-180)
    separation=np.linalg.norm((points[:,None,:]-points[None,:,:])[:,:,[0,2]],axis=2)
    legal=separation>=1.-1e-9;emitted=0;groups=set()
    for i in range(len(pool)):
        # Try distinct complete pairs instead of freezing the first accepted two.
        seconds=sorted(np.flatnonzero(legal[i]),key=lambda j:(not 16.<=delta[i,j]<=60.,abs(delta[i,j]-35.),int(j)))
        for j in seconds:
            thirds=np.flatnonzero(legal[i]&legal[j]);axis=(points[j]-points[i])[[0,2]]
            def score(k):
                distance=min(abs(radii[k]-radii[i]),abs(radii[k]-radii[j]))
                angular=min(delta[k,i],delta[k,j]);offset=(points[k]-points[i])[[0,2]];side=abs(float(axis[0]*offset[1]-axis[1]*offset[0]))/np.linalg.norm(axis)
                return (side<.25,distance<.6,angular<8.,-min(side,1.),-min(distance,1.),int(k))
            for k in sorted(thirds,key=score):
                group=(int(i),int(j),int(k))
                if group in groups:continue
                groups.add(group)
                yield {a['endpoint_id']:points[index].tolist() for a,index in zip(assets,group)}
                emitted+=1
                if emitted>=maximum_groups:return
