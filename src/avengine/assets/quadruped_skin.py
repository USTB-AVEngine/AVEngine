"""Continuous capsule envelopes for a target's own measured quadruped limbs."""
from __future__ import annotations
import numpy as np

SIDES=("FL","FR","HL","HR")
def smoothstep(x):
    u=np.clip(x,0,1)
    return u*u*(3-2*u)
def segment_distance(vertices,a,b):
    delta=b-a
    u=np.clip((vertices-a)@delta/max(float(delta@delta),1e-20),0,1)
    return np.linalg.norm(vertices-(a+u[:,None]*delta),axis=1)
def capsule_skin_weights(vertices,names,heads,feet,height,head_min,*,paw_only=False):
    v=np.asarray(vertices,float);feet=np.asarray(feet,float);heads=np.asarray(heads,float)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or height<=0:
        raise ValueError("finite target metric geometry required")
    if feet.shape!=(4,3):raise ValueError("four own semantic feet required")
    w=np.zeros((len(v),len(names)));body=names.index("Body");w[:,body]=1.
    dist2=np.linalg.norm(v[:,None,[0,2]]-feet[None,:,[0,2]],axis=2);nearest=dist2.argmin(1)
    separation=np.linalg.norm(feet[:,None,[0,2]]-feet[None,:,[0,2]],axis=2);np.fill_diagonal(separation,np.inf)
    for pi,side in enumerate(SIDES):
        ids=np.flatnonzero(nearest==pi);points=v[ids];foot=feet[pi]
        hi,ki,ai=[names.index(n+"_"+side) for n in ("Hip","Knee","Ankle")]
        distance=np.minimum.reduce([segment_distance(points,foot,heads[ai]),segment_distance(points,heads[ai],heads[ki]),segment_distance(points,heads[ki],heads[hi])])
        radius=min(height*.13,float(separation[pi].min())*.4)
        if paw_only:
            # Broad paw envelopes keep the animal's whole own paw rigid;
            # smooth Voronoi extinction still prevents influence crossing.
            radius=height*.25
        radial=1-smoothstep((distance-radius*.7)/(radius*.8))
        y=points[:,1]-foot[1]
        strength=radial*(1-smoothstep((y-height*.30)/(height*.18)))
        if paw_only:
            # A second envelope recipe: each distal limb moves rigidly with
            # its ankle, blending into the body over 26% of target height.
            strength=radial*(1-smoothstep((y-height*.04)/(height*.26)))
        # Signed distance to every nearest-foot Voronoi bisector. Each limb's
        # field and derivative vanish at its ownership boundary, even where
        # fitted limb axes are asymmetric. This avoids a left/right bone jump.
        competitors=np.arange(4)!=pi
        margin=(dist2[ids][:,competitors]**2-dist2[ids,pi,None]**2)/(2*separation[pi,competitors])
        boundary_width=min(height*.15,float(separation[pi].min())*.48)
        strength*=smoothstep(np.maximum(margin.min(1),0)/boundary_width)
        ankle=1-smoothstep((y-height*.12)/(height*.18))
        hip=smoothstep((y-height*.32)/(height*.16))
        knee=np.maximum(0,1-ankle-hip)
        if paw_only:
            ankle=np.ones_like(y);hip=np.zeros_like(y);knee=np.zeros_like(y)
        w[ids,body]=1-strength
        w[ids,ai]=strength*ankle;w[ids,ki]=strength*knee;w[ids,hi]=strength*hip
    head=names.index("Head");hc=heads[head]
    hs=(1-smoothstep((v[:,2]-hc[2]+height*.06)/(height*.23)))*smoothstep((v[:,1]-float(head_min[1]))/(height*.10))
    amount=w[:,body]*hs;w[:,body]-=amount;w[:,head]=amount
    tail=names.index("Tail");tc=heads[tail]
    ts=smoothstep((v[:,2]-tc[2])/(height*.10))*smoothstep((v[:,1]-(v[:,1].min()+height*.35))/(height*.15))
    if np.any((ts>0)&(hs>0)):raise ValueError("head/tail capsule regions overlap")
    amount=w[:,body]*ts;w[:,body]-=amount;w[:,tail]=amount
    active=(w>1e-12).sum(1)
    if active.max()>4:raise ValueError("capsule field exceeds four continuous influences")
    if not np.allclose(w.sum(1),1,atol=1e-12):raise ValueError("skin normalization failed")
    return w
