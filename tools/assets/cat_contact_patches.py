"""Clip intersecting surface triangles to their actual contact footprints.

The vectorized convex clipping retains source barycentric coordinates. Large
triangles contribute only their intersecting fragment, rather than a whole-face
area or the distance to vertices outside the opposing triangle's footprint.
"""
import numpy as np
CAPACITY = 8

def clip_barycentric(poly,counts,coefficients):
    n=len(poly);slots=np.arange(CAPACITY)[None,:]
    valid=slots<counts[:,None]
    previous=(slots-1)%np.maximum(counts[:,None],1)
    prior=np.take_along_axis(poly,previous[:,:,None],axis=1)
    distance=np.einsum("nvc,nc->nv",poly,coefficients)
    prior_distance=np.take_along_axis(distance,previous,axis=1)
    inside=distance>=-1e-13;prior_inside=prior_distance>=-1e-13
    crossing=valid&(inside!=prior_inside)&(np.abs(distance)>1e-13)&(np.abs(prior_distance)>1e-13)
    divisor=prior_distance-distance
    fraction=np.divide(prior_distance,divisor,out=np.zeros_like(distance),where=np.abs(divisor)>1e-30)
    intersection=prior+(poly-prior)*fraction[:,:,None]
    values=np.stack([intersection,poly],axis=2).reshape(n,CAPACITY*2,3)
    mask=np.stack([crossing,valid&inside],axis=2).reshape(n,CAPACITY*2)
    destinations=np.cumsum(mask,axis=1)-1
    new_counts=mask.sum(1)
    if np.any(new_counts>CAPACITY):raise ValueError("triangle contact polygon exceeded geometric capacity")
    result=np.zeros_like(poly)
    rows=np.broadcast_to(np.arange(n)[:,None],mask.shape)
    result[rows[mask],destinations[mask]]=values[mask]
    return result,new_counts

def footprints(source,target):
    source=np.asarray(source,float);target=np.asarray(target,float);n=len(source)
    poly=np.zeros((n,CAPACITY,3));poly[:,:3]=np.eye(3);counts=np.full(n,3)
    normal=np.cross(target[:,1]-target[:,0],target[:,2]-target[:,0])
    length=np.linalg.norm(normal,axis=1);normal/=np.maximum(length[:,None],1e-30)
    counts[length<1e-20]=0
    for edge in range(3):
        start=target[:,edge];vector=target[:,(edge+1)%3]-start
        coefficients=np.sum(np.cross(vector[:,None],source-start[:,None])*normal[:,None],axis=2)
        poly,counts=clip_barycentric(poly,counts,coefficients)
    signed=np.sum((source-target[:,0,None])*normal[:,None],axis=2)
    return poly,counts,signed

def clipped_internal_depths(source,target):
    depths=[]
    for a,b in ((source,target),(target,source)):
        poly,counts,signed=footprints(a,b)
        distances=np.einsum("nvc,nc->nv",poly,signed)
        valid=np.arange(CAPACITY)[None,:]<counts[:,None]
        positive=np.where(valid,np.maximum(distances,0),0).max(1)
        negative=np.where(valid,np.maximum(-distances,0),0).max(1)
        depths.append(np.minimum(positive,negative))
    return np.minimum(*depths)

def clipped_penetrating_patches(source,target,outward_sign):
    poly,counts,signed=footprints(source,target)
    signed=signed*np.asarray(outward_sign)[:,None]
    poly,counts=clip_barycentric(poly,counts,-signed)
    coordinates=np.einsum("nvc,ncd->nvd",poly,source)
    normal_depth=-np.einsum("nvc,nc->nv",poly,signed)
    valid=np.arange(CAPACITY)[None,:]<counts[:,None]
    depth=np.where(valid,np.maximum(normal_depth,0),0).max(1)
    first=coordinates[:,0,None];left=coordinates[:,1:-1]-first;right=coordinates[:,2:]-first
    areas=.5*np.linalg.norm(np.cross(left,right),axis=2)
    triangles=np.arange(1,CAPACITY-1)[None,:]+1<counts[:,None]
    area=np.where(triangles,areas,0).sum(1)
    return area,depth
