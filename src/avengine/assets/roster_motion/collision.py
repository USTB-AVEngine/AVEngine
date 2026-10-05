"""Triangle versus box SAT for seating and furniture clearance checks."""
import numpy as np
def intersections(triangles,center,size,yaw=0,tolerance=.004):
 c,s=np.cos(yaw),np.sin(yaw);r=np.array([[c,0,s],[0,1,0],[-s,0,c]]);t=(np.asarray(triangles)-center)@r;h=np.maximum(np.asarray(size)/2-tolerance,1e-6)
 broad=np.all(t.min(1)<=h,axis=1)&np.all(t.max(1)>=-h,axis=1);ids=np.flatnonzero(broad);t=t[broad]
 if not len(t):return ids
 edges=np.roll(t,-1,axis=1)-t;axes=[np.broadcast_to(np.eye(3),(len(t),3,3)),np.cross(edges[:,0],edges[:,1])[:,None,:]]
 axes.append(np.cross(edges[:,:,None,:],np.eye(3)[None,None,:,:]).reshape(len(t),9,3));a=np.concatenate(axes,axis=1);p=np.einsum('nvi,nai->nva',t,a);rad=(abs(a)*h).sum(-1);over=np.all((p.min(1)<=rad+1e-8)&(p.max(1)>=-rad-1e-8),axis=1);return ids[over]


def clipped_surface_minimum(triangles,center_xz,size_xz,minimum_y=.25):
 """Lowest exact triangle point inside an axis-aligned seat footprint.

 Clip each polygon against the footprint and a lower-height plane. This includes
 edge/plane intersections that a vertex-only slab test can miss.
 """
 triangles=np.asarray(triangles);center_xz=np.asarray(center_xz);size_xz=np.asarray(size_xz)
 lo=center_xz-size_xz/2;hi=center_xz+size_xz/2
 broad=(triangles[:,:,0].max(1)>=lo[0])&(triangles[:,:,0].min(1)<=hi[0])&(triangles[:,:,2].max(1)>=lo[1])&(triangles[:,:,2].min(1)<=hi[1])&(triangles[:,:,1].max(1)>=minimum_y)
 best=None
 for triangle in triangles[broad]:
  poly=list(triangle)
  for axis,bound,sign in [(0,lo[0],1),(0,hi[0],-1),(2,lo[1],1),(2,hi[1],-1),(1,minimum_y,1)]:
   if not poly:break
   nxt=[]
   for a,b in zip(poly,poly[1:]+poly[:1]):
    da=sign*(a[axis]-bound);db=sign*(b[axis]-bound)
    if da>=-1e-10:nxt.append(a)
    if (da<0)!=(db<0):nxt.append(a+(b-a)*(da/(da-db)))
   poly=nxt
  if poly:
   point=min(poly,key=lambda v:v[1])
   if best is None or point[1]<best[1]:best=np.asarray(point)
 return best
