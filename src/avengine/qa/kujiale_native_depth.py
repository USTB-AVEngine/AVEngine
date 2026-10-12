"""SPEAR native ray-distance depth in meters and perspective pixel geometry."""
import math
import numpy as np

REVISION="native-radial-meters-v3"

def basis(view):
    yaw,pitch=map(math.radians,[view["yaw_deg"],view["pitch_deg"]])
    forward=np.array([math.cos(pitch)*math.cos(yaw),math.sin(pitch),math.cos(pitch)*math.sin(yaw)])
    right=np.array([-math.sin(yaw),0.,math.cos(yaw)])
    return forward,right,np.cross(right,forward)

def world_pixels(depth,view,mask):
    """Return selected pixel centers in world meters; depth is radial range."""
    yy,xx=np.nonzero(mask);h,w=depth.shape
    f=w/(2*math.tan(math.radians(view.get("hfov_deg",85))/2))
    forward,right,up=basis(view)
    u=(xx+.5-w/2)/f;v=(h/2-yy-.5)/f
    rays=forward+u[:,None]*right+v[:,None]*up
    rays/=np.linalg.norm(rays,axis=1)[:,None]
    return np.asarray(view["position_m"])+np.asarray(depth)[yy,xx,None]*rays

def far_range(vertices,view):
    return float(np.linalg.norm(np.asarray(vertices)-np.asarray(view["position_m"]),axis=1).max())+.1

def visibility(scene,isolated,far_m):
    mask=np.isfinite(isolated)&(isolated>0)&(isolated<far_m)
    visible=mask&np.isfinite(scene)&(scene>0)&(np.abs(scene-isolated)<=.005)
    return mask,visible
