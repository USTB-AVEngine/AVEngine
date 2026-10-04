"""Perspective-correct visible shirt ROI from native camera and glTF UVs."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from v7_human_roster_materials import read_glb
from v7_human_roster_pose import accessor

def roi(path,maskpath,prefix,view,semantic):
 d,b=read_glb(path);tex=np.asarray(Image.open(maskpath).convert('L'));height,width=semantic.shape;zbuffer=np.full((height,width),np.inf);out=np.zeros((height,width),bool);yaw=view['yaw_rad'];c=np.cos(yaw);s=np.sin(yaw);R=np.array([[c,0,s],[0,1,0],[-s,0,c]]);translation=np.asarray(view['object_position_m'])
 V=np.asarray(view['rgb_view_matrix']);P=np.asarray(view['rgb_projection_matrix'])
 for node in d['nodes']:
  if 'mesh'not in node:continue
  for prim in d['meshes'][node['mesh']]['primitives']:
   points=accessor(d,b,prim['attributes']['POSITION']);world=points@R.T+translation;cam=np.c_[world,np.ones(len(world))]@V.T;clip=cam@P.T;screen=np.stack([(clip[:,0]/clip[:,3]+1)*width/2,(1-clip[:,1]/clip[:,3])*height/2],axis=1);depth=-cam[:,2]
   inds=accessor(d,b,prim['indices']).ravel().reshape(-1,3).astype(int);body=d['materials'][prim['material']].get('name','').split('.')[0]==prefix+'_body';uv=accessor(d,b,prim['attributes']['TEXCOORD_0']) if body else None
   for tri in inds:
    if (depth[tri]<=0).any():continue
    pts=screen[tri];xmin=max(0,int(np.floor(pts[:,0].min())));xmax=min(width-1,int(np.ceil(pts[:,0].max())));ymin=max(0,int(np.floor(pts[:,1].min())));ymax=min(height-1,int(np.ceil(pts[:,1].max())))
    if xmin>xmax or ymin>ymax:continue
    x0,y0=pts[0];x1,y1=pts[1];x2,y2=pts[2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
    if abs(den)<1e-7:continue
    yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1];xx=xx+.5;yy=yy+.5
    w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;w2=1-w0-w1;inside=(w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)
    weights=np.stack([w0,w1,w2],axis=-1)/depth[tri];idepth=weights.sum(axis=-1);z=1/np.maximum(idepth,1e-10);current=zbuffer[ymin:ymax+1,xmin:xmax+1];update=inside&(z<current);current[update]=z[update];sub=out[ymin:ymax+1,xmin:xmax+1];sub[update]=False
    if body and update.any():
     uvs=np.einsum('hwk,kj->hwj',weights,uv[tri])/np.maximum(idepth[...,None],1e-10);u=np.clip((uvs[...,0]*tex.shape[1]).astype(int),0,tex.shape[1]-1);v=np.clip((uvs[...,1]*tex.shape[0]).astype(int),0,tex.shape[0]-1);selected=tex[v,u]>127;sub[update]=selected[update]
 out &=semantic;out=np.asarray(Image.fromarray(out.astype('uint8')*255).filter(ImageFilter.MinFilter(3)))>0
 return out
