"""Find source-generated transverse bird rods from geometric line support and actual UV colors."""
import math
import numpy as np
from v7_animal_variants_color import rgb_to_lab

def rod_faces(v,faces,rgb,feet):
    span=float(np.ptp(v[:,1]));lab=rgb_to_lab(rgb);c=np.hypot(lab[:,1],lab[:,2]);h=np.mod(np.degrees(np.arctan2(lab[:,2],lab[:,1])),360);wood=((h>=40)&(h<=110)&(c>=8)&(c<=60)&(lab[:,0]<88))|((c<14)&(lab[:,0]<70));top=v[v[:,1]>v[:,1].min()+.78*span];low=v[v[:,1]<v[:,1].min()+.18*span];body_axis=top[:,[0,2]].mean(0)-low[:,[0,2]].mean(0);body_axis/=max(np.linalg.norm(body_axis),1e-9);rng=np.random.default_rng(20261012);best=None
    center_y=float(np.median(feet[:,1])+.035*span)
    for y in np.linspace(center_y-.09*span,center_y+.07*span,9):
        candidates=np.flatnonzero(wood&(np.abs(v[:,1]-y)<span*.030))
        if len(candidates)<100:continue
        ids=rng.choice(candidates,min(len(candidates),2500),replace=False);pts=v[ids][:,[0,2]]
        for _ in range(80):
            pair=pts[rng.choice(len(pts),2,replace=False)];direction=pair[1]-pair[0];length=np.linalg.norm(direction)
            if length<.12*span:continue
            direction/=length
            if abs(direction@body_axis)>.70:continue
            q=pts-pair[0];along=q@direction;distance=np.abs(q[:,0]*direction[1]-q[:,1]*direction[0]);inside=distance<.024*span
            if inside.sum()<80:continue
            coverage=float(np.quantile(along[inside],.995)-np.quantile(along[inside],.005))
            if coverage<.42*span:continue
            score=coverage*inside.sum();mid=pair[0]+direction*np.mean(along[inside]);candidate=(score,mid,direction,float(y),coverage,int(inside.sum()),len(ids))
            if best is None or score>best[0]:best=candidate
    if best is None:return np.zeros(len(faces),bool),{'status':'no_long_transverse_rod_detected','method':'deterministic 3D horizontal-line search with source texture wood/neutral pixels, perpendicular to head-tail axis','body_axis_xz':body_axis.tolist()}
    score,mid,direction,y,coverage,inliers,samples=best;center=np.array([mid[0],y,mid[1]]);axis=np.array([direction[0],0,direction[1]]);q=v-center;along=q@axis;distance=np.linalg.norm(q-along[:,None]*axis,axis=1);protected=np.min(np.stack([np.linalg.norm(v[:,[0,2]]-point[[0,2]],axis=1) for point in feet]),axis=0)<span*.05;rod=(distance<span*.050)&wood&~protected;remove=rod[faces].sum(1)>=2;detail={'status':'transverse_rod_detected','method':'deterministic line consensus, actual brown/dark-neutral UV colors; near-foot neighborhoods retained; no horizontal slicing through bird tail','axis':axis.tolist(),'center':center.tolist(),'line_length_raw':coverage,'radius_raw':span*.05,'inlier_vertices':inliers,'sampled_vertices':samples,'removed_faces':int(remove.sum()),'body_axis_xz':body_axis.tolist(),'axis_body_dot':float(abs(direction@body_axis))};return remove,detail
