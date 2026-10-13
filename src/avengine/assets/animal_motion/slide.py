"""Explicitly unanimated translation of the registered target standing mesh."""
from types import SimpleNamespace
from pathlib import Path
import json,math
import numpy as np
from avengine.assets.roster_motion.io import read_glb
from avengine.assets.roster_motion.pose import pose
from avengine.assets.roster_motion.contact import yaw_matrix
from avengine.assets.animal_motion.profiles import PAWS
from avengine.assets.animal_motion.policy import SLIDE_MAX_SPEED_M_S,MotionUnavailable


def registration_record(record):
    data=json.loads(Path(record['registry']).read_text())
    for major in data['animals']:
        for row in major.get('minor_variants',[major]):
            if row['asset_id']==record['asset_id']:return row
    raise ValueError('animal_registration_missing')


class StandingSlide:
    def __init__(self,record,*,speed=.05,floor=None):
        if not 0<speed<=SLIDE_MAX_SPEED_M_S:raise ValueError('slide_speed_exceeds_80mm_per_second')
        self.record=record;self.speed=float(speed);self.period=None;self.floor=floor or (lambda p:0.)
        row=registration_record(record);files=record['files'];key=next((k for k in ('habitat_glb','runtime_glb') if k in files),None)
        if key is None:raise MotionUnavailable('registered_standing_mesh_missing')
        path=Path(files[key]['path']);d,b=read_glb(path)
        if d.get('skins'):raise MotionUnavailable('standing_derivative_is_not_a_static_mesh')
        parts=pose(d,b,None,ground=False,canonical_flip=False)[0];vertices=np.concatenate([p['positions'] for p in parts]);offsets=np.cumsum([0,*[len(p['positions']) for p in parts]])
        landmarks=row.get('landmarks_local_m') or {};meta=row.get('source_static_metadata') or {}
        mouth=np.asarray(row.get('emit_point_local_m',meta.get('mouth_m')),float)
        if mouth.shape!=(3,) or not np.isfinite(mouth).all():raise MotionUnavailable('registered_mouth_missing')
        head=np.asarray(row.get('head_center_local_m',mouth),float)
        if 'tail_tip' in landmarks:tail=np.asarray(landmarks['tail_tip'],float);tail_method='registered_target_tail_tip'
        else:
            # The posture is already registered +Y-up/-Z-front. Infer only a
            # target-specific rear marker from its own upper-rear vertices.
            upper=vertices[vertices[:,1]>vertices[:,1].min()+np.ptp(vertices[:,1])*.25]
            rear=upper[upper[:,2]>=np.quantile(upper[:,2],.99)];tail=np.mean(rear,axis=0);tail_method='own_mesh_upper_rear_1percent_centroid; tail-base proxy'
        front=(head-tail)[[0,2]]
        if np.linalg.norm(front)<.05:raise MotionUnavailable('standing_head_tail_axis_ambiguous')
        correction=math.atan2(front[0],-front[1]);rotation=yaw_matrix(correction)
        feet=[np.asarray(landmarks[f'foot_{i}']) for i in range(4)] if all(f'foot_{i}' in landmarks for i in range(4)) else None
        if feet is None:
            # Legacy Rocketbox idle has four well-separated source paws but
            # no explicit registration markers. Four original sole regions.
            x,z=(vertices.min(0)+vertices.max(0))[[0,2]]/2
            regions=[(vertices[:,0]<x)&(vertices[:,2]<z),(vertices[:,0]>=x)&(vertices[:,2]<z),(vertices[:,0]<x)&(vertices[:,2]>=z),(vertices[:,0]>=x)&(vertices[:,2]>=z)]
            if any(not r.any() for r in regions):raise MotionUnavailable('four_standing_paw_regions_missing')
            feet=[vertices[r][np.argmin(vertices[r,1])] for r in regions]
        centers=np.asarray(feet);separation=np.linalg.norm(centers[:,None,[0,2]]-centers[None,:,[0,2]],axis=-1);np.fill_diagonal(separation,np.inf);indices=[]
        for i,p in enumerate(feet):
            radius=min(.05,separation[i].min()/3);ids=np.flatnonzero(np.linalg.norm(vertices[:,[0,2]]-p[[0,2]],axis=1)<=radius)
            if not len(ids):raise MotionUnavailable('standing_paw_sole_unresolved')
            indices.append(int(ids[np.argmin(vertices[ids,1])]))
        if len(set(indices))!=4:raise MotionUnavailable('standing_four_paws_not_distinct')
        for p in parts:
            p['positions']=p['positions']@rotation.T
            if p['normals'] is not None:p['normals']=p['normals']@rotation.T
        vertices=np.concatenate([p['positions'] for p in parts]);mouth=rotation@mouth;head=rotation@head;tail=rotation@tail
        # Assign front/hind then left/right by actual target sole positions.
        order=sorted(indices,key=lambda i:vertices[i,2]);front_pair=sorted(order[:2],key=lambda i:vertices[i,0]);hind_pair=sorted(order[2:],key=lambda i:vertices[i,0]);indices=[*front_pair,*hind_pair]
        self.vertices=vertices;self.parts=parts;self.foot_indices=dict(zip(PAWS,indices));self.mouth=mouth;self.head=head;self.tail=tail
        self.support_plane={'method':'unchanged registered standing geometry; per-frame rigid support translation only; no gait claim','heading_correction_rad':correction}
        self.rig=SimpleNamespace(document=d,binary=b,path=path,provenance={'registration':record['registry'],'tail_definition':tail_method,'original_standing_geometry_preserved':True})

    def frame(self,t):
        root=np.array([0.,0.,-self.speed*t]);world=self.vertices+root
        near=world[world[:,1]<=world[:,1].min()+.03]
        # A callable with many() can supply all exact stage intersections.
        levels=self.floor.many(near) if hasattr(self.floor,'many') else np.array([self.floor(p) for p in near])
        root[1]=float(np.max(levels-near[:,1]))+.0004
        feet={}
        for side,index in self.foot_indices.items():
            p=self.vertices[index]+root
            feet[side]={'in_contact':True,'support_segment':0,'support_vertex':index,'support_world_m':p.tolist(),'support_target_world_m':p.tolist(),'sole_world_m':p.tolist(),'floor_delta_m':float(p[1]-self.floor(p)),
                        'intentional_slide':True,'support_lock_error_m':None}
        headmat=np.eye(4);headmat[:3,3]=self.head
        forward=self.head-self.tail
        row={'time_s':float(t),'source_phase_s':None,'world_root_m':root.tolist(),'feet':feet,'mouth_m':self.mouth.tolist(),'head_m':self.head.tolist(),'tail_m':self.tail.tolist(),
             'mouth_bone_matrix':headmat.tolist(),'mouth_bone_local_m':(self.mouth-self.head).tolist(),'body_floor_min_m':float((self.vertices+root)[:,1].min()),
             'body_head_tail_pitch_deg':float(np.degrees(np.arctan2(abs(forward[1]),np.linalg.norm(forward[[0,2]])))),'moving':True,'motion_mode':'slide_no_animation','joint_animation_present':False}
        return self.parts,row
