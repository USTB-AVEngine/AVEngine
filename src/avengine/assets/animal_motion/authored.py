"""Replay target-owned Blender-baked walking without applying a second IK solve."""
import math,numpy as np
from .profiles import PAWS,paw_points
from .walk import smooth5
class AuthoredWalk:
    def __init__(self,rig,rigging,*,floor=None):
        self.rig=rig;self.period=float(rigging['gait_period_s'])
        self.scale=float(rigging.get('source_scale_correction',1.))
        self.speed=float(rigging.get('actual_authored_speed_m_s',rigging['root_speed_m_s']))
        self.lift=.008*self.scale;self.shift=np.asarray(rigging.get('source_translation_correction_m',[0,0,0]))
        if not .03<=float(rigging['root_speed_m_s'])<=.15 or not 1.6<=self.period<=4:raise ValueError('authored_source_motion_parameters_out_of_range')
        self.floor=floor or (lambda p:0.)
        ts=rig.samples();self.start=float(ts[0]);self.duration=float(ts[-1]-ts[0])
        if self.duration<=0:raise ValueError('authored_clip_has_no_duration')
        self.offsets=dict(zip(PAWS,(.25,.75,0,.5)))
        self.reference_g=rig.globals(0,'Idle');self.reference_parts=rig.skin(self.reference_g)
        self.reference_feet=paw_points(self.reference_parts,rig.masks)
        self.support_plane={'method':'target-owned Blender baked IK; source-scale corrected; no second IK; actual native floor remains authority',
                            'source':rigging.get('motion_runtime_glb'),'uniform_source_scale':self.scale,'source_floor_plane':rigging['support_plane']}
        self.heading_correction=0.;self.transform=np.eye(4)
    def frame(self,t):
        phase=self.start+(float(t)%self.duration);g=self.rig.globals(phase)
        travel=np.array([0.,0.,-self.speed*(phase-self.start)])
        for m in g.values():m[:3,3]-=travel
        root=np.array([0.,0.,-self.speed*float(t)])
        parts=self.rig.skin(g);measured=paw_points(parts,self.rig.masks);marks=self.rig.landmarks(g);feet={}
        for side in PAWS:
            q=float(t)/self.period+self.offsets[side];cycle=math.floor(q);fraction=q-cycle;active=fraction>=.25
            plant=(cycle+.25-self.offsets[side])*self.period
            ref=self.reference_feet[side];pi,vi=ref['part'],ref['vertex']
            target=ref['point'].copy();target[2]-=self.speed*(plant+.375*self.period)
            if not active:target[2]+=self.speed*self.period*(1-smooth5(fraction/.25))
            target[1]=self.shift[1]+.0004*self.scale+(self.lift*math.sin(math.pi*fraction/.25)**2 if not active else 0)
            actual=parts[pi]['positions'][vi]+root;sole=measured[side]['point']+root
            feet[side]={'in_contact':active,'support_segment':cycle if active else None,'support_vertex':[pi,vi],
                        'support_world_m':actual.tolist(),'support_target_world_m':target.tolist(),
                        'support_lock_error_m':float(np.linalg.norm(actual-target)),'sole_world_m':sole.tolist(),
                        'floor_delta_m':float(sole[1]-self.floor(sole))}
        vertices=np.concatenate([p['positions'] for p in parts]);forward=marks['head']-marks['tail']
        pitch=float(np.degrees(np.arctan2(abs(forward[1]),np.linalg.norm(forward[[0,2]]))))
        row={'time_s':float(t),'source_phase_s':phase,'world_root_m':root.tolist(),'feet':feet,
             'mouth_m':marks['mouth'].tolist(),'head_m':marks['head'].tolist(),'tail_m':marks['tail'].tolist(),
             'head_matrix':g[self.rig.head].tolist(),'mouth_bone_matrix':g[self.rig.mouth_bone].tolist(),
             'mouth_bone_local_m':self.rig.mouth_local.tolist(),'body_floor_min_m':float(vertices[:,1].min()),
             'body_head_tail_pitch_deg':pitch,'pelvis_lowering_m':max(0.,float(self.reference_g[self.rig.body][1,3]-g[self.rig.body][1,3])),
             'ik_reach_error_m':max(feet[s]['support_lock_error_m'] for s in PAWS),
             'source_heading_correction_rad':0.,'moving':True,'motion_generator':'target_owned_baked_blender_walking'}
        self.last_globals=g;return parts,row
