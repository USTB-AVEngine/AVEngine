"""World-space constrained slow gait on the original target rig and skin.

The authored Walking torso, head, tail and limb bend planes remain the source.
A four-beat walk repairs unreliable donor contact labels. Three paws support
at all times; each swing is a smooth 8 mm lift, and the step length derives
from the requested ground speed. Neither time reversal nor reflection is used.
"""
import copy,math
import numpy as np
from avengine.assets.roster_motion.contact import yaw_matrix,rotation_between,rotate_subtree,subtree
from avengine.assets.animal_motion.profiles import PAWS,paw_points


def solve_leg(d,g,chain,target,linear):
    hip,knee,ankle,_=chain
    h=g[hip][:3,3].copy();k=g[knee][:3,3].copy();a=g[ankle][:3,3].copy()
    l1=np.linalg.norm(k-h);l2=np.linalg.norm(a-k);delta=target-h;want=np.linalg.norm(delta)
    if min(l1,l2,want)<1e-7:raise ValueError('degenerate_target_limb')
    reach=max(0.,want-l1-l2);dist=np.clip(want,abs(l1-l2)+1e-7,l1+l2-1e-7);axis=delta/want
    along=(l1*l1-l2*l2+dist*dist)/(2*dist)
    bend=(k-h)-axis*np.dot(k-h,axis)
    if np.linalg.norm(bend)<1e-7:bend=np.cross(axis,[1.,0.,0.])
    if np.linalg.norm(bend)<1e-7:bend=np.cross(axis,[0.,0.,1.])
    bend/=np.linalg.norm(bend);newk=h+axis*along+bend*math.sqrt(max(0,l1*l1-along*along))
    rotate_subtree(d,g,hip,rotation_between(k-h,newk-h))
    k=g[knee][:3,3].copy();a=g[ankle][:3,3].copy()
    rotate_subtree(d,g,knee,rotation_between(a-k,h+axis*dist-k))
    # Global bone matrices carry uniform target scale. Inverting that matrix
    # is essential; using its transpose would multiply the scale twice.
    rotate_subtree(d,g,ankle,linear@np.linalg.inv(g[ankle][:3,:3]))
    return reach


def smooth5(u):return u*u*u*(10+u*(-15+6*u))


class ConstrainedWalk:
    def __init__(self,rig,*,speed=.08,period=2.4,lift=.008,floor=None):
        if not .03<=speed<=.15:raise ValueError('slow_walk_speed_out_of_range')
        if not 1.6<=period<=4.:raise ValueError('slow_walk_period_out_of_range')
        if not .003<=lift<=.01:raise ValueError('slow_walk_lift_out_of_range')
        self.rig=rig;self.speed=float(speed);self.period=float(period);self.lift=float(lift)
        self.floor=floor or (lambda point:0.);self.offsets=dict(zip(PAWS,(.25,.75,0.,.5)))
        ts=rig.samples();self.start=float(ts[0]);self.duration=float(ts[-1]-ts[0])
        idle=rig.globals(0,'Idle');marks=rig.landmarks(idle);forward=marks['head']-marks['tail']
        self.heading_correction=math.atan2(forward[0],-forward[2]);rot=yaw_matrix(self.heading_correction)
        m=np.eye(4);m[:3,:3]=rot;idle={j:m@g for j,g in idle.items()}
        parts=rig.skin(idle);feet=paw_points(parts,rig.masks)
        # Constant four-semantic-paw plane leveling precedes animation IK.
        points=np.array([feet[s]['point'] for s in PAWS]);coeff=np.linalg.lstsq(np.c_[points[:,0],points[:,2],np.ones(4)],points[:,1],rcond=None)[0]
        normal=np.array([-coeff[0],1.,-coeff[1]]);normal/=np.linalg.norm(normal)
        level=rotation_between(normal,[0.,1.,0.]);level_angle=float(np.degrees(np.arccos(np.clip(level[1,1],-1,1))))
        if level_angle>10:raise ValueError('target_support_plane_requires_more_than_10deg')
        self.transform=np.eye(4);self.transform[:3,:3]=level@rot
        leveled=points@level.T;self.transform[1,3]=-float(np.mean(leveled[:,1]))
        self.reference_g={j:self.transform@g for j,g in rig.globals(0,'Idle').items()}
        self.reference_parts=rig.skin(self.reference_g);self.reference_feet=paw_points(self.reference_parts,rig.masks)
        self.reference_body=rig.landmarks(self.reference_g)['body']
        self.support_plane={'constant_rotation_deg':level_angle,'source_paw_points':points.tolist(),'matrix':self.transform.tolist(),'method':'least squares four semantic paw support plane before motion IK'}
        self.foot_linear={s:self.reference_g[c[2]][:3,:3].copy() for s,c in rig.chains.items()}
        self.anchors={s:f['point'].copy() for s,f in self.reference_feet.items()}
        lengths=[np.linalg.norm(self.reference_g[c[1]][:3,3]-self.reference_g[c[0]][:3,3])+np.linalg.norm(self.reference_g[c[2]][:3,3]-self.reference_g[c[1]][:3,3]) for c in rig.chains.values()]
        if speed*period>min(lengths)*1.3:raise ValueError('requested_stride_too_long_for_target_limbs')

    def foot_target(self,side,t):
        offset=self.offsets[side];q=t/self.period+offset;cycle=math.floor(q);phase=q-cycle
        plant_time=(cycle+.25-offset)*self.period
        next_anchor=self.anchors[side]+[0.,0.,-self.speed*(plant_time+.375*self.period)]
        if phase>=.25:
            point=next_anchor;active=True;segment=cycle
        else:
            u=phase/.25;point=next_anchor+[0.,0.,self.speed*self.period*(1-smooth5(u))];active=False;segment=None
        point=np.asarray(point,float);point[1]=self.floor(point)+.0004
        if not active:point[1]+=self.lift*math.sin(math.pi*phase/.25)**2
        return point,active,segment

    def frame(self,t):
        rig=self.rig;phase=self.start+(t/self.period%1.)*self.duration
        original={j:self.transform@m for j,m in rig.globals(phase).items()}
        body=rig.landmarks(original)['body'];delta=body-self.reference_body
        # Remove only authored horizontal root motion; route is authoritative.
        for m in original.values():m[[0,2],3]-=delta[[0,2]]
        root=np.array([0.,0.,-self.speed*t]);g=copy.deepcopy(original)
        # Lock the whole distal paw relative to its ankle. Authored toe curl
        # during stance changed the lowest material point by 16--21 mm even
        # when the ankle was planted. Keep the target's own Idle paw shape;
        # preserve all upstream Walking articulation and original skin weights.
        for chain in rig.chains.values():
            ankle=chain[2];reference_inverse=np.linalg.inv(self.reference_g[ankle])
            for joint in subtree(rig.document,ankle):
                if joint!=ankle:g[joint]=g[ankle]@reference_inverse@self.reference_g[joint]
        parts=rig.skin(g);targets={};contacts={};refs={}
        for side,chain in rig.chains.items():
            ref=self.reference_feet[side];pi,vi=ref['part'],ref['vertex']
            local=(np.linalg.inv(self.reference_g[chain[2]])@np.r_[self.reference_parts[pi]['positions'][vi],1.])[:3]
            target,active,segment=self.foot_target(side,t)
            targets[side]=target-root-self.foot_linear[side]@local
            contacts[side]=(target,active,segment);refs[side]=(pi,vi)
        lower=0.
        for side,c in rig.chains.items():
            h=g[c[0]][:3,3];total=np.linalg.norm(g[c[1]][:3,3]-h)+np.linalg.norm(g[c[2]][:3,3]-g[c[1]][:3,3]);target=targets[side]
            horizontal=np.linalg.norm((h-target)[[0,2]])
            lower=max(lower,float(h[1]-target[1]-math.sqrt(max(0,(total*.99)**2-horizontal**2))))
        if lower>.08:raise ValueError('pelvis_lowering_exceeds_80mm:'+str(lower))
        for m in g.values():m[1,3]-=lower
        reach=0.
        for side,c in rig.chains.items():reach=max(reach,solve_leg(rig.document,g,c,targets[side],self.foot_linear[side]))
        for iteration in range(8):
            parts=rig.skin(g);measured=paw_points(parts,rig.masks)
            for side,c in rig.chains.items():
                pi,vi=refs[side];target,active,_=contacts[side];actual=parts[pi]['positions'][vi]+root
                adjustment=target-actual
                # Keep the actual lowest paw skin outside the floor. Do not
                # replace the fixed support vertex or reset the plant anchor.
                low=measured[side]['point']+root;needed=self.floor(low)+.0002-low[1]
                adjustment[1]=max(adjustment[1],needed)
                targets[side]+=adjustment
                reach=max(reach,solve_leg(rig.document,g,c,targets[side],self.foot_linear[side]))
        parts=rig.skin(g);measured=paw_points(parts,rig.masks);marks=rig.landmarks(g);feet={}
        for side in PAWS:
            pi,vi=refs[side];target,active,segment=contacts[side];actual=parts[pi]['positions'][vi]+root;sole=measured[side]['point']+root
            feet[side]={'in_contact':active,'support_segment':segment,'support_vertex':[pi,vi],'support_world_m':actual.tolist(),'support_target_world_m':target.tolist(),
                'support_lock_error_m':float(np.linalg.norm(actual-target)),'sole_world_m':sole.tolist(),'floor_delta_m':float(sole[1]-self.floor(sole))}
        vertices=np.concatenate([p['positions'] for p in parts]);forward=marks['head']-marks['tail'];pitch=float(np.degrees(np.arctan2(abs(forward[1]),np.linalg.norm(forward[[0,2]]))))
        row={'time_s':float(t),'source_phase_s':float(phase),'world_root_m':root.tolist(),'feet':feet,'mouth_m':marks['mouth'].tolist(),
             'head_m':marks['head'].tolist(),'tail_m':marks['tail'].tolist(),'head_matrix':g[rig.head].tolist(),
             'mouth_bone_matrix':g[rig.mouth_bone].tolist(),'mouth_bone_local_m':rig.mouth_local.tolist(),
             'body_floor_min_m':float(vertices[:,1].min()),'body_head_tail_pitch_deg':pitch,'pelvis_lowering_m':lower,'ik_reach_error_m':reach,
             'source_heading_correction_rad':self.heading_correction,'moving':True}
        self.last_globals=g
        return parts,row
