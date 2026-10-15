"""Owner 15:0x natural placement: metre spacing and varied human headings."""
import math
import numpy as np


def is_human(asset):
    return asset.get('kind')in {'human','performer'}or asset.get('species_id')=='human'


def is_animal(asset):
    return asset.get('kind')=='animal'


def actor_root_track(entry):
    motion=entry.get('motion_episode',{})
    if motion.get('world_roots_m')is not None:return np.asarray(motion['world_roots_m'],float)
    return np.asarray([entry['resolved_root_m']],float)


def spacing_violation(scene,asset,root,placed):
    if not(scene.get('automatic_construction')or scene.get('owner_social_layout')):return None
    root=np.asarray(root,float)
    for other in placed:
        declaration=other['declaration']
        required=1.0 if (is_human(asset)and(is_human(declaration)or is_animal(declaration)))or(is_animal(asset)and is_human(declaration))else 0.
        if not required:continue
        track=actor_root_track(other);distance=float(np.linalg.norm(track[:,[0,2]]-root[[0,2]],axis=1).min())
        if distance+1e-9<required:
            return {'other_asset_id':declaration['asset_id'],'distance_m':distance,'required_m':required,
                'centres':'actual body roots projected onto the horizontal floor; all reserved motion samples'}
    return None


def varied_human_yaw_offset(asset):
    index=int(asset.get('native_instance_id',1))-1
    return math.radians((0.,70.,145.,-145.,110.)[index%5])


def distributed_point(candidate,chosen,minimum=1.):
    if any(math.hypot(candidate[0]-p[0],candidate[1]-p[1])<minimum for p in chosen):return False
    if len(chosen)>=2:
        a,b=chosen[:2];dx=b[0]-a[0];dz=b[1]-a[1]
        # A third person occupies another part of the floor, rather than the
        # same line across the camera. This is a proposal rule, not QA truth.
        if abs(dx*(candidate[1]-a[1])-dz*(candidate[0]-a[0]))/max(1e-9,math.hypot(dx,dz))<.35:return False
    return True


def social_proposal_score(scene,asset,representative,listener,forward):
    if not(scene.get('automatic_construction')or scene.get('owner_social_layout'))or not is_human(asset):return None
    index=(int(asset.get('native_instance_id',1))-1)%5
    target=(-28.,28.,5.,-18.,18.)[index];distance=(2.8,3.4,1.8,4.8,5.2)[index]
    delta=np.asarray(representative)-np.asarray(listener);forward=np.asarray(forward)
    angle=math.degrees(math.atan2(float(delta@np.cross(forward,[0.,1.,0.])),float(delta@forward)))
    return abs(angle-target)/15+abs(float(np.linalg.norm(delta[[0,2]]))-distance)
