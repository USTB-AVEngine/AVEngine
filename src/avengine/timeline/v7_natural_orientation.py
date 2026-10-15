"""Scene-seeded natural initial headings, independent of actor ordering and voice."""
import hashlib,json,math
import numpy as np


def sample_initial_orientation(declaration,scene_seed,root,listener,room_center=None,other_people=(),furniture=(),retry=0):
    root=np.asarray(root,float);listener=np.asarray(listener,float)
    key=[int(scene_seed),declaration.get('asset_id','person'),[round(float(root[i]),3)for i in (0,2)],int(retry)]
    digest=hashlib.blake2b(json.dumps(key,separators=(',',':')).encode(),digest_size=8).hexdigest()
    rng=np.random.default_rng(int(digest,16));modes=['random'];weights=[.35];targets={}
    if room_center is not None:
        modes.append('room_center');weights.append(.35);targets['room_center']=[room_center]
    partners=[]
    for person in other_people:
        point=np.asarray(person['position_m'] if isinstance(person,dict)else person,float)
        if np.linalg.norm((point-root)[[0,2]])<.5:continue
        if isinstance(person,dict)and person.get('yaw_rad')is not None:
            toward=(root-point)[[0,2]];toward/=np.linalg.norm(toward)
            yaw=float(person['yaw_rad']);front=np.array([-math.sin(yaw),-math.cos(yaw)])
            if float(toward@front)<math.cos(math.radians(65)):continue
        partners.append(point)
    if partners:modes.append('conversation');weights.append(.2);targets['conversation']=partners
    if furniture:modes.append('furniture');weights.append(.1);targets['furniture']=list(furniture)
    weights=np.asarray(weights,float);mode=str(rng.choice(modes,p=weights/weights.sum()));target=None
    if mode!='random':
        target=np.asarray(targets[mode][int(rng.integers(len(targets[mode])))],float);delta=(target-root)[[0,2]]
        if np.linalg.norm(delta)<.25:mode='random';target=None
    if mode=='random':yaw=float(rng.uniform(-math.pi,math.pi))
    else:yaw=math.atan2(-delta[0],-delta[1])+float(rng.normal(0.,math.radians(16)))
    yaw=(yaw+math.pi)%(2*math.pi)-math.pi
    return {'yaw_rad':yaw,'mode':mode,'target_m':None if target is None else target.tolist(),
            'seed_fingerprint':digest,'uses_actor_order_or_speaking_order':False}


def all_people_face_camera(people,listener,half_angle_deg=25.):
    if len(people)<2:return False
    listener=np.asarray(listener,float)
    for person in people:
        root=np.asarray(person['position_m'],float);delta=(listener-root)[[0,2]];norm=np.linalg.norm(delta)
        if norm<1e-6:continue
        yaw=float(person['yaw_rad']);front=np.array([-math.sin(yaw),-math.cos(yaw)])
        if float(front@(delta/norm))<math.cos(math.radians(half_angle_deg)):return False
    return True
