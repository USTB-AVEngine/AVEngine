"""Reject physically reversed and sideways UE skeletal actors on the same route."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from avengine.assets.animal_motion.checks import direction_check
from avengine.assets.animal_motion.native import overlay
from avengine.assets.animal_motion.ue_runtime import (
    launch_native, camera_helpers, apply_camera, apply_native_rgb_profile,
    to_ue, from_ue, close_native,
)
from tools.ue.capture_animal_motion import matrix, xyz


def main():
    parser=argparse.ArgumentParser()
    for key in ('root','house','motion','imported','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    house=json.loads(args.house.read_text())
    motion=json.loads(args.motion.read_text())
    imported=json.loads(args.imported.read_text())
    ue=house['input']['ue_config'].copy()
    ue.update(uproject=str(args.root/'ue/SpearSim/SpearSim.uproject'),
        output_root=str(args.root),graphics_adapter=0,physical_gpu_index=0,
        network_guard_so=str(args.root/'ue/network_guard.so'),unpause_animation_sampling=True)
    instance,game,launch=launch_native(ue,house['input']['ue_map_path'])
    spawn,_,rgb_reader,*_=camera_helpers()
    actor=component=cam=components=None
    result={'status':'pending','excluded_from_admission':True,'backend':'UE/SPEAR',
        'asset_id':motion['asset_id'],'launch':launch,'controls':[]}
    def step(fn=None):
        with instance.begin_frame():
            if fn:fn()
        with instance.end_frame():pass
    try:
        root0=np.array(motion['root_m'])
        center=root0+np.array(motion['frames'][-1]['world_root_m'])/2+[0,.3,0]
        eye=center+[.15,1.4,.65]
        forward=center-eye
        view={'position_m':eye.tolist(),'yaw_deg':math.degrees(math.atan2(forward[2],forward[0])),
            'pitch_deg':math.degrees(math.atan2(forward[1],np.linalg.norm(forward[[0,2]]))),
            'hfov_deg':85.}
        def setup():
            nonlocal actor,component,cam,components
            actor=game.unreal_service.spawn_actor(uclass='ASkeletalMeshActor',spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'})
            actor.SetActorEnableCollision(bNewActorEnableCollision=False)
            component=game.unreal_service.get_component_by_class(actor=actor,uclass='USkeletalMeshComponent')
            component.SetMobility(NewMobility='Movable')
            component.SetSkeletalMeshAsset(NewMesh=game.unreal_service.load_object(uclass='USkeletalMesh',name=imported['skeletal_mesh']))
            component.SetCollisionEnabled(NewType='NoCollision')
            component.set_property_value(property_name='VisibilityBasedAnimTickOption',property_value='AlwaysTickPoseAndRefreshBones')
            component.SetComponentTickEnabled(bEnabled=True)
            clip=game.unreal_service.load_object(uclass='UAnimSequence',name=imported['animation'])
            component.PlayAnimation(NewAnimToPlay=clip,bLooping=False)
            component.SetPlayRate(Rate=0.)
            component.Stop()
            cam,components=spawn(game,horizontal_fov_deg=85,width=640,height=480)
            apply_camera(cam,view)
            apply_native_rgb_profile(components['rgb'],ue)
        step(setup)
        for _ in range(45):step()
        for degrees in (180,90):
            rows=[]
            for index in (0,len(motion['frames'])//2,len(motion['frames'])-1):
                frame=motion['frames'][index]
                root=root0+np.array(frame['world_root_m'])
                def place():
                    actor.K2_SetActorLocationAndRotation(NewLocation=dict(zip(('X','Y','Z'),to_ue(root))),
                        NewRotation={'Roll':0.,'Pitch':0.,'Yaw':float(degrees)},bSweep=False,bTeleport=True)
                    component.SetPosition(InPos=frame['time_s'],bFireNotifies=False)
                step(place)
                step()
                with instance.begin_frame():
                    raw_head=component.GetSocketTransform(InSocketName=imported['bone_mapping'][motion['head_bone']],TransformSpace='RTS_World',as_dict=True)
                    raw_tail=component.GetSocketTransform(InSocketName=imported['bone_mapping'][motion['tail_bone']],TransformSpace='RTS_World',as_dict=True)
                    root_read=from_ue(xyz(actor.K2_GetActorLocation(as_dict=True)))
                    time_read=float(component.GetPosition())
                with instance.end_frame():
                    rgb=rgb_reader(components['rgb']).copy()
                head=from_ue(matrix(raw_head)[:3,3])
                tail=from_ue(matrix(raw_tail)[:3,3])
                row={'time_s':time_read,'root_world_m':root_read,'head_world_m':head,'tail_world_m':tail}
                if abs(time_read-frame['time_s'])>.002:
                    raise ValueError('negative_control_native_animation_time_differs')
                cf=forward/np.linalg.norm(forward)
                cr=np.cross(cf,[0,1,0]);cr/=np.linalg.norm(cr);cu=np.cross(cr,cf)
                path=args.output/f'negative_{degrees}_{index:04d}.png'
                overlay(rgb[...,::-1],eye,(cf,cr,cu),head,tail,np.asarray(root_read)+[0,.02,0],np.array([0,0,-.08])).save(path)
                row['image']=str(path)
                rows.append(row)
            check=direction_check([f['time_s'] for f in rows],[f['root_world_m'] for f in rows],
                                  [f['head_world_m'] for f in rows],[f['tail_world_m'] for f in rows])
            result['controls'].append({'body_rotation_error_deg':degrees,'frames':rows,'check':check,
                'physically_rotated_native_drawable':True})
        result['status']='pass' if all(c['check']['status']=='fail' for c in result['controls']) else 'fail'
    except Exception as error:
        import traceback
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        (args.output/'native_motion.json').write_text(json.dumps(result,indent=2))
        if actor is not None:
            try:step(lambda:actor.K2_DestroyActor())
            except Exception:pass
        try:close_native(instance,cam,components,game=game)
        except Exception as error:result['cleanup_error']=repr(error)
        (args.output/'native_motion.json').write_text(json.dumps(result,indent=2))
    print(result['status'],flush=True)
    if result['status']!='pass':raise SystemExit(2)


if __name__=='__main__':
    main()
