"""Native Habitat replay and full original-triangle animal motion checks."""
from pathlib import Path
import json,math,subprocess,time,os
import numpy as np
from avengine.assets.animal_motion.stage import OriginalVerticalStageRay,original_native_stage_mesh
from avengine.assets.roster_motion.episode import fcl_model,triangle_mesh,externalize_unchanged_images
from avengine.assets.roster_motion.contact import yaw_matrix
from avengine.assets.roster_motion.pose import rigid_glb
from avengine.assets.animal_motion.profiles import Rig,PAWS
from avengine.assets.animal_motion.walk import ConstrainedWalk
from avengine.assets.animal_motion.checks import frame_checks,direction_negative_controls

class StageClearance:
    def __init__(self,mesh,reference):
        import fcl
        self.fcl=fcl;self.vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
        t=self.vertices[faces];normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm=np.linalg.norm(normal,axis=1)
        ground=(np.abs(normal[:,1])>.98*np.maximum(norm,1e-20))&(np.max(np.abs(t[:,:,1]-reference),axis=1)<.02)
        self.faces=faces[~ground];self.lo=t[~ground].min(1);self.hi=t[~ground].max(1);self.floor_triangles=int(ground.sum())
    def check(self,vertices,faces):
        eps=.0200001;inside=np.all(self.lo<=vertices.max(0)+eps,axis=1)&np.all(self.hi>=vertices.min(0)-eps,axis=1)
        if not inside.any():return eps
        fcl=self.fcl;selected=self.faces[inside];unique,indices=np.unique(selected,return_inverse=True)
        stage=fcl_model(self.vertices[unique],indices.reshape(-1,3));obj=fcl_model(vertices,faces)
        if fcl.collide(obj,stage,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult()):raise ValueError('original_nonfloor_triangle_collision')
        distance=float(fcl.distance(obj,stage,fcl.DistanceRequest(),fcl.DistanceResult()))
        if distance<=.02:raise ValueError('original_nonfloor_triangle_clearance_20mm:'+str(distance))
        return min(eps,distance)


def camera(video,eye,target):
    f=np.asarray(target)-eye;f=f/np.linalg.norm(f)
    r=np.cross(f,[0.,1.,0.]);r=r/np.linalg.norm(r);u=np.cross(r,f)
    state=video.hs.AgentState();state.position=np.asarray(eye,dtype=np.float32)
    state.rotation=video.qt.from_rotation_matrix(np.stack((r,u,-f),axis=1));video.sim.get_agent(0).set_state(state)
    return np.asarray(eye),f,r,u


def overlay(rgb,eye,basis,head,tail,root,velocity):
    from PIL import Image,ImageDraw
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);f,r,u=basis;hh,ww=rgb.shape[:2];focal=ww/(2*math.tan(math.radians(85/2)))
    def project(p):
        d=np.asarray(p)-eye;depth=d@f
        if depth<=.03:return None
        return (float(ww/2+focal*(d@r)/depth),float(hh/2-focal*(d@u)/depth))
    def arrow(a,b,color,width=3):
        x,y=project(a),project(b)
        if x is None or y is None:return
        draw.line([x,y],fill=color,width=width);v=np.array(x)-y;n=np.linalg.norm(v)
        if n>0:
            v=v/n*10;p=np.array([-v[1],v[0]])*.4;draw.polygon([y,tuple(np.array(y)+v+p),tuple(np.array(y)+v-p)],fill=color)
    arrow(tail,head,(0,255,140));arrow(root,root+np.asarray(velocity)*3,(255,70,55))
    draw.text((8,8),'GREEN: tail -> head | RED: velocity x3s',fill=(255,255,255))
    return im


def instantiate(video,path,root,semantic=700001):
    from avengine.assets.habitat_capture import _apply_root_with_habitat
    tm=video.sim.get_object_template_manager();a=tm.create_new_template(str(path));a.render_asset_handle=str(path);a.collision_asset_handle=str(path)
    a.compute_COM_from_shape=False;a.is_collidable=False;a.shader_type='pbr';tid=tm.register_template(a)
    obj=video.sim.get_rigid_object_manager().add_object_by_template_id(tid);obj.motion_type=video.hs.physics.MotionType.KINEMATIC;obj.semantic_id=semantic
    _apply_root_with_habitat(obj,root,qt=video.qt,mn=video.mn)
    return obj


def run_habitat(record,house,inputs,route,output,*,speed=.08,period=2.4,seconds=4.8,mode="walk_animation",frame_indices=None,pose_stream=None,skin_stream=None):
    from avengine.capture.continuous_native_video import NativeContinuousVideo
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    start=time.process_time();p0=np.asarray(route['root_m'],float);yaw=float(route['yaw_rad']);rotation=yaw_matrix(yaw)
    eye=np.asarray(house['room']['placement']['camera_m'],float)
    target=p0+rotation@np.array([0.,.3,-speed*seconds/2]);forward=target-eye;forward/=np.linalg.norm(forward)
    scene={'listener_m':eye.tolist(),'camera_forward':forward.tolist(),'sources':{}}
    video=NativeContinuousVideo(house,scene,inputs,[],width=512,height=384,source_material_cache_directory=output/'material_cache')
    result={'asset_id':record['asset_id'],'status':'pending','backend':'Habitat CPU llvmpipe','house':house['house'],'motion_mode':mode,'frames':[],'audio_status':'not_run'}
    encoder=None;skin_frames=[];skin_indices=[]
    try:
        path=Path(house['input'].get('visual_source') or Path(house['input']['scene_directory'])/(house['input']['scan_id']+'.glb'))
        mesh,transform=original_native_stage_mesh(video,path);rays=OriginalVerticalStageRay(path,mesh=mesh,transform_evidence=transform);clearance=StageClearance(mesh,p0[1])
        class RouteFloor:
            def many(self,points):return rays.many(np.asarray(points)@rotation.T+p0)-p0[1]
            def __call__(self,p):return float(self.many([p])[0])
        floor=RouteFloor()
        if mode=='walk_animation':
            rig=Rig.load(record);walk=ConstrainedWalk(rig,speed=speed,period=period,floor=floor)
        elif mode=='slide_no_animation':
            from avengine.assets.animal_motion.slide import StandingSlide
            walk=StandingSlide(record,speed=speed,floor=floor);rig=walk.rig
        else:raise ValueError('unsupported_animal_motion_mode')
        ed=externalize_unchanged_images(rig.document,rig.binary,output/'textures')
        eye,cf,cr,cu=camera(video,eye,target);n=round(seconds*30)+1;selected={0,n//4,n//2,3*n//4,n-1};images=[]
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','512x384','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(output/'motion.mp4')]
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        indices=list(range(n)) if frame_indices is None else list(frame_indices)
        if len(indices)<2 or indices!=sorted(set(indices)) or indices[0]<0 or indices[-1]>=n:raise ValueError('invalid_native_frame_indices')
        for i in indices:
            t=i/30
            parts,row=walk.frame(float(t));v,f=triangle_mesh(rig.document,rig.binary,parts);root=p0+rotation@np.asarray(row['world_root_m']);world=v@rotation.T+root
            row['minimum_nonfloor_clearance_m']=clearance.check(world,f)
            near=world[world[:,1]<=p0[1]+.08]
            if not len(near):raise ValueError('no_near_floor_skin_vertices')
            floor_values=rays.many(near);row['body_floor_min_m']=float(np.min(near[:,1]-floor_values))
            snap=np.asarray(video.pathfinder.snap_point(root));nav=float(np.linalg.norm((snap-root)[[0,2]]))
            if not np.isfinite(nav):raise ValueError('native_navigation_missing')
            row['navmesh_horizontal_error_m']=nav
            matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=root
            path=output/'frames'/('standing.glb' if mode=='slide_no_animation' else f'{i:04d}.glb')
            retained=pose_stream is None or i in selected or mode=='slide_no_animation'
            if not retained:
                path=output/'frames'/f'{i:04d}.stream.glb'
                working=Path(pose_stream);working.parent.mkdir(parents=True,exist_ok=True)
                rigid_glb(ed,rig.binary,parts,working)
                os.link(working,path)
            elif not path.exists():rigid_glb(ed,rig.binary,parts,path)
            if skin_stream is not None:
                skin_frames.append(v.astype(np.float32));skin_indices.append(i)
            obj=instantiate(video,path,matrix)
            native=np.asarray(obj.root_scene_node.absolute_transformation(),float)
            if pose_stream is not None:
                # Read the rendered resource's local bounding box. A fresh path
                # must load this pose, even when its private backing inode is reused.
                obj.root_scene_node.compute_cumulative_bb()
                bb=obj.root_scene_node.cumulative_bb
                actual=np.stack((np.asarray(bb.min,float),np.asarray(bb.max,float)))
                expected=np.stack((v.min(0),v.max(0)))
                bound_error=float(np.max(np.abs(actual-expected)))
                if bound_error>1e-5:raise ValueError('native_drawable_pose_bounds_mismatch:'+str(bound_error))
                row['native_drawable_bounds_error_m']=bound_error
            # Attachment has two native levels: actual animated head/mouth
            # bone and its local mouth offset, following the target head.
            bone=obj.root_scene_node.create_child();bone.transformation=video.mn.Matrix4(np.asarray(row['mouth_bone_matrix']))
            mouth=bone.create_child();mouth.translation=video.mn.Vector3(row['mouth_bone_local_m'])
            actual_mouth=np.asarray(mouth.absolute_transformation(),float)[:3,3]
            head_node=obj.root_scene_node.create_child();head_node.translation=video.mn.Vector3(row['head_m'])
            tail_node=obj.root_scene_node.create_child();tail_node.translation=video.mn.Vector3(row['tail_m'])
            row.update(frame_index=i,root_world_m=native[:3,3].tolist(),head_world_m=np.asarray(head_node.absolute_translation).tolist(),tail_world_m=np.asarray(tail_node.absolute_translation).tolist(),
                       mouth_world_m=actual_mouth.tolist(),pose_glb=str(path) if retained else None,pose_storage='retained' if retained else 'streamed_inplace_not_retained',native_root_error_m=float(np.max(np.abs(native-matrix))),
                       native_emitter_readback_error_m=float(np.linalg.norm(actual_mouth-(native@np.r_[row['mouth_m'],1.])[:3])))
            for foot in row['feet'].values():
                for k in ('support_world_m','support_target_world_m','sole_world_m'):foot[k]=(p0+rotation@np.asarray(foot[k])).tolist()
                wp=np.asarray(foot['sole_world_m']);ray=video.hs.geo.Ray(video.mn.Vector3(wp+[0.,.2,0.]),video.mn.Vector3(0.,-1.,0.))
                hits=[h for h in video.sim.cast_ray(ray,.85).hits if int(h.object_id)!=obj.object_id]
                if not hits:raise ValueError('native_paw_floor_ray_missing')
                fy=float(hits[0].point[1]);exact=float(rays.many([wp])[0])
                if abs(fy-exact)>.0001:raise ValueError('original_and_native_floor_ray_disagree')
                foot.update(floor_delta_m=float(wp[1]-fy),native_floor_y_m=fy,original_floor_y_m=exact)
            observations=video.sim.get_sensor_observations();rgb=np.asarray(observations['rgb'])[...,:3];mask=np.asarray(observations['semantic'])==700001
            row['visible_pixels']=int(mask.sum())
            yy,xx=np.where(mask);row['complete_silhouette_in_frame']=bool(len(xx)>100 and min(xx.min(),yy.min(),511-xx.max(),383-yy.max())>=3)
            encoder.stdin.write(rgb.tobytes())
            if i in selected:
                ip=output/f'frame_{i:04d}.png';overlay(rgb,eye,(cf,cr,cu),row['head_world_m'],row['tail_world_m'],np.asarray(row['root_world_m'])+[0.,.02,0.],rotation@np.array([0.,0.,-speed])).save(ip);images.append(str(ip))
            result['frames'].append(row);video.sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
            if i%30==0:print('NATIVE_FRAME',record['asset_id'],i,'skin_floor',row['body_floor_min_m'],'pixels',row['visible_pixels'],flush=True)
        encoder.stdin.close();code=encoder.wait();encoder=None
        if code:raise RuntimeError('video_encoding_failed')
        checks=frame_checks(result['frames'],native=True,mode=mode);negative=direction_negative_controls(result['frames'])
        if not all(f['complete_silhouette_in_frame'] for f in result['frames']):checks['status']='fail';checks['failures'].append('native_silhouette_clipped_or_under_100_pixels')
        result.update(status='geometry_pass_audio_pending' if checks['status']==negative['status']=='pass' else 'geometry_fail',checks=checks,direction_negative_controls=negative,
             renderer=video.receipt,stage_transform=transform,source_master=str(rig.path),source_provenance=rig.provenance,support_plane=walk.support_plane,
             listener_m=eye.tolist(),camera_forward=cf.tolist(),images=images,frame_count=len(result['frames']),native_stage_floor_triangles=clearance.floor_triangles,
             max_emitter_error_m=max(f['native_emitter_readback_error_m'] for f in result['frames']),min_visible_pixels=min(f['visible_pixels'] for f in result['frames']))
    except Exception as error:
        import traceback
        result.update(status='failed',error=repr(error),failure_kind='motion_rejected' if isinstance(error,ValueError) else 'execution_error',traceback=traceback.format_exc())
    finally:
        if encoder is not None:
            if encoder.stdin:encoder.stdin.close()
            encoder.wait()
        video.close()
        if skin_stream is not None and skin_frames:
            np.savez_compressed(Path(skin_stream),vertices=np.stack(skin_frames),indices=np.asarray(skin_indices),faces=f)
            result['skin_stream']=str(skin_stream);result['skin_stream_note']='Private overwriteable QA stream; its SHA and per-frame results are retained by the bounded replay coordinator.'
        result['cpu_s']=time.process_time()-start;(output/'native_motion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
