"""Native v6 pixel truth using one camera/context and OBJECTS_ONLY target passes.

The static stage is omitted by the existing native draw flag. Non-target AOs
are temporarily moved outside view and restored before the world is observed
again. No simulation step or saved RGB/mask correction is performed.
"""
from __future__ import annotations
import numpy as np
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth

VISIBILITY_REVISION='native_same_context_objects_only_v2'


class NativeVisibilityStream:
    def __init__(self,primary,debug_directory=None):
        self.primary=primary;self.debug_directory=debug_directory;self.last={};self.truth=None
        self.rendered_target_masks=0;self.reused_target_masks=0
        self.root_max_error=self.joint_max_error=self.camera_max_error=self.link_max_error=0.
        self.restored_modal_frames=0
        self.occluder_frames=[]

    def _target_mask(self,target):
        p=self.primary;hidden=[a for a in p.actors if a is not target]
        previous=[{'root':np.asarray(a['object'].root_scene_node.absolute_transformation()).copy(),
                   'joints':np.asarray(a['object'].joint_positions).copy(),
                   'links':{k:np.asarray(a['object'].get_link_scene_node(k).absolute_transformation()).copy() for k in a['object'].get_link_ids()}}
                  for a in hidden]
        sensor=p.sim._sensors['semantic']._sensor_object
        camera=np.asarray(sensor.render_camera.camera_matrix).copy()
        try:
            for i,a in enumerate(hidden):a['object'].translation=p.mn.Vector3(2000+i*10,2000,2000)
            data=np.empty((p.height,p.width),dtype=np.uint32)
            view=p.mn.MutableImageView2D(p.mn.PixelFormat.R32UI,(p.width,p.height),memoryview(data).cast('B'))
            with sensor.render_target:
                p.sim.renderer.draw(sensor.render_camera,p.sim.get_active_scene_graph(),
                    p.hs.gfx.Camera.Flags.OBJECTS_ONLY|p.hs.gfx.Camera.Flags.FRUSTUM_CULLING)
                sensor.render_target.read_frame_object_id(view)
            data=np.flipud(data).copy();sid=int(target['semantic_id'])
            foreign=set(np.unique(data))-{0,2**32-1,sid}
            if foreign:raise ValueError('target-only native pass contains another actor or stage')
            return np.where(data==sid,sid,0).astype(np.int32)
        finally:
            from avengine.assets.habitat_capture import _apply_root_with_habitat
            for a,old in zip(hidden,previous,strict=True):
                if a.get('native_actor_kind')=='registered_rigid':
                    # Target passes changed translation only. Retain the native
                    # rotation exactly instead of normalizing its quaternion twice.
                    a['object'].translation=p.mn.Vector3(old['root'][:3,3])
                else:
                    _apply_root_with_habitat(a['object'],a['applied_root_input'],qt=p.qt,mn=p.mn)
                obj=a['object']
                root_error=float(np.max(np.abs(np.asarray(obj.root_scene_node.absolute_transformation())-old['root'])))
                joint_delta=np.abs(np.asarray(obj.joint_positions)-old['joints'])
                joint_error=float(np.max(joint_delta)) if joint_delta.size else 0.
                link_error=max((float(np.max(np.abs(np.asarray(obj.get_link_scene_node(k).absolute_transformation())-v))) for k,v in old['links'].items()),default=0.)
                self.root_max_error=max(self.root_max_error,root_error);self.joint_max_error=max(self.joint_max_error,joint_error);self.link_max_error=max(self.link_max_error,link_error)
                if root_error or joint_error or link_error:raise ValueError('target-only pass failed exact native actor restoration')
            error=float(np.max(np.abs(np.asarray(sensor.render_camera.camera_matrix)-camera)))
            self.camera_max_error=max(self.camera_max_error,error)
            if error:raise ValueError('target-only pass changed the native camera')

    def capture(self,index,modal,actor_rows):
        p=self.primary;actor_ids={r['endpoint_id']:str(r['actor_id']) for r in actor_rows}
        # Own copies avoid comparing two views of a mutable sensor buffer.
        observed={k:np.asarray(v).copy() for k,v in p.sim.get_sensor_observations().items()}
        if not np.array_equal(observed['semantic'],modal):raise ValueError('caller modal pixels do not match current native state')
        masks={};ids={}
        camera=p.sim.get_agent(0).get_state()
        for entry in p.actors:
            obj=entry['object'];actor=actor_ids[entry['endpoint_id']];sid=int(entry['semantic_id'])
            pose=(np.asarray(obj.root_scene_node.absolute_transformation()).copy(),np.asarray(obj.joint_positions).copy(),
                  np.asarray(camera.position).copy(),np.asarray([camera.rotation.w,*camera.rotation.imag]).copy())
            old=self.last.get(actor)
            if old is not None and all(np.array_equal(a,b) for a,b in zip(pose,old[0],strict=True)):
                mask=old[1];self.reused_target_masks+=1
            else:
                mask=self._target_mask(entry);self.last[actor]=(pose,mask);self.rendered_target_masks+=1
            masks[actor]=[mask];ids[actor]=sid
        # IDs are measured from the context pixels inside each actual target
        # projection. Unregistered stage pixels are counted explicitly, never
        # inferred as a unique occluder or silently replaced by zero.
        registry_ids={sid:actor for actor,sid in ids.items()}
        for actor,values in masks.items():
            target=values[0]==ids[actor]
            hidden=target & (np.asarray(modal)!=ids[actor])
            values_seen,counts=np.unique(np.asarray(modal)[hidden],return_counts=True)
            known=[];unknown=0
            for sid,count in zip(values_seen,counts,strict=True):
                other=registry_ids.get(int(sid))
                if other is not None and other!=actor:known.append(other)
                else:unknown+=int(count)
            self.occluder_frames.append({'target_instance_id':actor,'frame_index':index,
                'occluder_instance_ids':sorted(set(known)),'unknown_occluder_pixels':unknown,
                'context_pixels':int(np.count_nonzero(np.asarray(modal)==ids[actor])),
                'target_only_pixels':int(np.count_nonzero(target)),
                'occluded_pixels':int(np.count_nonzero(hidden)),
                'identity_source':'actual_context_semantic_IDs_within_native_OBJECTS_ONLY_target_mask'})
        restored=p.sim.get_sensor_observations()
        if any(not np.array_equal(observed[k],restored[k]) for k in observed):
            raise ValueError('target-only controls changed published native RGB/depth/semantic')
        self.restored_modal_frames+=1
        common={'renderer_backend':'habitat_native','rgb_renderer_backend':'habitat_native',
            'camera_contract_id':'v7_native_camera_listener','semantic_id_namespace':'habitat_object_semantic_id_v1',
            'resolution_hw':[p.height,p.width],'frame_indices':[index],'camera_pose_ids':[f'v7-camera-frame-{index:06d}']}
        inconsistent=any(np.any((np.asarray(modal)==ids[a])&(v[0]!=ids[a])) for a,v in masks.items())
        if self.debug_directory is not None and inconsistent:
            from pathlib import Path
            root=Path(self.debug_directory);root.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(root/f'visibility_{index:04d}.npz',modal=modal,**{f'target_{a}':v[0] for a,v in masks.items()})
        result=compile_pixel_visibility_truth(normal_semantic_masks=[np.asarray(modal).astype(np.int32,copy=False)],
            target_only_semantic_masks_by_instance=masks,semantic_ids_by_instance=ids,
            normal_context={'pass_kind':'modal_scene',**common},target_only_contexts_by_instance={
                actor:{'pass_kind':'target_only','target_instance_id':actor,**common} for actor in ids})
        if self.truth is None:self.truth=result
        else:
            self.truth['frame_indices'].extend(result['frame_indices']);self.truth['camera_pose_ids'].extend(result['camera_pose_ids'])
            for actor,value in result['per_instance'].items():
                existing=self.truth['per_instance'][actor];existing['frames'].extend(value['frames'])
                for key,count in value['state_counts'].items():existing['state_counts'][key]+=count
        return result

    def receipt(self):
        return {'mode':'native_same_context_OBJECTS_ONLY_target_pass','published_RGB_scene':'retained admitted house',
            'compute_device':'CPU','root_max_error':self.root_max_error,'joint_max_error':self.joint_max_error,
            'camera_max_error':self.camera_max_error,'link_max_error':self.link_max_error,
            'rendered_native_masks':self.rendered_target_masks,'exact_pose_reused_masks':self.reused_target_masks,
            'restored_modal_RGB_depth_semantic_frames':self.restored_modal_frames,
            'dense_mask_history_retained':False,'occlusion_criterion':'unchanged avengine.qa.pixel_visibility'}

    def close(self):pass
