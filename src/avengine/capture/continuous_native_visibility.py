"""Stream exact v6 modal/target-only pixel truth from native v7 actor state.

Only the auxiliary NONE-stage simulator removes occluders. Published RGB and
world state stay in the admitted house simulator. Dense mask histories are
never retained: native exact-pose masks reuse only the last state per actor.
"""
from __future__ import annotations
import numpy as np
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth


class NativeVisibilityStream:
    def __init__(self,primary,debug_directory=None):
        from avengine.timeline.visual import _instantiate_actor_with_semantic_template
        self.primary=primary;hs=primary.hs;mn=primary.mn
        self.debug_directory=debug_directory
        cfg=hs.SimulatorConfiguration();cfg.scene_id='NONE';cfg.gpu_device_id=-1
        cfg.enable_physics=True;cfg.load_semantic_mesh=False
        sensor=hs.CameraSensorSpec();sensor.uuid='semantic';sensor.sensor_type=hs.SensorType.SEMANTIC
        sensor.resolution=[primary.height,primary.width];sensor.position=mn.Vector3(0.,0.,0.);sensor.hfov=85.
        agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[sensor]
        self.sim=hs.Simulator(hs.Configuration(cfg,[agent]));self.targets=[];self.last={}
        self.rendered_target_masks=0;self.reused_target_masks=0;self.root_max_error=0.;self.joint_max_error=0.
        for i,entry in enumerate(primary.actors):
            material=entry.get('source_material',{});bundle=entry['bundle']
            path=material.get('derived_ao_config_path',bundle.paths_by_role['habitat_ao_config'])
            manager=self.sim.metadata_mediator.ao_template_manager;loaded=manager.load_configs(str(path))
            if len(loaded)!=1:raise ValueError('target-only AO template is not unique')
            handle=manager.get_template_handle_by_id(loaded[0])
            kwargs={'shader_type':'pbr' if material else 'flat'}
            if material.get('derived_glb_path'):kwargs['render_asset_path']=material['derived_glb_path']
            obj,_=_instantiate_actor_with_semantic_template(self.sim,bundle=bundle,habitat_sim=hs,
                base_handle=handle,semantic_id=entry['semantic_id'],actor_index=i,**kwargs)
            obj.translation=mn.Vector3(2000.,2000.,2000.)
            self.targets.append({'object':obj,'entry':entry})
        self.truth=None

    def capture(self,index,modal,actor_rows):
        from avengine.assets.habitat_capture import _apply_root_with_habitat
        primary=self.primary;state=primary.sim.get_agent(0).get_state();self.sim.get_agent(0).set_state(state)
        masks={};ids={};actor_ids={r['endpoint_id']:str(r['actor_id']) for r in actor_rows}
        for target in self.targets:
            entry=target['entry'];obj=target['object'];actor=actor_ids[entry['endpoint_id']];sid=int(entry['semantic_id'])
            root=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
            joints=np.asarray(entry['object'].joint_positions,float)
            pose=(root.copy(),joints.copy(),np.asarray(state.position).copy(),np.asarray([state.rotation.w,*state.rotation.imag]).copy())
            old=self.last.get(actor)
            if old is not None and all(np.array_equal(a,b) for a,b in zip(pose,old[0],strict=True)):
                mask=old[1];self.reused_target_masks+=1
            else:
                _apply_root_with_habitat(obj,root,qt=primary.qt,mn=primary.mn)
                obj.joint_positions=joints
                root_error=float(np.max(np.abs(np.asarray(obj.root_scene_node.absolute_transformation())-root)))
                joint_error=float(np.max(np.abs(np.asarray(obj.joint_positions)-joints)))
                self.root_max_error=max(self.root_max_error,root_error);self.joint_max_error=max(self.joint_max_error,joint_error)
                if root_error>1e-5 or joint_error>1e-5:raise ValueError('target-only native pose differs from modal actor')
                raw=np.asarray(self.sim.get_sensor_observations()['semantic'])
                # Native blank-stage background is converted to namespace zero.
                foreign=set(np.unique(raw))-{0,-1,np.iinfo(np.uint32).max,sid}
                if foreign:raise ValueError('target-only pass contains another physical actor')
                mask=np.where(raw==sid,sid,0).astype(np.int32);obj.translation=primary.mn.Vector3(2000.,2000.,2000.)
                self.last[actor]=(pose,mask);self.rendered_target_masks+=1
            masks[actor]=[mask];ids[actor]=sid
        common={'renderer_backend':'habitat_native','rgb_renderer_backend':'habitat_native',
            'camera_contract_id':'v7_native_camera_listener','semantic_id_namespace':'habitat_object_semantic_id_v1',
            'resolution_hw':[primary.height,primary.width],'frame_indices':[index],'camera_pose_ids':[f'v7-camera-frame-{index:06d}']}
        if self.debug_directory is not None:
            from pathlib import Path
            path=Path(self.debug_directory);path.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path/f'visibility_{index:04d}.npz',modal=modal,**{f'target_{a}':v[0] for a,v in masks.items()})
        result=compile_pixel_visibility_truth(normal_semantic_masks=[np.asarray(modal).astype(np.int32,copy=False)],
            target_only_semantic_masks_by_instance=masks,semantic_ids_by_instance=ids,
            normal_context={'pass_kind':'modal_scene',**common},
            target_only_contexts_by_instance={actor:{'pass_kind':'target_only','target_instance_id':actor,**common} for actor in ids})
        if self.truth is None:self.truth=result
        else:
            self.truth['frame_indices'].extend(result['frame_indices']);self.truth['camera_pose_ids'].extend(result['camera_pose_ids'])
            for actor,value in result['per_instance'].items():
                existing=self.truth['per_instance'][actor];existing['frames'].extend(value['frames'])
                for key,count in value['state_counts'].items():existing['state_counts'][key]+=count
        return result

    def receipt(self):
        return {'mode':'native_NONE_stage_target_only_auxiliary','published_RGB_scene':'retained admitted house',
            'compute_device':'CPU','root_max_error':self.root_max_error,'joint_max_error':self.joint_max_error,
            'rendered_native_masks':self.rendered_target_masks,'exact_pose_reused_masks':self.reused_target_masks,
            'dense_mask_history_retained':False,'occlusion_criterion':'unchanged avengine.qa.pixel_visibility'}

    def close(self):self.sim.close()
