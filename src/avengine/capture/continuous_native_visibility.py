"""Stream exact v6 modal/target-only pixel truth from native v7 actor state.

Only the auxiliary NONE-stage simulator removes occluders. Published RGB and
world state stay in the admitted house simulator. Dense mask histories are
never retained: native exact-pose masks reuse only the last state per actor.
"""
from __future__ import annotations
import numpy as np

VISIBILITY_REVISION = "native_modal_target_only_controls_v1_unqualified"
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth


class NativeVisibilityStream:
    def __init__(self,primary,debug_directory=None):
        from avengine.timeline.visual import _instantiate_actor_with_semantic_template
        self.primary=primary;hs=primary.hs;mn=primary.mn
        self.debug_directory=debug_directory
        self.targets=[];self.last={}
        self.rendered_target_masks=0;self.reused_target_masks=0;self.root_max_error=0.;self.joint_max_error=0.
        self.camera_max_error=0.;self.link_max_error=0.
        from avengine.assets.source_materials import configure_source_pbr
        # Match the v6 native target-only method: one blank simulator per
        # target, with no other AO and no large-coordinate relocation.
        for i,entry in enumerate(primary.actors):
            cfg=hs.SimulatorConfiguration();cfg.scene_id='NONE';cfg.gpu_device_id=-1
            cfg.enable_physics=True;cfg.load_semantic_mesh=False
            sensor=hs.CameraSensorSpec();sensor.uuid='semantic';sensor.sensor_type=hs.SensorType.SEMANTIC
            sensor.resolution=[primary.height,primary.width];sensor.position=mn.Vector3(0.,0.,0.);sensor.hfov=85.
            agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[sensor]
            configuration=hs.Configuration(cfg,[agent]);sim=hs.Simulator(configuration)
            try:
                configure_source_pbr(configuration,hs,metadata_mediator=sim.metadata_mediator)
                material=entry.get('source_material',{});bundle=entry['bundle']
                path=material.get('derived_ao_config_path',bundle.paths_by_role['habitat_ao_config'])
                manager=sim.metadata_mediator.ao_template_manager;loaded=manager.load_configs(str(path))
                if len(loaded)!=1:raise ValueError('target-only AO template is not unique')
                handle=manager.get_template_handle_by_id(loaded[0])
                kwargs={'shader_type':'pbr' if material else 'flat'}
                if material.get('derived_path'):kwargs['render_asset_path']=material['derived_path']
                obj,_=_instantiate_actor_with_semantic_template(sim,bundle=bundle,habitat_sim=hs,
                    base_handle=handle,semantic_id=entry['semantic_id'],actor_index=i,**kwargs)
                self.targets.append({'sim':sim,'object':obj,'entry':entry})
            except BaseException:
                sim.close();self.close();raise
        self.truth=None

    def capture(self,index,modal,actor_rows):
        primary=self.primary;state=primary.sim.get_agent(0).get_state()
        masks={};ids={};actor_ids={r['endpoint_id']:str(r['actor_id']) for r in actor_rows}
        for target in self.targets:
            sim=target['sim'];sim.get_agent(0).set_state(state)
            observed=sim.get_agent(0).get_state().sensor_states['semantic'];expected=state.sensor_states['semantic']
            camera_error=max(float(np.max(np.abs(observed.position-expected.position))),
                float(np.max(np.abs(np.array([observed.rotation.w,*observed.rotation.imag])-np.array([expected.rotation.w,*expected.rotation.imag])))))
            self.camera_max_error=max(self.camera_max_error,camera_error)
            if camera_error>1e-6:raise ValueError('target-only native camera differs from modal camera')
            entry=target['entry'];obj=target['object'];actor=actor_ids[entry['endpoint_id']];sid=int(entry['semantic_id'])
            root=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
            joints=np.asarray(entry['object'].joint_positions,float)
            pose=(root.copy(),joints.copy(),np.asarray(state.position).copy(),np.asarray([state.rotation.w,*state.rotation.imag]).copy())
            old=self.last.get(actor)
            if old is not None and all(np.array_equal(a,b) for a,b in zip(pose,old[0],strict=True)):
                mask=old[1];self.reused_target_masks+=1
            else:
                # Reapply the exact original double-precision inputs in the
                # same call order as the modal renderer. Actual native
                # transforms below must still match; no analytic mask is used.
                from avengine.assets.habitat_capture import _apply_root_with_habitat
                obj.joint_positions=entry['applied_joint_input']
                _apply_root_with_habitat(obj,entry['applied_root_input'],qt=primary.qt,mn=primary.mn)
                root_error=float(np.max(np.abs(np.asarray(obj.root_scene_node.absolute_transformation())-root)))
                joint_error=float(np.max(np.abs(np.asarray(obj.joint_positions)-joints)))
                self.root_max_error=max(self.root_max_error,root_error);self.joint_max_error=max(self.joint_max_error,joint_error)
                if root_error>1e-5 or joint_error>1e-5:raise ValueError('target-only native pose differs from modal actor')
                source=entry['object']
                errors=[float(np.max(np.abs(np.asarray(obj.get_link_scene_node(k).absolute_transformation())-np.asarray(source.get_link_scene_node(k).absolute_transformation())))) for k in source.get_link_ids()]
                self.link_max_error=max(self.link_max_error,max(errors,default=0.))
                if max(errors,default=0.)>1e-5:raise ValueError('target-only native link transforms differ')
                raw=np.asarray(sim.get_sensor_observations()['semantic'])
                # Native blank-stage background is converted to namespace zero.
                foreign=set(np.unique(raw))-{0,-1,np.iinfo(np.uint32).max,sid}
                if foreign:raise ValueError('target-only pass contains another physical actor')
                mask=np.where(raw==sid,sid,0).astype(np.int32)
                self.last[actor]=(pose,mask);self.rendered_target_masks+=1
            masks[actor]=[mask];ids[actor]=sid
        common={'renderer_backend':'habitat_native','rgb_renderer_backend':'habitat_native',
            'camera_contract_id':'v7_native_camera_listener','semantic_id_namespace':'habitat_object_semantic_id_v1',
            'resolution_hw':[primary.height,primary.width],'frame_indices':[index],'camera_pose_ids':[f'v7-camera-frame-{index:06d}']}
        inconsistent=any(np.any((np.asarray(modal)==ids[a]) & (v[0]!=ids[a])) for a,v in masks.items())
        if self.debug_directory is not None and inconsistent:
            from pathlib import Path
            path=Path(self.debug_directory);path.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path/f'visibility_{index:04d}.npz',modal=modal,**{f'target_{a}':v[0] for a,v in masks.items()})
            from avengine.timeline.continuous_pilot import write_json
            write_json(path/f'visibility_{index:04d}_alignment.json',self.receipt())
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
            'camera_max_error':self.camera_max_error,'link_max_error':self.link_max_error,
            'rendered_native_masks':self.rendered_target_masks,'exact_pose_reused_masks':self.reused_target_masks,
            'dense_mask_history_retained':False,'occlusion_criterion':'unchanged avengine.qa.pixel_visibility'}

    def close(self):
        for target in self.targets:target['sim'].close()
        self.targets=[]
