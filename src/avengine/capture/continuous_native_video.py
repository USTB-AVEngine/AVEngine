"""Streaming Habitat RGB/depth/semantic with native M2 people and CPU EGL.

The renderer emits observed mouth trajectories before audio synthesis. It
never substitutes planned centres or an analytic body mask for native pixels.
No dense RGB/depth/semantic scene arrays are retained on persistent storage.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np

from avengine.timeline.continuous_scene import FPS, camera_basis, position_at
from avengine.timeline.continuous_pilot import write_json


class NativeContinuousVideo:
    def __init__(self, house, scene, inputs, assets, *, width=1280, height=720,
                 source_material_cache_directory=None):
        infra_gpu = os.environ.get('CCI_INFRA_RENDER_GPU')
        if infra_gpu is not None:
            if (os.environ.get('CCI_INFRASTRUCTURE_TRIAL') != '1' or
                    not infra_gpu.isdigit() or os.environ.get('CUDA_VISIBLE_DEVICES') != infra_gpu):
                raise RuntimeError('GPU rendering requires the explicit single-device infrastructure lease')
            if os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1' or os.environ.get('LD_PRELOAD'):
                raise RuntimeError('GPU infrastructure renderer cannot inherit a software EGL override')
        elif os.environ.get('LIBGL_ALWAYS_SOFTWARE') != '1' or not os.environ.get('LD_PRELOAD'):
            raise RuntimeError('CPU renderer requires the task-local software-only EGL selector')
        os.environ.setdefault('MESA_SHADER_CACHE_DISABLE','true')
        from avengine.acoustics.runtime import load_habitat_runtime
        from avengine.assets.contracts import load_and_validate_inputs
        from avengine.assets.habitat_capture import load_runtime_asset_bundle
        from avengine.timeline.visual import _instantiate_actor_with_semantic_template
        from avengine.assets.source_materials import (
            configure_source_pbr, prepare_source_glb, prepare_source_ao_config, read_glb,
        )
        hs, runtime=load_habitat_runtime(runtime_mode='current-installed', **inputs['runtime'])
        import magnum as mn
        import quaternion as qt
        self.hs,self.mn,self.qt=hs,mn,qt
        self.width,self.height=width,height
        cfg=hs.SimulatorConfiguration()
        cfg.scene_id=house['input'].get('visual_source') or str(Path(house['input']['scene_directory'])/(house['input']['scan_id']+'.glb'))
        cfg.gpu_device_id=int(os.environ.get('CCI_INFRA_HABITAT_DEVICE_ID',infra_gpu)) if infra_gpu is not None else -1
        cfg.enable_physics=True
        cfg.load_semantic_mesh=False
        sensors=[]
        for name,kind in [('rgb',hs.SensorType.COLOR),('depth',hs.SensorType.DEPTH),('semantic',hs.SensorType.SEMANTIC)]:
            spec=hs.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
            spec.resolution=[height,width];spec.position=mn.Vector3(0.,0.,0.)
            spec.hfov=85.;sensors.append(spec)
        agent=hs.agent.AgentConfiguration();agent.sensor_specifications=sensors
        read_glb(cfg.scene_id)  # Reject Basis before entering a native importer.
        configuration=hs.Configuration(cfg,[agent])
        self.sim=hs.Simulator(configuration)
        maps=Path('/proc/self/maps').read_text()
        if infra_gpu is None:
            if 'swrast_dri.so' not in maps or 'libEGL_nvidia' in maps or 'libnvidia-gl' in maps:
                self.sim.close();raise RuntimeError('native CPU renderer did not load only the software driver')
        elif 'swrast_dri.so' in maps or not any(x in maps for x in ('libEGL_nvidia','libnvidia-gl')):
            self.sim.close();raise RuntimeError('GPU infrastructure renderer did not load the NVIDIA EGL driver')
        # The scan may itself use native PBR. Configure sources after the scan
        # has loaded so its existing drawables keep the original PBR handle.
        self.material_rendering=configure_source_pbr(configuration,hs,
                                                   metadata_mediator=self.sim.metadata_mediator)
        f,r,u=camera_basis(scene['camera_forward'])
        state=hs.AgentState();state.position=np.asarray(scene['listener_m'],dtype=np.float32)
        state.rotation=qt.from_rotation_matrix(np.stack((r,u,-f),axis=1))
        self.sim.get_agent(0).set_state(state)
        pf=hs.PathFinder()
        if not pf.load_nav_mesh(house['input']['navmesh_source']):
            raise ValueError('read-only admitted navigation input is missing')
        self.pathfinder=pf
        self.actors=[]
        self.material_checks=[]
        cache=source_material_cache_directory or os.environ.get('AVENGINE_SOURCE_MATERIAL_CACHE')
        if cache is None:
            import tempfile
            cache=tempfile.mkdtemp(prefix='avengine-source-materials-')
        self.source_material_cache_directory=Path(cache)
        for i,asset in enumerate(assets):
            # Retain original admission, hashes and M2 semantics. A candidate
            # package is rendered only for a named qualification canary.
            validated=load_and_validate_inputs(asset['asset_manifest_path'],asset['base_m2_request_path'],allow_research_candidate=True)
            bundle=load_runtime_asset_bundle(validated)
            derived,material=prepare_source_glb(bundle.paths_by_role['visual'],asset_id=asset['asset_id'],
                entity_class=validated.asset.get('entity_class',asset.get('entity_class','articulated_human')),
                output_directory=self.source_material_cache_directory)
            ao_config=prepare_source_ao_config(bundle.paths_by_role['habitat_ao_config'],
                bundle.paths_by_role['habitat_urdf'],derived,derived.parent)
            material['derived_ao_config_path']=str(ao_config)
            manager=self.sim.metadata_mediator.ao_template_manager
            loaded=manager.load_configs(str(ao_config))
            if len(loaded)!=1:raise ValueError('M2 AO template is not unique')
            handle=manager.get_template_handle_by_id(loaded[0])
            obj,binding=_instantiate_actor_with_semantic_template(self.sim,bundle=bundle,habitat_sim=hs,
                base_handle=handle,semantic_id=1001+i,actor_index=i,shader_type="pbr",render_asset_path=derived)
            endpoint=asset['endpoint_id']
            foot=position_at(scene['sources'][endpoint]['trajectory'],0)-[0.,1.3,0.]
            snap=np.asarray(pf.snap_point(foot),float)
            if not np.isfinite(snap).all() or np.linalg.norm(snap-foot)>.15:
                raise ValueError('actor root has no admitted navigation support')
            anchors=[x for x in validated.asset['anchors'] if x['anchor_id']==asset.get('emitter_anchor_id','muzzle')]
            if len(anchors)!=1:raise ValueError('M2 mouth anchor is not unique')
            anchor=anchors[0]
            link=[int(k) for k in obj.get_link_ids() if obj.get_link_name(k)==anchor['joint_id']]
            if len(link)!=1:raise ValueError('native mouth link does not match M2')
            self.actors.append({'object':obj,'binding':binding,'inputs':validated,'bundle':bundle,
                'mouth_link':link[0],'anchor':anchor,'endpoint_id':endpoint,'floor_y':float(snap[1]),
                'semantic_id':1001+i,'declaration':asset,'source_material':material})
        self.receipt={'backend':'Habitat-Sim native M2','compute_device':'CPU','software_driver':'Mesa llvmpipe',
            'gpu_device_id':-1,'hardware_EGL_devices_excluded':True,'resolution_hw':[height,width],
            'hfov_deg':85.,'fps':FPS,'runtime':runtime,'stage':cfg.scene_id,
            'source_material_rendering':self.material_rendering,
            'source_materials':[a['source_material'] for a in self.actors],
            'asset_admission_states':[a['inputs'].asset.get('admission_state') for a in self.actors],
            'action_clocks':[{'endpoint_id':a['endpoint_id'],
                'sample_rate_hz':a['bundle'].action_sets_by_role[a['bundle'].action_roles_by_id['walk']].sample_rate_hz,
                'time_base_hz':a['bundle'].action_sets_by_role[a['bundle'].action_roles_by_id['walk']].time_base_hz} for a in self.actors]}

        if infra_gpu is not None:
            self.receipt.update(compute_device='GPU',software_driver=None,gpu_device_id=cfg.gpu_device_id,physical_gpu_lease=infra_gpu,physical_uuid=os.environ.get('CCI_INFRA_HABITAT_PHYSICAL_UUID'),
                                hardware_EGL_devices_excluded=False,infrastructure_trial_only=True,
                                training_eligible=False)

    def _apply(self, scene, frame_index):
        from avengine.assets.habitat_capture import _apply_root_with_habitat,transform_to_matrix
        from avengine.timeline.visual import _node_world_position
        rows=[];t=frame_index/FPS
        for entry in self.actors:
            track=scene['sources'][entry['endpoint_id']]['trajectory']
            p=position_at(track,t);previous=position_at(track,max(0,t-1/FPS));nextp=position_at(track,min(scene['duration_s'],t+1/FPS))
            direction=nextp-previous;direction[1]=0.
            moving=np.linalg.norm(direction)>1e-5
            action='walk' if moving else 'idle'
            bundle=entry['bundle'];clip=bundle.action_sets_by_role[bundle.action_roles_by_id[action]].action(action)
            # Read the actual M2 clock rather than assuming a baking rate.
            # Stopped actors hold idle pose 0; walking follows the asset loop.
            action_set=bundle.action_sets_by_role[bundle.action_roles_by_id[action]]
            offset=scene.get('actor_pose_phase_ticks',{}).get(entry['endpoint_id'],0)
            phase=(round(t*action_set.time_base_hz)+offset)%clip.loop_duration_ticks if moving else 0
            sample=max(0,int(np.searchsorted(clip.sample_ticks,phase,side='right'))-1) if moving else 0
            rotations=clip.rotations_xyzw[sample]
            joints=np.asarray(entry['binding'].map_pose(rotations),float)
            obj=entry['object'];obj.joint_positions=joints
            root=np.eye(4);root[:3,3]=[p[0],entry['floor_y'],p[2]]
            support=np.asarray(self.pathfinder.snap_point(root[:3,3]),float)
            if not np.isfinite(support).all() or np.linalg.norm(support-root[:3,3])>.15:
                raise ValueError('moving actor left admitted navigation support')
            if moving:
                forward=direction/np.linalg.norm(direction)
            else:
                forward=np.asarray(scene['listener_m'])-p;forward[1]=0;forward/=np.linalg.norm(forward)
            root[:3,:3]=np.stack((np.cross(forward,[0,1,0]),[0,1,0],-forward),axis=1)
            skin=root@np.asarray(bundle.actor_from_skin_root)
            _apply_root_with_habitat(obj,skin,qt=self.qt,mn=self.mn)
            # Keep only the current applied native inputs. Paired target-only
            # renders avoid another float32 quaternion normalization; readbacks
            # still certify every actual root, joint and link transform.
            entry['applied_root_input']=skin.copy();entry['applied_joint_input']=joints.copy()
            actual=np.asarray(obj.root_scene_node.absolute_transformation(),float)
            if np.max(np.abs(actual-skin))>1e-5 or np.max(np.abs(np.asarray(obj.joint_positions)-joints))>1e-5:
                raise ValueError('native actor root/joints differ from applied M2 state')
            from avengine.capture.mp3d_multi_actor import _emitter_position
            mouth=_emitter_position(obj,entry['mouth_link'],transform_to_matrix(entry['anchor']['joint_from_anchor']))
            rows.append({'endpoint_id':entry['endpoint_id'],'actor_id':scene['sources'][entry['endpoint_id']]['actor_id'],
                'semantic_id':entry['semantic_id'],'asset_id':entry['declaration']['asset_id'],
                'position_m':actual[:3,3].tolist(),'emitter_position_m':mouth.tolist(),
                'moving':bool(moving),'action_id':action,'action_sample_index':int(sample),'action_tick':phase,
                'root_max_error':float(np.max(np.abs(actual-skin))),
                'joint_max_error':float(np.max(np.abs(np.asarray(obj.joint_positions)-joints)))})
        return rows

    def _check_source_frame(self,rgb,semantic,frame_index):
        from avengine.assets.source_materials import measure_source_appearance
        for entry in self.actors:
            material=entry['source_material']
            measured=measure_source_appearance(rgb,semantic==entry['semantic_id'],
                palette=np.asarray(material['source_palette_srgb']),
                max_highlight_ratio=max(p['max_highlight_ratio'] for p in material['profiles']),strict=False)
            record={'endpoint_id':entry['endpoint_id'],'asset_id':entry['declaration']['asset_id'],
                    'frame_index':frame_index,**measured}
            self.material_checks.append(record)
            if measured['status']=='fail':
                raise ValueError('inserted source material acceptance failed: '+json.dumps(record))

    def encode(self,scene,path,review_frames=None,*,collect_pixel_visibility=False):
        path=Path(path);start=time.perf_counter();frames=[];mouths={a['endpoint_id']:[] for a in self.actors}
        write_json(path.parent/'source_materials.json',{'rendering':self.material_rendering,
                   'sources':[a['source_material'] for a in self.actors]})
        frame_count=round(scene['duration_s']*FPS)
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24',
            '-video_size',f'{self.width}x{self.height}','-framerate',str(FPS),'-i','pipe:0','-an',
            '-c:v','libx264','-threads','1','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(path)]
        from avengine.capture.continuous_native_visibility import NativeVisibilityStream
        visibility=NativeVisibilityStream(self,path.parent/"failed_visibility_masks") if collect_pixel_visibility else None
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for index in range(frame_count):
                actors=self._apply(scene,index)
                obs=self.sim.get_sensor_observations()
                rgb=np.asarray(obs['rgb'])[:,:,:3];depth=np.asarray(obs['depth']);semantic=np.asarray(obs['semantic'])
                if rgb.shape!=(self.height,self.width,3) or depth.shape!=(self.height,self.width) or semantic.shape!=depth.shape:
                    raise ValueError('native RGB/depth/semantic shapes disagree')
                if not np.isfinite(depth).all() or float(rgb.std())<2:
                    raise ValueError('native RGB/depth is empty or invalid')
                if index in {0,frame_count//2,frame_count-1}:
                    self._check_source_frame(rgb,semantic,index)
                for actor in actors:
                    mask=semantic==actor['semantic_id'];ys,xs=np.nonzero(mask)
                    actor['visible_pixels']=len(xs)
                    actor['mask_centroid_px']=[float(xs.mean()),float(ys.mean())] if len(xs) else None
                    actor['mask_bbox_xyxy']=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None
                    mouths[actor['endpoint_id']].append({'time_s':index/FPS,'position_m':actor['emitter_position_m']})
                if visibility is not None:visibility.capture(index,semantic,actors)
                frames.append({'frame_index':index,'time_s':index/FPS,'actors':actors,'rgb_std':float(rgb.std())})
                if index in set(review_frames or [0,45,75,90,120,149,225,300,375,449]):
                    from PIL import Image
                    Image.fromarray(rgb).save(path.parent/f'native_frame_{index:04d}.png')
                    np.savez_compressed(path.parent/f'native_masks_{index:04d}.npz',semantic=semantic,depth=depth)
                process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('native frame stream encoding failed')
        except BaseException:
            if any(r['status']=='fail' for r in self.material_checks):
                write_json(path.parent/'source_material_checks.json',{'status':'fail','checks':self.material_checks})
            if process.stdin and not process.stdin.closed:process.stdin.close()
            process.wait()
            raise
        finally:
            if visibility is not None:visibility.close()
        for values in mouths.values():values.append({'time_s':scene['duration_s'],'position_m':values[-1]['position_m']})
        material_checks={'status':'pass' if any(r['status']=='pass' for r in self.material_checks) else 'not_run',
                         'checks':self.material_checks}
        write_json(path.parent/'source_material_checks.json',material_checks)
        visibility_receipt={'pixel_visibility_truth':visibility.truth if visibility is not None else {'status':'not_run_unqualified_native_stream'},
                            'alignment':visibility.receipt() if visibility is not None else {'status':'not_run','full_stream_qualification':'failed; sampler controls retained'}}
        write_json(path.parent/'native_visibility_evidence.json',visibility_receipt)
        result={**visibility_receipt,'native_visibility_alignment':visibility_receipt['alignment'],
                'renderer':self.receipt,'frames':frames,'observed_mouth_trajectories':mouths,
                 'source_material_checks':material_checks,
                'frame_count':len(frames),'wall_seconds':time.perf_counter()-start}
        write_json(path.parent/'native_visual_evidence.json',result)
        return result

    def close(self):self.sim.close()
