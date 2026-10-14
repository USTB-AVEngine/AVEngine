"""Recorded/static alpha rendered in the admitted isolated UE/SPEAR project.

Original native MICs, 2cm UE contact tolerance and native modal/target depth
checks are retained. Kujiale now requires the unified histogram exposure and
measured authored-light map; previous EV3 maps cannot enter fresh generation. The Habitat alpha renderer's
root correction and PBR overrides are never used by this adapter.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import math
import numpy as np
from avengine.capture.continuous_ue_video import UENativeContinuousVideo
from avengine.capture.kujiale_spear_runtime import apply_camera,from_ue
from avengine.capture.v7_alpha_native import yaw_at
from avengine.timeline.continuous_pilot import write_json

ALPHA_UE_VISUAL_REVISION='v7-alpha-ue-design-whitebalance-histogram-v2'


def assert_ue_alpha_visual(visual):
    a=visual['native_visibility_alignment']
    if visual['frame_count']!=450 or len(visual['camera_frame_readbacks'])!=450:
        raise ValueError('incomplete native UE alpha world')
    if visual['source_material_checks']['status']!='pass' or visual['native_body_foot_status']!='pass':
        raise ValueError('original native highlight/contact checks failed')
    if a['status']!='pass' or a['target_only_root_max_error_m']>.0005 or a['target_only_animation_max_error_s']>1e-4 or a['camera_position_max_error_m']>.0005:
        raise ValueError('native UE modal/target pose readbacks disagree')
    if a['rendered_native_target_depth_passes']!=450*len(visual['frames'][0]['actors']):
        raise ValueError('native UE target-only frames incomplete')
    return True


class NativeUEAlphaVideo(UENativeContinuousVideo):
    def __init__(self,house,scene,inputs,assets,**kwargs):
        if house['input'].get('visual_backend')!='ue_spear':raise ValueError('explicit native UE backend required')
        for spec in scene['sources'].values():
            if any(not np.array_equal(k['position_m'],spec['trajectory'][0]['position_m']) for k in spec['trajectory']):
                raise ValueError('alpha static-human contract forbids translation')
        house=deepcopy(house)
        if house.get('family')=='kujiale':
            from avengine.capture.kujiale_native_exposure import bind_histogram_profile,require_authored_light_audit,require_histogram_room_visual_qa
            house['input']['ue_config']=bind_histogram_profile(house['input']['ue_config'],scene_seed=scene['scene_id'])
            require_authored_light_audit(house['input']['ue_config'],house['input']['ue_map_path'])
            room_id=house['room'].get('room_label') or house['room'].get('room_id')
            require_histogram_room_visual_qa(house['input']['ue_config'],room_id,house['input']['ue_map_path'])
        house['input']['ue_config']['allow_invisible_source_material_samples']=True
        self.alpha_time=0.;self.alpha_lower_endpoint=None;self.alpha_camera_rows={}
        super().__init__(house,scene,inputs,assets,**kwargs)
        if self.native_floor_surface is None:raise ValueError('alpha needs audited spatial native floor support')

    def _camera(self):
        yaw=yaw_at(self.scene,self.alpha_time)
        return {'position_m':self.scene['listener_m'],'kind':'perspective','yaw_deg':yaw-90.,'pitch_deg':0.}

    def _state(self,asset,native_id,index):
        state=super()._state(asset,native_id,index)
        if state['moving'] or state['action_id']!='idle' or state['action_phase']!=0.:
            raise ValueError('UE alpha pose must be the admitted original idle pose0')
        if asset['endpoint_id']==self.alpha_lower_endpoint:
            state['root_transform']['translation_m'][1]-=.1
        return state

    def _expected_camera_yaw(self,time_s):
        return yaw_at(self.scene,time_s)

    def _camera_readback(self,time_s):
        from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
        pose=read_scene_component_pose(self.camera.K2_GetRootComponent())
        actual_yaw=float(pose['rotation_deg'][2])+90.
        position=from_ue(pose['location_cm']);expected=self._expected_camera_yaw(time_s)
        delta=(actual_yaw-expected+180.)%360.-180.
        if abs(delta)>1e-4 or np.max(np.abs(np.asarray(position)-self.scene['listener_m']))>.0005:
            raise ValueError('native UE camera readback differs from the alpha pan')
        angle=math.radians(actual_yaw)
        return {'time_s':float(time_s),'position_m':position,'yaw_deg':actual_yaw,
                'forward_m':[math.sin(angle),0.,-math.cos(angle)],
                'native_component_pose':pose,'measurement':'actual UE K2 scene component transform'}

    def _apply(self,index):
        self.alpha_time=index/15
        apply_camera(self.camera,self._camera())
        actors=super()._apply(index)
        self.alpha_camera_rows[index]={'frame_index':index,**self._camera_readback(index/15)}
        return actors

    def _depth_contact_probe(self,actor,lower=False):
        from tools.rooms.check_kujiale_native_motion_contact import depth_support
        runtime=self.runtimes[actor['native_actor_id']];root=np.asarray(actor['position_m']);rows=[]
        self.alpha_lower_endpoint=actor['endpoint_id'] if lower else None
        try:
            for name,offset,yaw in [('front',[0.,1.,-3.],90.),('rear',[0.,1.,3.],-90.)]:
                origin=root+offset
                for _ in range(3):
                    with self.instance.begin_frame():
                        self._apply(0)
                        apply_camera(self.foot_camera,{'position_m':origin.tolist(),'kind':'perspective','yaw_deg':yaw,'pitch_deg':0.})
                        self.foot_components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
                        self.foot_components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[runtime['visual_actor']])
                    with self.instance.end_frame():depth=self.read_depth(self.foot_components['depth'])
                rows.append({'view':name,**depth_support(depth,float(origin[1]),float(root[1]))})
        finally:
            self.alpha_lower_endpoint=None
            with self.instance.begin_frame():self._apply(0)
            with self.instance.end_frame():pass
        if any(r['status']!='measured' for r in rows):raise ValueError('UE full-body contact probe missing or cropped')
        gap=min(r['gap_m'] for r in rows)
        return {'status':'pass' if abs(gap)<=.02 else 'fail','gap_m':gap,'contact_tolerance_m':.02,'views':rows,'root_y_adjustment_m':-.1 if lower else 0.}

    def qualify_standing(self,directory):
        from tools.rooms.check_v7_kujiale_native_assets import vector_field
        from tools.rooms.check_kujiale_native_capsule_contact import query_capsule
        directory=Path(directory);directory.mkdir(parents=True)
        for _ in range(3):
            with self.instance.begin_frame():actors=self._apply(0)
            with self.instance.end_frame():pass
        checks=[]
        for actor in actors:
            native=self.runtimes[actor['native_actor_id']]
            good=self._depth_contact_probe(actor);bad=self._depth_contact_probe(actor,lower=True)
            root=np.asarray(actor['position_m'])
            probe={'witness_index':0,'role':actor['endpoint_id'],'capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':[(root[0])*100,root[2]*100,(root[1]+.75)*100]}}
            capsule=query_capsule(self.native_floor_surface.mesh,probe)
            with self.instance.begin_frame():bounds=native['visual_actor'].GetActorBounds(bOnlyCollidingComponents=False,bIncludeFromChildActors=True,as_dict=True)
            with self.instance.end_frame():pass
            center=np.asarray(from_ue(vector_field(bounds,'Origin')));extent=np.asarray(from_ue(vector_field(bounds,'BoxExtent')))
            detected=bad['status']=='fail' and bad['gap_m']<-.02
            checks.append({'status':'pass' if good['status']=='pass' and detected and capsule['status']=='pass' else 'blocked',
                'endpoint_id':actor['endpoint_id'],'asset_id':actor['asset_id'],'action':'idle_pose0',
                'contact_tolerance_m':.02,'tolerance_changed':False,'root_y_adjustment_m':0.,
                'native_sole_depth':good,'native_capsule_static_geometry':capsule,
                'native_full_skin_bounds_m':[(center-extent).tolist(),(center+extent).tolist()],
                'bounds_authority':'UE GetActorBounds includes every original native visual component; conservative bounds, not a sampled vertex reconstruction',
                'known_bad_10cm_floor_contacts':bad,'known_bad_detected':detected,
                'scope':'retained UE front/rear native skin depth support plus frozen capsule vs actual UE triangles; exact hand/cloth/furniture contact not certified'})
        write_json(directory/'standing_admission.json',{'checks':checks,'status':'pass' if all(r['status']=='pass' for r in checks) else 'blocked'})
        self.alpha_standing_refs={a['endpoint_id']:{'position_m':a['position_m'],'reference':c['native_sole_depth']} for a,c in zip(actors,checks,strict=True)}
        self.alpha_camera_rows={}
        return checks

    def _uncropped_foot_probe(self,actor):
        # This cache is only for the explicitly static original idle pose0.
        # Every frame already reads the exact native root and animation clock;
        # no moving pose or distinct source can borrow a support measurement.
        saved=self.alpha_standing_refs[actor['endpoint_id']]
        animation=actor['native_animation']
        position_seconds=animation['observed_position_seconds']
        if actor['moving'] or actor['action_id']!='idle' or actor['action_phase']!=0. or abs(position_seconds)>1e-4 or np.max(np.abs(np.asarray(actor['position_m'])-saved['position_m']))>.0005:
            raise ValueError('static foot reference cannot be reused for a changed native pose')
        ref=saved['reference'];root_y=actor['position_m'][1]
        return {'status':ref['status'],'gap_m':ref['gap_m'],'lowest_rendered_body_y_m':root_y+ref['gap_m'],
            'support_floor_y_m':root_y,'raster_depth_tolerance_m':.02,
            'method':'original native front/rear full-body depth, exact same stopped idle pose0 verified at this frame',
            'reference_pose':'standing_admission idle pose0 at frame0','pose_readback_verified':True,
            'reference_views':ref['views'],'measurement_reused':True}

    def encode(self,scene,path,review_frames=None,*,collect_pixel_visibility=True):
        checks=self.qualify_standing(Path(path).parent/'standing')
        if not all(r['status']=='pass' for r in checks):raise ValueError('UE alpha standing placement failed original contact/capsule checks')
        result=super().encode(scene,path,review_frames=list(range(0,450,15))+[449],collect_pixel_visibility=collect_pixel_visibility)
        result.update(camera_frame_readbacks=[self.alpha_camera_rows[i] for i in range(450)],
                      standing_admission=checks,alpha_visual_revision=ALPHA_UE_VISUAL_REVISION)
        protocol=[]
        # RGB/depth media is complete; these 100ms rows need actual component
        # transforms only. Keep all native readbacks without extra image draws.
        with self.instance.begin_frame():
            for component in list(self.components.values())+list(self.foot_components.values()):
                component.set_property_value(property_name='bCaptureEveryFrame',property_value=False)
        with self.instance.end_frame():pass
        for tick in range(301):
            self.alpha_time=tick/10
            with self.instance.begin_frame():
                apply_camera(self.camera,self._camera());protocol.append(self._camera_readback(self.alpha_time))
            with self.instance.end_frame():pass
        result['protocol_camera_readbacks']=protocol
        result['alignment']=result['native_visibility_alignment']
        assert_ue_alpha_visual(result)
        write_json(Path(path).parent/'native_alpha_visual_evidence.json',result)
        return result
