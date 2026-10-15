#!/usr/bin/env python3
"""Native SPEAR/UE multi-view individual admission in the production Kujiale map."""
from __future__ import annotations
import argparse,json,math,os,sys,time,traceback,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy
import numpy as np
from avengine.capture.kujiale_spear_runtime import (launch_native,camera_helpers,apply_camera,to_ue,from_ue,apply_native_rgb_profile,configure_scene_lighting,close_native)
from avengine.qa.kujiale_asset_geometry import import_vertices,rotation
from avengine.qa.kujiale_native_depth import REVISION,world_pixels,far_range,visibility
from avengine.qa.kujiale_asset_color import measure,summarize_views,functional_exclusion


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def vector(raw,key):
    item=raw[key];return [float(item[k]) for k in ['X','Y','Z']]


class Capture:
    def __init__(self,args):
        self.args=args;self.root=args.root;self.manifest=json.loads(args.manifest.read_text());self.revision=self.manifest['revision'];self.run_label=args.run_label or args.house_id+'_'+str(args.gpu)
        value=json.loads(args.houses.read_text());houses=value['houses'] if isinstance(value,dict) else value
        self.house=deepcopy(next(x for x in houses if x['house']==args.house_id));ue=self.house['input']['ue_config']
        ue.update(uproject=str(self.root/'ue_v1/runtime/SpearSim/SpearSim.uproject'),output_root=str(self.root),unreal_editor=str(self.root/'scripts'/f'ue_gpu{args.gpu}'),physical_gpu_index=args.gpu,graphics_adapter=0)
        source_light=Path(ue['native_lighting_audit'])
        own_light=self.root/'inputs/authored_lights'/args.house_id/'native_lighting_audit.json'
        own_light.parent.mkdir(parents=True,exist_ok=True)
        import shutil
        shutil.copy2(source_light,own_light)
        ue['native_lighting_audit']=str(own_light)
        ue['unpause_animation_sampling']=args.mode=='motion'
        self.ue=ue;self.instance=self.camera=self.components=None;self.active=[];self.human=None;self.warmed=False
        sys.path.insert(0,str(Path(ue['spear_ext_dir']).resolve()))
        self.spawn,self.depth,self.rgb,*_=camera_helpers()
        from avengine.capture.kujiale_native_exposure import bind_histogram_profile,PARAMETERS
        self.ue=bind_histogram_profile(ue,scene_seed='kujiale-ue-assets-'+args.house_id);self.parameters=PARAMETERS
        self.instance,self.game,self.launch=launch_native(self.ue,self.house['input']['ue_map_path'])
        with self.instance.begin_frame():
            self.camera,self.components=self.spawn(self.game,horizontal_fov_deg=85,width=2560,height=1440)
            self.exposure=apply_native_rgb_profile(self.components['rgb'],self.ue)
            self.lights=configure_scene_lighting(self.game,json.loads(Path(ue['acceptance_plan_path']).read_text()),self.ue)
            for name in ['object_ids']:self.components[name].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
        with self.instance.end_frame():pass
        self.assert_gpu()
        self.thresholds={'neutral_chroma_max':10,'black_L_max':20,'white_L_min':45,'main_share_min':.55}
        write(self.root/'evidence'/f'native_launch_{self.revision}_{self.run_label}.json',{'status':'launched','launch':self.launch,'exposure':self.exposure,'lights':self.lights,'color_family_thresholds':self.thresholds,'color_method':'UE rendered CIELAB families; label comparison only, owner decides mismatches; no Habitat texture differences'})

    def assert_gpu(self):
        raw=subprocess.check_output(['nvidia-smi','-q','-x'],text=True);cards=ET.fromstring(raw).findall('gpu');found=[]
        for index,card in enumerate(cards):
            for p in card.findall('processes/process_info'):
                pid=p.findtext('pid');cmd=Path(f'/proc/{pid}/cmdline')
                if cmd.exists() and self.launch['temporary_directory'] in cmd.read_bytes().replace(b'\0',b' ').decode(errors='replace'):found.append({'visible_index':index,'uuid':card.findtext('uuid'),'pid':int(pid)})
        write(self.root/'evidence'/f'gpu_actual_{self.revision}_{self.run_label}.json',{'requested_physical_gpu':self.args.gpu,'observed_physical_gpus':found,'command':'nvidia-smi -q -x + own unique absolute log directory in /proc/PID/cmdline'})
        expected=self.launch['gpu_resource_preflight']['gpu_uuid']
        if not found or any(x['uuid']!=expected for x in found):raise RuntimeError('native process physical GPU UUID differs from assigned card: '+str(found))

    def frame(self,callback=None,read=False):
        with self.instance.begin_frame():
            if callback:callback()
        with self.instance.end_frame():
            if read:return self.rgb(self.components['rgb']),self.depth(self.components['depth'])

    def spawn_item(self,row,receipt,placement):
        from avengine.backends.spear_ue.research_runtime import spawn_attached_static_actor
        actors=[]
        for i,obj in enumerate(x for x in receipt['static'] if x['class']=='StaticMesh'):
            actor=spawn_attached_static_actor(self.game,actor_id=row['asset_id']+'_'+str(i),static_mesh_object_path=obj['path'],
                position_ue_cm=to_ue(placement['root_m']),yaw_ue_degrees=-math.degrees(placement['yaw_rad']),actor_scale=1.,
                emitter_local_ue_cm=to_ue(row.get('emitter_local_m') or [0,0,0]))
            actors.append(actor)
        return actors

    def plan_human(self,placement):
        from avengine.capture.native_floor_reference import NativeFloorReference
        from tools.rooms.check_kujiale_native_capsule_contact import query_capsule
        floor=NativeFloorReference(self.ue['native_floor_surface']);room=self.house['checked_rooms'][0];floor.validate_probes(room['floor_reference'])
        reference=json.loads((self.root.parent/'ue_round14/inputs/assets.json').read_text())['assets'][0]
        root=np.asarray(placement['root_m']);candidates=[]
        for probe in room['floor_reference']['probes']:
            try:p=np.asarray(floor.sample(probe['nav_point_m']))
            except ValueError:continue
            if np.linalg.norm((p-root)[[0,2]])<1.3:continue
            check=query_capsule(floor.mesh,{'witness_index':0,'role':'co_present_human','capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':to_ue(p+[0,.75,0])}})
            if check['status']=='pass':candidates.append((p,check,dict(floor.last_support_readback)))
        if not candidates:raise RuntimeError('no room-floor co-present human clearance')
        p,check,support=min(candidates,key=lambda t:float(np.linalg.norm(t[0]-root)))
        return {'reference':reference,'root_m':p.tolist(),'support':support,'native_capsule':check}

    def spawn_human(self,plan):
        reference=plan['reference'];p=np.asarray(plan['root_m']);check=plan['native_capsule']
        klass=self.game.unreal_service.load_class(uclass='AActor',name=reference['ue_generated_class']);actor=self.game.unreal_service.spawn_actor(uclass=klass,spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'})
        actor.SetActorEnableCollision(bNewActorEnableCollision=False);actor.K2_GetRootComponent().SetMobility(NewMobility='Movable')
        actor.K2_SetActorLocationAndRotation(NewLocation=dict(zip(['X','Y','Z'],to_ue(p))),NewRotation={'Roll':0.,'Pitch':0.,'Yaw':90.},bSweep=False,bTeleport=True)
        self.human=actor;return {'asset_id':reference['asset_id'],'root_m':p.tolist(),'support':plan['support'],'native_capsule':check}

    def body_mask(self,row,mask,depth,view,vertices,placement):
        if row['category'] in ['dog','cat'] or not mask.any():return mask
        coords=world_pixels(depth,view,mask)
        local=(coords-np.asarray(placement['root_m']))@rotation(placement['yaw_rad']);lo,hi=vertices.min(axis=0),vertices.max(axis=0);q=(local-lo)/np.maximum(hi-lo,1e-6)
        if row['category']=='bird':selected=local[:,1]>float((row.get('perch') or {}).get('perch_top_m',.35))
        else:
            front=(np.asarray(view['position_m'])-np.asarray(placement['root_m']))@rotation(placement['yaw_rad'])
            selected=~functional_exclusion(row,q,front_facing=np.full(mask.sum(),front[0]>0))
        result=mask.copy();result[mask]=selected;return result

    def view(self,row,view,placement,vertices,dest,index,warmup):
        def setup():
            apply_camera(self.camera,view)
            self.components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_RenderScenePrimitives')
            self.components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[])
        self.frame(setup)
        for _ in range(warmup):self.frame()
        rgb,scene=self.frame(read=True)
        def isolate():
            self.components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
            self.components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[n['visual_actor'] for n in self.active])
        self.frame(isolate);_,isolated=self.frame(read=True);self.frame(setup)
        world=vertices@rotation(placement['yaw_rad']).T+np.asarray(placement['root_m'])
        far_m=far_range(world,view)
        mask,visible=visibility(scene,isolated,far_m)
        count=int(mask.sum())
        frac=float(visible.sum()/count) if count else 0.;border=bool(mask[:5].any() or mask[-5:].any() or mask[:,:5].any() or mask[:,-5:].any())
        highlight=float(np.mean(np.all(rgb[visible]>247,axis=1))) if visible.any() else None
        body=self.body_mask(row,visible,isolated,view,vertices,placement);color=measure(rgb[:,:,::-1],body,self.thresholds)
        import cv2
        dest.mkdir(parents=True,exist_ok=True);rgbfile=dest/f'view_{index}.jpg';maskfile=dest/f'view_{index}_mask.png';cv2.imwrite(str(rgbfile),cv2.resize(rgb,(1280,720)),[cv2.IMWRITE_JPEG_QUALITY,90]);cv2.imwrite(str(maskfile),cv2.resize(mask.astype('uint8')*255,(1280,720),interpolation=cv2.INTER_NEAREST))
        status='pass' if count>=80 and frac>=.99 and not border else 'fail'  # highlights are record-only by owner
        rec={'status':status,'view':view,'rgb':str(rgbfile),'mask':str(maskfile),'whole_asset_pixels':count,'native_geometry_far_range_m':far_m,'isolated_background_range_m':float(np.nanmax(isolated)),'measurement_revision':REVISION,'native_depth_units':'radial meters','occlusion_tolerance_m':.005,'visible_fraction':frac,'crop_border':border,'highlight_ratio':highlight,'highlight_limit':.02,'highlight_previous_limit_exceeded':highlight is not None and highlight>.02,'highlight_gate_enforced':False,'color':color,
             'visibility_method':'paired actual UE scene and isolated native metric depth; occlusion fraction, five-pixel complete-frame margin','capture_resolution':[2560,1440],'delivery_resolution':[1280,720]}
        np.savez_compressed(dest/f'view_{index}_depth.npz',isolated=isolated.astype('float16'),scene=scene.astype('float16'))
        write(dest/f'view_{index}.json',rec);return rec

    def read_emitter(self,row,placement):
        from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
        from tools.rooms.check_v7_kujiale_native_assets import vector_field
        if row['silent_decor']:return {'status':'not_applicable_silent_decor'}
        if row['emitter_local_m'] is None:return {'status':'fail','reason':'registration has no physical emitter'}
        with self.instance.begin_frame():pose=read_scene_component_pose(self.active[0]['emitter_component'])
        with self.instance.end_frame():pass
        actual=np.asarray(from_ue(pose['location_cm']));expected=np.asarray(placement['root_m'])+rotation(placement['yaw_rad'])@np.asarray(row['emitter_local_m']);error=float(np.max(np.abs(actual-expected)))
        return {'status':'pass' if error<=.0005 else 'fail','expected_m':expected.tolist(),'actual_m':actual.tolist(),'maximum_error_m':error,'tolerance_m':.0005,'native_component_pose':pose}

    def cleanup(self):
        def apply():
            if self.human is not None:self.human.K2_DestroyActor();self.human=None
            for n in self.active:
                n['visual_actor'].K2_DestroyActor()
                if n['anchor'].uobject!=n['visual_actor'].uobject:n['anchor'].K2_DestroyActor()
            self.active=[]
        self.frame(apply)

    def run(self):
        rows=[];assets=self.manifest['assets']
        if self.args.shard_count:assets=[x for i,x in enumerate(assets) if i%self.args.shard_count==self.args.shard_index]
        if self.args.only:assets=[x for x in assets if x['asset_id'] in self.args.only]
        if self.args.limit:assets=assets[:self.args.limit]
        for row in assets:
            ev=self.root/'evidence/native'/self.revision/self.args.house_id/(row['asset_id']+'.json')
            if not self.args.force and ev.exists() and json.loads(ev.read_text()).get('status')=='pass_static_native' and json.loads(ev.read_text()).get('measurement_revision')==REVISION:rows.append(json.loads(ev.read_text()));continue
            planpath=self.root/'evidence/plans'/self.revision/self.args.house_id/(row['asset_id']+'.json')
            if not planpath.exists():continue
            plan=json.loads(planpath.read_text());rec={'asset_id':row['asset_id'],'category':row['category'],'house':self.args.house_id,'status':'fail','measurement_revision':REVISION,'views':[],'input_manifest':str(self.args.manifest),'native_map_path':self.house['input']['ue_map_path'],'launch_receipt':str(self.root/'evidence'/f'native_launch_{self.revision}_{self.run_label}.json')}
            try:
                if plan['status']!='pass_geometry_plan':raise RuntimeError('native geometry plan failed: '+str(plan.get('error')))
                imported=json.loads((self.root/'evidence/import'/self.revision/(row['asset_id']+'.json')).read_text());vertices=import_vertices(imported)
                for attempt,placement in enumerate(plan['placements']):
                    human_plan=self.plan_human(placement)
                    with self.instance.begin_frame():self.active=self.spawn_item(row,imported,placement);rec['co_present_human']=self.spawn_human(human_plan)
                    with self.instance.end_frame():pass
                    views=[]
                    for index,view in enumerate(placement['camera_candidates'][:18]):
                        result=self.view(row,view,placement,vertices,self.root/'renders'/self.revision/'depth_v3'/self.args.house_id/row['asset_id']/f'attempt_{attempt}',index,240 if not self.warmed else 24)
                        self.warmed=True;views.append(result)
                        if sum(x['status']=='pass' for x in views)>=3:break
                    passed=[v for v in views if v['status']=='pass']
                    rec.update(views=views,placement=placement,size={'status':plan['size_status'],'native_dimensions_m':plan['native_dimensions_m'],'relative_errors':plan['dimensions_relative_error'],'tolerance_fraction':.15},emitter=self.read_emitter(row,placement))
                    self.cleanup()
                    if len(passed)>=3:break
                if len([v for v in rec['views'] if v['status']=='pass'])<3:raise RuntimeError('fewer than three complete visible geometry views')
                if rec['emitter']['status'] not in ['pass','not_applicable_silent_decor']:raise RuntimeError('physical emitter readback failed')
                color=summarize_views([x['color'] for x in rec['views'] if x['status']=='pass'],self.thresholds);aliases={'gray':'gray_silver','silver':'gray_silver','brown':'wood_brown'};expected=aliases.get(row['label_color'],row['label_color'])
                rec.update(status='pass_static_native',measured_color=color,registered_label_color=row['label_color'],color_tag_status='unlabelled_companion' if expected is None else ('match' if color.get('label')==expected else 'mismatch_owner_decision'),registered_labels_modified=False,ue_asset_paths=[x['path'] for x in imported['static'] if x['class']=='StaticMesh'],native_motion_status='not_run' if row['walk_allowed'] else 'not_applicable_static')
            except Exception:
                rec['error']=traceback.format_exc()
                if self.active or self.human:self.cleanup()
            write(ev,rec);rows.append(rec);write(self.root/'evidence'/f'native_{self.revision}_{self.run_label}.json',rows)
            print('NATIVE',self.args.house_id,row['asset_id'],rec['status'],rec.get('color_tag_status'),flush=True)
        return rows

    def close(self):
        if self.instance is not None:
            try:
                if self.active or self.human:self.cleanup()
            finally:close_native(self.instance,self.camera,self.components)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--houses',type=Path,required=True);p.add_argument('--house-id',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--force',action='store_true');p.add_argument('--run-label');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int);p.add_argument('--only',nargs='*');p.add_argument('--limit',type=int);p.add_argument('--mode',choices=['admission','floor_reproduction','trial','motion','production_binding'],default='admission');a=p.parse_args();capture=None
    try:
        capture=Capture.__new__(Capture);Capture.__init__(capture,a)
        if a.mode=='floor_reproduction':
            from capture_kujiale_floor_reproduction import run
            run(capture)
        elif a.mode=='trial':
            from capture_kujiale_asset_trial import run
            run(capture)
        elif a.mode=='production_binding':
            from capture_kujiale_production_binding import run
            run(capture)
        elif a.mode=='motion':
            from capture_kujiale_dog_motion import run
            run(capture)
        else:capture.run()
    except BaseException:write(a.root/'evidence'/f'native_{a.house_id}_fatal.json',{'error':traceback.format_exc()});raise
    finally:
        if capture is not None:capture.close()

if __name__=='__main__':main()
