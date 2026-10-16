"""Known gray/brown human shirt probes, kept inside this task for independent calibration."""
import argparse,copy,hashlib,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from v7_human_roster_materials import read_glb,write_glb,append_image,material_body_image,geometry_animation_digest,srgb_to_linear,linear_to_srgb
from v7_human_roster_habitat import camera,make_object
from v7_human_roster_roi import roi
from avengine.assets.source_materials import configure_source_pbr
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
from v7_animal_variants_color import measure

def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(T,R):
    selection=json.loads((R/'registrations/selection.json').read_text());records=[]
    for aid,split in [('v7_female_adult_01','calibration'),('v7_male_adult_02','held_out')]:
        row=next(x for x in selection['avatars'] if x['asset_id']==aid);meta=json.loads((R/f'assets/runtime/{aid}/white/poses.json').read_text());source=R/meta['habitat_glb'];d,b=read_glb(source);ii=material_body_image(d,row['prefix']);view=d['bufferViews'][d['images'][ii]['bufferView']];pixels=np.array(Image.open(io.BytesIO(b[view.get('byteOffset',0):view.get('byteOffset',0)+view['byteLength']])).convert('RGB'));shirt=np.array(Image.open(R/'masks'/aid/'shirt_main_color.png').convert('L'))>0;assert pixels.shape[:2]==shirt.shape
        luminance=srgb_to_linear(pixels[shirt])@np.array([.2126,.7152,.0722]);shade=np.clip(luminance/np.median(luminance),.30,2.25)
        for color,palette in [('gray',[120,120,120]),('brown',[122,76,42]),('brown_dark',[80,60,42])]:
            edited=pixels.copy();edited[shirt]=linear_to_srgb(srgb_to_linear(np.array(palette))[None,:]*shade[:,None]);buf=io.BytesIO();Image.fromarray(edited).save(buf,format='PNG');nd=copy.deepcopy(d);nb=append_image(nd,b,ii,buf.getvalue());out=T/'evidence/color_probes/assets'/aid/color/'runtime.glb';write_glb(out,nd,nb);assert geometry_animation_digest(nd,nb)==geometry_animation_digest(d,b);proof={'asset_id':aid,'prefix':row['prefix'],'probe_id':color,'expected_label':'brown' if color.startswith('brown') else color,'palette_srgb':palette,'split':split,'runtime_glb':str(out),'runtime_sha256':sha(out),'source_glb':str(source),'source_sha256':sha(source),'geometry_skin_animation_unchanged':True,'changed_outside_shirt_mask_pixels':int(np.any(pixels!=edited,axis=2)[~shirt].sum()),'source_uv_shirt_mask':str(R/'masks'/aid/'shirt_main_color.png'),'source_uv_shirt_mask_sha256':sha(R/'masks'/aid/'shirt_main_color.png'),'license':'Rocketbox MIT; task-authored shirt appearance','purpose':'color calibration only; does not add animals or change existing human roster'};assert proof['changed_outside_shirt_mask_pixels']==0;records.append(proof)
    dump(T/'evidence/color_probes/plan.json',records);return records

def render(T,R):
    probes=json.loads((T/'evidence/color_probes/plan.json').read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());cal=json.loads((T/'evidence/color_calibration.json').read_text());records=[]
    for v in views:
        house=v['house'];view=json.loads((R/f'evidence/habitat_view_{house}.json').read_text());r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,T/'evidence/color_probes'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
        for agent in cfg.agents:
            for spec in agent.sensor_specifications:spec.resolution=[640,640]
        with r.habitat_sim.Simulator(cfg) as sim:
            shader=configure_source_pbr(cfg,r.habitat_sim,metadata_mediator=sim.metadata_mediator);root=np.array(view['object_position_m']);eye=np.array(view['camera_position_m']);target=root+[0,.98,0];sim.initialize_agent(0,camera(r,eye,target))
            for p in probes:
                base=Path(p['runtime_glb']);obj=make_object(sim,r,base,root,view['yaw_rad'],semantic=7600101);obs=sim.get_sensor_observations();rgb=np.array(obs[uuids['rgb']])[...,:3];body=np.array(obs[uuids['semantic']])==7600101;shirt=roi(base,Path(p['source_uv_shirt_mask']),p['prefix'],view,body);shirt &= body;dest=T/'evidence/color_probes'/house/p['asset_id']/p['probe_id'];dest.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(dest/'rgb.png');Image.fromarray(np.uint8(shirt)*255).save(dest/'shirt_mask.png');m=measure(rgb,shirt,cal['thresholds']);rec={**p,'house':house,'rgb_path':str(dest/'rgb.png'),'mask_path':str(dest/'shirt_mask.png'),'rgb_sha256':sha(dest/'rgb.png'),'mask_sha256':sha(dest/'shirt_mask.png'),**m,'correct':m['measured_label']==p['expected_label'] and m['status']=='pass','mask_method':'perspective-correct UV shirt mask with whole-human depth and current native semantic visibility; eroded one pixel','shader':shader,'scene':str(scene)};records.append(rec);sim.get_rigid_object_manager().remove_object_by_id(obj.object_id);print('COLOR_PROBE',p['asset_id'],p['expected_label'],house,rec['correct'],m['dominant_fraction'],flush=True)
    dump(T/'evidence/color_probe_measurements.json',records)
    validate_existing(T)

def validate_existing(T):
    cal=json.loads((T/'evidence/color_calibration.json').read_text());records=json.loads((T/'evidence/color_probe_measurements.json').read_text())
    all_rows=[x for x in cal['measurements'] if not x.get('probe_id')]+records
    loaded=[(row,np.array(Image.open(row['rgb_path']).convert('RGB')),np.array(Image.open(row['mask_path']).convert('L'))>0) for row in all_rows]
    options=[]
    for minimum,hue_max,black_max,white_min,neutral_max in __import__('itertools').product([4.,5.,6.,8.,10.,12.,15.],[80.,90.,95.,100.],[20.,25.,30.],[55.,60.],[8.,10.,12.,15.,18.]):
        threshold={**cal['thresholds'],'brown_min_c':minimum,'brown_neutral_l_max':48.,'brown_hue_max':hue_max,'black_l_max':black_max,'white_l_min':white_min,'neutral_c_max':neutral_max};correct=sum(measure(rgb,mask,threshold)['status']=='pass' and measure(rgb,mask,threshold)['measured_label']==row['expected_label'] for row,rgb,mask in loaded if row['split']=='calibration');options.append(((correct,-abs(minimum-12)-abs(hue_max-95)-abs(black_max-20)-abs(white_min-55)-abs(neutral_max-10)),threshold))
    _,selected=max(options,key=lambda x:x[0]);cal['thresholds']=selected
    cal['measurements']=[{**row,**measure(rgb,mask,selected),'correct':measure(rgb,mask,selected)['status']=='pass' and measure(rgb,mask,selected)['measured_label']==row['expected_label']} for row,rgb,mask in loaded]
    cal['groups']={split:{'count':sum(x['split']==split for x in cal['measurements']),'correct':sum(x['split']==split and x['correct'] for x in cal['measurements'])} for split in ['calibration','held_out']};cal['known_shirt_labels']=sorted(set(cal['known_shirt_labels'])|{'gray','brown'});cal['not_present_in_shirt_set']=['orange','golden'];cal['unrepresented_boundaries']='orange/golden not selected as animal labels in this task';cal['method']+='; independent known gray/brown UV shirt variants rendered natively for one fitting and one held-out identity, same thresholds';cal['status']='pass' if all(x['count']==x['correct'] for x in cal['groups'].values()) else 'fail';dump(T/'evidence/color_calibration.json',cal)
    print('EXTENDED_CALIBRATION',cal['status'],cal['groups'],selected,flush=True)
    if cal['status']!='pass':raise RuntimeError('extended native shirt validation failed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--render',action='store_true');p.add_argument('--validate-existing',action='store_true');a=p.parse_args();
    if a.validate_existing:return validate_existing(a.task_root)
    render(a.task_root,a.reference_root) if a.render else build(a.task_root,a.reference_root)
if __name__=='__main__':main()
