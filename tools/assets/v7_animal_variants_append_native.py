"""Native reruns for authorized static derivatives, retaining previous evidence."""
import argparse,json,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
from avengine.assets.source_materials import configure_source_pbr
from v7_animal_variants_habitat import qualify,dump
from v7_animal_variants_v6_candidates import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);p.add_argument('--phase',required=True);a=p.parse_args();T=a.task_root;P=T/'append_20261012';cal=T/'evidence/color_calibration.json';(P/'evidence/color_calibration.json').write_bytes(cal.read_bytes());RUN=P/'runs'/a.phase;(RUN/'evidence').mkdir(parents=True,exist_ok=True);(RUN/'evidence/color_calibration.json').write_bytes(cal.read_bytes());queue=json.loads(a.queue.read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());placements=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/v7_human_roster_20261004/round3/evidence/chair_native_placements.json').read_text());records=[]
 for view in views:
  house=view['house'];r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=view['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(view['room_manifest']),Path(view['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,P/'evidence/native'/a.phase/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
  for agent in cfg.agents:
   for spec in agent.sensor_specifications:spec.resolution=[640,640]
  with r.habitat_sim.Simulator(cfg) as sim:
   shader=configure_source_pbr(cfg,r.habitat_sim,metadata_mediator=sim.metadata_mediator)
   for mp in queue:
    meta=json.loads(Path(mp).read_text());aid=meta['asset_id'];(RUN/'assets'/aid).mkdir(parents=True,exist_ok=True);rec={'asset_id':aid,'house':house,'metadata':str(mp),'runtime_sha256':sha(meta['runtime_glb']),'color_calibration_sha256':sha(cal),'phase':a.phase,'qa_protocol':'native_four_views_v7_append_unchanged_gates'}
    try:
     assert rec['runtime_sha256']==meta['runtime_sha256'];rec.update(qualify(sim,r,uuids,RUN,meta,house,placements,shader,contact_threshold_m=.012 if meta['mode']=='v6_idle_static_derivative' else None))
     if meta['mode']=='v6_idle_static_derivative':
      rec['v6_original_C_four_paw_threshold_m']=.012
      if rec['maximum_support_delta_m']>.012+1e-8 or meta['normalization'].get('body_up_status')!='pass':rec['status']='fail';rec['error']='v6 original C 12 mm four-paw/body-up gate'
     rec['mesh_budget_status']='pass' if meta['triangles']<=100000 and Path(meta['runtime_glb']).stat().st_size<=40000000 else 'fail'
     if rec['mesh_budget_status']!='pass':rec['status']='fail'
    except BaseException as ex:
     rec.update(status='fail',error=repr(ex),traceback=traceback.format_exc())
     for obj in list(sim.get_rigid_object_manager().get_objects_by_handle_substring('').values()):sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
    dump(P/'evidence/native'/a.phase/house/(aid+'.json'),rec);records.append(rec);dump(P/'reports'/(a.phase+'_native_results.json'),records);print('APPEND_NATIVE',a.phase,aid,house,rec['status'],rec.get('maximum_support_delta_m'),rec.get('dominant_label'),rec.get('pooled_body_color',{}).get('dominant_fraction'),rec.get('error'),flush=True)
 return int(any(x['status']!='pass' for x in records))
if __name__=='__main__':raise SystemExit(main())
