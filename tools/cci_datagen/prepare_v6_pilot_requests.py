#!/usr/bin/env python3
"""Prepare 8 x 8 train-house requests with the existing v6 QA06 moving recipe.

This is a temporary infrastructure pilot, not the replacing v7 sampler. All
copied configuration paths and provenance are explicit; source inputs stay immutable.
"""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from resources import write_json
from generate import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--simulation',required=True);p.add_argument('--references',required=True);p.add_argument('--mirror-root',default='/');p.add_argument('--native-root',default='/data/avengine_external');p.add_argument('--runtime-config-directory');p.add_argument('--output',required=True)
 a=p.parse_args();mirror=Path(a.mirror_root);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);refs=Path(a.references);source=json.loads(Path(a.template).read_text());rec=json.loads((refs/'recommended_initial12.json').read_text());houses=[h for h in rec['houses'] if h['split']=='train']
 if len(houses)!=8:raise ValueError('the v6 pilot uses exactly the approved original 8 training houses')
 def mapped(value):
  if isinstance(value,str) and value.startswith('/data/'):
   if value.startswith('/data/avengine_external/') and ('runtime-prefixes/' in value or 'rlr-sdk/' in value):return str(Path(a.native_root)/value.removeprefix('/data/avengine_external/'))
   return str(mirror/value.lstrip('/'))
  if isinstance(value,list):return [mapped(v) for v in value]
  if isinstance(value,dict):return {k:mapped(v) for k,v in value.items()}
  return value
 from avengine.acoustics.runtime import load_habitat_runtime
 native=Path(a.native_root)
 hs,_=load_habitat_runtime(runtime_mode='current-installed',runtime_prefix=str(native/'runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z'),magnum_python_site=str(native/'runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages'),rlr_sdk_root=str(native/'rlr-sdk/RLRAudioPropagationPkg'))
 entries=[];house_scenes={}
 for h in houses:
  hid=h['house'];b=mirror/'data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/measurement_final/houses'/hid;room=mapped(json.loads((b/'room_manifest.json').read_text()));d=out/'rooms'/hid;d.mkdir(parents=True,exist_ok=True)
  scene=room['scene']['scene_id'];house_scenes[hid]=scene;dataset=d/'scene_dataset_config.json';write_json(dataset,{'stages':{'paths':{'.glb':[scene]}}});room['scene']['dataset_config_path']=str(dataset)
  for asset in room['assets']:
   if asset['role']=='scene_dataset_config':asset['path']=str(dataset)
  assets={v['role']:v['path'] for v in room['assets']}
  manifest=d/'room_manifest.json';write_json(manifest,room)
  witness=json.loads((refs/'batch2'/hid/'result.json').read_text())['input']['acoustic_rooms'][0]['placement'];floor=None
  pf=hs.PathFinder()
  if not pf.load_nav_mesh(room['scene']['navmesh_path']):raise ValueError('failed to load the accepted house navmesh')
  snapped=pf.snap_point(witness['camera_m']);floor=float(snapped[1])
  if not __import__('math').isfinite(floor):raise ValueError('accepted room camera has no measured navigable floor')
  floor_doc=d/'floor_reference.json';write_json(floor_doc,{'floor_height_m':floor,'snapped_point_m':[float(x) for x in snapped],'camera_witness_m':witness['camera_m'],'navmesh_path':room['scene']['navmesh_path'],'navmesh_sha256':sha(room['scene']['navmesh_path']),'measurement_kind':'habitat_pathfinder_snap_point_v1','gpu_renderer_created':False})
  entries.append({'room_id':hid,'family':'hm3d','renderer':'habitat','room_manifest':str(manifest),'acoustic_package':str(b/'package/manifest.json'),'floor_reference':{'path':str(floor_doc),'status':'measured','measurement_kind':'habitat_pathfinder_snap_point_v1'},'static_geometry':{'representation':'real_surface_mesh','source':'acoustic_package_arrays','source_manifest':str(b/'package/manifest.json'),'vertices':str(b/'package/acoustic/vertices.npy'),'triangles':str(b/'package/acoustic/triangles.npy'),'triangle_material_ids':str(b/'package/acoustic/triangle_material_ids.npy'),'coordinate_frame':{'linear_unit':'meter','up_axis':'+Y','forward_axis':'-Z','handedness':'right'},'render_surface':True},'semantics':room['semantics'],'pilot_scope':'legacy v6 request preparation; raw compiled geometry QA remains visible'})
 catalog=out/'train8_room_catalog.json';write_json(catalog,{'rooms':entries,'claim_boundary':'temporary v6 infrastructure pilot; no held-out rooms, no formal data admission'})
 sim=json.loads(Path(a.simulation).read_text());sim['simulation']['thread_count']=1;simulation=out/'simulation_single_thread.json';write_json(simulation,sim)
 template=mapped(source);paths=[]
 if a.runtime_config_directory:
  # Reuse the four separately remapped config copies, not raw JSON paths.
  mapping=json.loads((Path(a.runtime_config_directory)/'PATH_MAPPING.json').read_text())
  print('runtime_mapping_keys',list(mapping))
  records=mapping.get('rows',mapping.get('files',mapping.get('documents',[])))
  if isinstance(records,dict):records=list(records.values())
  for row in records:
   origin=row.get('source',row.get('source_path',''));target=row.get('effective',row.get('target',row.get('target_path',row.get('output',''))))
   if not target:continue
   for field in ('prepared_manifest','source_registry','sound_pool'):
    if template.get(field)==origin or source.get(field)==origin:template[field]=target
 for index in range(64):
  h=houses[index%8];hid=h['house'];request=copy.deepcopy(template);seed=2026100300+index;wid=f'cci_v6_pilot64_r2_{index:03d}_{hid}'
  request.update(episode_id=wid,world_id=wid,room_id=hid,seed=seed,room_catalog=str(catalog),simulation_request=str(simulation),sampling_candidate_index=0)
  runtime=request['runtime'];runtime.pop('unreal_editor',None);runtime.pop('uproject',None);runtime.pop('spear_ext_dir',None);runtime.pop('ddc_directory',None);runtime.pop('ddc_profile',None)
  native=Path(a.native_root);runtime.update(runtime_prefix=str(native/'runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z'),magnum_python_site=str(native/'runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages'),rlr_sdk_root=str(native/'rlr-sdk/RLRAudioPropagationPkg'),rlr_threads=1,cci_preflight_scene=house_scenes[hid])
  runtime['path_bindings']={'AVENGINE_HM3D_ROOT':str(mirror/'data/datasets/habitat_data'),'AVENGINE_HM3D_ACOUSTIC_ROOT':str(mirror/'data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/measurement_final/houses'/hid/'package'),'AVENGINE_QA_V3_ROOT':str(mirror/'data/jzy/tmp'),'AVENGINE_MULTI_HOME_ROOT':str(mirror/'data/datasets/avengine_workspaces/multi_home_activity_20260905')}
  request['cci_pilot_provenance']={'template':str(Path(a.template).resolve()),'template_sha256':sha(a.template),'recipe':'unchanged existing v6 QA06 moving 10-second recipe, original 8 training houses x 8 seeds','not_formal_v7':True,'test_members_generated':False}
  dest=out/'requests'/f'{wid}.json';write_json(dest,request);paths.append(str(dest))
 write_json(out/'pilot64_request_paths.json',paths);write_json(out/'PILOT_REQUESTS_PROVENANCE.json',{'world_count':64,'training_houses':8,'held_out_houses':0,'recipe_template_sha256':sha(a.template),'simulation_source_sha256':sha(a.simulation),'requests':[{'path':r,'sha256':sha(r)} for r in paths],'planning_validated':False,'rendering_validated':False,'note':'requests are prepared; CPU/native planning and GPU/audio delivery need actual validation'})
 print('prepared',len(paths),'train-only pilot requests',out)
if __name__=='__main__':main()
