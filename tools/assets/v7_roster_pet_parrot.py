"""Freeze the CC-BY author parrot idle pose on a silent matte perch."""
import argparse,json,sys,math,hashlib
from pathlib import Path
import bpy
from mathutils import Vector
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);T=a.task_root;W=T/'worktree';R=T/'round4'
sys.path.insert(0,str(W/'tools/assets'))
from v7_roster_disk import check
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,rigid_glb,accessor
check(T,64*1024**2);D=R/'assets/parrot';D.mkdir(parents=True,exist_ok=True)
source=R/'assets/Parrot_Rig_1.0.glb';d,b=read_glb(source)
action=next(a for a in d['animations']if a['name']=='Idle Loop')
posed,ev=pose(d,b,action,0,canonical_flip=True,ground=False,emitter_bone='beak')
allv=np.concatenate([r['positions']for r in posed]);lo=allv.min(0);hi=allv.max(0)
scale=.30/(hi[1]-lo[1]);center=np.array([(lo[0]+hi[0])/2,lo[1],(lo[2]+hi[2])/2]);top=.55
footqa=[];beak=[]
for row in posed:
 n=d['nodes'][row['node']];prim=d['meshes'][n['mesh']]['primitives'][row['primitive']]
 skin=d['skins'][n['skin']];js=accessor(d,b,prim['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,prim['attributes']['WEIGHTS_0'])
 v=(row['positions']-center)*scale+[0,top,0];row['positions']=v
 for side in ['l','r']:
  slots=[k for k,idx in enumerate(skin['joints'])if d['nodes'][idx]['name']in ['foot_'+side,'ball_'+side]]
  weight=np.sum(ws*np.isin(js,slots),axis=1);pts=v[weight>.25]
  if len(pts):
   lowest=pts[pts[:,1]<pts[:,1].min()+.0015]
   footqa.append({'side':side,'weighted_vertices':len(pts),'minimum_gap_to_perch_m':float(pts[:,1].min()-top),'contact_region_points_m':lowest.tolist()})
 slots=[k for k,idx in enumerate(skin['joints'])if d['nodes'][idx]['name']=='beak']
 beak.extend(v[np.sum(ws*np.isin(js,slots),axis=1)>.25].tolist())
beak=np.asarray(beak);assert len(beak)>0
anchor=beak[np.argmin(beak[:,2])].tolist()
for mat in d['materials']:
 mat.pop('normalTexture',None)  # Native Habitat rejects differing texture matrices.
 pbr=mat.setdefault('pbrMetallicRoughness',{});pbr['metallicFactor']=0;pbr['roughnessFactor']=.85;pbr.pop('metallicRoughnessTexture',None)
 for key in ['KHR_materials_clearcoat','KHR_materials_sheen','KHR_materials_specular']:mat.get('extensions',{}).pop(key,None)
body=D/'body_temporary.glb';rigid_glb(d,b,posed,body)
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(body))
bird=[o for o in bpy.data.objects if o.type=='MESH']
for o in bird:
 o.name='PetParrot_'+o.name
 for mat in o.data.materials:
  if not mat:continue
  mat.use_nodes=True;bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
  bs.inputs['Metallic'].default_value=0;bs.inputs['Roughness'].default_value=.85;bs.inputs['Specular IOR Level'].default_value=.15
wood=bpy.data.materials.new('ParrotPerchMatteWood');wood.use_nodes=True;bs=wood.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.25,.13,.07,1);bs.inputs['Metallic'].default_value=0;bs.inputs['Roughness'].default_value=.85;bs.inputs['Specular IOR Level'].default_value=.15
def cylinder(name,radius,depth,loc):
 bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=radius,depth=depth,location=loc);o=bpy.context.object;o.name=name;o.data.materials.append(wood);be=o.modifiers.new('EdgeBevel','BEVEL');be.width=.002;be.segments=2
feet=np.concatenate([r['contact_region_points_m']for r in footqa]);cx,cz=(feet[:,[0,2]].min(0)+feet[:,[0,2]].max(0))/2
width=max(.12,float(np.ptp(feet[:,0]))+.012);depth=max(.08,float(np.ptp(feet[:,2]))+.012)
cylinder('SilentPetBirdBase',.10,.016,(cx,-cz,.008));cylinder('SilentPetBirdPost',.008,top-.036,(cx,-cz,(top-.036)/2+.016))
bpy.ops.mesh.primitive_cube_add(size=1,location=(cx,-cz,top-.01));rail=bpy.context.object;rail.name='SilentPetBirdRail';rail.dimensions=(width,depth,.02);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);rail.data.materials.append(wood)
for row in footqa:
 pts=np.asarray(row['contact_region_points_m']);row['contact_points_inside_perch']=bool(np.all(abs(pts[:,0]-cx)<width/2)and np.all(abs(pts[:,2]-cz)<depth/2))
 assert abs(row['minimum_gap_to_perch_m'])<.001 and row['contact_points_inside_perch']
bpy.ops.object.select_all(action='DESELECT')
for o in bpy.data.objects:
 if o.type=='MESH':o.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(D/'runtime.glb'),export_format='GLB',use_selection=True,export_animations=False,export_yup=True,export_apply=True)
meta={'asset_id':'v7_ccby_pet_parrot_chris_woods_01','name_zh':'家养鹦鹉类（风格化，物种未定）','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_license':'CC-BY-4.0','source_author':'Chris Woods','source_url':'https://chris-woods.itch.io/parrot-rig','source_pose':'Idle Loop at t=0','source_animation_names':[a['name']for a in d['animations']],'movement':'static','walk_allowed':False,'height_bird_m':.30,'perch_top_m':top,'total_height_m':.85,'canonical_front':'-Z from source beak +Z, 180-degree yaw','units':'metres +Y up -Z forward','beak_local_glb_m':anchor,'beak_method':'frontmost vertex with >0.25 skin weight on author beak bone; frozen exposed beak surface, no image annotation','beak_source_weighted_vertices':len(beak),'foot_perch_qa':footqa,'perch_patch':{'center_m':[float(cx),top,float(cz)],'width_m':width,'depth_m':depth},'materials':{'metallic':0,'roughness':.85,'blender_specular_ior_level':.15,'changes':'removed metallic/roughness texture and incompatible independently transformed normal texture; retained base colour and original geometry'},'perch_produces_sound':False,'audio_admission_owner':'v7-families','modifications':'freeze authored idle pose; scale/orient/ground on task-authored perch; matte PBR; no breed/anatomy generation','habitat_status':'pending_native','ue_status':'pending_native'}
json_write(D/'source.json',meta);print('PET_PARROT_BUILT',json.dumps({'anchor':anchor,'feet':[(r['side'],r['minimum_gap_to_perch_m'])for r in footqa],'runtime_bytes':(D/'runtime.glb').stat().st_size}),flush=True)
