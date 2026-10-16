"""Prepare the explicitly allowed library sparrow fallback, static on a silent perch."""
import sys,argparse,json,math,hashlib
from pathlib import Path
import bpy
from mathutils import Vector,Matrix
p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);T=a.task_root;sys.path.insert(0,str(T/'worktree/tools/assets'));from v7_roster_disk import check;check(T,32*1024**2);D=T/'round3/assets/bird_sparrow_fallback';D.mkdir(parents=True,exist_ok=True)
source=Path('/data/datasets/rocketbox/Microsoft-Rocketbox/Assets/Animals/Bird_Sparrow_01')
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(source/'Export/Bird_Sparrow_01.fbx'))
arm=next(o for o in bpy.data.objects if o.type=='ARMATURE');arm.data.pose_position='REST'
bones=[{'name':b.name,'head':list(arm.matrix_world@b.head_local),'tail':list(arm.matrix_world@b.tail_local)}for b in arm.data.bones]
dg=bpy.context.evaluated_depsgraph_get();meshes=[]
for o in list(bpy.data.objects):
 if o.type!='MESH':continue
 e=o.evaluated_get(dg);m=bpy.data.meshes.new_from_object(e,depsgraph=dg);no=bpy.data.objects.new('Sparrow_'+o.name,m);bpy.context.collection.objects.link(no);no.matrix_world=o.matrix_world.copy();meshes.append(no)
verts=[o.matrix_world@v.co for o in meshes for v in o.data.vertices];lo=[min(v[i]for v in verts)for i in range(3)];hi=[max(v[i]for v in verts)for i in range(3)]
print('BIRD_SOURCE_DIMENSIONS',lo,hi,'BONES',json.dumps(bones),flush=True)
for o in list(bpy.data.objects):
 if o not in meshes:bpy.data.objects.remove(o,do_unlink=True)
height=hi[2]-lo[2];scale=.18/height;center=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]];R=Matrix.Rotation(math.pi,4,'Z');Q=Matrix.Translation(Vector((0,0,.27)))@Matrix.Scale(scale,4)@R@Matrix.Translation(-Vector(center))
toes=[Q@Vector(b['head'])for b in bones if ' Toe'in b['name']]
perch_y=sum(v.y for v in toes)/len(toes)
for o in meshes:
 o.matrix_world=Q@o.matrix_world
 for mat in o.data.materials:
  if mat is None:continue
  mat.use_nodes=True;bs=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
  if bs:bs.inputs['Metallic'].default_value=0;bs.inputs['Roughness'].default_value=.8
  for node in mat.node_tree.nodes:
   if node.type=='TEX_IMAGE'and node.image:
    filename=Path(node.image.filepath.replace(chr(92),'/')).name;candidates=[x for x in(source/'Textures').iterdir()if x.name.lower()==filename.lower()]
    if candidates:node.image.filepath=str(candidates[0]);node.image.reload()
def material(name,color):
 m=bpy.data.materials.new(name);m.use_nodes=True;b=m.node_tree.nodes.get('Principled BSDF');b.inputs['Base Color'].default_value=(*color,1);b.inputs['Metallic'].default_value=0;b.inputs['Roughness'].default_value=.85;return m
wood=material('SilentPerchMatteWood',(.23,.12,.06))
def cylinder(name,rad,depth,loc,rotation=None):
 bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=rad,depth=depth,location=loc);o=bpy.context.object;o.name=name
 if rotation:o.rotation_euler=rotation
 o.data.materials.append(wood);be=o.modifiers.new('SoftEdges','BEVEL');be.width=.002;be.segments=2;return o
cylinder('SilentPerchBase',.10,.015,(0,0,.0075));cylinder('SilentPerchPost',.009,.245,(0,.015,.1375));bpy.ops.mesh.primitive_cube_add(size=1,location=(0,perch_y,.26));rail=bpy.context.object;rail.name='SilentPerchFlatRail';rail.dimensions=(.18,.075,.02);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);rail.data.materials.append(wood);be=rail.modifiers.new('SoftRailEdges','BEVEL');be.width=.002;be.segments=2
objects=[o for o in bpy.data.objects if o.type=='MESH'];bpy.ops.object.select_all(action='DESELECT')
for o in objects:o.select_set(True)
bpy.context.view_layer.objects.active=objects[0]
bpy.ops.export_scene.gltf(filepath=str(D/'runtime.glb'),export_format='GLB',use_selection=True,export_animations=False,export_yup=True,export_apply=True)
mouth=next((b for b in bones if 'mouth'in b['name'].lower()or 'beak'in b['name'].lower()),None)
raw=Q@Vector(mouth['head'])if mouth else None;point=[raw.x,raw.z,-raw.y]if raw else None
rec={'asset_id':'v7_rocketbox_bird_sparrow_fallback_01','taxonomy':'鸟 / 麻雀（兜底，非家养鹦鹉或金丝雀）','source':str(source),'source_license':'MIT','movement':'static','height_bird_m':.18,'perch_top_m':.27,'perch_support_patch_glb_m':{'center':[0,.27,-perch_y],'width_x':.18,'depth_z':.075},'perch_contact_method':'flat matte wooden perch top at the normalized lowest foot plane; native whole-base contact and foot-region visual review separate','units':'metres +Y up -Z forward','materials':{'metallic':0,'roughness_bird':.8,'roughness_perch':.85},'produces_sound':True,'perch_produces_sound':False,'audio_admission_owner':'v7-families','beak_local_glb_m':point,'beak_status':'library mouth bone candidate; native visual review pending'if point else'pending_native_annotation','bone_inventory':bones,'original_source_bounds_blender_m':{'min':lo,'max':hi},'source_mesh_geometry_provenance':'original Rocketbox Sparrow; static transforms only, no parrot breed claim','source_sha256':hashlib.sha256((source/'Export/Bird_Sparrow_01.fbx').read_bytes()).hexdigest(),'habitat_status':'pending_native','ue_status':'pending_native'}
(D/'source.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');print('SPARROW_FALLBACK_BUILT',point)
