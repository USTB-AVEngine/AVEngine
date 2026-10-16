"""Create five matte chair styles, joined meshes, in canonical metric GLB."""
import bpy,json,sys,math,os
from pathlib import Path
T=Path(sys.argv[sys.argv.index('--')+1]);sys.path.insert(0,str(T/'worktree/tools/assets'))
from v7_roster_disk import check
R=T/'round3';(R/'assets/chairs').mkdir(parents=True,exist_ok=True);rows=[]
def mat(name,rgb):
 m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*rgb,1);p.inputs['Metallic'].default_value=0;p.inputs['Roughness'].default_value=.85;p.inputs['Specular IOR Level'].default_value=.15;return m
BOXES=[]
def box(name,c,s,m,bevel=.012,yaw=0):
 BOXES.append({'name':name,'center_m':list(c),'size_m':list(s),'yaw_rad':yaw})
 bpy.ops.mesh.primitive_cube_add(size=1,location=(c[0],-c[2],c[1]));o=bpy.context.object;o.name=name;o.dimensions=(s[0],s[2],s[1]);o.rotation_euler[2]=yaw;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(m)
 if bevel:
  mod=o.modifiers.new('SoftEdges','BEVEL');mod.width=min(bevel,min(s)*.3);mod.segments=3;bpy.context.view_layer.objects.active=o;bpy.ops.object.modifier_apply(modifier=mod.name)
 for f in o.data.polygons:f.use_smooth=True
 mod=o.modifiers.new('WeightedNormals','WEIGHTED_NORMAL');mod.keep_sharp=True;bpy.ops.object.modifier_apply(modifier=mod.name)
 return o
def cyl(name,c,r,h,m,axis='y'):
 bpy.ops.mesh.primitive_cylinder_add(vertices=40,radius=r,depth=h,location=(c[0],-c[2],c[1]));o=bpy.context.object;o.name=name
 if axis=='x':o.rotation_euler[1]=math.pi/2
 o.data.materials.append(m)
 for p in o.data.polygons:p.use_smooth=True
 return o
def build(style,seat,w,d,back,arms,office=False,stool=False):
 BOXES.clear();bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
 wood=mat(style+'_wood',(.24,.12,.055));fabric=mat(style+'_fabric',(.32,.36,.37)if style!='armchair'else(.20,.29,.28));dark=mat(style+'_matte_plastic',(.055,.065,.070));objects=[];supports=[]
 if stool:
  objects.append(box('seat',(0,seat-.018,0),(w,.036,d),wood,.018))
 else:
  objects.append(box('seat',(0,seat-.04,0),(w,.08,d),fabric,.034))
 if office:
  objects.append(cyl('column',(0,(seat-.09)/2,0),.035,seat-.09,dark))
  for i in range(5):
   a=2*math.pi*i/5;x,z=.27*math.cos(a),.27*math.sin(a);objects.append(box('base_arm',(x/2,.055,z/2),(.29,.035,.045),dark,.008,yaw=-a));objects.append(cyl('wheel',(x,.024,z),.024,.055,dark,'x'));supports.append([x,0,z])
 else:
  for x in [-w/2+.055,w/2-.055]:
   for z in [-d/2+.055,d/2-.055]:objects.append(box('leg',(x,(seat-.08)/2,z),(.05,seat-.08,.05),wood,.009));supports.append([x,0,z])
 if back:
  objects.append(box('back',(0,seat+(back-seat)/2,d/2+.085),(w,.0+back-seat,.09),fabric if style in ['armchair','sofa_chair','office']else wood,.025))
 if arms:
  for x in [-w/2-.045,w/2+.045]:
   objects.append(box('arm',(x,seat+.16,0),(.09,.075,d+.10),fabric if style in ['armchair','sofa_chair']else dark,.025));objects.append(box('arm_support',(x,seat+.08,d/2-.03),(.04,.16,.04),dark,.006))
 if style=='dining':
  # A slight gap in the back and two slats make it read as a dining chair.
  for x in [-w/2+.06,w/2-.06]:objects.append(box('back_post',(x,(seat+back)/2,d/2+.08),(.04,back-seat,.04),wood,.008))
 bpy.ops.object.select_all(action='DESELECT')
 for o in objects:o.select_set(True)
 bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.join();o=bpy.context.object;o.name='v7_chair_'+style;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);out=R/'assets/chairs'/('v7_chair_'+style+'.glb');proof=check(T,8*1024**2);bpy.ops.export_scene.gltf(filepath=str(out),export_format='GLB',use_selection=True,export_animations=False,export_yup=True)
 rows.append({'asset_id':'v7_chair_'+style,'style':style,'seat_height_m':seat,'seat_width_m':w,'seat_depth_m':d,'seat_center_m':[0,seat,0],'front':'-Z','up':'+Y','units':'metres','support_points_m':supports,'solid_boxes':list(BOXES),'habitat_glb':str(out),'ue_status':'pending_native_import','movement':'static','produces_sound':False,'foa_emitter':None,'materials':{'metallic':0,'roughness':.85,'blender_specular_ior_level':.15},'source':'task-authored procedural mesh; no external bitmap or furniture mesh','release_license':'owner_dataset_license_not_selected','disk_preflight':proof,'vertices':len(o.data.vertices),'triangles_estimate':sum(len(p.vertices)-2 for p in o.data.polygons)})
build('dining',.45,.49,.43,.94,False)
build('office',.45,.49,.45,1.03,True,office=True)
build('armchair',.42,.57,.48,.94,True)
build('stool',.42,.43,.39,0,False,stool=True)
build('sofa_chair',.40,.66,.54,.92,True)
(R/'registrations').mkdir(exist_ok=True);(R/'registrations/chair_roster.json').write_text(json.dumps({'schema':'v7_chair_roster_v1','chairs':rows},ensure_ascii=False,indent=2)+'\n')
print('CHAIRS',[(r['asset_id'],r['vertices'])for r in rows],flush=True)
