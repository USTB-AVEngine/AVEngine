"""Measure original bird foot regions against its own derived perch, without altering either source."""
import sys,argparse,json,math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector,Matrix
p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);T=p.parse_args(sys.argv[sys.argv.index('--')+1:]).task_root
meta=json.loads((T/'round3/assets/bird_sparrow_fallback/source.json').read_text());source=Path(meta['source'])/'Export/Bird_Sparrow_01.fbx'
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(source));arm=next(o for o in bpy.data.objects if o.type=='ARMATURE');arm.data.pose_position='REST'
lo=meta['original_source_bounds_blender_m']['min'];hi=meta['original_source_bounds_blender_m']['max'];center=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]];Q=Matrix.Translation(Vector((0,0,.27)))@Matrix.Scale(.18/(hi[2]-lo[2]),4)@Matrix.Rotation(math.pi,4,'Z')@Matrix.Translation(-Vector(center))
patch=meta['perch_support_patch_glb_m'];dg=bpy.context.evaluated_depsgraph_get();rows=[]
for side in ['L','R']:
 pts=[];groups=set()
 for o in bpy.data.objects:
  if o.type!='MESH':continue
  indices={g.index for g in o.vertex_groups if ' '+side+' 'in g.name.replace('_',' ')and any(x in g.name.lower()for x in ['foot','toe'])};groups.update(g.name for g in o.vertex_groups if g.index in indices)
  e=o.evaluated_get(dg);m=e.to_mesh();assert len(m.vertices)==len(o.data.vertices)
  for v in o.data.vertices:
   if sum(g.weight for g in v.groups if g.group in indices)>.15:
    w=Q@(o.matrix_world@m.vertices[v.index].co);pts.append([w.x,w.z,-w.y])
  e.to_mesh_clear()
 arr=np.asarray(pts);r={'side':side,'groups':sorted(groups),'weighted_vertices':len(pts),'status':'fail'}
 if len(pts):
  height=float(arr[:,1].min());contact=arr[arr[:,1]<height+.0015];center=np.asarray(patch['center']);inside=(abs(contact[:,0]-center[0])<=patch['width_x']/2+.001)&(abs(contact[:,2]-center[2])<=patch['depth_z']/2+.001);gap=height-patch['center'][1]
  r.update(foot_min_y_m=height,perch_gap_m=gap,low_contact_vertices=len(contact),low_vertices_inside_perch=int(inside.sum()),lowest_vertices_glb_m=contact.tolist(),status='pass'if abs(gap)<.012 and inside.all()else'fail')
 rows.append(r)
rec={'asset_id':meta['asset_id'],'method':'original FBX REST skin weights foot/toe >0.15; lowest 1.5mm vertex region against derived flat perch rectangle; no original mutation','feet':rows,'status':'pass'if all(r['status']=='pass'for r in rows)else'fail','native_base_contact_proof':'round3/evidence/bird_native_results.json'}
(T/'round3/evidence/bird_foot_perch.json').write_text(json.dumps(rec,indent=2)+'\n');print('BIRD_FOOT_PERCH',json.dumps(rec))
