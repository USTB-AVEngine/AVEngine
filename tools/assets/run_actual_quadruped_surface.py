"""Send the actual native float32 skin to Blender's unchanged BVH penetration gate."""
import json,pathlib,subprocess
import numpy as np
from avengine.assets.roster_motion.episode import triangle_mesh
def run(rig,walk,times,output,blender,registration_request,*,validation_policy='strict'):
    output=pathlib.Path(output)
    reference=walk.reference_parts;v,f=triangle_mesh(rig.document,rig.binary,reference)
    names={n.get('name'):i for i,n in enumerate(rig.document['nodes'])}
    tail=names.get('Tail',names.get('TailRoot',rig.tail))
    regions=[rig.body,rig.head,tail,*[rig.chains[s][2] for s in ('FL','FR','HL','HR')]]
    rest=output/'actual_skin_rest.npz'
    np.savez_compressed(rest,positions=v.astype('<f4'),faces=f,region_matrices=np.array([walk.reference_g[j] for j in regions]))
    metadata=output/'actual_skin_stream.json'
    row=json.loads(pathlib.Path(registration_request).read_text())['registration']
    transform=walk.transform@rig.basis
    original=row['landmarks_local_m'].copy()
    for key,value in list(row['landmarks_local_m'].items()):
        if isinstance(value,list) and len(value)==3:row['landmarks_local_m'][key]=(transform[:3,:3]@np.asarray(value)+transform[:3,3]).tolist()
    lo=np.asarray(original['head_bbox_min']);hi=np.asarray(original['head_bbox_max'])
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@transform[:3,:3].T+transform[:3,3]
    row['landmarks_local_m']['head_bbox_min']=corners.min(0).tolist();row['landmarks_local_m']['head_bbox_max']=corners.max(0).tolist()
    metadata.write_text(json.dumps({'rest_npz':str(rest),'registration':row,'source_master':str(rig.path),'frame_count':len(times),
       'source_coordinate_system':'Habitat +Y up -Z front','position_encoding':'little-endian float32 as rigid_glb',
       'source_evaluator':'all original skin influences; exact source-defined Blender-baked Walking','only_geometry_transport_changed':True},indent=2))
    surface=output/'surface.json'
    cmd=[str(blender),'-b','--factory-startup','--disable-autoexec','-t','1','--python-exit-code','2','--python',
         str(pathlib.Path(__file__).with_name('blender_check_quadruped_surface.py')),'--','--glb',str(rig.path),
         '--output',str(surface),'--frames',str(len(times)),'--skin-stream-meta',str(metadata),'--validation-policy',validation_policy]
    child=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for t in times:
            parts,row=walk.frame(float(t));v,f1=triangle_mesh(rig.document,rig.binary,parts)
            if not np.array_equal(f,f1):raise ValueError('actual_pose_changed_triangle_topology')
            matrices=np.array([walk.last_globals[j] for j in regions],dtype='<f8')
            child.stdin.write(np.array([t],dtype='<f8').tobytes())
            child.stdin.write(matrices.tobytes());child.stdin.write(v.astype('<f4').tobytes())
    except BrokenPipeError:pass
    finally:
        try:child.stdin.close()
        except BrokenPipeError:pass
        code=child.wait()
    return code,json.loads(surface.read_text())
