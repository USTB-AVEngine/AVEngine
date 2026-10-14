"""Measure saved actual Habitat skin; no new animation, renderer, or RIR run."""
import argparse,json,pathlib,subprocess,time
import numpy as np
from avengine.assets.animal_motion.profiles import Rig,PAWS,paw_points
from avengine.assets.animal_motion.walk import ConstrainedWalk
from avengine.assets.roster_motion.episode import triangle_mesh
from avengine.assets.roster_motion.io import read_glb
from avengine.assets.roster_motion.pose import accessor,globals_at


def actual_pose(path):
    document,binary=read_glb(path);vertices=[];faces=[];offset=0
    for node in document['nodes']:
        if 'mesh' not in node:continue
        if any(key in node for key in ('matrix','translation','rotation','scale')):
            raise ValueError('saved_rigid_pose_has_unexpected_node_transform')
        for primitive in document['meshes'][node['mesh']]['primitives']:
            positions=accessor(document,binary,primitive['attributes']['POSITION'])
            indices=accessor(document,binary,primitive['indices']).astype(np.int64).reshape(-1,3)
            vertices.append(positions);faces.append(indices+offset);offset+=len(positions)
    return np.concatenate(vertices).astype('<f4'),np.concatenate(faces)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inventory',type=pathlib.Path,required=True);parser.add_argument('--motion',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True);parser.add_argument('--blender',default='/data/jzy/blender/blender-4.5.13-linux-x64/blender');args=parser.parse_args()
    native=json.loads(args.motion.read_text());record=next(a for a in json.loads(args.inventory.read_text())['assets'] if a['asset_id']==native['asset_id']);frames=native['frames'];assert native['motion_mode']=='walk_animation' and len(frames)==native['frame_count']
    args.output.mkdir(parents=True,exist_ok=False);rig=Rig.load(record);assert rig.path.resolve()==pathlib.Path(native['source_master']).resolve()
    # Only reference anatomy and source torso rotation are needed. The observed
    # deformed vertices come directly from every saved native pose, not this walk.
    walk=ConstrainedWalk(rig,speed=.03,period=2.4)
    assert np.allclose(walk.transform,native['support_plane']['matrix'],atol=1e-9,rtol=0),'saved_native_reference_basis_differs'
    default=globals_at(rig.document,rig.binary)
    binding_globals={j:walk.transform@rig.basis@matrix for j,matrix in default.items()}
    rest_parts=[];recovered={}
    for part in walk.reference_parts:
        node=rig.document['nodes'][part['node']];primitive=rig.document['meshes'][node['mesh']]['primitives'][part['primitive']]
        matrix=walk.transform@rig.basis@default[part['node']];positions=accessor(rig.document,rig.binary,primitive['attributes']['POSITION'])
        rest_parts.append({**part,'positions':positions@matrix[:3,:3].T+matrix[:3,3]})
        skin=rig.document['skins'][node['skin']];ibm=accessor(rig.document,rig.binary,skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        for joint,inverse in zip(skin['joints'],ibm):
            candidate=matrix@np.linalg.inv(inverse)
            if joint in recovered and not np.allclose(recovered[joint],candidate,atol=1e-6,rtol=0):raise ValueError('inconsistent_original_inverse_binding')
            recovered[joint]=candidate
    binding_globals.update(recovered)
    rest,faces=triangle_mesh(rig.document,rig.binary,rest_parts)
    reconstructed,rest_faces=triangle_mesh(rig.document,rig.binary,rig.skin(binding_globals));assert np.array_equal(faces,rest_faces)
    rest_error=float(np.linalg.norm(rest-reconstructed,axis=1).max());assert rest_error<1e-6,'inverse_bind_does_not_reproduce_source_geometry'
    regions=[rig.body,rig.head,rig.tail,*[rig.chains[s][2] for s in PAWS]];binding=np.array([binding_globals[j] for j in regions]);rest_paws=paw_points(rest_parts,rig.masks)
    rest_file=args.output/'actual_skin_rest.npz';np.savez_compressed(rest_file,positions=rest.astype('<f4'),faces=faces.astype(np.int64),region_matrices=binding)
    head=np.concatenate([part['positions'][mask] for part,mask in zip(rest_parts,rig.head_masks) if mask.any()]);marks={f'foot_{i}':rest_paws[side]['point'].tolist() for i,side in enumerate(PAWS)};marks.update(head_bbox_min=head.min(0).tolist(),head_bbox_max=head.max(0).tolist())
    metadata={'rest_npz':str(rest_file),'registration':{'landmarks_local_m':marks},'source_master':str(rig.path),'native_motion':str(args.motion),'frame_count':len(frames),'source_coordinate_system':'Habitat +Y up -Z front','position_encoding':'exact saved rigid GLB float32','region_definition':'own original Head skin bounding box; own four Idle paw points and Tail landmark; same regional BVH classifier','region_matrix_scope':'Body rotation exactly from saved source_phase_s and unchanged original torso Walking; other matrices are not consumed because same-region pairs are excluded','no_animation_or_native_render_recomputed':True,'reference_pose':'unmodified source primitive binding geometry, not an Idle animation sample','inverse_bind_source_reconstruction_error_m':rest_error}
    meta_file=args.output/'actual_skin_stream.json';meta_file.write_text(json.dumps(metadata,indent=2));surface=args.output/'surface.json';cmd=[args.blender,'-b','--factory-startup','--disable-autoexec','-t','1','--python-exit-code','2','--python',str(pathlib.Path(__file__).with_name('check_motion_only_surface_blender.py')),'--','--glb',str(rig.path),'--output',str(surface),'--frames',str(len(frames)),'--skin-stream-meta',str(meta_file),'--validation-policy','owner_relaxed_20261014'];child=subprocess.Popen(cmd,stdin=subprocess.PIPE);sent=0;started=time.time()
    try:
        for index,frame in enumerate(frames):
            assert abs(frame['time_s']-index/30)<1e-10
            vertices,current_faces=actual_pose(frame['pose_glb']);assert np.array_equal(current_faces,faces),'native_pose_topology_changed';assert len(vertices)==len(rest)
            source=rig.globals(frame['source_phase_s']);body=walk.transform@source[rig.body]
            matrices=binding.copy();matrices[0]=body
            child.stdin.write(np.array([frame['time_s']],dtype='<f8').tobytes());child.stdin.write(matrices.astype('<f8').tobytes());child.stdin.write(vertices.tobytes());sent+=1
    except BrokenPipeError:pass
    finally:
        try:child.stdin.close()
        except BrokenPipeError:pass
        code=child.wait()
    receipt={'asset_id':native['asset_id'],'native_motion':str(args.motion),'surface':str(surface),'source_master':str(rig.path),'exit_code':code,'frames_sent':sent,'elapsed_s':time.time()-started,'original_files_modified':False,'audio_status':'deferred_to_production'};(args.output/'execution.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
    raise SystemExit(code)
if __name__=='__main__':main()
