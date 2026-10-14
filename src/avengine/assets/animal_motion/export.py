"""Export the repaired target skin and a metric Walking clip for native UE."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from avengine.assets.roster_motion.pose import globals_at,accessor,pose
from avengine.assets.roster_motion.contact import serialize_action
from avengine.assets.roster_motion.io import write_glb


def trs(matrix):
    m=np.asarray(matrix,float);scale=np.linalg.norm(m[:3,:3],axis=0);rotation=m[:3,:3]/scale
    if np.linalg.det(rotation)<.99999 or np.max(np.abs(rotation.T@rotation-np.eye(3)))>1e-5:
        raise ValueError('metric_export_requires_proper_rigid_uniform_basis')
    return {'translation':m[:3,3].tolist(),'rotation':Rotation.from_matrix(rotation).as_quat().tolist(),'scale':scale.tolist()}


def metric_document(rig,basis):
    """Bake a coordinate change in POSITION and inverse binds, preserving LBS."""
    d=copy.deepcopy(rig.document);binary=bytearray(rig.binary);old=rig.document
    def add(x,kind):
        arr=np.asarray(x,np.float32);binary.extend(bytes((-len(binary))%4));view=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':arr.nbytes});binary.extend(arr.tobytes());index=len(d['accessors']);a={'bufferView':view,'componentType':5126,'count':len(arr),'type':kind}
        if kind=='VEC3':a.update(min=arr.min(0).tolist(),max=arr.max(0).tolist())
        d['accessors'].append(a);return index
    inverse=np.linalg.inv(basis)
    for mesh in d['meshes']:
        for p in mesh['primitives']:
            if p.get('targets'):raise ValueError('morph_targets_need_explicit_metric_conversion')
            v=accessor(old,rig.binary,p['attributes']['POSITION']);v=v@basis[:3,:3].T+basis[:3,3];p['attributes']['POSITION']=add(v,'VEC3')
            if 'NORMAL' in p['attributes']:
                v=accessor(old,rig.binary,p['attributes']['NORMAL'])@np.linalg.inv(basis[:3,:3]);v/=np.linalg.norm(v,axis=1,keepdims=True);p['attributes']['NORMAL']=add(v,'VEC3')
    joint_ids=sorted(set(j for s in old['skins'] for j in s['joints']));mapping={j:i for i,j in enumerate(joint_ids)};parents={j:i for i,n in enumerate(old['nodes']) for j in n.get('children',[])}
    bind={j:basis@g for j,g in globals_at(old,rig.binary,None).items()};nodes=[];roots=[]
    for j in joint_ids:
        parent=parents.get(j)
        while parent is not None and parent not in mapping:parent=parents.get(parent)
        matrix=np.linalg.inv(bind[parent])@bind[j] if parent is not None else bind[j]
        node={'name':old['nodes'][j].get('name',f'joint_{j}'),**trs(matrix)};nodes.append(node)
        if parent is None:roots.append(mapping[j])
        else:nodes[mapping[parent]].setdefault('children',[]).append(mapping[j]) if mapping[parent]<len(nodes) else None
    # Node ordering in generated rigs is often child-before-parent.
    for n in nodes:n.pop('children',None)
    for j in joint_ids:
        parent=parents.get(j)
        while parent is not None and parent not in mapping:parent=parents.get(parent)
        if parent is not None:nodes[mapping[parent]].setdefault('children',[]).append(mapping[j])
    mesh_nodes=[]
    for n in old['nodes']:
        if 'mesh' in n:
            idx=len(nodes);nodes.append({'name':n.get('name','target_mesh'),'mesh':n['mesh'],'skin':n['skin']});roots.append(idx);mesh_nodes.append(idx)
    origin=len(nodes);nodes.append({'name':'OwnedMetricMotionOrigin','children':roots});d['nodes']=nodes;d['scenes']=[{'nodes':[origin]}];d['scene']=0;d['animations']=[]
    for s,oskin in zip(d['skins'],old['skins']):
        ibm=accessor(old,rig.binary,oskin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        ibm=ibm@inverse;s['inverseBindMatrices']=add(ibm.transpose(0,2,1).reshape(-1,16),'MAT4');s['joints']=[mapping[j] for j in oskin['joints']];s.pop('skeleton',None)
    d['buffers'][0]['byteLength']=len(binary)
    return d,bytes(binary),mapping,[*mesh_nodes,origin]


def export_walk(rig,walk,times,path):
    basis=walk.transform@rig.basis;d,b,mapping,helpers=metric_document(rig,basis);globals_frames=[];rows=[];parts_reference=[]
    for t in times:
        parts,row=walk.frame(float(t));g=walk.last_globals
        globals_frames.append({**{mapping[j]:g[j] for j in mapping},**{j:np.eye(4) for j in helpers}});rows.append(row);parts_reference.append(parts)
    d,b=serialize_action(d,b,'Walking_Repaired',times,globals_frames)
    errors=[]
    for i in (0,len(times)//2,len(times)-1):
        parts,_=pose(d,b,d['animations'][-1],float(times[i]),ground=False,canonical_flip=False)
        error=float(np.max(np.abs(np.concatenate([p['positions'] for p in parts])-np.concatenate([p['positions'] for p in parts_reference[i]]))));errors.append(error)
    if max(errors)>5e-5:raise ValueError('export_changed_target_posed_geometry:'+str(errors))
    write_glb(path,d,b)
    return {'clip':'Walking_Repaired','frames':rows,'native_metric_export_max_vertex_error_m':max(errors),'head_bone':rig.document['nodes'][rig.head]['name'],
            'tail_bone':rig.document['nodes'][rig.tail]['name'],'mouth_bone':rig.document['nodes'][rig.mouth_bone]['name'],'mouth_bone_local_m':rig.mouth_local.tolist(),
            'paw_bones':{s:rig.document['nodes'][c[2]]['name'] for s,c in rig.chains.items()},'bone_names':[rig.document['nodes'][j]['name'] for j in mapping],
            'basis_baked_in_mesh_and_inverse_binds':basis.tolist(),'skin_weights_preserved':True,'triangle_topology_preserved':True,'method':'coordinate change baked into positions, normals and inverse binds; owned repaired bone animation; original target mesh retained'}
