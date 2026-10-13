"""Owner-authorized phase-two derivatives; original registrations stay read-only."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
from avengine.assets.roster_motion.io import read_glb,write_glb
from avengine.capture.v7_visible_mesh import indexed_static_mesh

FIXES={
 'v7_lx_hand_drum_player_original_v1':'rigid_ground',
 'v7_lx_megaphone_speaker_original_v1':'rigid_ground',
 'v7_lx_pianist_original_v1':'production_matte',
 'v7_bird_budgerigar_blue_20261012':'palette_diagnostic_no_colour',
 'v7_object_smoke_detector_ceiling_octagonal_disc_gray_20261012':'ceiling_normal'
}

def rigid_ground_rotation(vertices):
    from scipy.optimize import minimize_scalar
    from scipy.spatial.transform import Rotation
    v=np.asarray(vertices,float); mid=(v[:,0].min()+v[:,0].max())/2
    masks=[v[:,0]<mid,v[:,0]>=mid]
    def gap(angle):
        r=Rotation.from_rotvec([0,0,angle]).as_matrix(); w=v@r.T
        return float(w[masks[1],1].min()-w[masks[0],1].min())
    solved=minimize_scalar(lambda a:gap(a)**2,bounds=(-.18,.18),method='bounded',options={'xatol':1e-12})
    r=Rotation.from_rotvec([0,0,float(solved.x)]).as_matrix()
    if abs(gap(solved.x))>.001:raise ValueError('bounded rigid grounding cannot level both original sole regions')
    return r,{'angle_z_rad':float(solved.x),'original_gap_m':gap(0),'derived_gap_m':abs(gap(solved.x))}

def derive_asset(asset,directory):
    a=deepcopy(asset); path=Path(a['habitat_glb']); doc,blob=read_glb(path)
    mesh=indexed_static_mesh(path);v=np.asarray(mesh.vertices)
    fix=FIXES[a['asset_id']];matrix=np.eye(4);details={}
    if fix=='rigid_ground':
        matrix[:3,:3],details=rigid_ground_rotation(v)
        matrix[1,3]=-float((v@matrix[:3,:3].T)[:,1].min())
    elif fix=='ceiling_normal':
        from avengine.assets.roster_motion.contact import rotation_between
        _,axes=np.linalg.eigh(np.cov(v.T)); normal=axes[:,0]
        if normal[2]>0:normal=-normal
        matrix[:3,:3]=rotation_between(normal,np.array([0.,1.,0.]))
        w=v@matrix[:3,:3].T;matrix[1,3]=-float(w[:,1].min())
        details={'original_installation_normal_m':normal.tolist(),'derived_installation_normal_m':[0,1,0],'method':'rigid align least-variance plate normal; native ceiling/support gates still required'}
    elif fix=='production_matte':
        before=deepcopy(doc['materials'])
        for material in doc['materials']:
            pbr=material.setdefault('pbrMetallicRoughness',{})
            pbr['metallicFactor']=0.;pbr['roughnessFactor']=1.;pbr.pop('metallicRoughnessTexture',None)
            material.pop('extensions',None)
        details={'material_before':before,'material_after':deepcopy(doc['materials']),'albedo_images_unchanged':True,'geometry_unchanged':True,'max_highlight_ratio':.02}
    elif fix=='palette_diagnostic_no_colour':
        a['label_color']=None
        a['color_question_policy']='phase-two owner-authorized no-colour fallback; registered colour retained; material fidelity gate unchanged'
        details={'registered_colour':a['registered_label_color'],'pilot_colour':None,'texture_bytes_unchanged':True,'material_palette_reference_revision':'per-primitive-stratified-srgb-v1'}
    if not np.array_equal(matrix,np.eye(4)):
        original_roots=list(doc['scenes'][doc.get('scene',0)]['nodes'])
        node=len(doc['nodes']);doc['nodes'].append({'name':'OwnedPhase2RigidCorrection','matrix':matrix.T.ravel().tolist(),'children':original_roots})
        doc['scenes'][doc.get('scene',0)]['nodes']=[node]
        for key in ['emitter_local_m','mouth_local_glb_m']:
            if a.get(key) is not None:a[key]=(matrix@np.r_[a[key],1.])[:3].tolist()
        if a.get('landmarks_local_m'):
            for k,val in a['landmarks_local_m'].items():
                if isinstance(val,list) and len(val)==3:a['landmarks_local_m'][k]=(matrix@np.r_[val,1.])[:3].tolist()
    directory=Path(directory)/a['asset_id'];directory.mkdir(parents=True,exist_ok=True)
    out=directory/'runtime.glb'
    if out.exists():raise FileExistsError('do not overwrite an existing derivative')
    write_glb(out,doc,blob)
    derived=indexed_static_mesh(out)
    a['original_registered_habitat_glb']=str(path);a['original_registered_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    a['habitat_glb']=str(out);a['actual_artifact_sha256']={str(out):hashlib.sha256(out.read_bytes()).hexdigest()}
    a['owned_derivative']={'fix':fix,'source':str(path),'source_sha256':a['original_registered_sha256'],'derived_sha256':a['actual_artifact_sha256'][str(out)],'rigid_transform_m':matrix.tolist(),'details':details,'native_requalification_required':True}
    if a['kind']=='object':
        a['original_registered_geometry']=deepcopy(a['geometry'])
        a['geometry']['dimensions_m']=np.ptp(derived.vertices,axis=0).tolist()
    return a

def stratified_material_palette(path):
    """Keep every original material in the measurement reference.

    The legacy global stride can omit the final factor-only perch entirely.
    glTF baseColorFactor is linear, while bitmap texels and RGB are sRGB.
    Neither rendered pixels nor any appearance threshold is changed.
    """
    from avengine.assets.source_materials import source_palette
    doc,blob=read_glb(Path(path));rows=[]
    primitives=[p for m in doc.get('meshes',[]) for p in m['primitives'] if 'material' in p]
    allowance=max(1,512//max(1,len(primitives)))
    for primitive in primitives:
        sub=deepcopy(doc);sub['meshes']=[{'primitives':[primitive]}]
        mat=doc['materials'][primitive['material']];pbr=mat.get('pbrMetallicRoughness',{})
        pal=source_palette(sub,blob,Path(path))
        if not pbr.get('baseColorTexture'):
            pal=np.where(pal<=.0031308,12.92*pal,1.055*pal**(1/2.4)-.055)
        indices=np.linspace(0,len(pal)-1,min(len(pal),allowance),dtype=int)
        rows.append(pal[indices])
    if not rows:raise ValueError('original material palette absent')
    return np.concatenate(rows)
