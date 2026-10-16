"""Repair a derived casing's PBR finish, preserving functional faces and actual triangle geometry."""
import argparse,copy,json,hashlib,datetime,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from PIL import Image
import trimesh
from scipy.spatial import cKDTree
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
from avengine.assets.source_materials import prepare_source_glb, read_glb
from measure_static_upright_correction import load_glb
from measure_static_resting_pose import measure as measure_pose
from v7_object_prepare import dump, write_glb
from v7_object_native import functional_exclusion
from v7_object_progress import append as progress

# glTF factors are linear RGB. This changes actual dielectric albedo, not the colour classifier.
FINISHES={'black':[.012,.012,.012,1.], 'white':[.70,.70,.70,1.],
          'gray_silver':[.14,.14,.14,1.], 'red':[.58,.012,.012,1.],
          'green':[.025,.30,.04,1.], 'wood_brown':[1.,1.,1.,1.]}

def canonical_triangles(vertices,faces):
    t=np.round(np.asarray(vertices)[np.asarray(faces)],6)
    order=np.lexsort((t[:,:,2],t[:,:,1],t[:,:,0]),axis=1)
    t=np.take_along_axis(t,order[:,:,None],axis=1).reshape((-1,9))
    return t[np.lexsort(tuple(t[:,i] for i in reversed(range(9))))]

def geometry_match(vertices_a,faces_a,vertices_b,faces_b,tolerance=1e-6):
    """Match actual triangles in both directions without lexicographic float-order assumptions."""
    if len(faces_a)!=len(faces_b):return {'status':'fail','reason':'face_count_changed'}
    a=np.asarray(vertices_a)[np.asarray(faces_a)];b=np.asarray(vertices_b)[np.asarray(faces_b)]
    errors=[]
    for old,new in [(a,b),(b,a)]:
        _,nearest=cKDTree(new.mean(axis=1)).query(old.mean(axis=1),k=min(4,len(new)))
        if nearest.ndim==1:nearest=nearest[:,None]
        candidates=new[nearest]
        distances=np.linalg.norm(old[:,None,:,None,:]-candidates[:,:,None,:,:],axis=-1)
        forward=distances.min(axis=3).max(axis=2)
        backward=distances.min(axis=2).max(axis=2)
        error=np.maximum(forward,backward).min(axis=1)
        errors.append(float(error.max()))
    return {'status':'pass' if max(errors)<=tolerance else 'fail',
            'face_count':len(faces_a),'triangle_bidirectional_max_vertex_error_m':max(errors),'tolerance_m':tolerance}

def body_faces(mesh,meta):
    triangles=np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    center=triangles.mean(axis=1)
    bbox=np.asarray(meta['geometry']['bbox_m']);dims=np.maximum(bbox[1]-bbox[0],1e-9)
    q=(center-bbox[0])/dims
    n=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
    n[np.sum(n*(center-bbox.mean(axis=0)),axis=1)<0]*=-1
    axis=1 if meta.get('functional_front_axis')=='positive-y' else 0
    return ~functional_exclusion(meta,q,n[:,axis]>.35)

def walnut_albedo(image):
    """Retain UV luminance/grain variation while removing baked lighting and wrong back-case hue."""
    a=np.asarray(image.convert('RGBA'),dtype=float)/255
    srgb=a[:,:,:3];linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    y=linear@np.array([.2126,.7152,.0722])
    variation=.65+.50*np.sqrt(np.clip(y,0,1))
    target=variation[:,:,None]*np.array([.30,.105,.03])
    rgb=np.where(target<=.0031308,12.92*target,1.055*np.maximum(target,0)**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(np.dstack([rgb,a[:,:,3]]),0,1)*255),'RGBA')

def repair_mesh(source,out,meta,color,linear_factor=None,dark_functional=False):
    if color not in FINISHES:raise ValueError('unsupported dielectric casing finish: '+color)
    # glTF names need not be unique. Trimesh graph keys are names, so duplicate
    # wrapper names can discard legitimate nested transforms. Indices are unchanged.
    document,binary=read_glb(source)
    for index,node in enumerate(document['nodes']):node['name']='v7_finish_input_node_'+str(index)
    renamed=out.with_name('input_unique_node_names.glb')
    write_glb(renamed,document,binary)
    src=trimesh.load(str(renamed),force='scene',process=False)
    dst=trimesh.Scene();preserved=changed=0
    factor=FINISHES[color] if linear_factor is None else list(linear_factor)
    if len(factor)!=4 or min(factor)<0 or max(factor)>1:raise ValueError('linear RGBA outside [0,1]')
    for index,node in enumerate(src.graph.nodes_geometry):
        matrix,name=src.graph[node];mesh=src.geometry[name].copy();mesh.apply_transform(matrix)
        body=body_faces(mesh,meta)
        for label,selection in [('functional',~body),('casing',body)]:
            ids=np.flatnonzero(selection)
            if not len(ids):continue
            part=mesh.submesh([ids],append=True,repair=False)
            if label=='casing':
                texture=None;uv=None
                if color=='wood_brown':
                    original=getattr(mesh.visual,'material',None)
                    image=getattr(original,'baseColorTexture',None)
                    if image is None:image=getattr(original,'image',None)
                    uv=getattr(part.visual,'uv',None)
                    if image is None or uv is None:raise ValueError('wood finish requires existing texture and UV grain information')
                    texture=walnut_albedo(image)
                part.visual=TextureVisuals(uv=uv,material=PBRMaterial(name='v7_real_dielectric_'+color,
                    baseColorTexture=texture,baseColorFactor=factor,metallicFactor=0.,roughnessFactor=.98,
                    emissiveFactor=[0.,0.,0.],doubleSided=True))
                changed+=len(ids)
            else:
                if dark_functional:
                    if meta['object_type'] not in ['tablet','television','printer']:raise ValueError('inactive LCD state only applies to display/control devices')
                    part.visual=TextureVisuals(material=PBRMaterial(name='v7_inactive_dark_functional',
                        baseColorFactor=[.012,.012,.012,1.],metallicFactor=0.,roughnessFactor=.98,
                        emissiveFactor=[0.,0.,0.],doubleSided=True))
                    preserved+=len(ids)
                    dst.add_geometry(part,geom_name=f'{index}_{label}',node_name=f'{index}_{label}')
                    continue
                original_material=getattr(mesh.visual,'material',None)
                if original_material is not None:
                    part.visual=TextureVisuals(uv=getattr(part.visual,'uv',None),material=copy.deepcopy(original_material))
                preserved+=len(ids)
            dst.add_geometry(part,geom_name=f'{index}_{label}',node_name=f'{index}_{label}')
    if not changed:raise ValueError('no measured casing faces to repair')
    out.write_bytes(dst.export(file_type='glb'))
    before_v,before_f=load_glb(source);after_v,after_f=load_glb(out)
    comparison=geometry_match(before_v,before_f,after_v,after_f)
    if comparison['status']!='pass':raise ValueError('finish export changed actual triangle geometry: '+str(comparison))
    return {'casing_faces':changed,'preserved_functional_faces':preserved,
            'actual_geometry_unchanged_atol_m':1e-6,'geometry_comparison':comparison,'linear_rgba':factor,'source_node_names_copy':str(renamed),'functional_state':'inactive_dark_screen' if dark_functional else 'source_functional_material_preserved',
            'rule':'same functional-exclusion policy as native body ROI; local facet geometry selects casing'}

def repair_one(T,request,activate=False):
    aid=request['asset_id'];D=T/'assets'/aid;meta=json.loads((D/'asset.json').read_text())
    source=Path(meta['derived_path']);assert source.resolve().is_relative_to(T.resolve())
    if meta.get('silent_decor'):raise ValueError('no finish variants for silent air conditioners')
    if meta.get('preflight_issues'):raise ValueError('material repair cannot bypass geometry failure')
    version=request.get('version','finish_v1');R=D/version;R.mkdir(exist_ok=True)
    if (R/'receipt.json').exists():raise ValueError('repair version already exists; inspect it before another version')
    color=request['color'];painted=R/'casing.glb';operations=repair_mesh(source,painted,meta,color,request.get('linear_rgba'),request.get('dark_functional',False))
    result,production=prepare_source_glb(painted,asset_id=aid+'_'+version,
        entity_class='rigid_static_object',output_directory=R)
    v,f=load_glb(result);oldv,oldf=load_glb(source)
    comparison=geometry_match(oldv,oldf,v,f)
    assert comparison['status']=='pass','production material repair changed geometry: '+str(comparison)
    operations['after_production_geometry_comparison']=comparison
    m=copy.deepcopy(meta);m['derived_path']=str(result)
    m['geometry']['bbox_m']=np.stack([v.min(axis=0),v.max(axis=0)]).tolist()
    m['geometry']['dimensions_m']=np.ptp(v,axis=0).tolist();m['geometry']['faces']=len(f);m['geometry']['glb_bytes']=result.stat().st_size
    m['geometry']['resting_pose']=measure_pose(result,attachment_surface='floor' if m['attachment_surface']=='table' else m['attachment_surface'])
    if m.get('emitter'):m['emitter']['nearest_vertex_distance_m']=float(np.linalg.norm(v-np.asarray(m['emitter']['local_m']),axis=1).min())
    m['material_receipt']=production
    proof={'asset_id':aid,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_derived_glb':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'derived_glb':str(result),'derived_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
           'desired_real_casing_color':color,'operation':'dielectric casing PBR albedo and matte roughness; functional material retained',
           'operations':operations,'production_material_receipt':production,
           'pre_repair_metadata':str(R/'before_asset.json'),'activated':activate,
           'native_status':'not_run; new MP3D and HM3D evidence required'}
    dump(R/'before_asset.json',meta);dump(R/'receipt.json',proof)
    m['finish_repairs']=list(m.get('finish_repairs',[]))+[{'receipt':str(R/'receipt.json'),'color':color,'version':version,'dark_inactive_functional':request.get('dark_functional',False)}]
    dump(R/'candidate_asset.json',m)
    if activate:dump(D/'asset.json',m)
    progress(T,'机壳材质派生修复 '+aid+'：'+('已启用，等待两房重验' if activate else '候选已生成，未启用'),str(R/'receipt.json'),'重新做全套原生验收；不沿用旧 GLB 的通过结论')
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--activate',action='store_true')
    p.add_argument('--output-name',default='prepared_finish_repairs')
    a=p.parse_args();rows=[];failed=[]
    for request in json.loads(a.recipe.read_text()):
        try:rows.append(repair_one(a.task_root,request,a.activate))
        except Exception as e:
            failed.append({'asset_id':request['asset_id'],'reason':type(e).__name__+':'+str(e)})
            progress(a.task_root,'机壳材质修复失败 '+request['asset_id'],str(a.recipe),'保留原派生文件与失败原因；不能计为通过')
    if a.activate and (a.task_root/'prepared_existing.json').exists():
        replacements={m['asset_id']:m for m in rows if m.get('origin')=='existing'}
        existing=json.loads((a.task_root/'prepared_existing.json').read_text())
        dump(a.task_root/'prepared_existing.json',[replacements.get(m['asset_id'],m) for m in existing])
    dump(a.task_root/(a.output_name+'.json'),rows)
    dump(a.task_root/(a.output_name+'_failures.json'),failed)
    print(json.dumps({'prepared':len(rows),'failures':failed},ensure_ascii=False))
    return 1 if failed else 0
if __name__=='__main__':raise SystemExit(main())
