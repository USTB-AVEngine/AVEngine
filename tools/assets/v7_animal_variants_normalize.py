"""Normalize owned static Pixal meshes, with measured support and pose gates."""
import argparse,copy,hashlib,io,json,math,os,struct,subprocess,sys,time,traceback,itertools
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from avengine.assets.source_materials import read_glb,image_bytes,prepare_source_glb,inspect_source_glb

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_glb(p,d,b):
    d['buffers']=[{'byteLength':len(b)}];j=json.dumps(d,separators=(',',':')).encode();j+=b' '*((-len(j))%4);b+=bytes((-len(b))%4);p.write_bytes(struct.pack('<III',0x46546c67,2,28+len(j)+len(b))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(b),0x004e4942)+b)
def png_compat(source,out):
    d,b=read_glb(source);b=bytearray(b)
    for i,im in enumerate(d.get('images',[])):
        image=Image.open(io.BytesIO(image_bytes(d,bytes(b),i,source))).convert('RGB');image.thumbnail((2048,2048));buf=io.BytesIO();image.save(buf,format='PNG');raw=buf.getvalue();b.extend(bytes((-len(b))%4));vi=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':len(b),'byteLength':len(raw)});b.extend(raw);im.clear();im.update(bufferView=vi,mimeType='image/png')
    for tx in d.get('textures',[]):
        ext=tx.get('extensions',{}).pop('EXT_texture_webp',None)
        if ext:tx['source']=ext['source']
        if not tx.get('extensions'):tx.pop('extensions',None)
    for k in ['extensionsUsed','extensionsRequired']:
        if k in d:d[k]=[x for x in d[k] if x!='EXT_texture_webp']
        if not d.get(k):d.pop(k,None)
    write_glb(out,d,bytes(b))
def four_paw_layout(points,height):
    if len(points)!=4:return False
    ordered=points[np.argsort(points[:,2])];front=ordered[:2];rear=ordered[2:]
    return bool(rear[:,2].mean()-front[:,2].mean()>height*.35 and np.ptp(front[:,2])<height*.25 and np.ptp(rear[:,2])<height*.25 and np.ptp(front[:,0])>height*.04 and np.ptp(rear[:,0])>height*.04 and abs(front[:,0].mean()-rear[:,0].mean())<height*.25)

def paw_supports(v):
    height=np.ptp(v[:,1]);bottom=v[:,1].min();last=[]
    for fraction in [.03,.05,.065,.08,.10,.12,.16,.20]:
        pts=v[v[:,1]<=bottom+fraction*height]
        step=max(height*.012,.003);lo=pts[:,[0,2]].min(0);coords=np.floor((pts[:,[0,2]]-lo)/step).astype(int);grid=np.zeros(tuple(coords.max(0)+3),bool);grid[coords[:,0]+1,coords[:,1]+1]=True;labs,n=label(grid,np.ones((3,3)));groups=[]
        for k in range(1,n+1):
            subset=pts[labs[coords[:,0]+1,coords[:,1]+1]==k]
            if len(subset)>=max(25,len(pts)*.025):
                y=subset[:,1].min();support=subset[subset[:,1]<y+height*.008].mean(0);support[1]=y;groups.append({'vertices':len(subset),'point':support,'height_band_fraction':fraction})
        groups.sort(key=lambda x:x['vertices'],reverse=True);last=groups
        if len(groups)==5:
            all_points=np.array([g['point'] for g in groups]);qualified=[]
            for subset in itertools.combinations(range(5),4):
                points=all_points[list(subset)]
                if not four_paw_layout(points,height):continue
                omitted=next(i for i in range(5) if i not in subset);extra=all_points[omitted];ordered=points[np.argsort(points[:,2])];front=ordered[:2];rear=ordered[2:];center=(front[:,0].mean()+rear[:,0].mean())/2;half_width=min(np.ptp(front[:,0]),np.ptp(rear[:,0]))*.45
                if abs(extra[0]-center)<half_width and extra[2]>rear[:,2].mean()+height*.05:qualified.append(subset)
            if len(qualified)==1:
                groups=[{**groups[i],'additional_rear_centerline_support_ignored':'one extra support lies behind four paired lateral paw regions; retained geometry, consistent with a low tail tip'} for i in qualified[0]]
        if len(groups)==4:
            points=np.array([g['point'] for g in groups]);distance=np.linalg.norm(points[:,None,[0,2]]-points[None,:,[0,2]],axis=-1)+np.eye(4)*1e6
            if distance.min()>height*.035 and four_paw_layout(points,height):return points,[{**g,'point':g['point'].tolist()} for g in groups]
    return np.array([g['point'] for g in last]),[{**g,'point':g['point'].tolist()} for g in last]
def level_rotation(points):
    center=points.mean(0);_,_,vh=np.linalg.svd(points-center);normal=vh[-1]
    if normal[1]<0:normal=-normal
    axis=np.cross(normal,[0,1,0]);angle=math.acos(np.clip(normal[1],-1,1))
    return Rotation.from_rotvec(axis/max(np.linalg.norm(axis),1e-12)*angle).as_matrix(),math.degrees(angle)
def one(T,item,raw):
    D=T/'assets'/item['asset_id'];N=D/'normalization'/raw.parent.name;N.mkdir(parents=True,exist_ok=True);compat=N/'input_png.glb';lod=N/'lod.glb';
    if not compat.exists():png_compat(raw,compat)
    cmd=['nice','-n','10','/data/jzy/.local/bin/blender','--background','--factory-startup','--threads','2','--python',str(Path(__file__).with_name('v7_animal_variants_lod_blender.py')),'--','--input',str(compat),'--output',str(lod),'--receipt',str(N/'lod_receipt.json')]
    if not (lod.exists() and (N/'lod_receipt.json').exists()):
        with (N/'blender.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    scene=trimesh.load(lod,force='scene',process=False);mesh=scene.to_mesh() if hasattr(scene,'to_mesh') else scene.dump(concatenate=True)
    if len(mesh.faces)>100000:raise RuntimeError('triangle_budget')
    v=mesh.vertices.copy();height=np.ptp(v[:,1]);xz=v[:,[0,2]];_,_,vh=np.linalg.svd((xz-xz.mean(0))[::max(1,len(xz)//10000)],full_matrices=False);axis=vh[0];upper=v[:,1]>=np.quantile(v[:,1],.78);head_sign=np.sign(np.median((xz[upper]-xz.mean(0))@axis)) or 1;front=np.array([axis[0],0,axis[1]])*head_sign;theta=math.pi-math.atan2(front[0],front[2]);rot=Rotation.from_rotvec([0,theta,0]).as_matrix();v=(v-v.mean(0))@rot.T
    marks={};details={};is_bird=item['category']=='bird'
    if not is_bird:
        points,groups=paw_supports(v);details['raw_support_clusters']=groups
        if len(points)!=4 or not four_paw_layout(points,height):raise RuntimeError('four_paw_layout_or_clusters_invalid:'+str(len(points)))
        lev,tilt=level_rotation(points);v=v@lev.T;points=points@lev.T;torso=np.abs(v[:,2]-np.median(v[:,2]))<np.ptp(v[:,2])*.18;floor_shift=-v[:,1].min();v[:,1]+=floor_shift;points[:,1]+=floor_shift;shoulder=float(np.quantile(v[torso,1],.95)-points[:,1].min());scale=item['physical_size']['target_m']/shoulder;v*=scale;points*=scale;shift=np.array([0,-v[:,1].min(),0]);v+=shift;points+=shift
        headset=v[(v[:,2]<np.quantile(v[:,2],.25))&(v[:,1]>.50*item['physical_size']['target_m'])&(v[:,1]<1.15*item['physical_size']['target_m'])]
        if len(headset)<20:raise RuntimeError('head_landmark_missing')
        mouth=headset[headset[:,2]<=np.quantile(headset[:,2],.02)].mean(0);marks.update({f'foot_{i}':p.tolist() for i,p in enumerate(points)});marks['mouth']=mouth.tolist();marks['head_center']=headset.mean(0).tolist();marks['head_bbox_min']=headset.min(0).tolist();marks['head_bbox_max']=headset.max(0).tolist();residual=float(np.ptp(points[:,1]));details.update(scale=scale,shoulder_height_m=item['physical_size']['target_m'],support_plane_tilt_deg=tilt,four_paw_height_range_m=residual,minimum_mesh_y_m=float(v[:,1].min()),horizontal_length_to_height_ratio=float(np.ptp(v[:,2])/np.ptp(v[:,1])))
        gate=tilt<12 and residual<=.020 and v[:,1].min()>=-.008 and np.ptp(v[:,2])/np.ptp(v[:,1])>.80
        if not gate:raise RuntimeError('static_pose_support_gate:'+json.dumps(details))
        mesh.vertices=v;normalized=trimesh.Scene(mesh);body_floor=0.;perch=None
    else:
        from v7_animal_variants_bird_geometry import normalize as normalize_bird
        normalized,mesh,stand,v,marks,details,perch=normalize_bird(mesh,item)
        stand.export(D/'perch_only.glb');body_floor=perch['perch_top_m'];details['normalizer_version']='textured_wood_and_two_feet_v1'
    intermediate=N/'normalized_pbr.glb';normalized.export(intermediate);prepared,receipt=prepare_source_glb(intermediate,asset_id=item['asset_id'],entity_class='articulated_animal',output_directory=N/'materials');runtime=D/'runtime.glb';runtime.write_bytes(Path(prepared).read_bytes());dump(D/'material_receipt.json',receipt);check=inspect_source_glb(runtime);final=trimesh.load(runtime,force='scene',process=False);bounds=final.bounds;count=sum(len(x.faces) for x in final.geometry.values());body=D/'body_only.glb';trimesh.Scene(mesh).export(body)
    if count>100000 or runtime.stat().st_size>40*1024**2:raise RuntimeError('final_asset_budget')
    mouth=np.array(marks['mouth']);head=np.array(marks['head_center']);head_radius=max(.025,float(np.linalg.norm(np.array(marks['head_bbox_max'])-np.array(marks['head_bbox_min'])))*.55) if 'head_bbox_max' in marks else max(.025,float(np.ptp(v[:,1]))*.28);head_distance=float(np.linalg.norm(mouth-head))
    meta={**item,'status':'pending_native_acceptance','units':'metres','coordinates':{'up':'+Y','front':'-Z','origin':'floor support center'},'runtime_glb':str(runtime),'body_only_glb':str(body),'dimensions_m':np.ptp(bounds,axis=0).tolist(),'bounds_min_m':bounds[0].tolist(),'bounds_max_m':bounds[1].tolist(),'emit_point_local_m':marks['mouth'],'head_center_local_m':marks['head_center'],'emit_head_distance_m':head_distance,'emit_head_radius_m':head_radius,'emit_head_status':'pass' if head_distance<=head_radius else 'fail','landmarks_local_m':marks,'body_floor_y_m':body_floor,'triangles':count,'glb_bytes':runtime.stat().st_size,'runtime_sha256':sha(runtime),'raw_sha256':sha(raw),'normalization':details,'perch':perch,'geometry_skin_animation_unchanged_by_materials':receipt['geometry_skin_animation_unchanged'],'material_inspection':check,'source_generator':'TencentARC/Pixal3D','license':{'reference_weights':'Apache-2.0','isnet_weights':'Apache-2.0','pixal_code_and_weights':'MIT','dinov3_weights':'DINOv3 License','naf_code_and_weights':'Apache-2.0','downstream_ai_training_admission':'pending_native_checks'}}
    dump(D/'asset.json',meta);dump(N/'normalization_receipt.json',{'status':'pass','input_sha256':sha(raw),'compat_sha256':sha(compat),'lod_sha256':sha(lod),'normalized_sha256':sha(intermediate),'runtime_sha256':sha(runtime),'details':details});return meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--only');p.add_argument('--seed',type=int);p.add_argument('--category',choices=['dog','cat','bird']);p.add_argument('--only-missing',action='store_true');p.add_argument('--result-name',default='normalization_results.json');a=p.parse_args();plan=json.loads((a.task_root/'plan.json').read_text());records=[];start=time.time()
    for item in plan['animals']:
        if item['mode']!='generated_static' or (a.only and a.only!=item['asset_id']) or (a.category and a.category!=item['category']):continue
        existing=a.task_root/'assets'/item['asset_id']/'asset.json'
        if a.only_missing and existing.exists() and json.loads(existing.read_text()).get('status') in ['pending_native_acceptance','admitted']:continue
        D=a.task_root/'assets'/item['asset_id'];seg=D/'reference/segmentation_receipt.json'
        while not seg.exists():time.sleep(3)
        if json.loads(seg.read_text())['status']!='pass':records.append({'asset_id':item['asset_id'],'status':'rejected_matting'});continue
        raw=D/'generation'/f"seed_{a.seed if a.seed is not None else item['mesh_seeds'][0]}"/'raw.glb';gen=raw.parent/'generation_receipt.json'
        while not gen.exists():
            state=json.loads((a.task_root/'evidence/mesh_status.json').read_text())
            entry=next((x for x in state['jobs'] if x['asset_id']==item['asset_id']),None)
            if entry and entry.get('selected_glb'):raw=Path(entry['selected_glb']);gen=raw.parent/'generation_receipt.json';break
            if entry or state.get('status') not in ['running',None]:break
            if time.time()-start>4.5*3600:raise RuntimeError('normalization wait deadline')
            time.sleep(10)
        rec={'asset_id':item['asset_id'],'raw_glb':str(raw)}
        try:
            if not gen.exists():raise RuntimeError('generation receipt missing')
            meta=one(a.task_root,item,raw);rec.update(status='pass_pending_native',triangles=meta['triangles'],bytes=meta['glb_bytes'])
        except BaseException as e:
            rec.update(status='rejected_normalization',error=repr(e),traceback=traceback.format_exc());dump(D/'normalization_failure.json',rec)
            if (D/'asset.json').exists():
                stale=json.loads((D/'asset.json').read_text());stale.update(status='rejected_normalization',latest_normalization_failure=rec);dump(D/'asset.json',stale)
        records.append(rec);dump(a.task_root/'evidence'/a.result_name,records);print('NORMALIZE',item['asset_id'],rec['status'],rec.get('error',''),flush=True)
    dump(a.task_root/'evidence'/a.result_name,records)
if __name__=='__main__':main()
