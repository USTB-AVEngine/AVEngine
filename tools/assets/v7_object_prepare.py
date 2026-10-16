"""Derive production PBR static objects and verify support pose, dimensions and emitter geometry."""
import argparse,copy,json,struct,sys,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from avengine.assets.source_materials import read_glb,prepare_source_glb
from measure_static_upright_correction import load_glb
from measure_static_resting_pose import measure

def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def write_glb(p,d,b):
    d=copy.deepcopy(d);d['buffers'][0]['byteLength']=len(b);j=json.dumps(d,separators=(',',':')).encode();j+=b' '*(-len(j)%4);b+=b'\0'*(-len(b)%4)
    p.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(j)+len(b))+struct.pack('<I4s',len(j),b'JSON')+j+struct.pack('<I4s',len(b),b'BIN\0')+b)
def wrapped(source,target,M):
    d,b=read_glb(source);node=len(d['nodes']);d['nodes'].append({'name':'v7_derived_physical_correction','matrix':M.T.ravel().tolist(),'children':d['scenes'][d.get('scene',0)]['nodes']});d['scenes'][d.get('scene',0)]['nodes']=[node];write_glb(target,d,b)
def align(a,b):
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);v=np.cross(a,b);c=float(a@b);s=np.linalg.norm(v)
    if s<1e-8:return np.eye(3)
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+K+K@K*((1-c)/(s*s))
def surface_for(a):
    k=a['identity']['object_type'];form=a['realized_attributes'].get('form_factor','');decl=a['geometry'].get('resting_pose',{}).get('attachment_surface','floor')
    if decl in ['wall','ceiling']:return decl
    if k in ['alarm_clock','blender','microwave_oven','printer','smart_speaker','desk_telephone','cellphone','bookshelf_speaker','soundbar','television','radio','tablet','kitchen_timer','mantel_clock']:return 'table'
    return 'floor'
# Broad engineering ranges, metres, depth (+X), height (+Y), width (+Z).
# Source profile target height is retained; corrections only address clear reconstruction distortion.
RANGES={
'air_conditioner':([.15,.20,.30],[.75,1.1,1.25]),'alarm_clock':([.03,.055,.065],[.20,.28,.30]),
'blender':([.10,.23,.10],[.35,.55,.33]),'microwave_oven':([.25,.20,.40],[.60,.48,.80]),
'printer':([.20,.12,.25],[.65,.65,.65]),'bookshelf_speaker':([.12,.16,.12],[.42,.55,.35]),
'floorstanding_speaker':([.15,.65,.16],[.50,1.35,.50]),'smart_speaker':([.055,.06,.055],[.25,.32,.25]),
'soundbar':([.035,.035,.30],[.20,.16,1.30]),'television':([.04,.30,.45],[.50,1.2,1.9]),
'desk_telephone':([.10,.07,.12],[.33,.28,.32]),'landline_phone':([.025,.15,.08],[.15,.40,.23]),
'cellphone':([.004,.12,.055],[.018,.19,.10]),'smoke_detector':([.025,.025,.065],[.20,.20,.20]),
'doorbell_chime_unit':([.018,.065,.035],[.20,.70,.40]),'bathtub':([.60,.35,1.30],[1.1,.80,2.1]),
'floor_drain':([.03,.015,.03],[.35,.40,.30]),'sink_with_tap':([.35,.65,.35],[.80,1.10,.90]),
'fireplace':([.28,.50,.35],[1.0,1.6,1.5]),'toilet':([.50,.65,.30],[.95,1.8,.65]),
'wall_clock':([.018,.16,.16],[.09,.48,.48]),'mantel_clock':([.05,.12,.12],[.18,.38,.40]),
'carbon_monoxide_alarm':([.025,.06,.07],[.08,.18,.22]),'security_keypad':([.02,.10,.09],[.07,.25,.25]),
'kitchen_timer':([.02,.05,.05],[.10,.15,.15]),'radio':([.05,.10,.14],[.20,.30,.45]),'tablet':([.006,.16,.11],[.18,.30,.27])}

def prepare_one(a,T,source_root,source=None):
    aid=a['asset_id'];D=T/'assets'/aid;D.mkdir(parents=True,exist_ok=True);k=a['identity']['object_type'];surface=surface_for(a)
    source=source or source_root/a['path']/a['geometry'].get('finalized_glb','finalized.glb')
    v,f=load_glb(source);original=np.ptp(v,axis=0);M=np.eye(4);corrections=[]
    p0=measure(source,attachment_surface='floor' if surface=='table' else surface)
    # A support-face weighted normal is used only for an existing measured lean.
    tilt=p0.get('base_normal_tilt_deg') if surface in ['floor','table'] else p0.get('mounting_plane_normal_tilt_deg')
    if tilt is not None and tilt>3 and surface in ['floor','table']:
        tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(n,axis=1);n/=np.maximum(area[:,None],1e-15);keep=(tri[:,:,1].min(axis=1)<v[:,1].min()+.05*original[1])&(np.abs(n[:,1])>.5)
        n[n[:,1]<0]*=-1;up=np.sum(n[keep]*area[keep,None],axis=0)
        if np.linalg.norm(up)>0:M[:3,:3]=align(up,np.array([0,1,0]));corrections.append({'operation':'support_normal_level','before_tilt_deg':tilt})
    vv=v@M[:3,:3].T;dims=np.ptp(vv,axis=0);lo,hi=np.asarray(RANGES[k]);lo=lo.copy();hi=hi.copy()
    if k=='cellphone' and a['realized_attributes'].get('resting_form')=='lying_screen_up':lo=lo[[1,0,2]];hi=hi[[1,0,2]]
    if k=='floor_drain' and a['realized_attributes'].get('form_factor')=='floor_drain':lo[[0,2]]=.07
    if k=='air_conditioner' and a['realized_attributes'].get('form_factor')=='wall_split':lo[2]=.60;hi[0]=.40;hi[1]=.40
    if k=='smoke_detector' and surface=='wall':hi[0]=.075
    desired=np.clip(dims,lo*1.02,hi*.98);scale=desired/dims
    if np.max(np.abs(scale-1))>.02:
        M[:3,:3]=np.diag(scale)@M[:3,:3];corrections.append({'operation':'axis_scale_to_broad_real_dimensions','scale_xyz':scale.tolist(),'range_min_m':lo.tolist(),'range_max_m':hi.tolist()})
    vv=v@M[:3,:3].T;M[1,3]=-vv[:,1].min();M[0,3]=-np.mean([vv[:,0].min(),vv[:,0].max()]);M[2,3]=-np.mean([vv[:,2].min(),vv[:,2].max()]);vv+=M[:3,3]
    corrected=D/'physical.glb';wrapped(source,corrected,M)
    derived,receipt=prepare_source_glb(corrected,asset_id=aid+'_matte',entity_class='rigid_static_object',output_directory=D)
    pose=measure(derived,attachment_surface='floor' if surface=='table' else surface);outv,outf=load_glb(derived);bbox=np.stack([outv.min(axis=0),outv.max(axis=0)]);dims=bbox[1]-bbox[0]
    if k=='doorbell_chime_unit':
        target=bbox[0]+np.asarray([.92,.5,.5])*dims;emit=outv[np.argmin(np.linalg.norm(outv-target,axis=1))];rationale='机身中心高度与左右中心，靠前的最近网格表面'
    elif a.get('emitter') and a['emitter'].get('offset_m') is not None:
        emit=M[:3,:3]@np.asarray(a['emitter']['offset_m'])+M[:3,3];rationale=a['emitter'].get('physical_rationale','既有登记的合理发声部位')
    else:
        fraction={'television':[.98,.12,.5],'bookshelf_speaker':[.98,.52,.5],'floorstanding_speaker':[.98,.6,.5],'smart_speaker':[.98,.5,.5],'soundbar':[.98,.5,.5],'desk_telephone':[.95,.65,.5],'landline_phone':[.95,.65,.5]}.get(k,[.92,.5,.5]);target=bbox[0]+np.asarray(fraction)*dims;emit=outv[np.argmin(np.linalg.norm(outv-target,axis=1))];rationale='按物件种类选前部部位，落到最近网格表面；音箱网罩、电视下沿、电话机身'
    nearest=float(np.min(np.linalg.norm(outv-emit,axis=1)));face_count=len(outf);size=derived.stat().st_size
    silent=k=='air_conditioner'
    issues=[]
    if pose.get('verdict') not in ['level','acceptable']:issues.append('support_pose:'+pose.get('verdict','missing'))
    if not np.all((dims>=lo)&(dims<=hi)):issues.append('dimensions_outside_range')
    if face_count>100000:issues.append('faces_exceed_100k')
    if size>40_000_000:issues.append('glb_exceeds_40MB')
    if not silent and nearest>.06:issues.append('emitter_far_from_surface')
    m={'asset_id':aid,'object_type':k,'form_factor':a['realized_attributes'].get('form_factor','unknown'),'intended_color':a['realized_attributes'].get('body_color',a['realized_attributes'].get('finish','unspecified')),'origin':'existing' if source_root else 'new','source_path':str(source),'derived_path':str(derived),'attachment_surface':surface,'geometry':{'bbox_m':bbox.tolist(),'dimensions_m':dims.tolist(),'faces':face_count,'glb_bytes':size,'original_dimensions_m':original.tolist(),'corrections':corrections,'transform_from_source':M.tolist(),'resting_pose':pose},'size_basis':{'source':'AVEngine asset.geometry.profile_physical_authority and target height; broad engineering ranges in tools/assets/v7_object_prepare.py','status':'工程常识范围，不宣称特定产品尺寸','tablet_note':'平板的包围盒包含后撑脚；不能把整机深度当作屏幕面板厚度' if k=='tablet' else None,'range_min_m':lo.tolist(),'range_max_m':hi.tolist()},'emitter':None if silent else {'local_m':emit.tolist(),'nearest_vertex_distance_m':nearest,'physical_rationale':rationale,'active':True},'silent_decor':silent,'material_receipt':receipt,'preflight_issues':issues,'license':{'image_model':'FLUX.2-klein-4B Apache-2.0','geometry_model':'Pixal3D MIT','source_model_license_path':'/data/models/hub/models--TencentARC--Pixal3D/snapshots/0b31f9160aa400719af409098bff7936a932f726/LICENSE','output_redistribution':'未验证；只登记项目内部科研用途'},'state':'prepared','functional_front_axis':'positive-y' if k=='cellphone' and a['realized_attributes'].get('resting_form')=='lying_screen_up' else 'positive-x'}
    dump(D/'asset.json',m);print('PREPARED',aid,issues,flush=True);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--index',type=Path,required=True);a=p.parse_args();idx=json.loads(a.index.read_text());rows=[]
    for obj in idx['assets']:
        if obj['entity_class']!='rigid_static_object' or obj['identity']['object_type'] in ['bathtub','floor_drain','sink_with_tap','toilet','fireplace']:continue
        try:rows.append(prepare_one(obj,a.task_root,a.index.parent))
        except Exception as e:
            r={'asset_id':obj['asset_id'],'object_type':obj['identity']['object_type'],'origin':'existing','state':'fail','preflight_issues':[type(e).__name__+':'+str(e)]};dump(a.task_root/'assets'/obj['asset_id']/'asset.json',r);rows.append(r);print('PREP_FAIL',r,flush=True)
    dump(a.task_root/'prepared_existing.json',rows)
if __name__=='__main__':main()
