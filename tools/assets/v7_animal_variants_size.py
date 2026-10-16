"""Consistent measured body sizes in owned GLB derivatives; no original asset mutation."""
import argparse,copy,collections,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from v7_human_roster_pose import read_glb,write_glb,pose
from v7_animal_variants_standing import size_metrics,sha,dump

V6_TARGETS={'abyssinian':.27,'burmese':.25,'siamese':.25,'shiba_inu':.39,'border_collie':.52,'pembroke_welsh_corgi':.28,'jack_russell_terrier':.30}

def geometry(path):
 d,b=read_glb(path);parts,_=pose(d,b,canonical_flip=False,ground=False);return np.concatenate([p['positions'] for p in parts])
def head_height(v,feet,head_bounds=None):
 if head_bounds is not None:
  lo,hi=map(np.asarray,head_bounds);region=v[np.all((v>=lo-1e-6)&(v<=hi+1e-6),axis=1)]
 else:
  fs=np.array(list(feet.values()));z=np.sort(fs[:,2]);front=z[:2].mean();span=z[2:].mean()-front;region=v[(v[:,2]<front+span*.15)&(v[:,1]>np.quantile(v[:,1],.45))]
 if len(region)<20:raise ValueError('head region insufficient for measured body-size consistency')
 return float(np.quantile(region[:,1],.998)-v[:,1].min())

def measurements(meta):
 v=geometry(meta['body_only_glb']);feet={k:np.array(p) for k,p in meta['landmarks_local_m'].items() if k.startswith('foot_')};m=size_metrics(v,feet,meta['category'])
 if meta['category']!='bird':
  bounds=(meta['landmarks_local_m']['head_bbox_min'],meta['landmarks_local_m']['head_bbox_max']) if meta['mode']=='v6_idle_static_derivative' else None
  m['full_body_p998_height_m']=m['robust_top_height_m'];m['robust_top_height_m']=head_height(v,feet,bounds);m['head_height_scope']='original semantic-head volume' if bounds else 'upper anterior region defined by qualified front paws; tail excluded'
 if meta['category']=='bird':
  _,axes=np.linalg.eigh(np.cov(v.T));proj=v@axes[:,-1];m['principal_body_length_m']=float(np.quantile(proj,.998)-np.quantile(proj,.002));m['body_height_without_perch_m']=float(np.ptp(v[:,1]));m['shoulder_proxy_m']=None
 return m

def metrics(category):return ['principal_body_length_m','body_height_without_perch_m'] if category=='bird' else ['fore_hind_paw_span_m','robust_top_height_m']
def scale_choice(before,medians,category):
 keys=metrics(category);ratio=np.array([before[k]/medians[k] for k in keys]);f=float(1/np.sqrt(ratio.prod()));s=np.array([f,f,f]);method='uniform scale fitted to two measured body medians'
 if np.any(np.abs(ratio*f-1)>.15):
  if category=='bird':raise ValueError('bird dimensions cannot meet 15% by uniform scale; requires pose diagnosis')
  y=medians[keys[1]]/before[keys[1]];z=medians[keys[0]]/before[keys[0]];s=np.array([np.sqrt(y*z),y,z]);method='derived mesh longitudinal/vertical proportion correction after uniform scale proved insufficient'
 return s,method

def wrapped_glb(source,dest,s):
 d,b=read_glb(source);d=copy.deepcopy(d);roots=d['scenes'][d.get('scene',0)]['nodes'];i=len(d['nodes']);d['nodes'].append({'name':'OwnedAnimalBodySizeCorrection','scale':s.tolist(),'children':roots});d['scenes'][d.get('scene',0)]['nodes']=[i];write_glb(dest,d,b)

def apply(T,extra_queue=None):
 P=T/'append_20261012';paths=[p for p in sorted((T/'assets').glob('*/asset.json'))];paths+=sorted((P/'v6_standing').glob('*/asset.json'))
 if extra_queue:paths += [Path(x) for x in json.loads(extra_queue.read_text())]
 source=[]
 for p in paths:
  meta=json.loads(p.read_text());source.append((p,meta,measurements(meta)))
 by=collections.defaultdict(list)
 for p,m,b in source:by[(m['category'],m['body_id'])].append((p,m,b))
 rows=[];queue=[];all_paths=[]
 for (category,body),group in by.items():
  med={k:float(np.median([b[k] for _,_,b in group])) for k in metrics(category)};plans=[]
  for p,m,b in group:
   s,method=scale_choice(b,med,category)
   if m['mode']=='rocketbox_coat_variant':s=np.ones(3);method='source rig/scale retained; identical coat body measures already pass'
   plans.append((p,m,b,s,method))
  # Choose one physical body target, applied equally to every color.
  common=1.;target=None
  if category!='bird' and not all(m['mode']=='rocketbox_coat_variant' for _,m,_,_,_ in plans):
   target=next((m['physical_size']['target_m'] for _,m,_,_,_ in plans if m.get('physical_size')),V6_TARGETS.get(body));proxy=float(np.median([b['shoulder_proxy_m']*s[1] for _,m,b,s,_ in plans]));common=float(target/proxy) if target else 1.
  for p,m,b,s,method in plans:
   s=s*common;D=P/'sized'/m['asset_id'];D.mkdir(parents=True,exist_ok=True);changed=bool(np.max(np.abs(s-1))>.003);s=s if changed else np.ones(3);new=copy.deepcopy(m)
   if changed:
    for key,name in [('runtime_glb','runtime.glb'),('body_only_glb','body_only.glb')]:wrapped_glb(m[key],D/name,s);new[key]=str(D/name)
    v=geometry(new['runtime_glb']);new.update(runtime_sha256=sha(new['runtime_glb']),glb_bytes=Path(new['runtime_glb']).stat().st_size,dimensions_m=np.ptp(v,axis=0).tolist(),bounds_min_m=v.min(0).tolist(),bounds_max_m=v.max(0).tolist())
    for key in ['emit_point_local_m','head_center_local_m']:new[key]=(np.array(m[key])*s).tolist()
    new['landmarks_local_m']={k:(np.array(v)*s).tolist() for k,v in m['landmarks_local_m'].items()}
    new['emit_head_distance_m']=float(np.linalg.norm(np.array(new['emit_point_local_m'])-np.array(new['head_center_local_m']))) if m['mode']!='v6_idle_static_derivative' else m['emit_head_distance_m']*float(s.max())
    new['emit_head_radius_m']=m.get('emit_head_radius_m',.08)*float(s.max())
    if new.get('perch'):
     for key,val in list(new['perch'].items()):
      if key=='perch_top_m':new['perch'][key]=val*float(s[1])
    new['prior_normalization']=m['normalization'];new['normalization']=copy.deepcopy(m['normalization']);new['normalization']['size_derivative_method']=method;new['normalization']['size_scale_xyz']=s.tolist()
    for key in ['maximum_contact_delta_m','four_paw_height_range_m']:
     if new['normalization'].get(key) is not None:new['normalization'][key]*=float(s[1])
   after=measurements(new);new['size_measurements']=after
   if category!='bird':new['normalization']['shoulder_height_m']=after['shoulder_proxy_m']
   new['size_consistency']={'measurement_method':'four authored/qualified support centers, front/rear mean separation; head region vertex Y 99.8th percentile above minimum; original semantic head for v6, anterior upper region for generated bodies, tail excluded; central torso upper 95% fur surface shoulder proxy' if category!='bird' else 'body-only geometry PCA principal extent between 0.2th/99.8th percentiles and body vertical height; perch excluded, independent of yaw','source_metadata':str(p),'source_runtime_sha256':m['runtime_sha256'],'before':b,'body_medians_before':med,'scale_xyz':s.tolist(),'uniform_scale':bool(np.ptp(s)<1e-9),'method':method,'shared_physical_target_proxy_m':target,'physical_size_caveat':'fur-surface shoulder proxy; cats representative estimates, exact breed withers unverified','after':after,'derived_original_unchanged':True,'changed':changed,'cause':('原模型长度/头高比例不一致；等比缩放无法同时满足两项，派生副本才作轴向修正' if not np.ptp(s)<1e-9 else '几何尺寸偏差由前后爪间距和排除尾巴的头高共同确认，使用等比缩放') if np.any(np.abs(np.array([b[k]/med[k] for k in med])-1)>.15) else '统一稳健测法和品种目标尺寸；原组内两项测量已在15%范围内','diagnostic_before_relative':{k:b[k]/med[k] for k in med},'full_height_to_head_ratio':None if category=='bird' else b['full_body_p998_height_m']/b['robust_top_height_m']}
   out=D/'asset.json';dump(out,new);all_paths.append(str(out));queue.extend([str(out)] if changed else []);rows.append({'asset_id':m['asset_id'],'category':category,'body_id':body,'metadata':str(out),**new['size_consistency']});print('BODY_SIZE',m['asset_id'],s.tolist(),method,flush=True)
 # Compute medians AFTER all derivatives, not by comparing only with obsolete targets.
 groups=collections.defaultdict(list)
 for row in rows:groups[(row['category'],row['body_id'])].append(row)
 failures=[]
 for (category,body),g in groups.items():
  med={k:float(np.median([r['after'][k] for r in g])) for k in metrics(category)}
  for r in g:
   r['body_medians_after']=med;r['relative_after']={k:r['after'][k]/med[k] for k in med};r['status']='pass' if all(abs(x-1)<=.15+1e-7 for x in r['relative_after'].values()) else 'fail'
   p=Path(r['metadata']);m=json.loads(p.read_text());m['size_consistency'].update(body_medians_after=med,relative_after=r['relative_after'],status=r['status']);dump(p,m)
   if r['status']=='fail':failures.append(r['asset_id'])
 dump(P/'reports/size_consistency.json',{'status':'pass' if not failures else 'fail','threshold_relative_to_body_median':.15,'rows':rows,'failures':failures});dump(P/'reports/size_native_queue.json',queue);dump(P/'reports/all_final_metadata.json',all_paths);return failures

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--extra-queue',type=Path);a=p.parse_args();return int(bool(apply(a.task_root,a.extra_queue)))
if __name__=='__main__':raise SystemExit(main())
