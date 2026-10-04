"""Native rendered shirt colour measurements; never used as a UE admission gate."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from PIL import Image
from v7_human_roster_materials import srgb_to_linear,json_write
from v7_human_roster_roi import roi

def lab(rgb):
 xyz=srgb_to_linear(rgb)@np.array([[.4124564,.2126729,.0193339],[.3575761,.7151522,.1191920],[.1804375,.0721750,.9503041]])
 xyz/=np.array([.95047,1,1.08883]);f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29);return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)

def run(T):
 selection=json.loads((T/'registrations/selection.json').read_text());rows={r['asset_id']:r for r in selection['avatars']};measure=[];views={h:json.loads((T/'evidence'/('habitat_view_'+h+'.json')).read_text())for h in ['mp3d','hm3d']}
 for p in sorted((T/'renders/habitat').glob('*/*/*/native.json')):
  r=json.loads(p.read_text());aid=r['asset_id'];row=rows.get(aid)
  if not row or not row['controlled_color']:continue
  view=views[r['house']];rgb=np.asarray(Image.open(p.with_name('rgb.png')).convert('RGB'));semantic=np.asarray(Image.open(p.with_name('mask.png')))>0;cached=T/'evidence/rendered_shirt_roi'/r['house']/(aid+'.png');cached.parent.mkdir(parents=True,exist_ok=True)
  if cached.exists():mask=np.asarray(Image.open(cached))>0
  else:
   mask=roi(T/r['glb'],T/'masks'/aid/'shirt_main_color.png',row['prefix'],view,semantic);Image.fromarray(mask.astype('uint8')*255).save(cached)
  mask &=semantic;Image.fromarray(mask.astype('uint8')*255).save(p.with_name('shirt_mask.png'));count=int(mask.sum())
  med=np.median(rgb[mask],axis=0) if count else None;value=lab(med) if med is not None else None;color=r['color'];named=False
  if med is not None:
   # Colour names are checked independently of pairwise separation.
   mx=float(med.max());mn=float(med.min());delta=mx-mn
   if color=='white':named=value[0]>=55 and delta<45
   elif color=='black':named=value[0]<=42 and delta<45
   else:
    import colorsys
    hue,sat,val=colorsys.rgb_to_hsv(*(med/255));hue*=360
    ranges={'red':lambda h:h<25 or h>340,'green':lambda h:75<=h<=165,'blue':lambda h:185<=h<=255,'yellow':lambda h:30<=h<=75}
    named=bool(ranges[color](hue) and sat>.30 and val>.15)
  item={'house':r['house'],'asset_id':aid,'color':color,'shirt_pixels':count,'median_srgb':None if med is None else med.tolist(),'median_lab_d65':None if value is None else value.tolist(),'nominal_color_match':bool(named),'color_name_status':'pass' if named and count>=80 else 'fail','pixel_source':str(p.with_name('rgb.png').relative_to(T)),'mask_source':str(p.with_name('shirt_mask.png').relative_to(T)),'roi_method':'perspective-correct UV + full human z-buffer + native semantic pixels, one-pixel erosion'}
  r.update(shirt_pixels=count,shirt_median_srgb=item['median_srgb'],shirt_roi_method=item['roi_method']);json_write(p,r);measure.append(item)
 pairs=[];groups={}
 for r in measure:groups.setdefault((r['house'],r['asset_id']),[]).append(r)
 for (house,aid),items in groups.items():
  for a,b in itertools.combinations(items,2):
   delta=None if a['median_lab_d65']is None or b['median_lab_d65']is None else float(np.linalg.norm(np.array(a['median_lab_d65'])-b['median_lab_d65']))
   pairs.append({'house':house,'asset_id':aid,'colors':[a['color'],b['color']],'delta_e_cie76':delta,'threshold':20.0,'status':'pass' if delta is not None and delta>=20 else 'fail'})
 report={'schema':'avengine_v7_human_rendered_color_qa_v1','metric':'CIELAB D65, delta E CIE76 on median native RGB inside visible shirt_main_color','backend':'Habitat native only','nominal_check':'HSV hue/saturation for chromatic colours; LAB L and neutral channel spread for white/black','minimum_shirt_pixels':80,'pair_threshold':20.0,'measurements':measure,'pairs':pairs,'measurement_count':len(measure),'pair_count':len(pairs),'failed_names':sum(x['color_name_status']!='pass'for x in measure),'failed_pairs':sum(x['status']!='pass'for x in pairs),'minimum_delta_e':min((x['delta_e_cie76']for x in pairs if x['delta_e_cie76']is not None),default=None)}
 json_write(T/'evidence/rendered_color_check.json',report);json_write(T/'evidence/habitat_native_results.json',[json.loads(p.read_text())for p in sorted((T/'renders/habitat').glob('*/*/*/native.json'))]);print(json.dumps({k:report[k]for k in ['measurement_count','pair_count','failed_names','failed_pairs','minimum_delta_e']},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
