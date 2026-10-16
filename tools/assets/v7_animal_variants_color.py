"""CIELAB pixel families calibrated on actual native human shirt masks."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from PIL import Image

FAMILIES=['black','white','gray','brown','orange','golden','yellow','green','blue','red','other']
DEFAULT={'black_l_max':30.,'white_l_min':60.,'neutral_c_max':18.,'dominant_fraction_min':.55,'min_pixels':200,'min_pair_delta_e76':15.,'yellow_hue_min':80.,'brown_min_c':12.,'brown_neutral_l_max':48.,'brown_hue_max':95.}
def rgb_to_lab(rgb):
    v=np.asarray(rgb,dtype=float)/255.;linear=np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4);xyz=linear@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T;xyz=xyz/np.array([.95047,1.,1.08883]);f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29);return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)
def pixel_families(lab,thresholds=DEFAULT):
    l,a,b=np.moveaxis(np.asarray(lab),-1,0);c=np.hypot(a,b);h=np.mod(np.degrees(np.arctan2(b,a)),360);out=np.full(l.shape,FAMILIES.index('other'),dtype=np.uint8);warm_brown=(h>=45)&(h<thresholds.get('brown_hue_max',95.))&(c>=thresholds['brown_min_c'])&(l<thresholds['brown_neutral_l_max']);neutral=(c<=thresholds['neutral_c_max'])&~warm_brown;out[neutral&(l<=thresholds['black_l_max'])]=FAMILIES.index('black');out[neutral&(l>=thresholds['white_l_min'])]=FAMILIES.index('white');out[neutral&(l>thresholds['black_l_max'])&(l<thresholds['white_l_min'])]=FAMILIES.index('gray');colored=~neutral
    rules={'red':(h<45)|(h>=330),'brown':((h>=45)&(h<thresholds['yellow_hue_min'])&(l<50))|warm_brown,'orange':(h>=45)&(h<70)&(l>=50),'golden':(h>=70)&(h<thresholds['yellow_hue_min'])&(l>=50),'yellow':(h>=thresholds['yellow_hue_min'])&(h<115)&~warm_brown,'green':(h>=115)&(h<225),'blue':(h>=225)&(h<330)}
    for name,condition in rules.items():out[colored&condition]=FAMILIES.index(name)
    return out

def measure(rgb,mask,thresholds=DEFAULT,excluded_mask=None):
    mask=np.asarray(mask,bool).copy();excluded=np.zeros_like(mask) if excluded_mask is None else np.asarray(excluded_mask,bool);excluded_count=int((mask&excluded).sum());mask &= ~excluded;pixels=np.asarray(rgb)[mask,:3];lab=rgb_to_lab(pixels);families=pixel_families(lab,thresholds);counts=np.bincount(families,minlength=len(FAMILIES));best=int(counts.argmax());fraction=float(counts[best]/max(len(pixels),1));median=np.median(lab[families==best],axis=0).tolist() if counts[best] else None
    return {'measured_label':FAMILIES[best],'dominant_fraction':fraction,'family_fractions':{name:float(counts[i]/max(len(pixels),1)) for i,name in enumerate(FAMILIES)},'body_pixels_after_exclusion':len(pixels),'excluded_eye_nose_beak_pixels':excluded_count,'dominant_median_lab':median,'status':'pass' if len(pixels)>=thresholds['min_pixels'] and fraction>=thresholds['dominant_fraction_min'] and FAMILIES[best]!='other' else 'fail'}

def calibration(root,out):
    original=json.loads((root/'evidence/rendered_color_check.json').read_text());data=original['measurements'];bodies=sorted({x['asset_id'] for x in data});training=set(bodies[:4]);loaded=[]
    for row in data:
        p=root/row['pixel_source'];m=root/row['mask_source'];rgb=np.array(Image.open(p).convert('RGB'));mask=np.array(Image.open(m).convert('L'))>0;loaded.append((row,rgb,mask))
    options=[]
    for b,w,c,y in itertools.product([25.,30.,35.],[55.,60.,65.],[12.,15.,18.],[75.,80.,85.,90.]):
        t={**DEFAULT,'black_l_max':b,'white_l_min':w,'neutral_c_max':c,'yellow_hue_min':y};train=[(r,measure(rgb,mask,t)) for r,rgb,mask in loaded if r['asset_id'] in training];correct=sum(m['status']=='pass' and m['measured_label']==r['color'] for r,m in train);score=(correct,-abs(b-30)-abs(w-60)-abs(c-18)-abs(y-80));options.append((score,t))
    _,selected=max(options,key=lambda x:x[0]);rows=[]
    for row,rgb,mask in loaded:
        m=measure(rgb,mask,selected);rows.append({'asset_id':row['asset_id'],'house':row['house'],'expected_label':row['color'],'split':'calibration' if row['asset_id'] in training else 'held_out','rgb_path':str(root/row['pixel_source']),'mask_path':str(root/row['mask_source']),'rgb_sha256':hashlib.sha256((root/row['pixel_source']).read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256((root/row['mask_source']).read_bytes()).hexdigest(),**m,'correct':m['measured_label']==row['color'] and m['status']=='pass'})
    groups={split:{'count':sum(x['split']==split for x in rows),'correct':sum(x['split']==split and x['correct'] for x in rows)} for split in ['calibration','held_out']};passed=all(x['count']>0 and x['count']==x['correct'] for x in groups.values());result={'schema':'avengine_v7_animal_color_calibration_v1','status':'pass' if passed else 'fail','method':'sRGB to CIELAB D65; chromatic hue/chroma families; neutral luminance; real native rendered shirt pixel masks; first four identities fit, remaining identities held out','thresholds':selected,'groups':groups,'known_shirt_labels':sorted({x['color'] for x in data}),'not_present_in_shirt_set':['gray','brown','orange','golden'],'unrepresented_boundaries':'gray luminance between black and white; brown low-light orange hue; not independently calibrated by a gray/brown shirt example','measurements':rows};out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print({k:result[k] for k in ['status','groups','thresholds']});return passed

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();return 0 if calibration(a.reference_root,a.output) else 1
if __name__=='__main__':raise SystemExit(main())
