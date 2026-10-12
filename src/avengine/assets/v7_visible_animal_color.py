"""Fixed animal colour boundaries from the closed animal task at e21aa93.

No calibration job is rerun; real scene body pixels are measured anew.
"""
import numpy as np

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
