"""Measure colour families from native UE body pixels; never change registry labels.

CIELAB routines adapted from the authorized v7 object branch. Calibration
uses declared UE references in the caller, without Habitat texture comparison.
"""
import numpy as np

def lab(rgb):
    v=np.asarray(rgb,dtype=float)/255
    v=np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
    xyz=v@np.array([[.4124564,.2126729,.0193339],[.3575761,.7151522,.1191920],[.1804375,.0721750,.9503041]])
    xyz/=np.array([.95047,1,1.08883]);f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)

def classify(rgb,thresholds):
    x=lab(rgb);L,a,b=x.T;C=np.hypot(a,b);h=np.degrees(np.arctan2(b,a))%360
    result=np.full(L.shape,'unknown',dtype='<U16');neutral=C<thresholds['neutral_chroma_max']
    result[neutral&(L<thresholds['black_L_max'])]='black'
    result[neutral&(L>=thresholds['black_L_max'])&(L<thresholds['white_L_min'])]='gray_silver'
    result[neutral&(L>=thresholds['white_L_min'])]='white'
    chromatic=~neutral
    for name,lo,hi in [('red',0,45),('yellow',65,110),('green',110,190),('blue',190,310),('red',310,360)]:
        result[chromatic&(h>=lo)&(h<hi)]=name
    result[chromatic&(h>=45)&(h<90)&(L<65)]='wood_brown'
    result[(C>=6)&(C<28)&(h>=45)&(h<110)&(L>=65)]='beige'
    return result

def measure(rgb,mask,thresholds):
    pixels=np.asarray(rgb)[np.asarray(mask,dtype=bool)]
    if len(pixels)<80:return {'status':'fail','reason':'fewer_than_80_body_pixels','body_pixels':len(pixels)}
    labels=classify(pixels,thresholds);names,counts=np.unique(labels,return_counts=True);hist={str(n):float(c/len(pixels)) for n,c in zip(names,counts)}
    label=max(hist,key=hist.get);share=hist[label];value=np.median(lab(pixels),axis=0)
    return {'status':'pass' if label!='unknown' and share>=thresholds['main_share_min'] else 'fail','label':label,'main_share':share,'body_pixels':len(pixels),'family_shares':hist,'median_lab_d65':value.tolist(),'metric':'CIELAB D65 L*, C*, atan2(b*,a*); native rendered body ROI'}

def summarize_views(colors,thresholds):
    """Require a dominant pooled body family and two independently measured views."""
    usable=[c for c in colors if c.get('body_pixels',0)>=80 and c.get('family_shares')]
    total=sum(c['body_pixels'] for c in usable)
    if not total:return {'status':'fail','label':None,'main_share':None,'reason':'body_ROI_missing'}
    pooled={}
    for c in usable:
        for family,share in c['family_shares'].items():pooled[family]=pooled.get(family,0)+share*c['body_pixels']/total
    label=max(pooled,key=pooled.get);votes=sum(c.get('status')=='pass' and c.get('label')==label for c in usable);share=pooled[label];passed=label!='unknown' and share>=thresholds['main_share_min'] and votes>=2
    return {'status':'pass' if passed else 'fail','label':label if passed else None,'main_share':share,'pooled_family_shares':pooled,'body_pixels_across_views':total,'dominant_agreeing_views':votes,'minimum_agreeing_views':2}

def functional_exclusion(meta,q,front_facing=None):
    x,y,z=np.asarray(q).T;k=meta['object_type'];form=meta.get('form_factor','');exclude=np.zeros(len(x),bool)
    # Inset functional surfaces can lie well behind the foremost bezel: x>0.8
    # incorrectly left entire clock dials/grilles in the main-housing sample.
    if k=='cellphone' and meta.get('functional_front_axis')=='positive-y':exclude=(y>.10)&(x>.08)&(x<.92)&(z>.08)&(z<.92)
    elif k in ['television','cellphone','tablet']:exclude=(x>.10)&(y>.13)&(y<.94)&(z>.075)&(z<.925)
    elif k=='microwave_oven':exclude=(x>.10)&(y>.12)&(y<.90)&(z>.08)&(z<.92)
    elif k=='wall_clock':
        inside=((y-.5)/.42)**2+((z-.5)/.42)**2<1 if 'round' in form else ((y>.12)&(y<.88)&(z>.12)&(z<.88))
        exclude=(x>.08)&inside
    elif k=='mantel_clock':
        inside=((y-.55)/.36)**2+((z-.5)/.41)**2<1 if 'round' in form else ((y>.22)&(y<.88)&(z>.13)&(z<.87))
        exclude=(x>.08)&inside
    elif k=='alarm_clock':
        inside=((y-.40)/.29)**2+((z-.5)/.41)**2<1 if 'twin_bell' in form else ((y>.13)&(y<.87)&(z>.10)&(z<.90))
        exclude=(x>.10)&inside
    elif k in ['bookshelf_speaker','floorstanding_speaker','soundbar','radio']:exclude=(x>.10)&(y>.04)&(y<.96)&(z>.04)&(z<.96)
    elif k=='blender':exclude=y>.44
    elif k=='printer':exclude=(x>.10)&(y>.55)&(y<.85)&(z>.60)&(z<.90)
    elif k=='desk_telephone':exclude=(x>.10)&(y>.12)&(y<.60)&(z>.20)&(z<.80)
    elif k=='landline_phone':exclude=(x>.10)&(y>.18)&(y<.78)&(z>.55)&(z<.94)
    elif k=='security_keypad' and 'flip_closed' in form:exclude=np.zeros(len(x),bool) # opaque closed casing lid is body, not exposed buttons
    elif k=='security_keypad':exclude=(x>.10)&(y>.22)&(y<.85)&(z>.20)&(z<.80)
    elif k=='doorbell_chime_unit':exclude=(x>.10)&(y>.25)&(y<.85)&(z>.3)&(z<.7)
    elif k in ['smoke_detector','carbon_monoxide_alarm','kitchen_timer']:exclude=(x>.10)&(y>.35)&(y<.65)&(z>.35)&(z<.65)
    if front_facing is not None and k!='blender':exclude &= np.asarray(front_facing,dtype=bool)
    return exclude
