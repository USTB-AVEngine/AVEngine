"""Object-task native CIELAB measurement; fixed calibration is a caller input."""
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


COLOR_EVIDENCE_REVISION='v7-visible-color-ROI-normalization-v2'


def color_roi_pixels(record):
    return int(record.get('roi_pixels',record.get('body_pixels',0)))


def rebuild_appearance_review(checks,assets,animal_minimum_pixels=200):
    """Qualify actual observations without changing registered colour labels."""
    reviews={};excluded=[]
    names={'blue':'蓝色','yellow':'黄色','green':'绿色','white':'白色','black':'黑色','gray':'灰色','gray_silver':'灰银色','wood_brown':'木棕色','beige':'米色','brown':'棕色','red':'红色'}
    for a in assets:
        label=a.get('label_color')
        if label is None:continue
        rows=[r for r in checks if r['asset_id']==a['asset_id']]
        minimum=animal_minimum_pixels if a['kind']=='animal' else 80
        matching=[r for r in rows if r['matched']]
        wrong=[r for r in rows if not r['matched'] and color_roi_pixels(r)>=minimum]
        if not matching or wrong:
            excluded.append({'asset_id':a['asset_id'],'actor_id':a['native_instance_id'],
                'registered_label_color':a['registered_label_color'],'expected_color':label,
                'reason':'measurable_color_mismatch' if wrong else 'no_matching_native_color_observation',
                'measurable_mismatch_frames':sorted({r['frame_index'] for r in wrong}),
                'minimum_ROI_pixels':minimum,'asset_label_changed':False})
            continue
        field='top_color' if a['kind'] in {'human','performer'} else 'coat_color' if a['kind']=='animal' else 'housing_color'
        reviews[str(a['native_instance_id'])]={'status':'reviewed','field':field,'value':label,
            'label_zh':names.get(label,label)+'的'+a['name_zh'],'label_en':label+' '+a['name_zh'],
            'frame_refs':sorted({r['frame_index'] for r in matching}),
            'review_method':'actual native frame ROI; fixed hue/luminance rules; normalized body_pixels/roi_pixels',
            'human_viewing_used':False}
    return {'actors':reviews},{'revision':COLOR_EVIDENCE_REVISION,'excluded':excluded,'asset_labels_changed':0}
