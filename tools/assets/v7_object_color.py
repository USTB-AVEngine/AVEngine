"""Calibrate CIELAB colour families on known native shirts, then measure object body pixels."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image

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

def calibrate(reference,out):
    records=[]
    for p in sorted((reference/'renders/habitat').glob('*/*/*/shirt_mask.png')):
        color=p.parent.name
        if color not in ['black','white','red','green','blue']:continue
        mask=np.asarray(Image.open(p))>0;rgb=np.asarray(Image.open(p.with_name('rgb.png')).convert('RGB'))
        if mask.sum()>=80:records.append((p,color,rgb,mask))
    train=[r for r in records if r[0].parts[-3].endswith(('_01','_05','_08','_13','_17'))]
    held=[r for r in records if not r[0].parts[-3].endswith(('_01','_05','_08','_13','_17'))]
    if not held:train=records[::2];held=records[1::2]
    best=None
    for chroma in [8,10,12,15,18,22]:
        for black in [28,32,36,40,42]:
            for white in [52,55,58,60,63,65]:
                t={'neutral_chroma_max':chroma,'black_L_max':black,'white_L_min':white,'main_share_min':.55}
                # Calibration on median samples is bounded; full-pixel validation follows.
                correct=sum(str(classify(np.median(rgb[mask],axis=0)[None],t)[0])==color for _,color,rgb,mask in train)
                if best is None or correct>best[0]:best=(correct,t)
    if best is None:raise RuntimeError('Known human shirt RGB/masks missing')
    t=best[1];rows=[]
    for p,color,rgb,mask in records:
        m=measure(rgb,mask,t);rows.append({'rgb':str(p.with_name('rgb.png')),'mask':str(p),'expected':color,'split':'train' if any(p==q[0] for q in train) else 'held_out',**m,'correct':m.get('label')==color and m['status']=='pass'})
    hold=[r for r in rows if r['split']=='held_out'];accuracy=sum(r['correct'] for r in hold)/len(hold)
    report={'status':'pass' if accuracy>=.95 else 'fail','reference_root':str(reference),'thresholds':t,'known_samples':len(rows),'held_out_samples':len(hold),'held_out_accuracy':accuracy,'required_held_out_accuracy':.95,'gray_beige_wood_calibration':'not represented among known shirts; engineering CIELAB definitions, separately reported','measurements':rows}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','thresholds','known_samples','held_out_samples','held_out_accuracy']}))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();calibrate(a.reference,a.out)

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
