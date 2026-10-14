"""Apply unchanged QA color categories to actual lit UE pixels and masks.

A pan can leave a valid target mask with zero saturated upper-body pixels.
The common screen's empty-list NumPy index raises before returning blocked.
This caller records that empty measurement as blocked and never gives it a
color cue. Every nonempty measurement still uses the unchanged common screen.
No texture sRGB comparison or native material overrides are introduced.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from avengine.qa.v7_visual_admission import measure_native_color,COLOR_HUE_DEGREES


def measure_ue_color(rgb,semantic,semantic_id,color):
    mask=np.asarray(semantic)==int(semantic_id);ys,xs=np.nonzero(mask)
    if len(xs):
        top=int(ys.min());height=int(ys.max())-top+1;yy=np.arange(mask.shape[0])[:,None]
        roi=mask&(yy>=top+.15*height)&(yy<top+.60*height)
        pixels=np.asarray(rgb)[:,:,:3][roi].astype(float)/255.
        high=pixels.max(1);delta=high-pixels.min(1)
        sat=np.divide(delta,high,out=np.zeros_like(delta),where=high>0)
        if not np.any((sat>=.25)&(high>=.12)&(delta>1e-8)):
            return {'status':'blocked','reason':'no_saturated_native_upper_body_pixels',
                    'native_visible_pixels':len(xs),'roi_pixels':len(pixels),'saturated_pixels':0,
                    'expected_color':color,'semantic_id':int(semantic_id),
                    'gate':{'minimum_colored_pixels':20,'minimum_fraction':.15,'minimum_margin':.10,'saturation_minimum':.25,'value_minimum':.12},
                    'human_viewing_used':False}
    return measure_native_color(rgb,semantic,semantic_id,color)


def derive_ue_appearance_evidence(directory,visual,assets):
    directory=Path(directory);by_asset={a['asset_id']:a for a in assets};reviews={};screens=[]
    for png in sorted(directory.glob('native_frame_*.png')):
        index=int(png.stem.rsplit('_',1)[1]);mask=directory/f'native_masks_{index:04d}.npz'
        if not mask.is_file() or not 0<=index<len(visual['frames']):continue
        rgb=np.asarray(Image.open(png));semantic=np.load(mask,allow_pickle=False)['semantic']
        for actor in visual['frames'][index]['actors']:
            asset=by_asset.get(actor['asset_id']);color=(asset or {}).get('appearance',{}).get('top_color')
            if color not in COLOR_HUE_DEGREES:continue
            result=measure_ue_color(rgb,semantic,actor['semantic_id'],color)
            screens.append({'frame_index':index,'actor_id':actor['actor_id'],'asset_id':actor['asset_id'],**result})
            if result['status']!='pass':continue
            key=str(actor['actor_id']);label=asset.get('appearance_zh',{'blue':'蓝色上衣的人','yellow':'黄色上衣的人','green':'绿色上衣的人','burgundy':'酒红色上衣的人'}[color])
            review=reviews.setdefault(key,{'status':'reviewed','value':color,'label_zh':label,'label_en':color,'frame_refs':[],'review_method':'automatic_native_color_pixels','human_viewing_used':False})
            if review['value']!=color:raise ValueError('native appearance changed within a physical actor')
            review['frame_refs'].append(index)
    return {'revision':'automatic_native_hue_review_v1','adapter':'UE actual lit RGB; empty ROI remains blocked',
            'review':{'actors':reviews},'screens':screens,'status':'pass' if reviews else 'not_run_or_no_eligible_color_pixels','human_viewing_used':False}
