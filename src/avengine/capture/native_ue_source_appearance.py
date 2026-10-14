"""Native UE source QA: retained highlights only, with original material readback.

Owner correction in task round 7 disables sRGB texture-color admission for UE.
BaseColor supplies only the unchanged highlight detector's brightness baseline.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def measure_ue_source_highlights(rgb, mask, *, max_highlight_ratio, reference_rgb):
    rgb=np.asarray(rgb)[...,:3].astype(float)/255.
    reference=np.asarray(reference_rgb)[...,:3].astype(float)/255.
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=rgb.shape[:2] or reference.shape!=rgb.shape:raise ValueError('native UE source image/mask shape differs')
    pixels=rgb[mask];count=len(pixels)
    if count<100:return {'status':'not_run','visible_pixels':count,'reason':'fewer_than_100_source_pixels','color_checks_used':False}
    expected_luma=reference[mask].mean(axis=1);luminance=pixels.mean(axis=1)
    # Exact frozen source_image_metrics_v3 highlight definition and class limit.
    highlight=(pixels.min(axis=1)>.70)&(luminance>expected_luma*1.35+.05)&(luminance-expected_luma>.12)
    ratio=float(highlight.mean());failures=['excess_source_highlights'] if ratio>max_highlight_ratio else []
    return {'status':'fail' if failures else 'pass','visible_pixels':count,'highlight_ratio':ratio,
            'thresholds':{'max_highlight_ratio':max_highlight_ratio},'failures':failures,
            'metric_revision':'native_ue_highlights_v1','color_checks_used':False,
            'reference_kind':'same UE BaseColor view only for retained highlight luminance baseline',
            'authority':'owner correction, user round 7: UE colors may differ under Lumen, EV3, white balance 4000K and shadows'}


def verify_original_ue_materials(game,runtime,asset_id,bindings_path):
    if not bindings_path:return {'status':'not_requested'}
    receipt=json.loads(Path(bindings_path).read_text())
    if receipt['status']!='pass':raise ValueError('original native material restoration is incomplete')
    materials=receipt['asset_bindings'][asset_id]
    component=runtime['component']
    if int(component.GetNumMaterials())!=len(materials):raise ValueError('original native material slot count differs')
    slots=[]
    for index,path in enumerate(materials):
        expected=int(game.unreal_service.load_object(uclass='UMaterialInterface',name=path,as_handle=True))
        observed=int(component.GetMaterial(ElementIndex=index,as_handle=True))
        if expected!=observed:raise ValueError('spawned native source does not use restored original material')
        slots.append({'slot':index,'original_material':path,'readback_matches':True})
    return {'status':'pass','asset_id':asset_id,'slots':slots,'material_overrides_applied':False,'restoration_receipt':receipt['restore_receipt']}
