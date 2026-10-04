"""Automatic appearance checks on actual RGB and native semantic pixels."""
from __future__ import annotations
import numpy as np

COLOR_HUE_DEGREES={"blue":(190.,260.),"yellow":(35.,75.),"green":(75.,175.),"burgundy":(330.,360.)}

def measure_native_color(rgb, semantic, semantic_id, expected_color, *, upper_body=True,
                         minimum_colored_pixels=20, minimum_fraction=.15, minimum_margin=.10):
    """A bounded color screen, not a person identity or sex classifier.

    The expected label is checked against measured pixels, including a wrong
    blue/yellow label control. Mask pixels must be from the actual native frame.
    Failed screens must not be converted into appearance reviews.
    """
    rgb=np.asarray(rgb);mask=np.asarray(semantic)==int(semantic_id)
    if rgb.ndim!=3 or rgb.shape[:2]!=mask.shape or rgb.shape[2]<3:
        raise ValueError("RGB/semantic dimensions disagree")
    ys,xs=np.nonzero(mask)
    if not len(xs):return {"status":"blocked","reason":"no_native_visible_pixels"}
    if upper_body:
        top=int(ys.min());height=int(ys.max())-top+1
        yy=np.arange(mask.shape[0])[:,None]
        mask &= (yy>=top+.15*height)&(yy<top+.60*height)
    pixels=rgb[:,:,:3][mask].astype(float)/255.
    maximum=pixels.max(1);minimum=pixels.min(1);delta=maximum-minimum
    saturation=np.divide(delta,maximum,out=np.zeros_like(delta),where=maximum>0)
    valid=(saturation>=.25)&(maximum>=.12)&(delta>1e-8)
    values=pixels[valid];d=delta[valid];imax=values.argmax(1) if len(values) else []
    hue=np.zeros(len(values),float)
    for channel in range(3):
        choose=imax==channel
        if channel==0:hue[choose]=((values[choose,1]-values[choose,2])/d[choose])%6
        elif channel==1:hue[choose]=(values[choose,2]-values[choose,0])/d[choose]+2
        else:hue[choose]=(values[choose,0]-values[choose,1])/d[choose]+4
    hue*=60
    fractions={name:float(np.mean((hue>=lo)&(hue<hi))) if len(hue) else 0.
               for name,(lo,hi) in COLOR_HUE_DEGREES.items()}
    if expected_color not in fractions:raise ValueError("automatic color domain has no requested label")
    confidence=fractions[expected_color]
    other=max([v for k,v in fractions.items() if k!=expected_color],default=0.)
    passed=len(hue)>=minimum_colored_pixels and confidence>=minimum_fraction and confidence-other>=minimum_margin
    return {"status":"pass" if passed else "blocked","reason":None if passed else "insufficient_or_ambiguous_native_color",
            "expected_color":expected_color,"semantic_id":int(semantic_id),"native_visible_pixels":len(xs),
            "roi_pixels":len(pixels),"saturated_pixels":len(hue),"color_fractions":fractions,
            "confidence":confidence,"margin":confidence-other,
            "gate":{"minimum_colored_pixels":minimum_colored_pixels,"minimum_fraction":minimum_fraction,
                    "minimum_margin":minimum_margin,"saturation_minimum":.25,"value_minimum":.12},
            "human_viewing_used":False}
