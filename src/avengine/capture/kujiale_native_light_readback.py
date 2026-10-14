"""Native authored light state and transient on/off image diagnostics."""
from pathlib import Path
import json
import numpy as np


def audit_native_light_effect(instance, game, camera, components, plan, output):
    import cv2
    from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
    from avengine.capture.kujiale_spear_runtime import capture_view
    lights=[];native=[]
    with instance.begin_frame():
        for actor_class in ['APointLight','ASpotLight','ARectLight','ADirectionalLight','ASkyLight']:
            actors=game.unreal_service.find_actors_by_class(uclass=actor_class,as_unreal_object=True)
            for actor in actors:
                for component in game.unreal_service.get_components_by_class(actor=actor,uclass='ULightComponentBase',as_unreal_object=True):
                    visible=bool(component.get_property_value(property_name='bVisible'))
                    row={'actor_class':actor_class,'native_pose':read_scene_component_pose(component),
                         'intensity':float(component.get_property_value(property_name='Intensity')),
                         'visible':visible}
                    for property_name in ['Mobility','bAffectsWorld','bCastShadows','IndirectLightingIntensity']:
                        try:row[property_name]=component.get_property_value(property_name=property_name)
                        except Exception as e:row[property_name+'_readback_error']=str(e)
                    native.append(row);lights.append((component,visible))
    with instance.end_frame():pass
    view=next(v for room in plan['rooms'] for v in room['views'] if v['kind']=='perspective')
    on,_=capture_view(instance,camera,components,view,warmup_frames=300)
    control,_=capture_view(instance,camera,components,view,warmup_frames=40)
    off=None
    try:
        with instance.begin_frame():
            for component,_visible in lights:component.SetVisibility(bNewVisibility=False,bPropagateToChildren=False)
        with instance.end_frame():pass
        off,_=capture_view(instance,camera,components,view,warmup_frames=40)
    finally:
        with instance.begin_frame():
            for component,visible in lights:component.SetVisibility(bNewVisibility=visible,bPropagateToChildren=False)
        with instance.end_frame():pass
    restored,_=capture_view(instance,camera,components,view,warmup_frames=300)
    output=Path(output);output.mkdir(exist_ok=False)
    for label,pixels in [('on',on),('control_on',control),('off',off),('restored',restored)]:
        cv2.imwrite(str(output/(label+'.jpg')),cv2.resize(pixels,(1280,720),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,92])
    delta=np.abs(control.astype(float)-off.astype(float));noise=np.abs(on.astype(float)-control.astype(float))
    result={'status':'measured','house':plan['house'],'map_path':plan['map_path'],'view':view,
            'native_light_components':native,'light_count':len(lights),
            'visible_positive_light_count':sum(r['visible'] and r['intensity']>0 for r in native),
            'lights_off_mean_abs_rgb_difference_255':float(delta.mean()),
            'control_on_mean_abs_rgb_difference_255':float(noise.mean()),
            'lights_off_pixels_channel_difference_gt5_fraction':float(np.mean(np.any(delta>5,axis=2))),
            'control_on_pixels_channel_difference_gt5_fraction':float(np.mean(np.any(noise>5,axis=2))),
            'scope':'transient all-authored-light visibility toggle in own game process; histogram parameters unchanged; admission views captured before diagnostic; original visibility restored',
            'new_admission_gate_added':False,'parameter_tuning_used':False,
            'paths':{label:str(output/(label+'.jpg')) for label in ['on','control_on','off','restored']}}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
