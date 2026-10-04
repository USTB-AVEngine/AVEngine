"""Task-isolated editor/game SPEAR capture for Kujiale native UE pixels.

Imports stay lazy so CPU planning and lightweight validation do not start UE.
The caller supplies an isolated project, IPC directory and graphics adapter.
"""
from __future__ import annotations
import math
from pathlib import Path
import socket
import sys
import time
import numpy as np


def to_ue(point):
    x,y,z=map(float,point)
    return [100*x,100*z,100*y]


def from_ue(point):
    x,y,z=map(float,point)
    return [x/100,z/100,y/100]


def launch_native(ue, map_path):
    ext=Path(ue['spear_ext_dir']).resolve()
    project=Path(ue['uproject']).resolve()
    output=Path(ue['output_root']).resolve()
    if output not in project.parents:
        raise ValueError('native project must be isolated inside the output root')
    sys.path.insert(0,str(ext))
    from avengine.backends.spear_ue import client
    from avengine.backends.spear_ue.research_runtime import cleanup_failed_constructor
    # rpc_port=0 asks the OS for an available port. Consecutive isolated UE
    # processes otherwise collide with the previous RPC socket's TIME_WAIT.
    port=int(ue.get('rpc_port',0))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',port))
        port=int(probe.getsockname()[1])
    temporary=output/'ue_v1/tmp'/f'spear_{port}_{time.time_ns()}'
    temporary.mkdir(parents=True)
    config=client.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE='editor'
    config.SPEAR.INSTANCE.EDITOR_EXECUTABLE=ue['unreal_editor']
    config.SPEAR.INSTANCE.EDITOR_UPROJECT=str(project)
    config.SPEAR.INSTANCE.EDITOR_LAUNCH_MODE='game'
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS=3600.
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS=300.
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP=True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP=map_path
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME=1/15
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT=port
    config.SPEAR.INSTANCE.TEMP_DIR=str(temporary)
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID=port*10000
    arguments=config.SPEAR.INSTANCE.COMMAND_LINE_ARGS
    arguments.renderoffscreen=None
    arguments.graphicsadapter=int(ue['graphics_adapter'])
    arguments.ddc='KUJIALEOwnDDC'
    arguments.notraceserver=None
    arguments.noexceptionhandler=None
    arguments.noinnerexception=None
    arguments.TraceAutoStart=0
    arguments.corelimit=8
    arguments.unattended=None
    arguments.nosound=None
    arguments.noautosave=None
    arguments.nop4=None
    arguments.userdir=str(output/'ue_v1/user')
    arguments.abslog=str(temporary/'UnrealEditor.log')
    from avengine.capture.kujiale_native_exposure import exposure_console_commands
    arguments.execcmds=','.join(exposure_console_commands(ue)+[
        'r.Lumen.DiffuseIndirect.Allow 1','r.Lumen.Reflections.Allow 1',
        'sg.ViewDistanceQuality 3','sg.ShadowQuality 3','sg.GlobalIlluminationQuality 3','sg.ReflectionQuality 3'])
    config.SPEAR.ENVIRONMENT_VARS.UE_LocalDataCachePath=str(output/'ue_v1/ddc')
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json'
    config.SPEAR.ENVIRONMENT_VARS.LD_PRELOAD=ue['network_guard_so']
    config.freeze()
    client.configure_system(config=config)
    try:instance=client.Instance(config=config)
    except BaseException:
        cleanup_failed_constructor(executable=Path(ue['unreal_editor']),temporary_directory=temporary)
        raise
    try:
        game=instance.get_game()
        sampling={'status':'not_requested'}
        if ue.get('unpause_animation_sampling'):
            with instance.begin_frame():
                statics=game.get_unreal_object(uclass='UGameplayStatics')
                statics.SetGamePaused(bPaused=False)
                paused=bool(statics.IsGamePaused())
                if paused:raise RuntimeError('native animation sampling world remains paused')
                sampling={'status':'pass','world_game_paused':paused,'timeline_phase_authority':'component.Stop then SetPosition; UE evaluates the sampled pose during the fixed native frame'}
            with instance.end_frame():pass
        return instance,game,{'temporary_directory':str(temporary),'rpc_port':port,'map_path':map_path,'animation_sampling':sampling}
    except BaseException:
        instance.close()
        raise


def camera_helpers():
    repository=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(repository/'tools/rooms'))
    from run_spear_residential_episode import _spawn_multimodal_camera,_depth_native,_rgb_bgr
    from run_spear_apartment_canary import _apply_camera,_actor_readback,_spawn_runtime_actors,_apply_actor_state,_destroy_runtime_actors
    return _spawn_multimodal_camera,_depth_native,_rgb_bgr,_apply_camera,_actor_readback,_spawn_runtime_actors,_apply_actor_state,_destroy_runtime_actors


def apply_camera(camera, view):
    position=view.get('position_ue_cm') or to_ue(view['position_m'])
    camera.K2_SetActorLocationAndRotation(
        NewLocation=dict(zip(('X','Y','Z'),position)),
        NewRotation={'Roll':0.,'Pitch':float(view.get('pitch_deg',0.)), 'Yaw':float(view.get('yaw_deg',0.))},
        bSweep=False,bTeleport=True)


def spawn_fill_lights(game, plan):
    """Use the retained v6 warm-indoor-fill point-light parameters everywhere.

    Placement follows each measured room's representative point. This is a
    declared visual production profile, independent of acoustic material truth.
    """
    from shapely.geometry import shape
    result=[]
    for room in plan['rooms']:
        point=shape(room['floor_polygon']).representative_point()
        pos=to_ue([point.x,room['source_floor_y_m']+2.5,point.y])
        actor=game.unreal_service.spawn_actor(uclass='APointLight',location=dict(zip(('X','Y','Z'),pos)))
        root=actor.K2_GetRootComponent();root.SetMobility(NewMobility='Movable')
        component=game.unreal_service.get_component_by_class(actor=actor,uclass='UPointLightComponent')
        component.SetIntensity(NewIntensity=350.)
        component.SetAttenuationRadius(NewRadius=450.)
        component.SetCastShadows(bNewValue=True)
        component.set_property_value(property_name='SourceRadius',property_value=15.)
        component.set_property_value(property_name='SoftSourceRadius',property_value=75.)
        component.set_property_value(property_name='bUseTemperature',property_value=True)
        component.set_property_value(property_name='Temperature',property_value=4000.)
        result.append({'room_id':room['room_id'],'position_ue_cm':pos,
            'intensity':float(component.get_property_value(property_name='Intensity')),
            'intensity_units':str(component.get_property_value(property_name='IntensityUnits')),
            'attenuation_radius_cm':float(component.get_property_value(property_name='AttenuationRadius')),
            'temperature_kelvin':4000.,'reference':'examples/rooms/spear_apartment_lighting_profiles.json warm_indoor_fill dining_soft_fill'})
    return result


def capture_view(instance,camera,components,view,warmup_frames=40):
    _spawn,depth_reader,rgb_reader,*_=camera_helpers()
    with instance.begin_frame():
        apply_camera(camera,view)
        for component in components.values():
            component.set_property_value(property_name='ProjectionType',property_value='Orthographic' if view.get('kind')=='orthographic' else 'Perspective')
            if view.get('kind')=='orthographic':
                component.set_property_value(property_name='bAutoCalculateOrthoPlanes',property_value=False)
                component.set_property_value(property_name='bUpdateOrthoPlanes',property_value=False)
                component.set_property_value(property_name='OrthoWidth',property_value=float(view['ortho_width_cm']))
    with instance.end_frame():pass
    # Real readback, rather than only UE ticks, drives streamed texture pages.
    for _ in range(warmup_frames):
        with instance.begin_frame():pass
        with instance.end_frame():rgb_reader(components['rgb'])
    with instance.begin_frame():pass
    with instance.end_frame():
        rgb=rgb_reader(components['rgb'])
        depth=depth_reader(components['depth'])
    return rgb,depth


def close_native(instance,camera,components):
    from avengine.backends.spear_ue.research_runtime import close_scene_capture
    if camera is not None and components:
        for name in ('object_ids','depth','rgb'):
            try:close_scene_capture(instance=instance,game=instance.get_game(),camera=None,capture=components[name])
            except Exception:pass
        try:
            with instance.begin_frame():camera.K2_DestroyActor()
            with instance.end_frame():pass
        except Exception:pass
    instance.close()


def apply_native_rgb_profile(capture,ue):
    """Apply one declared production camera exposure/white balance everywhere."""
    from avengine.capture.kujiale_native_exposure import is_histogram,apply_histogram_exposure
    if is_histogram(ue):return apply_histogram_exposure(capture)
    from avengine.backends.spear_ue.research_runtime import apply_capture_exposure
    result=apply_capture_exposure(capture,bias_ev=float(ue.get('exposure_bias_ev',2.)))
    if ue.get('white_balance_temp_k') is not None:
        temperature=float(ue['white_balance_temp_k'])
        capture.set_property_value(property_name='PostProcessSettings.bOverride_WhiteTemp',property_value=True)
        capture.set_property_value(property_name='PostProcessSettings.WhiteTemp',property_value=temperature)
        observed=float(capture.get_property_value(property_name='PostProcessSettings.WhiteTemp'))
        override=bool(capture.get_property_value(property_name='PostProcessSettings.bOverride_WhiteTemp'))
        if not override or abs(observed-temperature)>1e-5:raise ValueError('native white-balance readback differs')
        result.update(white_balance_temp_k=observed,white_balance_override=override)
    return result


def configure_scene_lighting(game,plan,ue):
    """New profile uses the measured imported design lights without room fills."""
    from avengine.capture.kujiale_native_exposure import is_histogram,require_authored_light_audit
    if not is_histogram(ue):return spawn_fill_lights(game,plan)
    audit=require_authored_light_audit(ue,plan['map_path'])
    return audit['light_components']
