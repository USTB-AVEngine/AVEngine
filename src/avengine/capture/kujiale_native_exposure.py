"""One native Kujiale histogram exposure profile; no per-room brightness fit."""
from __future__ import annotations
from copy import deepcopy
import math

PROFILE = 'kujiale-native-authored-lights-histogram-v1'
PARAMETERS = {
    'method': 'AEM_Histogram', 'min_ev100': -4.0, 'max_ev100': 16.0,
    'histogram_min_ev100': -10.0, 'histogram_max_ev100': 20.0,
    'low_percent': 10.0, 'high_percent': 90.0,
    'speed_up_f_stops_per_s': 3.0, 'speed_down_f_stops_per_s': 1.0,
    'exposure_bias_ev': 0.0, 'white_balance_temp_k': 4000.0,
    'warmup_frames': 300, 'fps': 15,
}


def is_histogram(ue):
    return ue.get('visual_profile') == PROFILE


def bind_histogram_profile(ue):
    """Ignore legacy EV3; this explicit named profile owns all exposure knobs."""
    result = deepcopy(ue)
    result.update(visual_profile=PROFILE, exposure_bias_ev=PARAMETERS['exposure_bias_ev'],
                  white_balance_temp_k=PARAMETERS['white_balance_temp_k'])
    return result


def exposure_console_commands(ue):
    if not is_histogram(ue):
        return ['r.EyeAdaptationQuality 0', 'r.DefaultFeature.AutoExposure 0']
    return ['r.EyeAdaptationQuality 2', 'r.DefaultFeature.AutoExposure 1',
            'r.DefaultFeature.AutoExposure.Method 0',
            'r.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange 1']


def apply_histogram_exposure(capture):
    """Set EV100 postprocess overrides and verify the actual component values.

    The launch commands must enable ExtendDefaultLuminanceRange; otherwise
    AutoExposureMin/MaxBrightness would be luminance, not the declared EV100.
    No output pixels or room/source materials are changed by this function.
    """
    capture_source = str(capture.get_property_value(property_name='CaptureSource')).split('::')[-1]
    if capture_source not in {'SCS_FinalColorLDR', 'SCS_FinalColorHDR', 'SCS_FinalToneCurveHDR'}:
        raise ValueError('histogram requires a native final-color postprocess capture source')
    flags = capture.get_property_value(property_name='ShowFlagSettings')
    if not isinstance(flags, list):
        raise ValueError('native capture show-flag array is unreadable')
    flags = [dict(row) for row in flags if row['ShowFlagName'] != 'EyeAdaptation']
    flags.append({'ShowFlagName': 'EyeAdaptation', 'Enabled': True})
    # Direct property writes do not refresh FEngineShowFlags. UE 5.5's
    # BlueprintSetter updates both settings and the actual capture show flags.
    # Apply to RGB only; never force exposure on native depth/ID passes.
    capture.SetShowFlagSettings(InShowFlagSettings=flags)
    read_flags = capture.get_property_value(property_name='ShowFlagSettings')
    if not any(row['ShowFlagName'] == 'EyeAdaptation' and row['Enabled'] for row in read_flags):
        raise ValueError('native RGB EyeAdaptation show-flag readback differs')
    properties = {
        'AutoExposureMethod': PARAMETERS['method'],
        'AutoExposureMinBrightness': PARAMETERS['min_ev100'],
        'AutoExposureMaxBrightness': PARAMETERS['max_ev100'],
        'HistogramLogMin': PARAMETERS['histogram_min_ev100'],
        'HistogramLogMax': PARAMETERS['histogram_max_ev100'],
        'AutoExposureLowPercent': PARAMETERS['low_percent'],
        'AutoExposureHighPercent': PARAMETERS['high_percent'],
        'AutoExposureSpeedUp': PARAMETERS['speed_up_f_stops_per_s'],
        'AutoExposureSpeedDown': PARAMETERS['speed_down_f_stops_per_s'],
        'AutoExposureBias': PARAMETERS['exposure_bias_ev'],
        'WhiteTemp': PARAMETERS['white_balance_temp_k'],
        'AutoExposureApplyPhysicalCameraExposure': False,
    }
    for name, value in properties.items():
        for suffix, item in [('bOverride_' + name, True), (name, value)]:
            capture.set_property_value(property_name='PostProcessSettings.' + suffix, property_value=item)
    for name, value in [('bAlwaysPersistRenderingState', True), ('bCaptureEveryFrame', True),
                        ('PostProcessBlendWeight', 1.0)]:
        capture.set_property_value(property_name=name, property_value=value)
    observed = {}
    for name, expected in properties.items():
        actual = capture.get_property_value(property_name='PostProcessSettings.' + name)
        override = capture.get_property_value(property_name='PostProcessSettings.bOverride_' + name)
        if isinstance(expected, bool):
            agrees = bool(actual) is expected
        elif isinstance(expected, str):
            agrees = str(actual).split('::')[-1] == expected
        else:
            agrees = math.isfinite(float(actual)) and abs(float(actual) - expected) <= 1e-5
        if not override or not agrees:
            raise ValueError(f'native histogram exposure readback differs: {name}')
        observed[name] = actual
    for name, expected in [('bAlwaysPersistRenderingState', True), ('bCaptureEveryFrame', True),
                           ('PostProcessBlendWeight', 1.0)]:
        actual = capture.get_property_value(property_name=name)
        if actual != expected:
            raise ValueError(f'native histogram capture state differs: {name}')
    return {'status': 'pass', 'profile': PROFILE, 'parameters': dict(PARAMETERS),
            'native_readbacks': observed, 'capture_source': capture_source,
            'native_rgb_eye_adaptation_show_flag': True, 'per_room_overrides': False,
            'parameter_calibration': 'UE 5.5 histogram defaults 10/90%, bounded indoor/outdoor EV100 -4..16; unified initial profile, real native distribution validation pending',
            'brightness_match_verified': False}


def require_authored_light_audit(ue, map_path):
    """A stale imported map cannot be relabeled as the new lighting profile."""
    import json
    from pathlib import Path
    if not is_histogram(ue):
        return None
    path = ue.get('native_lighting_audit')
    if not path:
        raise ValueError('histogram Kujiale requires measured authored-light import audit; legacy map is not qualified')
    root = Path(ue['output_root']).resolve()
    target = Path(path).resolve()
    if root not in target.parents:
        raise ValueError('authored-light audit must stay in own task root')
    audit = json.loads(target.read_text())
    if audit.get('status') != 'pass' or audit.get('map_path') != map_path:
        raise ValueError('authored-light map/readback audit failed or belongs to another map')
    if not audit.get('expected_light_count') or len(audit.get('light_components', [])) != audit['expected_light_count']:
        raise ValueError('authored-light component inventory is incomplete')
    if not audit.get('light_transform_assertions_passed') or audit.get('ad_hoc_fill_count') != 0:
        raise ValueError('authored lights require native transform assertion and no ad hoc fill')
    return audit


def require_histogram_room_visual_qa(ue, room_id, map_path):
    """Keep old EV3 room passes separate from this newly rendered profile."""
    import json
    from pathlib import Path
    value = ue.get('native_room_visual_qa')
    if not value:
        raise ValueError('new Kujiale profile requires this room rechecked with native pixels')
    root = Path(ue['output_root']).resolve(); path = Path(value).resolve()
    if root not in path.parents:
        raise ValueError('room visual QA must remain in own task root')
    qa = json.loads(path.read_text())
    capture_path = (Path(qa['inputs']['capture']) / 'result.json').resolve()
    if root not in capture_path.parents:
        raise ValueError('room capture escaped own task root')
    capture = json.loads(capture_path.read_text())
    room = next((r for r in qa['rooms'] if r['room_id'] == room_id), None)
    if room is None or room.get('automated_room_status') != 'pass':
        raise ValueError('room has not passed unchanged native visual/floor/geometry rules')
    if (capture.get('status') != 'native_pixels_and_floor_measured'
            or capture.get('exposure', {}).get('profile') != PROFILE
            or capture.get('launch', {}).get('map_path') != map_path):
        raise ValueError('room QA belongs to stale exposure or another native map')
    return {'status': 'pass', 'room_id': room_id, 'qa_path': str(path),
            'capture_path': str(capture_path), 'profile': PROFILE}
