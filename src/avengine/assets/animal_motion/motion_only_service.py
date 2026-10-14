"""Explicit motion-only request entry; audio remains a production responsibility."""
import json
from pathlib import Path
from avengine.assets.animal_motion.motion_only_policy import (
    POLICY_NAME, AUDIO_STATUS, validation_policy, qualify_motion, rigid_slide_surface_proof,
)
from avengine.assets.animal_motion.native import run_habitat
from avengine.assets.animal_motion.policy import select_motion, MotionUnavailable, attach_to_scene


def _evidence(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text())


def execute_motion_only_request(record, house, inputs, route, output, *, speed=.05,
                                seconds=4.8, reuse=None, policy=POLICY_NAME):
    """Choose qualified Walking first without computing or requiring an RIR.

    Per-mode reuse entries accept ``native``, ``surface`` and ``controls``.
    An optional old ``audio`` receipt is retained as history only. The existing
    strict entry remains the default and is not weakened by this entry.
    """
    from avengine.assets.animal_motion.service import validate_reused_route
    settings = validation_policy(policy)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    reuse = reuse or {}
    receipt_paths = {}
    attempts = {}

    def build(mode, actual_speed):
        if mode == 'walk_animation' and not record.get('has_walking'):
            raise MotionUnavailable('registered_target_has_no_skeleton_or_Walking')
        evidence = reuse.get(mode, {})
        directory = root / mode
        if evidence.get('native'):
            native_path = Path(evidence['native'])
            native = _evidence(native_path)
            if native['asset_id'] != record['asset_id'] or native['house'] != house['house'] or native['motion_mode'] != mode:
                raise ValueError('reused_native_asset_house_or_mode_differs')
            validate_reused_route(native, route, speed=actual_speed, seconds=seconds)
            if mode == 'walk_animation':
                key = next(k for k in ('motion_master_glb', 'motion_runtime_glb', 'walking_candidate_master') if k in record['files'])
                if Path(native['source_master']).resolve() != Path(record['files'][key]['path']).resolve():
                    raise ValueError('reused_native_motion_source_differs')
                if native.get('source_provenance', {}).get('explicit_motion_profile') != record.get('motion_profile'):
                    raise ValueError('reused_native_target_motion_profile_differs')
        else:
            native_path = directory / 'native_motion.json'
            native = run_habitat(record, house, inputs, route, directory,
                                 speed=actual_speed, seconds=seconds, mode=mode, physical_controls=True)
        if native.get('status') == 'failed' and native.get('failure_kind') != 'motion_rejected':
            raise RuntimeError('native_execution_error:' + str(native.get('error')))
        surface = _evidence(evidence.get('surface'))
        if surface is None and mode == 'slide_no_animation':
            surface = rigid_slide_surface_proof(native)
        controls = _evidence(evidence.get('controls'))
        qualification = qualify_motion(native, surface, controls, mode=mode)
        paths = {'native': str(native_path), 'surface': evidence.get('surface'),
                 'controls': evidence.get('controls'), 'audio_status': AUDIO_STATUS}
        if evidence.get('audio'):
            paths['historical_audio'] = str(evidence['audio'])
        receipt_paths[mode] = paths
        attempts[mode] = qualification
        return {'native': native, 'qualification': qualification, 'receipt_paths': paths}

    def qualify(payload, mode):
        result = payload['qualification']
        return {'status': 'pass' if result['status'] == 'pass' else 'fail',
                'failures': [*result.get('failures', []),
                             *['unverified:' + x for x in result.get('missing_evidence', [])]]}

    choice = select_motion(asset_id=record['asset_id'], requested_speed_m_s=speed,
                           attempt_animation=lambda value: build('walk_animation', value),
                           attempt_slide=lambda value: build('slide_no_animation', value),
                           qualify=qualify)
    receipt = {'asset_id': record['asset_id'], 'species': record['species'], 'house': house['house'],
               **choice.metadata(), 'status': 'pass' if choice.motion_mode != 'cannot_move' else 'blocked',
               'admission_scope': 'motion_only', 'validation_policy': settings,
               'audio_status': AUDIO_STATUS, 'audio_deferred_to_production': True,
               'strict_joint_audio_admission_claimed': False,
               'receipts': receipt_paths, 'motion_only_checks': attempts}
    if choice.payload:
        native = choice.payload['native']
        scene = {'sources': {'animal': {'asset_id': record['asset_id'],
                 'trajectory': [{'time_s': f['time_s'], 'position_m': f['mouth_world_m']} for f in native['frames']],
                 'native_frame_source': choice.payload['receipt_paths']['native'],
                 'animal_motion_only_evidence': choice.payload['receipt_paths']}},
                 'listener_m': native['listener_m'], 'camera_forward': native['camera_forward'],
                 'duration_s': native['frames'][-1]['time_s']}
        attach_to_scene(scene, 'animal', choice)
        scene['sources']['animal']['motion_metadata'].update(
            validation_policy=POLICY_NAME, admission_scope='motion_only',
            audio_status=AUDIO_STATUS, audio_deferred_to_production=True)
        # No unqualified animal_motion_audio binding is created here.
        (root / 'motion_scene.json').write_text(json.dumps(scene, ensure_ascii=False, indent=2))
    (root / 'admission.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt
