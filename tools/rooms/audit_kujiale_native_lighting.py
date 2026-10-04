#!/usr/bin/env python3
"""Read authored USD lights and retained native room pixels without starting UE."""
from __future__ import annotations
import argparse, collections, csv, datetime, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def serial(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [serial(x) for x in value]
    except TypeError:
        return str(value)


def light_records(path):
    from pxr import Usd, UsdGeom, UsdLux
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f'USD cannot open: {path}')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records = []
    units = UsdGeom.GetStageMetersPerUnit(stage)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        api = UsdLux.LightAPI(prim)
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = np.asarray(matrix.ExtractTranslation()) * units
        attributes = {a.GetName(): serial(a.Get()) for a in prim.GetAttributes()
                      if a.GetName().startswith('inputs:') or a.GetName() in
                      {'intensity', 'exposure', 'color', 'colorTemperature', 'enableColorTemperature'}}
        intensity = float(api.GetIntensityAttr().Get())
        exposure = float(api.GetExposureAttr().Get())
        texture_assets = []
        for name in ['inputs:texture:file', 'inputs:shaping:ies:file']:
            value = prim.GetAttribute(name).Get()
            if value:
                texture_assets.append({'attribute': name, 'authored': value.path,
                                       'resolved': value.resolvedPath,
                                       'exists': bool(value.resolvedPath and Path(value.resolvedPath).is_file())})
        records.append({'prim_path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                        'world_matrix_usd': serial(matrix), 'position_usd_m': translation.tolist(),
                        'expected_native_ue_cm': (translation * 100).tolist(),
                        'canonical_position_m': translation[[0, 2, 1]].tolist(),
                        'inputs_intensity': intensity, 'inputs_exposure': exposure,
                        'effective_usd_intensity': intensity * 2 ** exposure,
                        'intensity_units': 'USD radiometric; UE native importer conversion required, not assumed lumens',
                        'legacy_intensity_conflicts': 'intensity' in attributes and attributes['intensity'] != intensity,
                        'authored_attributes': attributes, 'external_light_assets': texture_assets,
                        'visible': str(UsdGeom.Imageable(prim).ComputeVisibility()),
                        'old_clear_all_xforms_origin_displacement_m': float(np.linalg.norm(translation))})
    return records


def image_metrics(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot read native JPEG: {path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    return {'mean_gray_255': float(gray.mean() * 255), 'brightness': float(gray.mean()),
            'rms_contrast': float(gray.std()),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'resolution_hw': list(gray.shape),
            'definition': 'OpenCV BGR2GRAY, full 720p native perspective JPEG, mean/255 and population std/255; gates use retained uncompressed native metrics'}


def aggregate(values):
    if not values:
        return {'status': 'unmeasured', 'n': 0}
    array = np.asarray(values)
    return {'status': 'measured', 'n': len(array), 'mean': float(array.mean()),
            'quantiles': {str(q): float(np.quantile(array, q)) for q in [.05, .5, .95]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root not in args.output.resolve().parents:
        raise ValueError('audit output must stay in own task root')
    args.output.mkdir(parents=True, exist_ok=False)
    rooms = json.loads((root / 'ue_round7/delivery_v1/room_results.json').read_text())
    houses = sorted({r['house'] for r in rooms})
    light_rows = []
    for house in houses:
        audit_path = root / 'ue_round6/houses' / house / 'native_editor_audit.json'
        native_audit = json.loads(audit_path.read_text())
        plan = json.loads((root / 'ue_round6/houses' / house / 'acceptance_plan.json').read_text())
        raw_path = Path('/data/datasets/interioragent') / house / f'{house}.usda'
        derived_path = Path(plan['build_directory']) / 'engine_expanded/render.usdc'
        raw = light_records(raw_path)
        derived = light_records(derived_path)
        raw_lookup = {r['prim_path']: r for r in raw}
        mismatches = []
        for record in derived:
            original = raw_lookup.get(record['prim_path'])
            if original is None or any(original[key] != record[key] for key in
                                       ['world_matrix_usd', 'inputs_intensity', 'inputs_exposure', 'authored_attributes']):
                mismatches.append(record['prim_path'])
        missing = sorted(set(raw_lookup) - {r['prim_path'] for r in derived})
        bake_path = root / 'ue_round6/houses' / house / 'bake_receipt.json'
        baked_receipt = json.loads(bake_path.read_text())
        baked_path = Path(native_audit['source_usd'])
        baked = light_records(baked_path) if baked_path.exists() else None
        light_rows.append({'house': house, 'source_usd': str(raw_path), 'derived_usd': str(derived_path),
                           'raw_light_count': len(raw), 'derived_light_count': len(derived),
                           'light_types': dict(collections.Counter(r['type'] for r in raw)),
                           'missing_in_render_usd': missing, 'changed_in_render_usd': mismatches,
                           'old_bake_schema': baked_receipt['schema'],
                           'old_bake_cleared_xformable_count': baked_receipt.get('all_xformable_orders_cleared'),
                           'old_baked_stage_retained': baked is not None,
                           'old_baked_lights': baked,
                           'native_light_components': native_audit.get('light_components'),
                           'native_light_actor_presence': 'measured' if 'light_components' in native_audit else 'unverified: historical editor audit recorded meshes/materials only',
                           'raw_nonzero_light_world_positions': sum(r['old_clear_all_xforms_origin_displacement_m'] > 1e-5 for r in raw),
                           'lights': raw})
        print(house, len(raw), 'USD lights', flush=True)
    write(args.output / 'authored_lights.json', {'schema': 'kujiale_authored_lighting_audit_v1',
          'house_count': len(houses), 'authored_light_count': sum(r['raw_light_count'] for r in light_rows),
          'houses': light_rows, 'ue_runtime_verified': False})
    result_rows = []
    for row in rooms:
        qa = json.loads(Path(row['qa_result']).read_text())
        capture_directory = Path(qa['inputs']['capture'])
        capture = json.loads((capture_directory / 'result.json').read_text())
        room = next(r for r in capture['rooms'] if r['room_id'] == row['room_label'])
        selected = [v for v in room['views'] if v['view']['kind'] != 'orthographic']
        tested = [v for v in room.get('tested_views', room['views']) if v['view']['kind'] != 'orthographic']
        images = selected[:3] if len(selected) >= 3 else tested[:3]
        perspectives = [{'view_id': v['view']['view_id'], 'view': v['view'],
                         'path': v['path_720p'], 'metrics': image_metrics(v['path_720p']),
                         'frozen_gate_metrics': v['metrics']} for v in images]
        orthographic = next((v['path_720p'] for v in room.get('tested_views', room['views'])
                             if v['view']['kind'] == 'orthographic'), None)
        result_rows.append({'house': row['house'], 'room_id': row['room_label'], 'room_type': row['room_type'],
                            'admitted_before': row['room_training_admitted'], 'old_reasons': row['reasons'],
                            'qa_result_before': row['qa_result'], 'orthographic_before': orthographic,
                            'camera_views_before': perspectives,
                            'before_mean_gray_255': float(np.mean([v['metrics']['mean_gray_255'] for v in perspectives])) if perspectives else None,
                            'before_rms_contrast': float(np.mean([v['metrics']['rms_contrast'] for v in perspectives])) if perspectives else None,
                            'paired_view_rule': 'same 3 admitted frozen viewpoints, or first 3 tested original candidates for rejected room',
                            'after_mean_gray_255': None, 'after_rms_contrast': None,
                            'render_after_status': 'blocked_shared_disk_floor',
                            'admitted_after': None, 'formal_generation_ready': False})
    reference = json.loads(args.reference.read_text())['datasets']['STARSS23']['appearance']
    summary = {'schema': 'kujiale_native_exposure_comparison_v1', 'room_count': len(result_rows),
               'reference_path': str(args.reference), 'reference_only': 'aggregate appearance; no individual recording used',
               'before_brightness': aggregate([v['metrics']['brightness'] for r in result_rows for v in r['camera_views_before']]),
               'before_rms_contrast': aggregate([v['metrics']['rms_contrast'] for r in result_rows for v in r['camera_views_before']]),
               'reference_brightness': reference['brightness_v02_crop'],
               'reference_rms_contrast': reference['rms_contrast_v02_crop'],
               'after_brightness': aggregate([]), 'after_rms_contrast': aggregate([]),
               'historical_admitted_rooms': sum(r['admitted_before'] for r in result_rows),
               'new_profile_verified_admitted_rooms': None, 'rooms': result_rows,
               'sampling_limitation': 'native static 3 viewpoints/room versus STARSS 10860 video frame samples; no temporal weighting equivalence claimed',
               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(args.output / 'room_brightness_before_after.json', summary)
    with (args.output / 'room_brightness_before_after.csv').open('x') as stream:
        keys = ['house', 'room_id', 'room_type', 'admitted_before', 'before_mean_gray_255',
                'before_rms_contrast', 'after_mean_gray_255', 'after_rms_contrast', 'admitted_after',
                'render_after_status', 'old_reasons']
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in result_rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['rooms', 'reference_brightness', 'reference_rms_contrast']}, indent=2))


if __name__ == '__main__':
    main()
