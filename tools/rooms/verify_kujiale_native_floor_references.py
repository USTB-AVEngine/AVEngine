#!/usr/bin/env python3
"""Verify spatial support references against audited, unmodified UE LOD0."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from avengine.capture.native_floor_reference import NativeFloorReference, file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for candidate in read(root / 'acoustic_pass_train_candidates.json')['houses']:
        house = candidate['house']
        directory = root / 'ue_round6/houses' / house
        recovered = root / 'ue_round6/recovered_houses' / (house + '_fresh_v2')
        if (recovered / 'result.json').is_file() and read(recovered / 'result.json')['status'] == 'completed':
            directory = recovered
        qa = read(directory / 'qa/result.json')
        audit_path = Path(qa['inputs']['editor_audit'])
        geometry = read(audit_path)['native_render_geometry']
        declaration = {
            'native_editor_audit': str(audit_path),
            'native_editor_audit_sha256': file_sha256(audit_path),
            'native_vertices_sha256': file_sha256(geometry['world_vertices_ue_cm_f32']),
            'native_triangles_sha256': file_sha256(geometry['triangles_u32']),
        }
        surface = NativeFloorReference(declaration)
        rooms = []
        for room in qa['rooms']:
            row = {'room_id': room['room_id'], 'original_native_room_status': room['automated_room_status']}
            try:
                reference = room['floor_reference']
                row['validation'] = surface.validate_probes(reference)
                row['floor_reference'] = {
                    **reference,
                    'production_support_method': 'per-point native render LOD0 ray; median offset is diagnostic only',
                    'native_floor_surface': declaration,
                    'spatial_native_floor_probes': [
                        {'witness_index': p['witness_index'], 'role': p['role'],
                         'nav_point_m': p['nav_point_m'], 'native_render_surface_m': surface.sample(p['nav_point_m'])}
                        for p in reference['probes']
                    ],
                }
                row['status'] = 'pass'
            except (ValueError, KeyError) as error:
                row.update(status='fail', reason=str(error))
            rooms.append(row)
        record = {'house': house, 'native_floor_surface': declaration, 'rooms': rooms}
        write(args.output / (house + '.json'), record)
        records.append(record)
        print(house, [(r['room_id'], r['status']) for r in rooms], flush=True)
    result = {
        'schema': 'kujiale_native_spatial_floor_reference_v1', 'records': records,
        'house_count': len(records), 'room_count': sum(len(r['rooms']) for r in records),
        'geometry_and_navigation_modified': False,
    }
    write(args.output / 'result.json', result)
    with (root / 'PROGRESS_zh.md').open('a') as stream:
        stream.write('\n\n## ' + datetime.datetime.now(datetime.timezone.utc).isoformat() + ' 空间地板参考验证\n\n')
        stream.write('- 完成 ' + str(result['house_count']) + ' 套 / ' + str(result['room_count']) + ' 间；逐点 UE 实导出 LOD0 射线与原生深度的九个见证比较。保留 2 cm 接触误差门槛，原网格与导航网格未改。结果：' + str(args.output / 'result.json') + '。\n')


if __name__ == '__main__':
    main()
