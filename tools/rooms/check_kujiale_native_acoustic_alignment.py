#!/usr/bin/env python3
"""Compare actual native LOD0 and final acoustic geometry at frozen room poses."""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if root not in output.parents:
        raise ValueError('alignment output must stay in this task root')
    output.mkdir(parents=True, exist_ok=False)
    from tools.rooms.prepare_v7_kujiale_native_generation import geometry_support_alignment

    overrides = {(row['house'], row['room_id']): row for row in
                 read(root/'ue_round6/native_room_overrides.json')['records']}
    houses = [row['house'] for row in read(root/'acoustic_pass_train_candidates.json')['houses']]
    started = time.monotonic()
    records = []
    for house in houses:
        directory = root/'ue_round6/houses'/house
        recovered = root/'ue_round6/recovered_houses'/(house+'_fresh_v2')
        if (recovered/'result.json').is_file() and read(recovered/'result.json')['status'] == 'completed':
            directory = recovered
        if read(directory/'result.json')['status'] != 'completed':
            raise ValueError('native import/capture/QA not completed: '+house)
        plan = read(directory/'acceptance_plan.json')
        audit = read(directory/'native_editor_audit.json')
        qa = {row['room_id']: row for row in read(directory/'qa/result.json')['rooms']}
        selected = []
        for room in plan['rooms']:
            key = (house, room['room_id'])
            current = qa[room['room_id']]
            if key in overrides:
                current = next(r for r in read(overrides[key]['qa_result'])['rooms']
                               if r['room_id'] == room['room_id'])
            witnesses = current['selected_witnesses'][:3]
            if len(witnesses) < 3:
                # Rejected views still have frozen placement candidates. They
                # are geometry witnesses only, and never become visual passes.
                witnesses = [{'index': i, 'placement': w} for i, w in
                             enumerate(room['witnesses'][:3])]
            selected.append({'room_id': room['room_id'],
                             'selected_witnesses': witnesses})
        result = geometry_support_alignment(plan, audit, selected)
        rows = []
        for room in selected:
            checks = [c for c in result['checks'] if c['room_id'] == room['room_id']]
            passed = len(checks) >= 9 and all(c['status'] == 'pass' for c in checks)
            rows.append({'room_id': room['room_id'], 'status': 'pass' if passed else 'fail',
                         'placement_count': len(room['selected_witnesses']),
                         'camera_and_two_emitters': True, 'checks': checks,
                         'max_surface_distance_error_m': max(
                             (c['max_surface_distance_error_m'] for c in checks), default=None),
                         'surface_hit_presence_mismatches': sum(
                             c['native_acoustic_hit_presence_mismatch'] for c in checks)})
        record = {'house': house, 'status': 'pass' if all(r['status'] == 'pass' for r in rows) else 'fail',
                  'rooms': rows, 'native_audit': str(directory/'native_editor_audit.json'),
                  'acoustic_manifest': result['acoustic_manifest'],
                  'source_acoustic_vertex_max_error_m': result['source_acoustic_vertex_max_error_m'],
                  'tolerance_m': result['tolerance_m'], 'alignment_fitting_used': False}
        (output/(house+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
        print(house, record['status'], [(r['room_id'], r['status']) for r in rows], flush=True)
        (output/'live_result.json').write_text(json.dumps({'records': records}, indent=2)+'\n')
    result = {'schema': 'kujiale_native_acoustic_support_alignment_v1', 'records': records,
              'house_count': len(records), 'room_count': sum(len(r['rooms']) for r in records),
              'passed_room_count': sum(r['status'] == 'pass' for h in records for r in h['rooms']),
              'method': 'same retained 0.5 mm rule, three frozen placements per room, camera/two emitters, 26 surrounding rays each',
              'geometry_or_navigation_or_frozen_thresholds_modified': False,
              'elapsed_seconds': time.monotonic()-started}
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    with (root/'PROGRESS_zh.md').open('a') as stream:
        stream.write('\n\n## '+datetime.datetime.now(datetime.timezone.utc).isoformat()+
                     ' 全候选房的原生/声学几何三摆位断言完成\n\n- '+
                     str(result['house_count'])+'套 / '+str(result['room_count'])+'间，'+
                     str(result['passed_room_count'])+'间通过；复用旧0.5mm规则，未拟合坐标，'+
                     '视觉失败房仍保持失败。来源 '+str(output/'result.json')+'。\n')


if __name__ == '__main__':
    main()
