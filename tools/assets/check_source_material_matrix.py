#!/usr/bin/env python3
"""Gate the complete source-material matrix and emit an offline review page.

This CPU-only check requires every inventoried source in every declared house,
checks native appearance results and immutable source/base-color bytes, and
returns nonzero on missing, offscreen, failing or inconsistent observations.
The page embeds native crop PNGs; no external resources or server are needed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from avengine.assets.source_materials import read_glb, texture_pixels


def verify_assets(inventory):
    verified = []
    for row in inventory['assets']:
        errors = []
        if row['status'] != 'prepared':
            verified.append({'asset_id': row['asset_id'], 'status': 'fail', 'errors': ['not_prepared']})
            continue
        source = Path(row['source']['path']); derived = Path(row['derived_path'])
        receipt = row['material_receipt']
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        derived_sha = hashlib.sha256(derived.read_bytes()).hexdigest()
        if source_sha != receipt['source_sha256']: errors.append('original_source_changed')
        if derived_sha != receipt['derived_sha256']: errors.append('derived_source_changed')
        a, bin_a = read_glb(source); b, bin_b = read_glb(derived)
        for key in ['meshes', 'nodes', 'skins', 'animations', 'accessors']:
            if a.get(key) != b.get(key): errors.append(key + '_changed')
        if not bin_b.startswith(bin_a): errors.append('original_bin_prefix_changed')
        if len(a['materials']) != len(b['materials']): errors.append('material_count_changed')
        else:
            for index, (m0, m1) in enumerate(zip(a['materials'], b['materials'], strict=True)):
                p0 = m0.get('pbrMetallicRoughness', {}); p1 = m1.get('pbrMetallicRoughness', {})
                if p0.get('baseColorFactor', [1]*4) != p1.get('baseColorFactor', [1]*4):
                    errors.append(f'material_{index}_base_color_factor_changed')
                t0 = p0.get('baseColorTexture'); t1 = p1.get('baseColorTexture')
                if t0 != t1: errors.append(f'material_{index}_base_color_binding_changed')
                elif t0 is not None and not np.array_equal(texture_pixels(a, bin_a, t0, source), texture_pixels(b, bin_b, t1, derived)):
                    errors.append(f'material_{index}_base_color_pixels_changed')
                for key in ['alphaMode', 'alphaCutoff', 'doubleSided']:
                    if m0.get(key) != m1.get(key): errors.append(f'material_{index}_{key}_changed')
        verified.append({'asset_id': row['asset_id'], 'status': 'fail' if errors else 'pass',
                         'source_sha256': source_sha, 'derived_sha256': derived_sha, 'errors': errors})
    return verified


def check_results(inventory, render_root, houses, result_name='result.json'):
    records = []; errors = []; views = {}
    for house in houses:
        for asset in inventory['assets']:
            asset_id = asset['asset_id']; folder = render_root/house/asset_id; path = folder/result_name
            if not path.is_file():
                errors.append({'house': house, 'asset_id': asset_id, 'error': 'missing_native_result'}); continue
            result = json.loads(path.read_text()); problems = []
            if result.get('asset_id') != asset_id or result.get('house') != house: problems.append('result_identity_differs')
            if result.get('status') != 'pass': problems.append('native_after_failed')
            for kind in [result['metrics']['after'], result['metrics']['after']['runtime_palette_check']]:
                if kind.get('status') != 'pass' or kind.get('visible_pixels', 0) < 100: problems.append('appearance_failed_or_not_run')
                if kind.get('failures'): problems.extend(kind['failures'])
            view = (result['camera_position'], result['camera_rotation'], result.get('object_position'))
            if house in views and view != views[house]: problems.append('house_camera_changed')
            views.setdefault(house, view)
            if result.get('scanned_background_changed_pixels') != 0: problems.append('scanned_background_changed')
            readbacks=result.get('phase_readbacks',{})
            if readbacks.get('after',{}).get('native_render_asset_fullpath') != asset.get('derived_path'):
                problems.append('native_derived_asset_not_loaded')
            if readbacks.get('before',{}).get('native_render_asset_fullpath') != asset.get('source',{}).get('path'):
                problems.append('native_original_asset_not_loaded')
            for phase in ['before', 'reference', 'after']:
                for name in ['rgb.png', 'mask.png', 'crop.png']:
                    image_path = folder/phase/name
                    if not image_path.is_file(): problems.append(f'missing_{phase}_{name}')
                    else:
                        with Image.open(image_path) as im:
                            if name == 'mask.png' and np.count_nonzero(np.asarray(im)) < 100:
                                problems.append(f'insufficient_{phase}_source_pixels')
                            if name != 'mask.png': im.verify()
            records.append({'house': house, 'asset_id': asset_id, 'result_path': str(path),
                            'folder': str(folder), 'errors': sorted(set(problems)), 'result': result})
            errors.extend({'house': house, 'asset_id': asset_id, 'error': p} for p in sorted(set(problems)))
    return records, errors


def offline_page(path, inventory, records, summary):
    escape = html.escape
    def embedded(p): return 'data:image/png;base64,' + base64.b64encode(Path(p).read_bytes()).decode()
    rows = []; assets = {a['asset_id']: a for a in inventory['assets']}
    for record in records:
        asset = assets[record['asset_id']]; result = record['result']; folder = Path(record['folder'])
        before = result['metrics']['before']; after = result['metrics']['after']
        bad_before = before['status'] != 'pass' or before['runtime_palette_check']['status'] != 'pass'
        label = '原先超门槛' if bad_before else '原先正常（本构图）'
        status = '通过' if not record['errors'] else '失败：' + ', '.join(record['errors'])
        images = []
        for phase, title in [('before', '修前'), ('after', '修后'), ('reference', '原贴图参考')]:
            p = folder/phase/'crop.png'
            images.append(f'<td><figure><img loading="lazy" src="{embedded(p)}" alt="{title}"><figcaption>{title}</figcaption></figure></td>' if p.is_file() else '<td>缺图</td>')
        def value(metric, key, percent=False):
            v = metric.get(key)
            return '未运行' if v is None else (f'{100*v:.3f}%' if percent else f'{v:.4f}')
        details = {'source': asset['source'], 'derived': asset['derived'],
                   'habitat_binding': asset['habitat'], 'runtime_materials': asset.get('runtime_materials'),
                   'material_profiles': asset['material_receipt']['profiles'],
                   'environment_reflections_after': False, 'native_result': result}
        rows.append(f'<tr data-house="{escape(record["house"])}" data-bad="{str(bad_before).lower()}"><td class="info"><b>{escape(record["asset_id"])}</b><p>{escape(record["house"])} · {escape(asset["entity_class"])}<br>{label} → {escape(status)}</p><small>高光 {value(before,"highlight_ratio",True)} → {value(after,"highlight_ratio",True)}<br>贴图 RGB RMSE {value(before,"texture_rgb_rmse")} → {value(after,"texture_rgb_rmse")}<br>色度 P95 {value(before,"texture_chromaticity_error_p95")} → {value(after,"texture_chromaticity_error_p95")}<br>有效像素 {after.get("visible_pixels",0)}；背景变动 {result.get("scanned_background_changed_pixels")}</small><details><summary>文件、格式、材质、门槛、原生读回</summary><pre>{escape(json.dumps(details,ensure_ascii=False,indent=2))}</pre></details></td>{"".join(images)}</tr>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>插入声源材质修前/修后原生对照</title><style>body{font:15px/1.55 system-ui,sans-serif;margin:24px;color:#17202a;background:#fafafa}h1{font-size:24px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #ddd;vertical-align:top}th{background:#edf2f7}.info{width:25%;overflow-wrap:anywhere}figure{margin:0}img{max-width:100%;width:300px;height:230px;object-fit:contain;background:#eee;cursor:zoom-in}figcaption{color:#566}details pre{white-space:pre-wrap;max-height:600px;overflow:auto;font-size:12px}label{margin-right:12px}input,select{padding:8px}dialog{max-width:95vw;border:0}dialog img{width:auto;height:auto;max-height:85vh;max-width:90vw}summary{cursor:pointer}header pre{white-space:pre-wrap}small{font-size:13px}</style><h1>插入声源材质：原生修前 / 修后 / 原贴图参考</h1><p>房子：MP3D 17DRP5sb8fy、HM3D TEEsavR23oF。每个房子固定相机和地面展示点，最长几何边归一化至 1 m（仅诊断，生产尺度不变）。图片均为当前安装 Habitat 管线的真实 RGB 输出，未人工补色或压低整体亮度。毛皮原色和花纹来自原 GLB；参考图使用独立 NONE 场景的 flat / no_lights 输出，直接读回原贴图 sRGB；原色校准结果见验收汇总。</p><p>高光指超过原贴图亮度的近白像素比例，毛皮门槛 2%，人皮肤/衣服 4%，普通道具 8%，含金属材质 12%；RGB RMSE 按 [0,1] 归一化、保留光照影响。色度差用 RGB 除以通道和以减少光照强度影响（P50≤0.12、P95≤0.28）；参考暗部的色度误差按 clip(参考通道均值 / 0.1, 0, 1)² 连续加权。非金属的中位绝对 RGB 误差门槛 0.35；金属原色代表镜面反射率，其 RGB 亮度差仅报告，仍检查色度与新增高光。渲染通道均值≤0.02 的近黑像素没有可定义的色度；有色原贴图的整件资产变黑仍会失败。白毛/白瓷原有白色不计作新增高光。有效源像素少于 100 时记未运行、矩阵验收失败；生产中遮挡样本保留未运行状态。</p><header><details open><summary>完整验收汇总</summary><pre>SUMMARY</pre></details></header><p><label>房子 <select id="house"><option value="">全部</option><option>mp3d</option><option>hm3d</option></select></label><label>筛选 <input id="query" placeholder="资产名"></label><label><input type="checkbox" id="bad">仅看原先超门槛</label></p><table><thead><tr><th>资产 / 数字 / 材质详情</th><th>修前</th><th>修后</th><th>原贴图参考</th></tr></thead><tbody>ROWS</tbody></table><dialog id="zoom"><button onclick="this.parentElement.close()">关闭</button><br><img alt="原生截图"></dialog><script>function filter(){const h=document.querySelector('#house').value,q=document.querySelector('#query').value.toLowerCase(),b=document.querySelector('#bad').checked;for(const r of document.querySelectorAll('tbody tr'))r.hidden=!!((h&&r.dataset.house!==h)||(q&&!r.textContent.toLowerCase().includes(q))||(b&&r.dataset.bad!=='true'))}for(const e of document.querySelectorAll('input,select'))e.addEventListener('input',filter);for(const i of document.querySelectorAll('tbody img'))i.onclick=()=>{const d=document.querySelector('#zoom');d.querySelector('img').src=i.src;d.showModal()};</script></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.replace('SUMMARY', escape(json.dumps(summary, ensure_ascii=False, indent=2))).replace('ROWS', ''.join(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--render-root', type=Path, required=True)
    parser.add_argument('--result-name', default='result.json')
    parser.add_argument('--reference-calibration', type=Path)
    parser.add_argument('--house', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--html', type=Path)
    args = parser.parse_args(); inventory = json.loads(args.inventory.read_text())
    records, errors = check_results(inventory, args.render_root, args.house, args.result_name)
    integrity = verify_assets(inventory)
    for item in integrity:
        errors.extend({'asset_id': item['asset_id'], 'error': e} for e in item['errors'])
    calibration = None
    if args.reference_calibration:
        calibration = json.loads(args.reference_calibration.read_text())
        if (calibration.get('max_channel_error') != 0 or calibration.get('pixels', 0) < 100
                or calibration.get('expected_srgb') != calibration.get('observed_median_srgb')):
            errors.append({'error': 'native_reference_color_calibration_failed'})
    before_bad = [r for r in records if r['result']['metrics']['before']['status'] != 'pass' or r['result']['metrics']['before']['runtime_palette_check']['status'] != 'pass']
    summary = {'status': 'fail' if errors else 'pass', 'inventory': str(args.inventory), 'render_root': str(args.render_root),
               'asset_count': len(inventory['assets']), 'houses': args.house,
               'result_name': args.result_name, 'reference_calibration': calibration,
               'expected_native_cases': len(inventory['assets'])*len(args.house), 'observed_native_cases': len(records),
               'before_failed_cases': len(before_bad), 'before_failed_asset_count': len({r['asset_id'] for r in before_bad}),
               'after_failed_cases': len([r for r in records if r['errors']]), 'errors': errors,
               'asset_integrity': integrity, 'native_results': [{k: v for k, v in r.items() if k != 'folder'} for r in records],
               'production_episode_counted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.html: offline_page(args.html, inventory, records, {k: v for k, v in summary.items() if k not in {'native_results', 'asset_integrity'}})
    print(json.dumps({k: v for k, v in summary.items() if k not in {'native_results', 'asset_integrity'}}, ensure_ascii=False))
    return int(bool(errors))


if __name__ == '__main__': raise SystemExit(main())
