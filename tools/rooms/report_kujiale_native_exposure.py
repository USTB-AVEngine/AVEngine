#!/usr/bin/env python3
"""Export measured baseline, pending native exposure status and offline witnesses."""
from __future__ import annotations
import argparse, base64, csv, html, json
from pathlib import Path
import cv2
from avengine.capture.kujiale_native_exposure import PARAMETERS, PROFILE


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args(); root = args.root.resolve(); run = root/'ue_round9'; out = run/'delivery_v1'
    out.mkdir(parents=True, exist_ok=False)
    measured = json.loads((run/'audit_v1/room_brightness_before_after.json').read_text())
    lighting = json.loads((run/'audit_v1/authored_lights.json').read_text())
    resource = json.loads((run/'evidence/report_resource_audit.json').read_text())
    jobs = json.loads((run/'pending_native_v3/jobs.json').read_text())
    rooms = measured['rooms']; brightness = measured['before_brightness']; real = measured['reference_brightness']
    contrast = measured['before_rms_contrast']; real_contrast = measured['reference_rms_contrast']
    mean = lambda item: item['mean']*255
    q = lambda item, value: item['quantiles'][str(value)]*255
    conclusion = ('尚不能确认改后亮度是否贴合真实画面：48g /data 空闲仅'
                  f'{resource["data_free_bytes"]/1e9:.2f} GB，低于80 GB要求，44间UE重渲未启动；'
                  '灯位变换修复、统一自动曝光、旧画面统计和CPU回归已完成，新准入数量未验证。')
    lines = [conclusion, '',
             f'历史准入仍为18套29间，59/59资产和7/7人物接触是上一设置的验收结果。本轮不把这些直接改标为新设置通过。源码提交 `{args.commit}`，未push。', '',
             '## 亮度与对比度', '',
             '|统计（视角图均值分布）|旧UE|STARSS 85°总体|改后UE|',
             '|---|---:|---:|---|',
             f'|灰度均值 0–255|{mean(brightness):.2f}|{mean(real):.2f}|未测|',
             f'|灰度5%分位|{q(brightness,.05):.2f}|{q(real,.05):.2f}|未测|',
             f'|灰度95%分位|{q(brightness,.95):.2f}|{q(real,.95):.2f}|未测|',
             f'|RMS对比度均值（gray std/255）|{contrast["mean"]:.4f}|{real_contrast["mean"]:.4f}|未测|',
             f'|RMS对比度5–95%|{contrast["quantiles"]["0.05"]:.4f}–{contrast["quantiles"]["0.95"]:.4f}|{real_contrast["quantiles"]["0.05"]:.4f}–{real_contrast["quantiles"]["0.95"]:.4f}|未测|', '',
             '来源：`ue_round9/audit_v1/room_brightness_before_after.json`。旧图是129个保留透视视角，OpenCV BGR2GRAY后统计全720p JPEG；画面门槛仍取原未压缩RGB/深度的冻结检查。STARSS来自 `'+measured['reference_path']+'` 的 `/datasets/STARSS23/appearance/{brightness_v02_crop,rms_contrast_v02_crop}`，10860帧总体；未打开任何单段真实录音/视频，也未逐段或逐间拟合。静态视角与真实视频帧的采样权重不同，不宣称同分布检验已经通过。', '',
             '0031/R1仅保留2个透视视角，0066/R1仅1个；原冻结计划也不足3处。逐间表保留这一限制，不补造见证、不放宽3视角要求。12/129旧视角低于STARSS总体5%亮度分位；0030/R0有43.1、117.0、44.0三视角，不能用房间均值掩盖暗视角。', '',
             '## 灯具导入', '',
             '21套原USD共317个光源：SphereLight 194、RectLight 81、DistantLight 21、DomeLight 21；派生render.usdc中数量全部保留。21个Dome HDR只改派生路径，实际内容SHA256一致（`evidence/dome_hdr_relocation_sha.json`）。', '',
             '确定的缺陷在旧bake_v2：仅把mesh世界变换烘焙进顶点，然后清空所有Xform，灯位/朝向没有保存。296个原始光源有非零世界位置。保留的0009旧烘焙USD中19个光源的位置全为原点；其余20套已清掉该中间件，仅能据同版本脚本和bake receipt确认处理路径，不能宣称已读回20套UE灯位。历史native_editor_audit未记录LightComponent，实际UE内灯组件是否齐全目前未验证。', '',
             '修复：bake_v3保留每个灯的原组合世界矩阵及原强度/尺寸/色温/曝光；UE原生USD导入器负责辐射单位转换，未猜流明系数。导入端按原Y轴适配同时处理灯位置、yaw/roll，读回朝向、数量、位置和强度；新路径取消按房间中心添加的350 unitless旧补光，也不插入新的任意补光。只写隔离项目新NativeR9/ImportedNativeR9资产；本轮未启动编辑器，因此这些原生断言待实际执行。', '',
             '|房子|原USD光源|派生光源|类型|本轮UE状态|', '|---|---:|---:|---|---|']
    for house in lighting['houses']:
        lines.append(f'|{house["house"]}|{house["raw_light_count"]}|{house["derived_light_count"]}|'+', '.join(f'{k}:{v}' for k,v in house['light_types'].items())+'|待重导入/重渲|')
    lines += ['', '## 统一曝光与生成后端', '',
              f'命名profile：`{PROFILE}`。全部房间统一AEM_Histogram；EV100最小−4、最大16；histogram范围−10..20；百分位10%..90%；上调3、下调1 f-stop/s；补偿0 EV；白平衡4000 K；300帧/20秒RGB读回预热。1440p渲染→720p交付、85°水平FOV不变。参数是统一初始候选，尚未经真实UE重渲分布校准，不能称为已验证最终参数。', '',
              '`r.EyeAdaptationQuality 2`、`r.DefaultFeature.AutoExposure 1`、`r.DefaultFeature.AutoExposure.Method 0`、`r.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange 1`。RGB通过UE 5.5 `SetShowFlagSettings`启用EyeAdaptation，保持持续view state及final-color postprocess来源；不对深度/ID强制曝光，不处理输出RGB来制造亮图。依据48g UE 5.5 Scene.h、SceneCaptureComponent.h、SceneCaptureRendering.cpp及本检出SPEAR组件源码。', '',
              'NativeUEAlphaVideo的Kujiale路径自动绑定此profile，要求实际新灯光地图审计和同profile该房间新pixel/floor/geometry QA；旧EV3图或旧29间白名单不能冒充本轮通过。人物、声源原MIC、出题/选题/封存代码、真值、评分器和冻结门槛均未修改。源纹理sRGB色差仍不用于UE。', '',
              '## 每间改前/改后', '',
              '|房子/房间|旧灰度均值|旧3视角灰度（不足会注明）|旧RMS|历史准入|新灰度/新准入|',
              '|---|---:|---|---:|---|---|']
    for row in rooms:
        views = ', '.join(f'{v["metrics"]["mean_gray_255"]:.1f}' for v in row['camera_views_before'])
        lines.append(f'|{row["house"]}/{row["room_id"]}|{row["before_mean_gray_255"]:.2f}|{views}|{row["before_rms_contrast"]:.4f}|'+('通过' if row['admitted_before'] else '拒绝')+'|未测/待重验|')
    lines += ['', '## 验证、存储与剩余工作', '',
              '专用CPU测试6项通过（日志`logs/specific_v1.log`）；首次完整fast_unit为371通过、36跳过、7029未选择（`logs/full_fast_v1.log`）；最终修改后的复核见`logs/full_fast_v3.log`和`evidence/full_fast_v3.xml`。所有317个灯的矩阵与21个HDR内容已CPU核对；真实UE编译、灯组件API、EyeAdaptation读回、亮度/对比度变化均未验证。', '',
              f'结束时/data空闲{resource["data_free_bytes"]:,}字节，本任务唯一inode分配{resource["own_allocation"]["allocated_bytes"]:,}字节。没有任何UE/GPU渲染、新生成场景或CCI传输。本轮继续运行作业数量见资源审计，应为0。没有删除原始资产、旧交付、他人缓存或其他检出文件。', '',
              '恢复条件：共享盘至少80GB空闲；既有guard使用更保守80GiB+2GB缓冲，再计导入预留。需重新读取实时盘/显存，第一次UE要求≥16GiB，随后按实测峰值+4GiB；nvidia-smi -q -x逐秒按核对过的自有进程树采样。本轮显存峰值未测，上一轮3.9GB不能当新灯光版本实测值。', '',
              '顺序：先两套新灯光地图/曝光canary并验证原生API及几何；通过后顺序重渲21套44间、执行原finalize与原接触/对齐依据；按同一原判据发布新准入并比较总体亮度/RMS，再做新profile正式场景canary。两间不足3原候选的限制保留。旧拒绝房是否获救、新29间是否全保留、吞吐和生成预算变化均待实测。', '',
              '新请求：`ue_round9/pending_native_v3/jobs.json`完整列21套44间及每套bake/capture/finalize命令；pending_native与pending_native_v2是准备失败草稿，不用。原生导入仍通过已有own guard执行import_v7_kujiale_editor.py及AVENGINE_USD_LEVEL_IMPORT_REQUEST，不直接裸跑编辑器。', '',
              '离线页：`ue_round9/delivery_v1/BEFORE_AFTER_pending.html`，图片全部内嵌，每间一行；本轮只有真实旧图，改后列明确空缺。未发送owner。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    for name in ['room_brightness_before_after.json','room_brightness_before_after.csv']:
        (out/name).write_bytes((run/'audit_v1'/name).read_bytes())
    write(out/'admission_recheck_pending.json', {'profile':PROFILE,'historical_houses':18,'historical_rooms':29,
          'new_admitted_houses':None,'new_admitted_rooms':None,'formal_generation_ready':False,
          'reason':'44 rooms not re-rendered; shared /data below 80GB floor',
          'rooms':[{'house':r['house'],'room_id':r['room_id'],'admitted_before':r['admitted_before'],
                    'admitted_after':None,'status':'pending_native_recheck','old_reasons':r['old_reasons']} for r in rooms]})
    def image(path):
        if not path:return '<div class="missing">旧图未保存</div>'
        target=Path(path).resolve()
        if root not in target.parents:raise ValueError('witness outside own root')
        pixels=cv2.imread(str(target));pixels=cv2.resize(pixels,(384,216),interpolation=cv2.INTER_AREA)
        ok,data=cv2.imencode('.jpg',pixels,[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:raise ValueError('offline witness encoding failed')
        return '<img src="data:image/jpeg;base64,'+base64.b64encode(data).decode()+'" alt="旧UE原生图缩略见证">'
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>酷家乐统一曝光：待重渲</title><style>body{font:14px system-ui;margin:16px;background:#f5f5f5;color:#222}.row{background:white;padding:12px;margin:12px 0}.pictures{display:grid;grid-template-columns:repeat(8,minmax(130px,1fr));gap:5px;overflow:auto}.pictures img{width:100%}figure{margin:0;min-width:130px}.missing{background:#ddd;padding:30px 5px;color:#555;min-height:50px}figcaption{font-size:12px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #aaa}.pending{color:#a50;font-weight:700}</style><h1>酷家乐曝光重验（待原生重渲）</h1>',
          '<p class="pending">'+html.escape(conclusion)+'</p>',
          f'<p>源码提交 {args.commit}。历史18套/29间；新设置未验收。灰度均值旧UE {mean(brightness):.2f}，5–95% {q(brightness,.05):.2f}–{q(brightness,.95):.2f}；STARSS {mean(real):.2f}，{q(real,.05):.2f}–{q(real,.95):.2f}。RMS对比度旧 {contrast["mean"]:.4f}，真实 {real_contrast["mean"]:.4f}。只用总体参照，无逐录音/逐房调参。</p>',
          '<p>317个原USD光源；旧烘焙清空灯变换的缺陷已修。候选直方图曝光EV100 −4..16、补偿0、WB4000K、10–90%直方图、300帧预热；新图尚无。页面缩略图来自实际旧原生JPEG，不作为压缩前门槛重判。</p>']
    for row in rooms:
        page.append('<article class="row"><b>'+html.escape(row['house']+'/'+row['room_id'])+'</b> · 历史'+('通过' if row['admitted_before'] else '拒绝')+f' · 旧均值 {row["before_mean_gray_255"]:.2f} / RMS {row["before_rms_contrast"]:.4f} · <span class="pending">新图/新准入待验</span><div class="pictures">')
        page.append('<figure>'+image(row['orthographic_before'])+'<figcaption>旧俯视</figcaption></figure>')
        for i in range(3):
            v=row['camera_views_before'][i] if i<len(row['camera_views_before']) else None
            page.append('<figure>'+image(v['path'] if v else None)+'<figcaption>旧相机'+str(i+1)+(f'：灰度 {v["metrics"]["mean_gray_255"]:.2f} / RMS {v["metrics"]["rms_contrast"]:.4f}' if v else '：原候选/旧图不足')+'</figcaption></figure>')
        for label in ['新俯视','新相机1','新相机2','新相机3']:
            page.append('<figure><div class="missing">未启动重渲：磁盘&lt;80GB</div><figcaption>'+label+'</figcaption></figure>')
        page.append('</div></article>')
    page.append('</html>');page_text=''.join(page);(out/'BEFORE_AFTER_pending.html').write_text(page_text)
    print(json.dumps({'report':str(out/'REPORT_zh.md'),'offline_page':str(out/'BEFORE_AFTER_pending.html'),
                      'embedded_images':page_text.count('data:image/jpeg;base64,'),'new_native_images':0,'new_admission':'unverified'}))


if __name__=='__main__':main()
