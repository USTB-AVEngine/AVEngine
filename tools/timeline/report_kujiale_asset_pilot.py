#!/usr/bin/env python3
"""Write the local native Kujiale asset pilot's Chinese, image-embedded report."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,base64,html,io,json,os,re,subprocess
from PIL import Image
from avengine.qa.kujiale_pilot_truth import audit_sealed_parent

def read(path):return json.loads(Path(path).read_text())
def image(path):
    path=Path(path)
    if not path.is_file():return ''
    with Image.open(path) as pic:
        pic=pic.convert('RGB');pic.thumbnail((960,540));buffer=io.BytesIO();pic.save(buffer,format='JPEG',quality=78)
    return '<img loading="lazy" src="data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--name',default='REPORT_zh.html');a=p.parse_args();root=a.root.resolve()
    if root.name!='ue_round21_pilot':raise ValueError('unexpected pilot output')
    selected=read(root/'inputs/assets_selected.json')['assets'];byid={x['asset_id']:x for x in selected}
    expected_animals={(x['category'],x['body_id']) for x in selected if x['category']!='object'}
    expected_objects={x['object_type'] for x in selected if x['category']=='object'}
    parent_dirs=sorted((root/'production/parents').glob('*'));results=[];passed=[];partial=[];history=[];assets_seen=set();houses=set()
    events=Counter();attempt_events=Counter();programs={};partial_assets=set();joint_assets={};house_samples={};metrics=defaultdict(list);colours={};samples=[];candidate_colour_checks={};candidate_colour_rejections={};truth_forms=0
    for parent in parent_dirs:
        if not (parent/'input_recipe.json').exists():continue
        recipe=read(parent/'input_recipe.json');row=recipe['seed_row'];sid=row['scene_id']
        terminal=root/'evidence/parent_results'/f'{sid}.json'
        result=read(terminal) if terminal.exists() else {'scene_id':sid,'house':row['house'],'status':'unfinished','reason':'原生作业尚未结束'}
        seal=parent/'alpha_seal_completed.json';sealed=False
        if seal.exists():
            pointer=read(seal);delivery=parent/pointer['attempt']
            try:
                truth=audit_sealed_parent(parent,delivery)
                result=dict(result,status='formal_alpha_admitted',independent_truth={'status':'pass','checked_forms':truth['checked_forms']})
                sealed=True;passed.append(result);truth_forms+=truth['checked_forms'];houses.add(row['house'])
                for win in pointer['result']['windows']:
                    events['logical_questions']+=win['logical_questions'];events['windows']+=1
            except Exception as exc:
                result=dict(result,status='independent_truth_failed',reason=str(exc))
        for proof in parent.glob('visual*/physical_environment/admission.json'):
            data=read(proof)
            if data.get('status')=='pass':
                for node in data['entities'].values():
                    joint_assets[node['asset_id']]={'scene_id':sid,'house':row['house'],'proof':str(proof),
                        'views':sum(v['status']=='pass' for v in node['views']),'attachment_gap_mm':node['attachment']['gap_m']*1000,
                        'maximum_highlight_ratio':max(v['highlight_ratio'] for v in node['views'] if v['status']=='pass')}
        for folder in sorted(parent.glob('visual*_attempt*')):
            pngs=sorted(folder.glob('native_frame_*.png'))
            if pngs:house_samples[row['house']]=(sid+' / '+row['house'],pngs[len(pngs)//2],sealed)
        results.append(result)
        logical=re.search(r'pilot_(\d{6})',sid)
        if logical and (parent/'dry_completed.json').exists():programs[int(logical.group(1))]=recipe['recorded_program']
        for pointer in parent.glob('windows_*_completed.json'):
            directory=parent/read(pointer)['attempt']
            for candidate in directory.glob('*/candidate_audit.json'):
                gate=read(candidate).get('current_scene_colour_gate',{})
                for checked in gate.get('checks',[]):candidate_colour_checks[checked['question_id']]=checked
                for rejected in gate.get('rejected',[]):candidate_colour_rejections[rejected['question_id']]=rejected
        for pointer in parent.glob('visual_v*_completed.json'):
            directory=parent/read(pointer)['attempt'];visual=read(directory/'native_registered_visual_evidence.json')
            physical=visual.get('native_physical_environment',{})
            expected_emitters={n['asset_id']:n.get('expected_emitter_m') for n in row['production_scenario']['native_environment']}
            partial.append({'scene_id':sid,'visual':pointer.stem,'native_physical_status':physical.get('status'),'frame_count':visual.get('frame_count')})
            for standing in visual.get('standing_admission',[]):metrics['human_floor_gap_mm'].append(abs(standing['native_sole_depth']['gap_m'])*1000)
            for entry in read(directory/'acoustic_placement_admission.json')['checks']:
                if entry['status']=='pass':metrics['RIR_angle_deg'].append(entry['direction']['check']['angle_deg'])
            for node in physical.get('current_joint_admission',{}).values():
                metrics['attachment_gap_mm'].append(node['attachment']['gap_m']*1000)
                actual=node['emitter'].get('actual_m');expected=expected_emitters.get(node['asset_id'])
                if actual is not None and expected is not None:metrics['emitter_readback_error_mm'].append(max(abs(x-y) for x,y in zip(actual,expected))*1000)
                metrics['size_relative_error'].extend(node['size']['relative_errors'])
                metrics['complete_views'].append(sum(v['status']=='pass' for v in node['views']))
                for view in node['views']:
                    if view['status']=='pass':metrics['UE_highlight_ratio'].append(view['highlight_ratio'])
            for frame in physical.get('frames',[]):
                for entity in frame['entities']:
                    actual=entity.get('emitter_position_m');expected=expected_emitters.get(entity['asset_id'])
                    if actual is not None and expected is not None:metrics['emitter_readback_error_mm'].append(max(abs(x-y) for x,y in zip(actual,expected))*1000)
                    metrics['native_root_error_mm'].append(entity['root_error_m']*1000)
                    if entity.get('highlight_ratio') is not None:metrics['UE_highlight_ratio'].append(entity['highlight_ratio'])
                    if entity['visible_pixels']>=80:partial_assets.add(entity['asset_id'])
                    if sealed and entity['visible_pixels']>=80:assets_seen.add(entity['asset_id'])
                    if entity.get('color'):colours[entity['asset_id']]=entity['color']
            for check in visual['source_material_checks']['checks']:
                if check['metrics'].get('highlight_ratio') is not None:metrics['UE_highlight_ratio'].append(check['metrics']['highlight_ratio'])
            pngs=sorted(directory.glob('native_frame_*.png'))
            if pngs and len(samples)<12:samples.append((sid+' / '+row['house'],pngs[len(pngs)//2],sealed))
        if sealed:
            program=recipe['recorded_program'];records={x['sound_asset_id']:x for x in program['records']}
            for event in program['events']:events[records[event['sound_asset_id']].get('sound_class') or 'human_speech']+=1
        # Native joint view evidence survives even when a later parent gate failed.
        if not sealed:
            for proof in parent.glob('visual_v*_attempt*/physical_environment/admission.json'):
                data=read(proof)
                for ep,node in data['entities'].items():
                    if len(samples)<12 and node.get('views'):samples.append((sid+' / '+node['asset_id']+'（未封存）',Path(node['views'][0]['rgb']),False))
        if result.get('status')!='formal_alpha_admitted':history.append((sid,row['house'],result.get('reason','未完成')))
    failed_measurements=[];all_measurements=defaultdict(list)
    for proof in root.glob('production/parents/*/visual*/acoustic_placement_admission.json'):
        for check in read(proof).get('checks',[]):
            value=check.get('direction',{}).get('check',{}).get('angle_deg')
            if value is not None:
                all_measurements['RIR_angle_deg'].append(value)
                if check.get('status')!='pass':failed_measurements.append({'scene_id':proof.parent.parent.name,'check':'RIR direction','value':value,'limit':5.,'source':str(proof)})
    for proof in root.glob('production/parents/*/visual*/source_material_checks.json'):
        for check in read(proof).get('checks',[]):
            value=check.get('metrics',{}).get('highlight_ratio')
            if value is not None:
                all_measurements['UE_highlight_ratio'].append(value)
                if value>.02:failed_measurements.append({'scene_id':proof.parent.parent.name,'check':'UE highlight','value':value,'limit':.02,'source':str(proof)})
    candidate_audit=root/'evidence/candidate_truth_audit_v1.json'
    candidate_truth=read(candidate_audit) if candidate_audit.exists() else {}
    seen_animals={(byid[x]['category'],byid[x]['body_id']) for x in assets_seen if x in byid and byid[x]['category']!='object'}
    seen_objects={byid[x]['object_type'] for x in assets_seen if x in byid and byid[x]['category']=='object'}
    unique_logical=set()
    for result in passed:
        match=re.search(r'pilot_(\d{6})',result['scene_id'])
        if match:unique_logical.add(int(match.group(1)))
    gaps={'animal_bodies':sorted(expected_animals-seen_animals),'object_types':sorted(expected_objects-seen_objects)}
    for program in programs.values():
        records={x['sound_asset_id']:x for x in program['records']}
        for event in program['events']:attempt_events[records[event['sound_asset_id']].get('sound_class') or 'human_speech']+=1
    jobs=read(root/'JOBS.json');gpu_seconds=sum(j.get('gpu_allocated_seconds',0) for j in jobs)
    now=datetime.now(timezone.utc);intervals=defaultdict(list)
    for job in jobs:
        if job.get('gpu_physical') not in (2,3):continue
        start=datetime.fromisoformat(job['started_utc']).timestamp()
        end=datetime.fromisoformat(job['finished_utc']).timestamp() if job.get('finished_utc') else now.timestamp()
        intervals[job['gpu_physical']].append((start,end))
    gpu_card_seconds={}
    for gpu,ranges in intervals.items():
        merged=[]
        for start,end in sorted(ranges):
            if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
            else:merged.append([start,end])
        gpu_card_seconds[str(gpu)]=sum(end-start for start,end in merged)
    own_bytes=0;allocated=0;inodes=set()
    for folder,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        for filename in files:
            path=Path(folder)/filename
            try:stat=path.lstat()
            except FileNotFoundError:continue
            if path.is_symlink():continue
            key=(stat.st_dev,stat.st_ino)
            if key in inodes:continue
            inodes.add(key);own_bytes+=stat.st_size;allocated+=stat.st_blocks*512
    unit_files=sorted((root/'evidence').glob('unit_all_*.xml'),key=lambda p:p.stat().st_mtime)
    unit={}
    if unit_files:
        import xml.etree.ElementTree as ET
        suite=ET.parse(unit_files[-1]).getroot().find('testsuite')
        unit=dict(suite.attrib);unit['source']=str(unit_files[-1])
    summaries={k:{'count':len(values),'maximum':max(values),'minimum':min(values)} for k,values in metrics.items() if values}
    ready=len(unique_logical)>=12 and len(houses)>=6 and not gaps['animal_bodies'] and not gaps['object_types'] and unit.get('failures')=='0' and unit.get('errors')=='0'
    conclusion=('通过：酷家乐动物、物件和人物同场试产完成。' if ready else '未通过完整试产验收；当前产物不具备接入量产条件。')+f' 全部门禁通过父场景 {len(unique_logical)}/12，训练房子 {len(houses)}/6。'
    summary={'status':'pass' if ready else 'not_ready','conclusion':conclusion,'parents_passed':len(unique_logical),'parents_requested':12,'houses_passed':sorted(houses),
             'passed_parents':passed,'attempt_results':results,'coverage':{'selected_appearances':len(selected),'native_visual_stage_appearances_seen':len(partial_assets),'sealed_appearances_seen':len(assets_seen),'animal_bodies_seen':len(seen_animals),'animal_bodies_expected':len(expected_animals),'object_types_seen':len(seen_objects),'object_types_expected':len(expected_objects),'gaps':gaps},
             'metrics':summaries,'failed_measurements':failed_measurements,'all_attempt_measured_maxima':{k:max(v) for k,v in all_measurements.items() if v},'candidate_truth':{k:candidate_truth.get(k) for k in ['status','scope','checked_forms','passed_forms','mismatches','errors']},'independent_truth_checked_forms':truth_forms,'independent_truth_mismatches':sum(len(read(p)['mismatches']) for p in root.glob('production/parents/*/independent_truth_audit.json')),
             'colour_checked_forms':len(candidate_colour_checks),'colour_discarded_forms':len(candidate_colour_rejections),'colour_discarded_logical_questions':len({qid.rsplit('__',1)[0] for qid in candidate_colour_rejections}),
             'event_counts_completed_dry_base_programs_unsealed':dict(attempt_events),'event_counts_sealed_parent_base_programs':dict(events),'unit_tests':unit,'gpu_assigned_card_union_seconds':gpu_card_seconds,'gpu_allocated_seconds_summed':gpu_seconds,'gpu_time_definition':'sum of assigned-card worker wall time, including startup, RIR, encoding and question work; not CUDA kernel timing',
             'own_output_logical_bytes':own_bytes,'own_output_allocated_bytes':allocated,'active_jobs':[j['name'] for j in jobs if j['status']=='running'],'source_labels_modified':False,'training_table_exported':False,'CCI_transferred':False}
    summary['joint_scene_physical_admission_appearances']=joint_assets
    summary['joint_scene_physical_admission_count']=len(joint_assets)
    summary['implementation_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    summary['camera_root_cause_evidence']=str(root/'evidence/camera_root_cause_v2.json')
    summary['audio_input_seal']=read(root/'evidence/input_seal.json')
    preferred=list(house_samples.values());paths={str(x[1]) for x in preferred}
    samples=(preferred+[x for x in samples if str(x[1]) not in paths])[:12]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    summary_path=root/'evidence'/('pilot_report_'+stamp+'.json');summary_path.open('x').write(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    rows=[['不可见声源',0 if passed else '未形成通过父场景','只对封存且独立真值通过的父场景成立'],
          ['附件贴附 ≤15 mm',summaries.get('attachment_gap_mm',{}).get('maximum','未验证'),'已完成原生视觉阶段的最大值'],
          ['人物房间地面 ≤15 mm',summaries.get('human_floor_gap_mm',{}).get('maximum','未验证'),'地板基准和语义支持面检查'],
          ['发声点读回',summaries.get('emitter_readback_error_mm',{}).get('maximum','未验证'),'逐帧原生组件读回；单位 mm'],
          ['RIR 方向 ≤5°',summaries.get('RIR_angle_deg',{}).get('maximum','未验证'),'通过位置最大角误差'],
          ['UE 高光 ≤2%',summaries.get('UE_highlight_ratio',{}).get('maximum','未验证'),'比率；已完成视觉阶段'],
          ['尺寸误差 ≤15%',summaries.get('size_relative_error',{}).get('maximum','未验证'),'原生 LOD0 与登记尺寸'],
          ['完整视角 ≥3',summaries.get('complete_views',{}).get('minimum','未验证'),'每个同场资产的完整视角数'],
          ['独立真值重算',truth_forms,'封存保留形式数'],
           ['候选池独立真值重算',candidate_truth.get('checked_forms','未验证'),'候选池通过不代表父场景封存通过'],
           ['当场颜色重测',len(candidate_colour_checks),'检查的候选题目形式数'],
          ['颜色不一致丢题',len(candidate_colour_rejections),'形式数；候选池，标签不改']]
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>酷家乐 UE 原生小批试产</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#172238;margin:24px auto;max-width:1280px;padding:0 20px}h1{font-size:26px}.top{background:#fff1d6;padding:20px;border-radius:12px}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{padding:9px;border:1px solid #d7dfeb;overflow-wrap:anywhere}th{background:#eaf0f9}img{max-width:100%;border-radius:8px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}figure{margin:0;background:white;padding:12px}code{overflow-wrap:anywhere}.note{color:#465875}</style><body>'
    page+='<h1>酷家乐 UE 原生：动物、物件与人物同场</h1><p class="top">'+html.escape(conclusion)+'</p>'
    page+='<p class="note">仅本机非商业研究试产。没有导出训练表，没有传 CCI，没有修改源登记。动物及物件静态出场；空调为静音摆设。失败阶段保留，原门槛未放宽。</p><h2>原生门禁数字</h2>'+table(['项目','实测 / 结论','数字范围与出处'],rows)
    page+='<p>详细数值来源：<code>'+html.escape(str(summary_path))+'</code>。未形成全部门禁通过的父场景时，部分阶段的通过数字不代表父场景已通过。</p>'
    page+='<h2>资产覆盖</h2>'+table(['范围','封存父场景实际出场','要求'],[['动物身体',len(seen_animals),len(expected_animals)],['物件种类',len(seen_objects),len(expected_objects)],['代表外观',len(assets_seen),len(selected)],['训练房子',len(houses),6]])
    page+='<p>同场物理验收证据已覆盖 '+str(len(joint_assets))+' 个代表外观；它只说明三视角与贴附阶段通过，不代替完整视频、题目与封存。</p>'+table(['资产','类别 / 身体或种类','同场物理阶段','完整视频出场','封存出场'],
        [[x['asset_id'],x['body_id'] if x['category']!='object' else x['object_type'],('三视角通过；贴附 '+format(joint_assets[x['asset_id']]['attachment_gap_mm'],'.3f')+' mm') if x['asset_id'] in joint_assets else '未完成',x['asset_id'] in partial_assets,x['asset_id'] in assets_seen] for x in selected])
    page+='<h3>覆盖缺口</h3><pre>'+html.escape(json.dumps(gaps,ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>每类事件数</h2><p>统计封存父场景的基础事件程序一次；四成员的反事实复用不重复计数。</p>'+table(['事件类别','事件数'],sorted(events.items()))
    page+='<p>已完成但未封存的干音程序，按逻辑父场景去重；以下数字不代表音频与全部门禁通过。</p>'+table(['事件类别','干音事件数'],sorted(attempt_events.items()))
    page+='<h2>失败清单</h2>'+table(['父场景 / 尝试','房子','失败原因'],history)
    page+='<h3>失败尝试的实测超限</h3>'+table(['父场景','项目','实测值','门槛','出处'],[(v['scene_id'],v['check'],v['value'],v['limit'],v['source']) for v in failed_measurements])
    page+='<h2>UE 原生样张</h2><div class="gallery">'+''.join('<figure>'+image(path)+'<figcaption>'+html.escape(label)+('；封存通过' if sealed else '；部分阶段 / 未通过全部封存门禁')+'</figcaption></figure>' for label,path,sealed in samples)+'</div>'
    page+='<h2>相机故障与修复</h2><p>原因是验收站位后给下发相机添加了偏航偏移，继承的读回检查仍按原时间表计算期望。已使两边使用同一偏航计划，保留 0.0001° 和 0.5 mm 容差。后续单场的读回错误未重现，但旧视角导致一个说话人偏离主视角 67.12°，有效绑定题 0/5；这些完成阶段与失败都保留。当前改用水平角平分，并限制两名说话人的相对站位，继续单场验收。</p><p>故障证据：<code>'+html.escape(str(root/'evidence/camera_root_cause_v2.json'))+'</code>。</p>'
    page+='<h2>验证及资源</h2><pre>'+html.escape(json.dumps({'unit_tests':unit,'GPU卡时间并集小时':{g:s/3600 for g,s in gpu_card_seconds.items()},'GPU实例累计墙钟小时':gpu_seconds/3600,'新增产物逻辑GiB':own_bytes/2**30,'新增产物分配GiB':allocated/2**30,'仍在运行':summary['active_jobs']},ensure_ascii=False,indent=2))+'</pre>'
    page+='<p>颜色规则：在当前场景画面中重测。对不上只丢掉依赖颜色的题，资产照常准入，登记标签保持原样。数据许可限非商业研究，不再分发。</p></body></html>'
    path=root/a.name
    if path.exists():raise ValueError('completed report is not overwritten; choose a new report filename')
    path.write_text(page)
    with (root/'PROGRESS.md').open('a') as f:f.write('- '+datetime.now(timezone.utc).isoformat()+'：生成中文内嵌图报告 '+str(path)+'；'+conclusion+'\n')
    print(json.dumps({'html':str(path),'summary':str(summary_path),'conclusion':conclusion},ensure_ascii=False))
if __name__=='__main__':main()
