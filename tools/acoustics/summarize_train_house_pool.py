#!/usr/bin/env python3
"""Prepare MP3D acoustic samples and report CPU training houses in batch1 schema.

Historical SO/M1 exposure is a descriptive training attribute. Original clean
HM3D validation houses and live development/test recommendations are excluded.
The six original MP3D diagnoses are retained without retesting or promotion.
"""
from __future__ import annotations
import argparse, copy, csv, json, subprocess, sys, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.accept_houses import choose_rooms, compact_room
from tools.acoustics.summarize_house_acceptance import collect

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)

@cache
def exposure_inputs(output):
    return (read(output/'evidence/inputs/inventory.json'), read(output/'evidence/exposure_audit.json'), set(read(output/'evidence/inputs/clean.json')))

def exposure(output,h):
    inv,m1,clean=exposure_inputs(output)
    sid=h['scan_id'];family=h['family']
    seen=sid in m1['m1_seen_by_scan_id']
    return dict(so_seen=sid in inv['so_overlap']['houses'][family],m1_seen=seen,m1_seen_scope=m1['scope'],m1_sources=m1['m1_seen_by_scan_id'].get(sid,[]),m1_other_historical_runs_verified=False,v6_seen=(h['house'] in inv['our_house_exposure']['physical_house_evidence'] or 'habitat_mp3d_example_'+sid in inv['our_house_exposure']['physical_house_evidence']),clean_curated=h['house'] in clean,so_anonymous_identity_verified=False)

def mp_houses(output):
    roomroot=output/'mp3d_rooms_final';inputs=read(roomroot/'inputs.json');houses=[]
    for house in inputs['houses']:
        d=read(roomroot/'houses'/(house+'.json'));rs=d['rows'];eligible=[r for r in rs if r['stage1']['status']=='pass'];r=rs[0];sid=house[len('mp3d_'):]
        h=dict(house=house,family='mp3d',scan_id=sid,original_split=None,scene_directory=r['scene_directory'],semantic_source=r['semantic_source'],annotation_source=r['annotation_source'],navmesh_source=r['navmesh_source'],candidate_room_count=len(eligible),candidate_area_sum_m2=sum(x['metrics']['floor_area_m2'] for x in eligible),room_types=sorted({x['room_type'] for x in eligible}),floor_count_proxy=len({x['native_region']['level_index'] for x in rs}),known_exposure=False,acoustic_rooms=[compact_room(x) for x in choose_rooms(eligible)],registry_source=str(roomroot/'rooms_registry.jsonl'),candidate_room_labels=[r['room_label'] for r in eligible],room_rule_counts=dict(Counter(r['stage1']['status'] for r in rs)),room_measurement_status=d['status'],room_measurement_error=d.get('error'),human_review_status='待 smy 审房间',split='train',house_gate=dict(status='not_run',reason='No historical whole-house visual production gate; task is CPU selection and RLR diagnosis'))
        h['exposure']=exposure(output,h);h['known_exposure']=h['exposure']['so_seen'] or h['exposure']['m1_seen'] or h['exposure']['v6_seen'];houses.append(h)
    return houses

def prepare_mp3d(a):
    original=read(a.output/'evidence/inputs/batch1_manifest.json');previous={h['house'] for h in original['houses'] if h['family']=='mp3d'}
    houses=mp_houses(a.output)
    selection={'houses':houses,'house_count':len(houses),'room_count':sum(h['candidate_room_count'] for h in houses),'prior_six_reused':sorted(previous),'sample_method':'unchanged small/large/type-diverse choose_rooms; <=3 before acoustic observation','no_model_scores_read':True}
    save(a.output/'mp3d_selection.json',selection)
    original['houses']=[h for h in houses if h['acoustic_rooms'] and h['house'] not in previous]
    original['selection']={'method':'all new MP3D room-rule passes, no exposure filter; six first-batch results retained','selected_before_acoustic_measurement':True}
    save(a.output/'mp3d_manifest.json',original)
    print('MP3D_ACOUSTIC_CANDIDATES',len(original['houses']),'ROOM_RULE_PASS_HOUSES',sum(h['candidate_room_count']>0 for h in houses),'ROOMS',selection['room_count'],flush=True)

def visual_mp3d(a):
    from avengine.acoustics.house_acceptance_visual import load_raw_scan
    from avengine.acoustics.house_acceptance_geometry import valid_native_triangles
    m=read(a.output/'mp3d_manifest.json');root=a.output/'mp3d_acoustics'
    def one(h):
        d=root/'houses'/h['house'];r=read(d/'result.json')
        if not r.get('foa_checks'):return h['house'],'native_input_review'
        mesh,_,_=load_raw_scan(Path(h['scene_directory'])/(h['scan_id']+'.glb'))
        invalid=int((~valid_native_triangles(mesh.vertices,mesh.faces)).sum())
        if invalid:
            save(d/'visual_checks.json',{'status':'review','reasons':['RAW_VISUAL_ZERO_AREA_TRIANGLES; no face removal authorized'],'raw_zero_area_triangle_count':invalid,'compute_device':'CPU'})
            return h['house'],'raw_input_review'
        cmd=[sys.executable,str(REPO/'tools/acoustics/accept_houses.py'),'visual','--manifest',str(a.output/'mp3d_manifest.json'),'--output',str(root),'--house',h['house']]
        with (a.output/'logs'/('visual_'+h['house']+'.log')).open('x') as f:
            cp=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        return h['house'],cp.returncode
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for item in pool.map(one,m['houses']):print('MP3D_VISUAL',item,flush=True)
    save(a.output/'evidence/mp3d_complete.json',{'complete':True,'new_houses':len(m['houses'])})

@cache
def hm_room_index(output):
    by={}
    for r in read(output/'hm3d_candidate_rooms.json'):by.setdefault(r['house'],[]).append(r)
    return by

def candidate_rooms(output,h):
    if h['family']=='hm3d':return hm_room_index(output).get(h['house'],[])
    labels=set(h['candidate_room_labels'])
    return [r for r in read(output/'mp3d_rooms_final/houses'/(h['house']+'.json'))['rows'] if r['room_label'] in labels]

def room_rule_summary(output,h):
    rooms=candidate_rooms(output,h)
    def extremum(key,fn):
        values=[(r.get('metrics') or {}).get(key) for r in rooms]
        return fn(v for v in values if v is not None) if any(v is not None for v in values) else None
    return dict(candidate_room_count=len(rooms),status_counts=h.get('room_rule_counts',{'pass':len(rooms)}),house_gate=h['house_gate'],minimum_floor_area_m2=extremum('floor_area_m2',min),minimum_short_side_m=extremum('short_side_m',min),minimum_nav_main_area_m2=extremum('nav_main_area_m2',min),maximum_black_fraction=extremum('black_fraction',max),source_registry=h['registry_source'],thresholds_source=str(output/'evidence/inputs/thresholds.frozen.yaml'))

def training_rows(a):
    output=a.output
    hm=collect(read(output/'hm3d_manifest.json'),output/'hm3d_acoustics')
    measured={r['house']:r for r in collect(read(output/'mp3d_manifest.json'),output/'mp3d_acoustics')}
    prior={r['house']:r for r in read(output/'evidence/inputs/batch1_results.json') if r['family']=='mp3d'}
    mp=[]
    for h in read(output/'mp3d_selection.json')['houses']:
        if h['house'] in prior:
            r=copy.deepcopy(prior[h['house']]);r['acoustic_evidence_reused_from']=r['result_directory'];r['prior_six_original_status']=r['status'];r['input']=dict(h,acoustic_rooms=prior[h['house']]['input']['acoustic_rooms']);r['candidate_room_count_current']=h['candidate_room_count']
        elif h['house'] in measured:r=measured[h['house']]
        else:
            status='fail' if h['room_rule_counts'].get('review',0)==0 else 'review'
            r=dict(house=h['house'],family='mp3d',status=status,native_status='not_run',acoustic_status='not_run',visual=dict(status='not_run',reasons=['NO_RULE_PASS_ROOM_FOR_ACOUSTIC_SAMPLING']),reasons=['NO_ROOM_RULE_PASS']+([h['room_measurement_error']] if h['room_measurement_error'] else []),rirs=[],input=h,result_directory=str(output/'mp3d_rooms_final/houses'/(h['house']+'.json')))
        mp.append(r)
    reserved=set(read(output/'hm3d_selection.json')['reserved_houses'])
    live_sources=[]
    testroot=output.parent/'v7_batch2_test_20261003'
    for p in testroot.glob('**/recommended_houses*.json'):
        reserved.update(h['house'] for h in read(p).get('houses',[]) if h.get('split') in ('test','development'));live_sources.append(str(p))
    for r in hm+mp:
        h=r['input'];r['exposure']=exposure(output,h);h['exposure']=r['exposure']
        r['known_exposure']=any(r['exposure'][k] for k in ['so_seen','m1_seen','v6_seen'])
        r.pop('batch_exclusion',None)
        r['batch_eligible']=r['status']=='pass' and r['house'] not in reserved and h['candidate_room_count']>0
        r['training_eligible']=r['batch_eligible'];r['split']='train' if r['training_eligible'] else None
        r['room_rules']=room_rule_summary(output,h)
        r['candidate_room_count']=h['candidate_room_count'];r['measured_room_labels']=sorted({x['room_label'] for x in r.get('rirs',[])})
        r['sampled_room_count']=len(r['measured_room_labels']);r['planned_sampled_room_count']=len(h['acoustic_rooms'])
        r['human_review_status']='smy_use' if r['family']=='hm3d' else '待 smy 审房间'
        r['native_habitat_rgb_verified']=False;r['production_readiness_verified']=False
        r['exposure_policy']='SO/M1/v6 exposure permitted for training; excluded holdout houses are reserved'
        if r['house'] in reserved:
            r['batch_exclusion']='CURRENT_HOLDOUT_RESERVED';r['reasons'].append('CURRENT_HOLDOUT_RESERVED')
    return hm+mp,dict(reserved=sorted(reserved),live_holdout_sources=live_sources)

def report(a):
    output=a.output;dest=output/'report_final';dest.mkdir(exist_ok=False)
    results,holdouts=training_rows(a);passed=[r for r in results if r['training_eligible']]
    by={f:dict(Counter(r['status'] for r in results if r['family']==f)) for f in ['hm3d','mp3d']}
    counts={f:dict(houses=sum(r['family']==f for r in passed),candidate_rooms=sum(r['candidate_room_count'] for r in passed if r['family']==f),acoustically_sampled_rooms=sum(r['sampled_room_count'] for r in passed if r['family']==f)) for f in ['hm3d','mp3d']}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    stats=dict(code_commit=commit,acoustic_code_commit='3414629',cpu_only=True,training_counts=counts,by_family_status=by,hm3d_funnel=read(output/'hm3d_selection.json')['funnel'],mp3d_funnel=dict(registered_houses=90,registered_regions=read(output/'mp3d_rooms_final/inputs.json')['room_count'],rule_pass_houses=sum(h['candidate_room_count']>0 for h in read(output/'mp3d_selection.json')['houses']),rule_pass_rooms=read(output/'mp3d_selection.json')['room_count'],new_acoustic_houses=len(read(output/'mp3d_manifest.json')['houses']),prior_six_reused=6),rirs=sum(len(r.get('rirs',[])) for r in results),new_rirs=sum(len(r.get('rirs',[])) for r in results if 'acoustic_evidence_reused_from' not in r),baseline_ray_count=sum(sum(x['ray_count'] for x in r.get('leakage',[])) for r in results),holdout_audit=holdouts,mp3d_human_labels_available=False,formal_visual_runtime='not_run: CPU-only',production_readiness_verified=False)
    save(dest/'summary.json',stats);save(dest/'all_house_results.json',results)
    recommendation=[];scope=[]
    for r in passed:
        h=r['input'];recommendation.append(dict(house=r['house'],family=r['family'],split='train',original_split=h['original_split'],candidate_room_count=h['candidate_room_count'],candidate_area_sum_m2=h['candidate_area_sum_m2'],room_types=h['room_types'],rt60_mid_median_s=r.get('rt60_mid_median_s'),result_path=str(Path(r['result_directory'])/'result.json'),visual_path=str(Path(r['result_directory'])/'visual_checks.json'),exposure=r['exposure'],human_review_status=r['human_review_status'],sampled_room_count=r['sampled_room_count'],candidate_room_labels=h['candidate_room_labels']))
        scope.append(dict(house=r['house'],family=r['family'],candidate_room_labels=h['candidate_room_labels'],acoustically_sampled_rooms=h['acoustic_rooms'],scope='house diagnosis from <=3 predetermined rooms; all other rule-pass rooms need new-position RIR spot checks before production'))
    save(dest/'recommended_houses.json',dict(houses=recommendation,house_counts={'train':len(passed),'development':0,'test':0},by_family=counts,known_exposure_excluded=False,training_only=True,mp3d_human_review_pending=True))
    save(dest/'training_houses_v1.json',recommendation);save(dest/'recommended_room_scope.json',scope)
    columns=['house','family','original_split','candidate_area_sum_m2','candidate_room_count','native_status','visual_status','status','batch_eligible','rt60_mid_median_s','max_escape_fraction','max_foa_angle_deg','reasons','result_directory']
    with (dest/'house_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in results:
            row={k:r.get(k) for k in columns};row.update({k:r['input'].get(k) for k in ['original_split','candidate_area_sum_m2','candidate_room_count']});row.update(visual_status=r['visual']['status'],reasons='; '.join(r['reasons']));w.writerow(row)
    extended=['house','family','candidate_room_count','sampled_room_count','training_eligible','so_seen','m1_seen','v6_seen','human_review_status','max_direct_delay_ms','distance_exponents','valid_rirs','distinct_listeners','house_gate_status','room_rule_pass_count','minimum_floor_area_m2','minimum_short_side_m','minimum_nav_main_area_m2','maximum_black_fraction']
    with (dest/'training_house_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=extended);w.writeheader()
        for r in results:
            w.writerow(dict(house=r['house'],family=r['family'],candidate_room_count=r['candidate_room_count'],sampled_room_count=r['sampled_room_count'],training_eligible=r['training_eligible'],so_seen=r['exposure']['so_seen'],m1_seen=r['exposure']['m1_seen'],v6_seen=r['exposure']['v6_seen'],human_review_status=r['human_review_status'],max_direct_delay_ms=max((abs(x['direct_delay_error_ms']) for x in r.get('rirs',[]) if x.get('direct_delay_error_ms') is not None),default=None),distance_exponents=';'.join(str(x.get('pressure_exponent')) for x in r.get('distance_attenuation',[])),valid_rirs=sum(x.get('status')=='valid' for x in r.get('rirs',[])),distinct_listeners=r.get('distinct_listener_positions',0),house_gate_status=r['room_rules']['house_gate']['status'],room_rule_pass_count=r['room_rules']['candidate_room_count'],**{k:r['room_rules'][k] for k in ['minimum_floor_area_m2','minimum_short_side_m','minimum_nav_main_area_m2','maximum_black_fraction']}))
    room_columns=['house','family','room_label','smy_status','acoustically_sampled','floor_area_m2','short_side_m','nav_main_area_m2','black_fraction','floor_y_m','floor_reference_path','room_registry_source']
    with (dest/'training_room_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=room_columns);w.writeheader()
        for r in passed:
            h=r['input'];sampled=set(r['measured_room_labels'])
            for room in candidate_rooms(output,h):
                m=room['metrics'];w.writerow(dict(house=r['house'],family=r['family'],room_label=room['room_label'],smy_status=r['human_review_status'],acoustically_sampled=room['room_label'] in sampled,**{k:m.get(k) for k in ['floor_area_m2','short_side_m','nav_main_area_m2','black_fraction','floor_y_m']},floor_reference_path=(m.get('floor_reference') or {}).get('path'),room_registry_source=h['registry_source']))
    hm,mp=counts['hm3d'],counts['mp3d']
    conclusion=f"结论：HM3D 可供训练 {hm['houses']} 套、{hm['candidate_rooms']} 间规则合格房（声学抽查 {hm['acoustically_sampled_rooms']} 间）；MP3D 自动通过 {mp['houses']} 套、{mp['candidate_rooms']} 间（声学抽查 {mp['acoustically_sampled_rooms']} 间），全部标记待 smy 审房间。"
    labels={'pass':'通过','review':'待审','fail':'不通过','not_run':'未运行'}
    num=lambda x:'—' if x is None else f'{x:.3f}'
    lines=[conclusion,'','## 范围与口径','',f'48g，独立分支 claude/v7-train-houses-20261003；代码提交 `{commit}`。声学五个文件保持 `3414629` 原版；不改任何门槛。筛选和验收全程 CPU，未读取模型分数，未 push、修改共享数据或删除退化面。数字见本目录 summary.json、house_results.csv、all_house_results.json 和上级逐屋产物。','',
        '房间数是通过房屋中房间规则合格的候选数；声学只覆盖预先选定的小/大/不同房型至多三间。候选房间不能逐间声称已测声学。CPU 原始 GLB 视图是首批同口径坐标/投影诊断，正式 Habitat RGB、材质实测和训练窗口生产未验证。', '', '## 漏斗','', '|来源/阶段|套|间|','|---|---:|---:|']
    for x in stats['hm3d_funnel']:lines.append(f"|HM3D {x['stage']}|{x['houses']}|{x['rooms']}|")
    mf=stats['mp3d_funnel'];lines += [f"|MP3D 注册|90|{mf['registered_regions']}|",f"|MP3D 房间规则通过|{mf['rule_pass_houses']}|{mf['rule_pass_rooms']}|",f"|HM3D 最终通过|{hm['houses']}|{hm['candidate_rooms']}|",f"|MP3D 最终自动通过|{mp['houses']}|{mp['candidate_rooms']}|",'',f'综合状态：{by}。总 RIR {stats["rirs"]} 条，其中本轮新测 {stats["new_rirs"]} 条；其余沿用第一批六套 MP3D。','',
        '## 首批复测与历史暴露','', '先重测 00059、00168，漏声计数、有效 RIR/听者数和冻结规则结论一致。8 线程原生间接声 Monte Carlo 数值不逐位相同；00059 一条直达偏差从 0.004636 ms 变为 0.879636 ms，两者均在 3 ms 内。完整差异见 ../evidence/calibration_comparison_v2.json/.csv；最初按直达数值完全相等做的更严格审计也保留。没有用匹配容差改变房屋门槛。', '',
        'SO/M1/v6 见过不排除训练。SO 身份来自 assets_inventory.json；M1 来自实际两份 Stage-A 40,000 行训练调度及引用元数据，见 ../evidence/exposure_audit.json。未命中表示本次审计未识别，其他历史运行与匿名 SO 未验证。全部干净原 HM3D val 和当前开发/测试已排除，最终再次读取补测试任务推荐名单；不会修改它的名单。','', '## MP3D 后端与验证','',
        '新适配器从 .house 的 O/C/R 记录和 semantic PLY face.object_id 取得对象、房间身份；PLY/house/GLB 从 z 朝上统一旋转为 (x,z,-y)。地面按语义地面三角形并集测量，原生 bbox 只作身份元数据；家具 footprint、主层 98%、navmesh 连通面积、摆放和决定函数与 HM3D 共用。每层先保存 PathFinder snap 与原始 GLB 向下射线地板参考，再做网格、净空和摆放；不手写或移动地板，不重建共享 navmesh。','',
        '俯视图为 CPU 纹理三角形光栅化，逐像素 UV 取样；Basis 压缩纹理由独立 CPU 适配器和 [Basis Universal 上游](https://github.com/BinomialLLC/basis_universal/tree/c54f46f8aaa44eff1bc6e1d919e61e18b2b8f7d1) 解码，版本/编译命令见 ../evidence/basis_cpu_runtime.json，未修改共享环境；同一 polygon_mask/black_metric 计算 RGB 三通道均小于 8 的比例。纹理缺失保留待审，不填中性色作为通过依据。明确记录高度截面与成像矩阵。该 CPU 图与 Habitat GPU PBR 图的数值等价性未验证；MP3D 还需 smy 审房间。三套实跑与已知房对拍见 ../evidence/mp3d_pilot_validation_v2.json。','',
        '原生零面积面/加载失败保持待审，未启用 --omit-degenerate-for-diagnostics。新声学测量和画面预检均保留原始几何；第一批六套原结论原样沿用，包含原有诊断限制。','', '## 每套结果（最小地面/边/可走单位 m²/m/m²；最黑为候选房间最大比例）','', '|房屋|来源|候选/抽查间|整屋/房规则|最小地面/边/可走|最黑%|声学/视图/综合|漏声%|FOA°|衰减指数|有效IR/听者|最大直达偏差ms|SO/M1|人工/原因|','|---|---|---:|---|---|---:|---|---:|---:|---|---|---:|---|---|']
    for r in results:
        delay=max((abs(x['direct_delay_error_ms']) for x in r.get('rirs',[]) if x.get('direct_delay_error_ms') is not None),default=None)
        ex=' / '.join(num(x.get('pressure_exponent')) for x in r.get('distance_attenuation',[])) or '—'
        leak=r.get('max_escape_fraction');valid=sum(x.get('status')=='valid' for x in r.get('rirs',[]))
        reason='; '.join(r['reasons']) or '各项通过';human='smy use' if r['family']=='hm3d' else '待 smy 审房间'
        rule=r['room_rules'];minimum='/'.join(num(rule[k]) for k in ['minimum_floor_area_m2','minimum_short_side_m','minimum_nav_main_area_m2']);black=rule['maximum_black_fraction'];gate=labels.get(rule['house_gate']['status'],rule['house_gate']['status']);rule_counts='/'.join(str(rule['status_counts'].get(k,0)) for k in ['pass','review','fail'])
        lines.append(f"|{r['house']}|{r['family']}|{r['candidate_room_count']}/{r['sampled_room_count']}|{gate}; P/R/F {rule_counts}|{minimum}|{num(black*100 if black is not None else None)}|{labels.get(r.get('native_status'),r.get('native_status'))}/{labels[r['visual']['status']]}/{labels[r['status']]}|{num(leak*100 if leak is not None else None)}|{num(r.get('max_foa_angle_deg'))}|{ex}|{valid}/{r.get('distinct_listener_positions',0)}|{num(delay)}|{int(r['exposure']['so_seen'])}/{int(r['exposure']['m1_seen'])}|{human}; {reason}|" )
    lines += ['', '## 验证、资源与交付', '', '后端/首批验收针对测试 18 项通过；fast_unit 289 通过、36 跳过（历史资源缺失），日志 ../logs/tests_backend_final_v2.log、tests_fast_unit_final_v2.log。共享门槛与声学源码的最终一致检查另见 ../evidence/final_validation.json。', '', 'CPU 分配见 ../evidence/cpu_allocation.json：现场 110/112 逻辑核低于 20% 使用率，声学最多 2 worker × 原版 8 线程，房间筛选最多 4 worker，分别限制在 16/4 核。MP3D 开始房间筛选和声学时各重新按空闲核分配，收据 cpu_allocation_mp3d_rooms.json、cpu_allocation_mp3d_acoustics.json。所有作业 PID/命令见 logs 与逐屋 .process.json；原始数据、第一批、其他检出与公共环境均只读。', '', 'owner 页面兼容输入为 house_results.csv、all_house_results.json、recommended_houses.json。完整训练名单为 training_houses_v1.json；每套附来源、房间数、各项验收、SO/M1 证据和人工状态。recommended_room_scope.json 保留实际抽查摆放范围；training_room_results.csv 列出逐间规则数值和声学是否抽查。']
    (dest/'REPORT_zh.md').write_text('\n'.join(lines)+'\n');(output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print(conclusion,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare-mp3d','visual-mp3d','report']);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args();{'prepare-mp3d':prepare_mp3d,'visual-mp3d':visual_mp3d,'report':report}[a.stage](a)
