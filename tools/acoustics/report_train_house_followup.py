#!/usr/bin/env python3
"""Publish the automatic pool with explicit historical and current gate evidence."""
from __future__ import annotations
import argparse,collections,copy,csv,json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save,registry
from tools.acoustics.summarize_house_acceptance import collect
from tools.acoustics.summarize_train_house_pool import exposure

CATEGORIES={
 'camera_pose_search':('相机/声源姿态搜索失败','旧流程仅搜旧路线的八个姿态；改用冻结房间规则的导航支撑、净空和三条视线见证，再实测原生声学及原始视图。旧特定路线不可用不等于整屋没有可用房间。'),
 'frame_parity':('原始场景与声学包帧一致性失败','有真实的旧射线不一致证据；不能仅凭报错断言坏房。保留旧 0.05 m 对拍值，重新用当前编译、量过的地板/导航支撑和选定房间做统一 CPU 验收；旧 Habitat 全场景对拍未重跑，根因未完全验证。'),
 'ambiguous_annotation':('语义标注颜色对应多个材质类别','资产标注冲突，当前同一编译器也拒绝；不猜类别、不修改原标注，保持待审。'),
 'machine_foa_audition':('动态成品 FOA 机器听审失败','旧机器听审对成品方向要求 90% 帧达到 5°；当前采用 owner 已认可的首批选房方向验收和 e3dc246 首到达窗口。不能把本轮选房通过写成旧动态成品也通过；生产成品仍未验证。'),
 'binaural_measurement_or_pose':('双耳渲染/相机姿态验收失败','旧双耳流程/姿态诊断拒绝；完整旧双耳生产链未在本轮复跑。当前按预先选定的可行房间及同一 FOA/RIR CPU 选房验收重新判定，旧双耳失败根因逐屋保留，未验证部分不称为已修好。'),
 'native_segfault':('原生声学加载崩溃','旧 AudioSensor 路线在加载语义场景时退出 -11；当前直接使用首批原生 RLR CPU 路线复测。当前仍无法加载的保持待审，不删除真实面。'),
 'interrupted_with_completed_receipt':('任务状态中断但完成回执通过','流程状态记账问题：已有完整端到端 receipt、机器听审 pass；不改共享 task.json，在本任务派生记录中恢复证据，再按统一声学规则复测。'),
 'machine_channel_report':('成品声道与报告不一致','旧成品/报告同步或声道测量异常；本轮使用首批 ACN 原始 FOA 通道顺序和冻结旋转测量。旧成品问题未宣称解决，当前 CPU 房屋筛选证据单独记录。')}

JUDGMENTS={
 'camera_pose_search':'旧路线的有限姿态搜索问题；本轮另选冻结规则见证并实测，旧路线并未修复',
 'frame_parity':'真实射线不一致证据；资产、坐标或测量根因未验证，旧生产对拍未修复',
 'ambiguous_annotation':'真实资产标注冲突；不等于物理房间不可用，保持待审',
 'machine_foa_audition':'旧动态成品确未达到其原门槛；物理房间或生产测量根因未验证',
 'binaural_measurement_or_pose':'4 套确证接受了零入镜比例姿态；其余旧双耳失败根因未验证',
 'native_segfault':'加载器/运行时崩溃；物理质量由原生 CPU 重测判定，崩溃根因未验证',
 'interrupted_with_completed_receipt':'确证流程状态记账问题；完成回执已通过，派生记录恢复证据',
 'machine_channel_report':'声道/报告确实不一致；同步或测量根因未验证'}

def annotate(r,h,output,rooms,reserved,gate_retest=False):
    r=copy.deepcopy(r);h=copy.deepcopy(h);r['input']=h
    r['exposure']=exposure(output,h);h['exposure']=r['exposure'];r['known_exposure']=any(r['exposure'][k] for k in ('so_seen','m1_seen','v6_seen'))
    r.pop('batch_exclusion',None)
    valid=sum(x.get('status')=='valid' for x in r.get('rirs',[]))
    if r['status']=='pass' and valid<6:r['status']='review';r['reasons'].append('FEWER_THAN_SIX_VALID_RIRS')
    current_gate=dict(status=r['status'],source=str(Path(r['result_directory'])/'result.json'),visual_source=str(Path(r['result_directory'])/'visual_checks.json'),scope='frozen automatic room rules plus unchanged first-batch CPU house acoustics/visual acceptance with e3dc246; not legacy GPU production replay')
    old_gate=h.get('historical_house_gate',h.get('house_gate'))
    h['historical_house_gate']=old_gate
    h['house_gate']=current_gate
    if r.get('derived_geometry'):h['derived_visual_source']=r['derived_geometry']['derived']
    r['stage4']=current_gate;r['historical_stage4']=old_gate;r['historical_gate_retested']=gate_retest
    r['historical_gate_original_status']=old_gate.get('status') if old_gate else None
    r['training_eligible']=r['status']=='pass' and h['candidate_room_count']>0 and h['house'] not in reserved
    r['batch_eligible']=r['training_eligible'];r['split']='train' if r['training_eligible'] else None
    r['holdout_reserved']=h['house'] in reserved
    if r['holdout_reserved']:r['batch_exclusion']='CURRENT_HOLDOUT_RESERVED'
    labels=dict(collections.Counter((x.get('human') or {}).get('verdict') or 'unlabeled' for x in rooms)) if h['family']=='hm3d' else {'unlabeled':len(rooms)}
    r['smy_labels']=labels;r['smy_verdict_by_room']={x['room_label']:(x.get('human') or {}).get('verdict','unlabeled') for x in rooms};r['human_labels_filter']=False
    r['human_review_status']='标签仅记录，不参与筛选' if h['family']=='hm3d' else '待 smy 审房间；标签不参与筛选'
    r['candidate_room_count']=h['candidate_room_count'];r['measured_room_labels']=sorted({m['room_label'] for m in r.get('rirs',[])})
    r['sampled_room_count']=len(r['measured_room_labels']);r['planned_sampled_room_count']=len(h['acoustic_rooms'])
    r['native_habitat_rgb_verified']=False;r['production_readiness_verified']=False
    r['geometry_derivation_label']='派生几何：清退化三角形' if r.get('derived_geometry') else None
    def extreme(k,fn):
        vs=[x['metrics'].get(k) for x in rooms if x['metrics'].get(k) is not None];return fn(vs) if vs else None
    r['room_rules']=dict(candidate_room_count=len(rooms),house_gate=h.get('house_gate'),source_registry=h['registry_source'],thresholds_source=str(output/'evidence/inputs/thresholds.frozen.yaml'),minimum_floor_area_m2=extreme('floor_area_m2',min),minimum_short_side_m=extreme('short_side_m',min),minimum_nav_main_area_m2=extreme('nav_main_area_m2',min),maximum_black_fraction=extreme('black_fraction',max))
    return r

def report(output,work):
    assert read(work/'evidence/calibration_corrected_exact.json')['exactly_equal']
    assert read(work/'evidence/hm3d_visual_completion.json')['completed']==159
    destination=work/'report_final';destination.mkdir(exist_ok=False)
    reg=registry(work);selection=read(work/'hm3d_selection.json');reserved=set(selection['reserved_houses'])
    results=[];blocked_results={};all_room_rows={}
    for mf,folder,blocked in [('hm3d_manifest.json','hm3d_current',False),('hm3d_blocked_manifest.json','hm3d_blocked',True)]:
        manifest=read(work/mf)
        for r in collect(manifest,work/folder):
            h=r['input'];rs=[x for x in reg[h['house']] if x['stage1']['status']=='pass'];a=annotate(r,h,output,rs,reserved,blocked)
            all_room_rows[h['house']]=rs
            if blocked:blocked_results[h['house']]=a
            if not a['holdout_reserved']:results.append(a)
    prior={r['house']:r for r in read(output/'evidence/inputs/batch1_results.json') if r['family']=='mp3d'}
    mp_manifest=read(output/'mp3d_manifest.json');mp_houses=read(output/'mp3d_selection.json')['houses']
    for h in mp_houses:
        rs=read(output/'mp3d_rooms_final/houses'/(h['house']+'.json'))['rows'];rs=[x for x in rs if x['stage1']['status']=='pass'];all_room_rows[h['house']]=rs
        if h['house'] in prior:
            original=prior[h['house']];h=copy.deepcopy(h);h['acoustic_rooms']=original['input']['acoustic_rooms']
            r=collect(dict(mp_manifest,houses=[h]),work/'mp3d_prior_corrected')[0];r['prior_six_original_status']=original['status'];r['acoustic_evidence_reused_from']=original['result_directory']
            # The task explicitly retains their original conclusions; the new
            # metric values are still uniformly recorded in the corrected copy.
            r['status']=original['status']
        elif h['acoustic_rooms']:
            folder='mp3d_current' if (work/'mp3d_current/houses'/h['house']/'result.json').exists() else 'mp3d_corrected'
            r=collect(dict(mp_manifest,houses=[h]),work/folder)[0]
        else:
            status='review' if h['room_rule_counts'].get('review',0)>0 else 'fail'
            r=dict(house=h['house'],family='mp3d',status=status,native_status='not_run',reasons=['NO_ROOM_RULE_PASS'],rirs=[],visual={'status':'not_run','reasons':['NO_ROOM_RULE_PASS']},result_directory=str(output/'mp3d_rooms_final/houses'/h['house']))
        results.append(annotate(r,h,output,rs,reserved))
    passed=[r for r in results if r['training_eligible']]
    counts={f:dict(houses=sum(r['family']==f for r in passed),candidate_rooms=sum(r['candidate_room_count'] for r in passed if r['family']==f),sampled_rooms=sum(r['sampled_room_count'] for r in passed if r['family']==f)) for f in ['hm3d','mp3d']}
    audit=[]
    for b in read(work/'evidence/historical_gate_classification_pretest.json'):
        r=blocked_results[b['house']];name,interpretation=CATEGORIES[b['category']]
        old_pose=None
        for evidence in b.get('reports',[]):
            data=evidence.get('data',{});index=data.get('accepted_index')
            if '/listener_pose/' in evidence['path'] and isinstance(index,int) and 0<=index<len(data.get('candidates',[])):
                old_pose=dict(path=evidence['path'],accepted_index=index,route_share_in_frame=data['candidates'][index]['route_share_in_frame'])
        audit.append(dict(house=b['house'],category=b['category'],category_zh=name,historical_status=b['historical_status'],historical_task=reg[b['house']][0]['stage4']['source'],historical_log=b['log'],decisive_log_lines=b['decisive_log_lines'],historical_report_paths=[e['path'] for e in b.get('reports',[])],historical_accepted_pose=old_pose,candidate_rooms=r['candidate_room_count'],native_status=r['native_status'],visual_status=r['visual']['status'],current_status=r['status'],training_eligible=r['training_eligible'],holdout_reserved=r['holdout_reserved'],rescued_current_automatic_rules=r['status']=='pass',legacy_gpu_end_to_end_rerun=False,real_problem_or_workflow=JUDGMENTS[b['category']],interpretation=interpretation,current_reasons=r['reasons'],result_directory=r['result_directory']))
    category_summary=[]
    for key,(name,interpretation) in CATEGORIES.items():
        subset=[r for r in audit if r['category']==key]
        category_summary.append(dict(category=key,category_zh=name,houses=len(subset),rescued_current_rules=sum(r['rescued_current_automatic_rules'] for r in subset),rescued_training_houses=sum(r['training_eligible'] for r in subset),review=sum(r['current_status']=='review' for r in subset),fail=sum(r['current_status']=='fail' for r in subset),real_problem_or_workflow=JUDGMENTS[key],interpretation=interpretation))
    save(destination/'historical_gate_failures_93.json',audit)
    save(destination/'historical_gate_failure_categories.json',category_summary)
    old_reports=work/'historical/report_final'
    derivation=read(work/'evidence/mp3d_derivation_summary_v4.json');previous7={r['house'] for r in read(old_reports/'mp3d_acoustic_pass_pending_review_v1.json')}
    revived7=[r for r in passed if r['house'] in previous7]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    summary=dict(code_commit=commit,acoustic_fix_commit='e3dc246',direct_fix_cherry_pick='8151374',cpu_only=True,human_labels_filter=False,training_counts=counts,old_training_counts=read(old_reports/'summary.json')['training_counts'],old_and_new_funnel=selection['funnel'],delta=selection,house_result_rows=len(results),by_family_status={f:dict(collections.Counter(r['status'] for r in results if r['family']==f)) for f in ['hm3d','mp3d']},historical_gate_houses=93,historical_gate_rescued_current_rules=sum(r['rescued_current_automatic_rules'] for r in audit),historical_gate_rescued_training=sum(r['training_eligible'] for r in audit),historical_gate_rescued_training_rooms=sum(r['candidate_rooms'] for r in audit if r['training_eligible']),historical_gate_failure_categories=category_summary,mp3d_derivation=derivation,mp3d_previous_seven_rescued_houses=len(revived7),mp3d_previous_seven_rescued_rooms=sum(r['candidate_room_count'] for r in revived7),mp3d_all_room_rule_pass_houses=85,mp3d_all_room_rule_pass_rooms=623,holdout_reserved=sorted(reserved),rirs=sum(len(r.get('rirs',[])) for r in results),valid_rirs=sum(m.get('status')=='valid' for r in results for m in r.get('rirs',[])),formal_visual_runtime='not_run: CPU-only',production_readiness_verified=False)
    save(destination/'summary.json',summary);save(destination/'all_house_results.json',results)
    recommendations=[]
    for r in passed:
        h=r['input'];recommendations.append(dict(house=r['house'],family=r['family'],split='train',original_split=h['original_split'],candidate_room_count=h['candidate_room_count'],candidate_area_sum_m2=h['candidate_area_sum_m2'],candidate_room_labels=h['candidate_room_labels'],room_types=h['room_types'],result_path=str(Path(r['result_directory'])/'result.json'),visual_path=str(Path(r['result_directory'])/'visual_checks.json'),smy_labels=r['smy_labels'],human_labels_filter=False,exposure=r['exposure'],sampled_room_count=r['sampled_room_count'],rt60_mid_median_s=r.get('rt60_mid_median_s'),geometry_derivation_label=r['geometry_derivation_label'],derived_visual_source=(r.get('derived_geometry') or {}).get('derived'),historical_gate_original_status=r['historical_gate_original_status'],stage4=r['stage4']))
    save(destination/'historical_gate_pass_only_training_houses_v1.json',[r for r in recommendations if r['family']=='hm3d' and r['historical_gate_original_status']=='pass']);save(destination/'training_houses_v1.json',recommendations);save(destination/'recommended_houses.json',dict(houses=recommendations,house_counts={'train':len(passed),'development':0,'test':0},by_family=counts,training_only=True,human_labels_filter=False))
    save(destination/'recommended_room_scope.json',[dict(house=r['house'],family=r['family'],candidate_room_labels=r['input']['candidate_room_labels'],acoustically_sampled_rooms=r['input']['acoustic_rooms'],smy_verdict_by_room=r['smy_verdict_by_room'],derived_visual_source=(r.get('derived_geometry') or {}).get('derived'),scope='<=3 predetermined rooms acoustically checked; other automatic room passes are house-level candidates') for r in passed])
    pending=[dict(house=r['house'],candidate_room_count=r['candidate_room_count'],sampled_room_count=r['sampled_room_count'],native_status=r['native_status'],visual_status=r['visual']['status'],combined_status=r['status'],training_eligible=False,raw_zero_area_triangle_count=(r.get('derived_geometry') or {}).get('removed_faces'),human_review_status=r['human_review_status'],exposure=r['exposure'],reasons=r['reasons'],acoustic_result_path=str(Path(r['result_directory'])/'result.json'),visual_result_path=str(Path(r['result_directory'])/'visual_checks.json')) for r in results if r['family']=='mp3d' and r['native_status']=='pass' and r['status']!='pass']
    save(destination/'mp3d_acoustic_pass_pending_review_v1.json',pending)
    pendingcols=['house','candidate_room_count','sampled_room_count','native_status','visual_status','combined_status','training_eligible','raw_zero_area_triangle_count','human_review_status','so_seen','m1_seen','reasons','acoustic_result_path','visual_result_path']
    with (destination/'mp3d_acoustic_pass_pending_review_v1.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=pendingcols);writer.writeheader()
        for r in pending:
            row={k:r.get(k) for k in pendingcols};row.update(so_seen=r['exposure']['so_seen'],m1_seen=r['exposure']['m1_seen'],reasons='; '.join(r['reasons']));writer.writerow(row)
    columns=['house','family','original_split','candidate_area_sum_m2','candidate_room_count','native_status','visual_status','status','batch_eligible','rt60_mid_median_s','max_escape_fraction','max_foa_angle_deg','reasons','result_directory','smy_labels']
    with (destination/'house_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in results:
            row={k:r.get(k) for k in columns};row.update({k:r['input'].get(k) for k in ['original_split','candidate_area_sum_m2','candidate_room_count']});row.update(visual_status=r['visual']['status'],reasons='; '.join(r['reasons']),smy_labels=json.dumps(r['smy_labels'],ensure_ascii=False,sort_keys=True));w.writerow(row)
    extended=['house','family','candidate_room_count','sampled_room_count','training_eligible','so_seen','m1_seen','smy_labels','historical_gate_status','current_stage4_status','max_direct_delay_ms','distance_exponents','valid_rirs','distinct_listeners','geometry_derivation_label']
    with (destination/'training_house_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=extended);w.writeheader()
        for r in results:w.writerow(dict(house=r['house'],family=r['family'],candidate_room_count=r['candidate_room_count'],sampled_room_count=r['sampled_room_count'],training_eligible=r['training_eligible'],so_seen=r['exposure']['so_seen'],m1_seen=r['exposure']['m1_seen'],smy_labels=json.dumps(r['smy_labels'],ensure_ascii=False),historical_gate_status=r['historical_gate_original_status'],current_stage4_status=r['stage4']['status'],max_direct_delay_ms=max((abs(m['direct_delay_error_ms']) for m in r.get('rirs',[]) if m.get('direct_delay_error_ms') is not None),default=None),distance_exponents=';'.join(str(x.get('pressure_exponent')) for x in r.get('distance_attenuation',[])),valid_rirs=sum(m.get('status')=='valid' for m in r.get('rirs',[])),distinct_listeners=r.get('distinct_listener_positions'),geometry_derivation_label=r['geometry_derivation_label']))
    roomcols=['house','family','room_label','smy_status','stage1_status','stage4_status','acoustically_sampled','floor_area_m2','short_side_m','nav_main_area_m2','black_fraction','floor_y_m','floor_reference_path','room_registry_source']
    with (destination/'training_room_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=roomcols);w.writeheader()
        for r in passed:
            for room in all_room_rows[r['house']]:
                m=room['metrics'];w.writerow(dict(house=r['house'],family=r['family'],room_label=room['room_label'],smy_status=(room.get('human') or {}).get('verdict','unlabeled'),stage1_status=room['stage1']['status'],stage4_status=r['stage4']['status'],acoustically_sampled=room['room_label'] in r['measured_room_labels'],**{k:m.get(k) for k in ['floor_area_m2','short_side_m','nav_main_area_m2','black_fraction','floor_y_m']},floor_reference_path=(m.get('floor_reference') or {}).get('path'),room_registry_source=r['input']['registry_source']))
    hm,mp=counts['hm3d'],counts['mp3d'];strict=[r for r in passed if r['family']=='hm3d' and r['historical_gate_original_status']=='pass'];lines=[f"结论：自动规则下 HM3D 可供训练 {hm['houses']} 套、{hm['candidate_rooms']} 间；MP3D 可供训练 {mp['houses']} 套、{mp['candidate_rooms']} 间。",'',f'机器 48g；独立检出 {REPO}，分支 claude/v7-train-houses-20261003；当前提交 `{commit}`；直达修正 `e3dc246` 已 cherry-pick 为 `8151374`。无 push，全程 CPU。数字出处：本目录 summary.json、house_results.csv、all_house_results.json；逐屋 result_directory 指向原生证据。','',
      '## 口径与漏斗','', 'smy 的 use/skip/unsure/未标注逐间、逐屋保留，不参与筛选。SO/M1/v6 暴露只记录，训练允许；33 套干净原 HM3D val 及实时开发、测试推荐名单仍全部预留。训练候选房间数是规则通过数；声学实测仅每屋预先选定至多 3 间。','',
      '旧端到端 task/status 原记录不改。本轮“救回”指重新通过 owner 冻结房间规则、首批同一整屋 CPU 声学/视图规则的房屋；当前 stage4 与历史 stage4 分列。旧 GPU 端到端生产、动态 5°机器听审和双耳成品链未重跑，不能把本轮通过写成旧成品也通过。','',
      '|阶段|旧 smy 口径：套/间|自动口径：套/间|','|---|---:|---:|','|登记|181 / 2599|181 / 2599|','|仅房间规则通过|—|179 / 1154|','|旧整屋门禁 + 房间规则|85 / 390|86 / 567|','|排除全部预留房后|65 / 294|66 / 424|',f"|本轮最终训练|33 / 151（旧测量）|{hm['houses']} / {hm['candidate_rooms']}|",f"|MP3D 最终训练|0 / 0|{mp['houses']} / {mp['candidate_rooms']}|",'',
      f"取消人审后新增 1 套、130 间候选；保留已测样本，8 套补测 10 间。HM3D 最终声学覆盖 {hm['sampled_rooms']} 间，MP3D 覆盖 {mp['sampled_rooms']} 间；其余规则合格房间不得宣称逐间已测声学。新增候选与已测波形均保存在 followup_auto_only_20261003，第一轮结果及波形保留。",'',
      f'仅保留历史 stage4.pass 而不采纳本轮救回记录时：{len(strict)} 套、{sum(r["candidate_room_count"] for r in strict)} 间，见 historical_gate_pass_only_training_houses_v1.json；它与本轮重测后的完整训练名单分别保存。','', '## 统一直达声修正','', '所有先前 65 套 HM3D、79 套新 MP3D 以及首批 6 套 MP3D 的保存波形均由 e3dc246 重算，原始结果保留。先取首批 00059、00168 在 v7-fill-test 的修正后波形重算：146 项逐项完全相等，见 ../evidence/calibration_corrected_exact.json；这是保存波形的代码对拍，没有声称独立 Monte Carlo 传播数值逐位相等。新增测量直接用同一修正。阈值、材质、传播设置不改。','',
      '## 93 套旧门禁失败/中断','',f"93 套中 {summary['historical_gate_rescued_current_rules']} 套重新通过本轮统一 CPU 规则，其中可入训练 {summary['historical_gate_rescued_training']} 套、{summary['historical_gate_rescued_training_rooms']} 间；其他通过但属于干净原 val 的仍预留。逐套旧原因、日志行、当前原因与去向：historical_gate_failures_93.json。",'', '|旧失败分类|套|本轮通过|可入训练|待审|不通过|','|---|---:|---:|---:|---:|---:|']
    for c in category_summary:lines.append(f"|{c['category_zh']}|{c['houses']}|{c['rescued_current_rules']}|{c['rescued_training_houses']}|{c['review']}|{c['fail']}|")
    for key,(name,text) in CATEGORIES.items():lines+=['',f'- {name}：判断：{JUDGMENTS[key]}。{text}']
    lines+=['','真实/当前测得的漏声、方向、距离、直达偏差、有效 RIR 不足等仍依原门槛失败或待审，不因旧原因属于流程问题就放行。历史几何对拍、双耳和动态成品的根因未完全确认之处在逐屋表明确记录。','',
      '## MP3D 清退化面与验证','',f"此前 7 套、45 间纯退化面候选，本轮救回 {len(revived7)} 套、{sum(r['candidate_room_count'] for r in revived7)} 间；本轮全部派生对拍结果见 ../evidence/mp3d_derivation_summary_v4.json。",'',
      '派生 GLB 仅重写三角索引，原顶点/法线/UV/材质/纹理二进制前缀不变；全零 primitive/mesh 删除后节点保留、mesh 索引合法重排。原始 GLB、semantic PLY、导航网格、声学包不动。删除条件：float64 叉积严格零；或 native float32 叉积严格零且 float64 面积 ≤ 4·eps(float32)·|e1|·|e2|。不存在一般小面积删面门槛；每套精确零数、浮点残差数、最大及总面积、删除索引都在 geometry_derivation.json/.removed_faces.npz。','',
      '声学逐项不变的对拍使用相同、未变的 semantic PLY 声学包及保存 RIR 两侧统一重算，逐字段 exact equality；没有宣称重新随机传播的晚反射也逐位相等。视觉同时输出真实纹理的全部候选房俯视图和首批同代码的 CPU 射线透视图；原生 raw BVH 报零面积时改用保留全部原始面的 CPU 参考射线；Trimesh 实测过慢时采用 Numba BVH 加速，先与已完成的 Trimesh 三张完整真实视图逐像素对拍，见 ../evidence/cpu_bvh_real_scan_comparison.json。两侧后端及原生错误均记录，不把它写成原生 raw BVH 已通过。俯视图逐像素为零差才入选；透视近邻面心着色可因删掉空面心产生颜色差，逐张最大/均值/RMS/像素差保存，投影结论与可见声源像素必须不变。正式 Habitat RGB 未运行。','',
      '|MP3D|规则房间|删面|严格零|浮点残差|俯视图差|透视像素差|对拍|','|---|---:|---:|---:|---:|---|---:|---|']
    for r in derivation['rows']:lines.append(f"|{r['house']}|{r.get('candidate_rooms','—')}|{r.get('removed_faces','—')}|{r.get('exact_zero_faces','—')}|{r.get('float32_roundoff_zero_faces','—')}|{'0' if r.get('all_overhead_pixels_identical') else '未完成/非零'}|{r.get('perspective_different_pixels','—')}|{r['status']} {r.get('error','')}|")
    lines+=['','MP3D 后端仍取 .house 的 O/C/R 和 semantic PLY face.object_id；z 朝上先转 (x,z,-y)，每个新房间先量地板偏移。全量 90 套/2206 间已有 85 套/623 间房间规则通过；3218 份地板参考、14366 次导航探测等原始证据见第一轮 evidence/mp3d_full_room_audit.json。6 套首批 MP3D 原结论仍保留，统一修正后的指标另存；MP3D 只进入训练。','',
      '## 每套结果','', '|房屋|来源|房间|声学|视图|综合|SO/M1|smy 标签|派生|','|---|---|---:|---|---|---|---|---|---|']
    for r in results:lines.append(f"|{r['house']}|{r['family']}|{r['candidate_room_count']}|{r['native_status']}|{r['visual']['status']}|{r['status']}|{int(r['exposure']['so_seen'])}/{int(r['exposure']['m1_seen'])}|{json.dumps(r['smy_labels'],ensure_ascii=False)}|{r['geometry_derivation_label'] or '—'}|")
    lines+=['','SO/M1 身份仍来自已保存的资产清单与两份 40,000 行 M1 调度审计；未命中仅表示该审计范围内未识别，其他历史运行、匿名暴露未验证。','', '## 验证、运行和限制','', '专项与 fast_unit 的实际命令、数量、退出码见 ../logs/tests_*.log 与 ../evidence/final_validation.json。只用自己的检出和产物目录；未改共享数据、原始标注/navmesh、第一批、其他检出、环境或服务。CPU 线程和 PID/结束状态见 ../evidence/finished_jobs_round2.json；未生成训练视频/音频窗口或模型分数。', '', '原始资产标注冲突、非退化原生加载失败保留待审；未抽样房间/位置的声学、正式 Habitat RGB、旧动态/双耳成品生产，以及模型训练均未验证。训练名单 v1 是选房产物。']
    (destination/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print('FINAL_TRAINING_COUNTS',json.dumps(counts),'GATE_RESCUES',summary['historical_gate_rescued_training'],'MP7_RESCUES',len(revived7),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();report(a.output,a.work)
if __name__=='__main__':main()
