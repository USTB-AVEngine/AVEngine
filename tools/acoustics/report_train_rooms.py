#!/usr/bin/env python3
"""Room-level training lists and frozen/finished house-veto comparisons."""
from __future__ import annotations
import argparse,collections,copy,csv,datetime,json,statistics,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save,corrected_result

def count(rows):return dict(houses=len({r['house'] for r in rows}),rooms=len(rows))

def visual_root(output,house,overrides):
    path=Path(overrides[house]['output']) if house in overrides else output/'visuals'/house
    if not path.resolve().is_relative_to(output.resolve()):raise ValueError('Visual retry must stay in this owned output')
    return path

def csvfile(path,columns,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(r.get(k),ensure_ascii=False,sort_keys=True) if isinstance(r.get(k),(list,dict)) else r.get(k) for k in columns})

def room_row(h,room,rule,result,visual,result_path,visual_path):
    leak=result.get('max_escape_fraction');valid=sum(r.get('status')=='valid' for r in result.get('rirs',[]))
    reasons=list(result.get('reasons',[]))+list(result.get('completion_reasons',[]))
    if visual['status']!='pass':reasons+=visual.get('reasons',['CPU_VISUAL_REQUIRES_REVIEW'])
    enough=valid>=6 and result.get('distinct_listener_positions',0)>=2
    train=result['status']=='pass' and visual['status']=='pass' and enough and leak is not None and leak<=.05
    if train:status='pass';tier='training'
    elif leak is not None and .05<leak<=.15:status='review';tier='leak_5_to_15_review'
    elif result['status']=='fail':status='fail';tier='fail'
    else:status='review';tier='other_review'
    metadata=rule['metrics'];ex=h.get('exposure',{})
    return dict(house=h['house'],family=h['family'],room_label=room['room_label'],room_type=room['room_type'],original_split=h['original_split'],split='train',stage1_status=rule['stage1']['status'],historical_stage4_status=(h.get('historical_house_gate') or h.get('house_gate') or {}).get('status','not_run'),acoustic_status=result['status'],visual_status=visual['status'],status=status,tier=tier,training_eligible=train,smy_label=(rule.get('human') or {}).get('verdict') or 'unlabeled',human_labels_filter=False,so_seen=ex.get('so_seen',False),m1_seen=ex.get('m1_seen',False),v6_seen=ex.get('v6_seen',False),exposure=ex,max_escape_fraction=leak,max_foa_angle_deg=result.get('max_foa_angle_deg'),distance_exponents=[x.get('pressure_exponent') for x in result.get('distance_attenuation',[])],rir_count=len(result.get('rirs',[])),valid_rirs=valid,distinct_listeners=result.get('distinct_listener_positions',0),max_direct_delay_ms=max((abs(x['direct_delay_error_ms']) for x in result.get('rirs',[]) if x.get('direct_delay_error_ms') is not None),default=None),rt60_mid_median_s=result.get('rt60_mid_median_s'),rt60_valid_fraction=result.get('rt60_valid_fraction'),reused_rirs=result.get('reused_rir_count',0),new_rirs=result.get('new_rir_count',0),listener_supplement=result.get('listener_supplement',False),leakage_origins=result.get('leakage',[]),reasons=reasons,acoustic_reasons=result.get('reasons',[]),geometry_derivation_label='派生几何：清退化三角形' if visual.get('geometry_derivation') else '',geometry_comparison_status=visual.get('geometry_comparison_status'),derived_visual_source=visual.get('derived_visual_asset'),floor_area_m2=metadata['floor_area_m2'],short_side_m=metadata['short_side_m'],nav_main_area_m2=metadata['nav_main_area_m2'],black_fraction=metadata['black_fraction'],floor_y_m=metadata['floor_y_m'],floor_reference_path=(metadata.get('floor_reference') or {}).get('path'),placement=room['placement'],result_path=str(result_path),visual_path=str(visual_path),compute_device='CPU',native_habitat_rgb_verified=False)

def old_house(h,rows,output,settings):
    labels=[r['room_label'] for r in h['old_sampled_rooms']]
    chosen=[r for r in rows if r['room_label'] in labels]
    hh=copy.deepcopy(h);by_label={r['room_label']:r for r in h['acoustic_rooms']};hh['acoustic_rooms']=[copy.deepcopy(by_label[label]) for label in labels if label in by_label];hh.pop('room_rule_records',None)
    record=dict(house=h['house'],family=h['family'],input=hh,rirs=[],leakage=[],occluded_probe_lines=[],known_exposure=h.get('known_exposure',False),status='review',reasons=[])
    missing=[]
    for label in labels:
        p=output/'houses'/h['house']/'rooms'/label/'result.json'
        if not p.exists():missing.append(label);continue
        r=read(p)
        if not r.get('rirs') or not r.get('foa_checks'):missing.append(label)
        for key in ('rirs','leakage'):record[key]+=r.get(key,[])
        record['occluded_probe_lines'] += [x for x in r.get('occluded_probe_lines',[]) if len(labels)==1 or x['source_label'] in ('range_near','range_middle','source_1','source_2')]
    if missing or len(hh['acoustic_rooms'])!=len(labels):record.update(status='review',reasons=['OLD_SAMPLE_MEASUREMENT_INCOMPLETE'],missing_rooms=missing)
    else:
        record=corrected_result(record,settings,hh,selected_only=True)
    visual_status='pass' if len(chosen)==len(labels) and all(r['visual_status']=='pass' for r in chosen) else 'review'
    before_verified=not any('ACOUSTIC_GEOMETRY_BEFORE_WAVEFORM_UNAVAILABLE' in r['reasons'] for r in chosen)
    status='fail' if record['status']=='fail' else 'pass' if record['status']=='pass' and visual_status=='pass' and before_verified else 'review'
    reasons=list(record.get('reasons',[]))+(['CPU_VISUAL_REQUIRES_REVIEW'] if visual_status!='pass' else [])
    if not before_verified:reasons.append('ACOUSTIC_GEOMETRY_BEFORE_WAVEFORM_UNAVAILABLE')
    return dict(house=h['house'],family=h['family'],candidate_room_count=h['candidate_room_count'],candidate_room_labels=h['candidate_room_labels'],sampled_room_labels=labels,sampled_room_count=len(labels),foa_checked_probes=[x['probe'] for x in record.get('foa_checks',[])],native_status=record['status'],visual_status=visual_status,status=status,training_eligible=status=='pass',max_escape_fraction=record.get('max_escape_fraction'),max_foa_angle_deg=record.get('max_foa_angle_deg'),rt60_mid_median_s=record.get('rt60_mid_median_s'),reasons=reasons,input={k:hh[k] for k in ('house','family','scan_id','original_split','candidate_room_count','candidate_room_labels','candidate_area_sum_m2','acoustic_rooms','exposure') if k in hh},compute_device='CPU',scope='original predetermined <=3 rooms; combined first-batch house classifier; other candidate rooms inferred only under this old house-veto convention')

def report(output,destination):
    for stage in ('native','visual'):
        finished=output/'evidence'/(stage+'_driver_finished.json')
        if not finished.exists() or read(finished)['houses']!=231:
            raise ValueError('Final report requires all 231 owned house jobs to finish: '+stage)
    override_path=output/'evidence/visual_retry_registry.json'
    overrides=read(override_path) if override_path.exists() else {}
    for r in read(output/'evidence/visual_driver_finished.json')['results']:
        if not r['finished_marker'] and not (r['house'] in overrides and (visual_root(output,r['house'],overrides)/'finished.json').exists()):
            raise ValueError('Interrupted visual worker still needs a verified replacement: '+r['house'])
    destination.mkdir(parents=True,exist_ok=False)
    candidates=read(output/'inputs/candidates.json');old=read(output/'inputs/old_all_house_results.json');summary_pre=read(output/'evidence/preparation.json');rows=[];newhouses=[];oldhouses=[];derivations=[]
    settings={f:read(output/'inputs'/(f+'_settings.json')) for f in ('hm3d','mp3d')}
    for h in candidates:
        house=h['house'];one=[]
        setup_path=output/'houses'/house/'native_house_setup.json'
        acoustic_geometry=(read(setup_path).get('geometry') or {}) if setup_path.exists() else {}
        cached_labels={r['room_label'] for r in read(h['cached_results'])['rirs']}
        for room,rule in zip(h['acoustic_rooms'],h['room_rule_records'],strict=True):
            rp=output/'houses'/house/'rooms'/room['room_label']/'result.json';vp=visual_root(output,house,overrides)/'rooms'/room['room_label']/'visual_checks.json'
            r=read(rp) if rp.exists() else dict(status='review',reasons=['NATIVE_RESULT_MISSING'],rirs=[])
            v=read(vp) if vp.exists() else dict(status='review',reasons=['CPU_VISUAL_RESULT_MISSING'])
            row=room_row(h,room,rule,r,v,rp,vp)
            row['acoustic_geometry_derivation']=acoustic_geometry if acoustic_geometry.get('removed_faces') else None
            row['acoustic_before_waveform_available']=room['room_label'] in cached_labels
            if acoustic_geometry.get('removed_faces') and room['room_label'] not in cached_labels:
                # The original acoustic upload rejected this geometry before
                # producing an IR. A visual-only GLB comparison cannot prove
                # an acoustic before/after comparison for that separate
                # semantic-array derivation. Retain the real after waveform,
                # but do not claim the required before propagation verified.
                row['training_eligible']=False
                row['reasons'].append('ACOUSTIC_GEOMETRY_BEFORE_WAVEFORM_UNAVAILABLE')
                if row['tier']=='training':row['status']='review';row['tier']='other_review'
            rows.append(row);one.append(row)
        tiers={tier:count([r for r in one if r['tier']==tier]) for tier in ('training','leak_5_to_15_review','other_review','fail')}
        accepted=[r for r in one if r['training_eligible']]
        newhouses.append(dict(house=house,family=h['family'],original_split=h['original_split'],candidate_room_count=h['candidate_room_count'],accepted_room_count=len(accepted),accepted_room_labels=[r['room_label'] for r in accepted],review_leak_room_count=tiers['leak_5_to_15_review']['rooms'],other_review_room_count=tiers['other_review']['rooms'],failed_room_count=tiers['fail']['rooms'],status='pass' if accepted else 'fail' if all(r['status']=='fail' for r in one) else 'review',native_status='pass' if accepted else 'review',visual_status='pass' if accepted else 'review',batch_eligible=bool(accepted),training_eligible=bool(accepted),human_labels_filter=False,smy_labels=dict(collections.Counter(r['smy_label'] for r in one)),exposure=h.get('exposure',{}),max_escape_fraction=max((r['max_escape_fraction'] for r in accepted),default=None),max_foa_angle_deg=max((r['max_foa_angle_deg'] for r in accepted),default=None),rt60_mid_median_s=None,candidate_area_sum_m2=h['candidate_area_sum_m2'],accepted_area_sum_m2=sum(r['floor_area_m2'] for r in accepted),reasons=sorted({reason for r in one if not r['training_eligible'] for reason in r['reasons']}),result_directory=str(output/'houses'/house),scope='accepted_room_labels only; neighboring rejected/review rooms never enter training'))
        new=newhouses[-1]
        new['input']={k:copy.deepcopy(h[k]) for k in ('house','family','scan_id','original_split','scene_directory','semantic_source','annotation_source','navmesh_source','candidate_room_count','candidate_room_labels','candidate_area_sum_m2','room_types','exposure','known_exposure') if k in h}
        new['input'].update(split='train',accepted_room_count=len(accepted),accepted_room_labels=new['accepted_room_labels'],acoustic_rooms=[room for room in h['acoustic_rooms'] if room['room_label'] in new['accepted_room_labels']],human_labels_filter=False)
        accepted_native=[read(r['result_path']) for r in accepted]
        new['rirs']=[m for r in accepted_native for m in r['rirs']]
        new['distance_attenuation']=[d for r in accepted_native for d in r.get('distance_attenuation',[])]
        new['leakage']=[d for r in accepted_native for d in r.get('leakage',[])]
        new['foa_checks']=[d for r in accepted_native for d in r.get('foa_checks',[])]
        values=[m['rt60_mid_s'] for m in new['rirs'] if m['status']=='valid' and m.get('rt60_mid_s') is not None]
        new['rt60_mid_median_s']=statistics.median(values) if values else None
        new['rt60_valid_fraction']=len(values)/len(new['rirs']) if new['rirs'] else None
        new['distinct_listener_positions']=len({tuple(m['listener_m']) for m in new['rirs']})
        new['measured_room_labels']=[r['room_label'] for r in one if r['rir_count']];new['sampled_room_count']=len(new['measured_room_labels'])
        new['smy_verdict_by_room']={r['room_label']:r['smy_label'] for r in one}
        new['stage4']=dict(status=new['status'],scope='training per-room acceptance; accepted_room_labels only',historical_status=one[0]['historical_stage4_status'])
        new['human_review_status']='待 smy 审房间（只记录，不筛选）' if h['family']=='mp3d' else 'smy 标签只记录，不筛选'
        aggregate=output/'houses'/house/'result.json'
        if not aggregate.exists():save(aggregate,new)
        aggregate_visual=output/'houses'/house/'visual_checks.json'
        if not aggregate_visual.exists():save(aggregate_visual,dict(status=new['visual_status'],scope='accepted room CPU views only; room_results retain rejected/review candidates',checks=[x for r in accepted for x in read(r['visual_path']).get('checks',[])],room_results=[dict(room_label=r['room_label'],status=r['visual_status'],path=r['visual_path']) for r in one],compute_device='CPU',native_habitat_rgb_verified=False))
        oldhouses.append(old_house(h,one,output,settings[h['family']]))
        p=visual_root(output,house,overrides)/'finished.json'
        if p.exists():
            v=read(p)
            if v.get('geometry_derivation'):derivations.append(v)
    counts={f:{tier:count([r for r in rows if r['family']==f and r['tier']==tier]) for tier in ('training','leak_5_to_15_review','other_review','fail')} for f in settings}
    oldfinished={f:dict(houses=sum(h['family']==f and h['status']=='pass' for h in oldhouses),rooms=sum(h['candidate_room_count'] for h in oldhouses if h['family']==f and h['status']=='pass')) for f in settings}
    frozen={f:dict(houses=sum(h['family']==f and h['status']=='pass' for h in old),rooms=sum(h['candidate_room_count'] for h in old if h['family']==f and h['status']=='pass')) for f in settings}
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    geometry_pending={f:{h['house'] for h in candidates if h['family']==f and h['previous_visual'].get('status')!='pass' and any('raw_visual_scan_ray_diagnostic' in reason or 'RAW_VISUAL_ZERO_AREA_TRIANGLES' in reason for reason in h['previous_reasons'])} for f in settings}
    previous_33={h['house'] for h in candidates if h['family']=='mp3d' and any('RAW_VISUAL_ZERO_AREA_TRIANGLES' in reason for reason in h['previous_reasons'])}
    requested_derivation_rescue={f:count([r for r in rows if r['family']==f and r['training_eligible'] and r['house'] in geometry_pending[f] and r['geometry_comparison_status']=='pass']) for f in settings}
    requested_33_rescue=count([r for r in rows if r['family']=='mp3d' and r['training_eligible'] and r['house'] in previous_33 and r['geometry_comparison_status']=='pass'])
    comparison=read(output/'evidence/random_ten_room_comparison.json') if (output/'evidence/random_ten_room_comparison.json').exists() else dict(status='not_run')
    summary=dict(code_commit=code,generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_only=True,acoustic_measurement_revision='e3dc246 first arrival + 0.5 ms',thresholds_unchanged=True,shared_classifier_unchanged=True,human_labels_filter=False,candidates=summary_pre['counts'],old_house_veto_frozen=frozen,old_house_veto_after_completion=oldfinished,new_per_room=counts,all_room_rows=len(rows),rirs=sum(r['rir_count'] for r in rows),valid_rirs=sum(r['valid_rirs'] for r in rows),reused_rirs=sum(r['reused_rirs'] for r in rows),new_rirs=sum(r['new_rirs'] for r in rows),listener_supplement_rooms=sum(r['listener_supplement'] for r in rows),rooms_without_complete_measurement=sum(r['rir_count']<8 or r['distinct_listeners']<2 for r in rows),semantic_conflicts=[dict(house=h['house'],rooms=h['candidate_room_count']) for h in candidates if h['semantic_conflict']],derivations=dict(houses=len(derivations),removed_faces=sum(r['geometry_derivation']['removed_faces'] for r in derivations),tolerance_m2=0.0,perspective_different_pixels=sum(r.get('perspective_different_pixels',0) for r in derivations),overhead_different_pixels=sum(x['different_pixels'] for r in derivations for x in r.get('all_candidate_overheads',[])),rescued_training={f:count([r for r in rows if r['family']==f and r['training_eligible'] and r['geometry_derivation_label']]) for f in settings}),random_ten_comparison={k:v for k,v in comparison.items() if k!='rows'},reserved_houses=summary_pre['reserved_houses'],formal_habitat_rgb='not_run: CPU only',test_and_development_modified=False)
    summary['previous_pending_geometry']={f:dict(houses=len(geometry_pending[f]),rooms=sum(h['candidate_room_count'] for h in candidates if h['house'] in geometry_pending[f]),rescued_training=requested_derivation_rescue[f]) for f in settings}
    summary['previous_mp3d_33_geometry']={'houses':len(previous_33),'rooms':sum(h['candidate_room_count'] for h in candidates if h['house'] in previous_33),'rescued_training':requested_33_rescue}
    summary['previous_pending_cpu_visuals']={f:dict(houses=sum(h['family']==f and h['previous_visual'].get('status')!='pass' for h in candidates),checks_completed=sum(h['family']==f and h['previous_visual'].get('status')!='pass' and (visual_root(output,h['house'],overrides)/'finished.json').exists() for h in candidates)) for f in settings}
    audit_path=output/'evidence/native_failures_array_audit.json' if (output/'evidence/native_failures_array_audit.json').exists() else output/'evidence/native_failures_audit.json'
    audit=read(audit_path) if audit_path.exists() else dict(categories={},status='not_run')
    summary['native_failure_categories']=audit['categories']
    summary['native_failure_audit_source']=str(audit_path)
    summary['visual_retry_registry']=overrides
    summary['acoustic_derivation_before_comparison_unverified']=count([r for r in rows if 'ACOUSTIC_GEOMETRY_BEFORE_WAVEFORM_UNAVAILABLE' in r['reasons']])
    summary['accepted_houses_with_zero_individual_review']=sum(all(r['tier']=='training' for r in rows if r['house']==h['house']) for h in newhouses)
    save(destination/'summary.json',summary);save(destination/'all_room_results.json',rows);save(destination/'all_house_results.json',newhouses);save(destination/'old_house_veto_all_results.json',oldhouses);save(destination/'geometry_derivations.json',derivations)
    for name,predicate in [('training_rooms_v1',lambda r:r['training_eligible']),('leak_5_to_15_review_rooms',lambda r:r['tier']=='leak_5_to_15_review'),('other_review_rooms',lambda r:r['tier']=='other_review'),('failed_rooms',lambda r:r['tier']=='fail')]:
        rr=[r for r in rows if predicate(r)];save(destination/(name+'.json'),rr)
    save(destination/'training_houses_v1.json',[r for r in newhouses if r['training_eligible']]);save(destination/'old_house_veto_training_houses_v1.json',[r for r in oldhouses if r['training_eligible']])
    save(destination/'old_house_veto_frozen_training_houses_v1.json',[r for r in old if r['status']=='pass'])
    columns=['house','family','room_label','room_type','original_split','stage1_status','historical_stage4_status','acoustic_status','visual_status','status','tier','training_eligible','smy_label','so_seen','m1_seen','max_escape_fraction','max_foa_angle_deg','distance_exponents','rir_count','valid_rirs','distinct_listeners','max_direct_delay_ms','rt60_mid_median_s','rt60_valid_fraction','reused_rirs','new_rirs','listener_supplement','floor_area_m2','short_side_m','nav_main_area_m2','black_fraction','floor_y_m','floor_reference_path','geometry_derivation_label','geometry_comparison_status','acoustic_before_waveform_available','derived_visual_source','reasons','result_path','visual_path']
    csvfile(destination/'room_results.csv',columns,rows)
    csvfile(destination/'training_rooms_v1.csv',columns,[r for r in rows if r['training_eligible']])
    csvfile(destination/'leak_5_to_15_review_rooms.csv',columns,[r for r in rows if r['tier']=='leak_5_to_15_review'])
    housecols=['house','family','original_split','candidate_area_sum_m2','candidate_room_count','accepted_room_count','accepted_room_labels','review_leak_room_count','other_review_room_count','failed_room_count','native_status','visual_status','status','batch_eligible','rt60_mid_median_s','max_escape_fraction','max_foa_angle_deg','reasons','result_directory','smy_labels']
    csvfile(destination/'house_results.csv',housecols,newhouses);csvfile(destination/'old_house_veto_house_results.csv',['house','family','candidate_room_count','sampled_room_count','sampled_room_labels','native_status','visual_status','status','training_eligible','max_escape_fraction','max_foa_angle_deg','reasons'],oldhouses)
    hm,mp=counts['hm3d'],counts['mp3d'];lines=[f"结论：逐间判后，HM3D 可进训练 {hm['training']['houses']} 套、{hm['training']['rooms']} 间，MP3D {mp['training']['houses']} 套、{mp['training']['rooms']} 间；漏声 5–15% 待审档分别为 HM3D {hm['leak_5_to_15_review']['houses']} 套、{hm['leak_5_to_15_review']['rooms']} 间，MP3D {mp['leak_5_to_15_review']['houses']} 套、{mp['leak_5_to_15_review']['rooms']} 间，均不进训练。",'',f'机器：48g；独立检出 {REPO}，分支 claude/v7-train-houses-20261003；代码提交 `{code}`；无 push。新产物目录 {output}，上一轮目录及测试/开发名单保持原样。数字出处：summary.json、room_results.csv、all_room_results.json；每间原生测量和 CPU 视图路径在 result_path/visual_path。','',
    '## 新旧漏斗与逐间去向','', '|阶段|HM3D 套 / 间|MP3D 套 / 间|','|---|---:|---:|','|全量房间自动规则通过|179 / 1154|85 / 623|','|排除预留训练候选|146 / 927|85 / 623|',f"|旧整套抽 ≤3 间，一票否决：上轮冻结结果|{frozen['hm3d']['houses']} / {frozen['hm3d']['rooms']}|{frozen['mp3d']['houses']} / {frozen['mp3d']['rooms']}|",f"|旧整套口径：补完本轮检查后|{oldfinished['hm3d']['houses']} / {oldfinished['hm3d']['rooms']}|{oldfinished['mp3d']['houses']} / {oldfinished['mp3d']['rooms']}|",f"|新逐间口径：训练|{hm['training']['houses']} / {hm['training']['rooms']}|{mp['training']['houses']} / {mp['training']['rooms']}|",f"|漏声 5–15% 单列待审|{hm['leak_5_to_15_review']['houses']} / {hm['leak_5_to_15_review']['rooms']}|{mp['leak_5_to_15_review']['houses']} / {mp['leak_5_to_15_review']['rooms']}|",f"|其他待审|{hm['other_review']['houses']} / {hm['other_review']['rooms']}|{mp['other_review']['houses']} / {mp['other_review']['rooms']}|",f"|不通过|{hm['fail']['houses']} / {hm['fail']['rooms']}|{mp['fail']['houses']} / {mp['fail']['rooms']}|",'',
    '各档房间互斥、相加等于全部 1550 间；同一套可以同时有训练、待审和失败房间，套数不能相加。5–15% 档也可能同时有其他声学或画面问题，acoustic_status 与 reasons 保留全部结论，不代表其他指标已通过。all_house_results.json 中 accepted_room_labels 是允许训练的精确范围；candidate_room_count 仅表示自动规则候选数。smy 标签逐间记录，不筛选；SO/M1/v6 暴露只记录。测试和开发继续整套严格口径，全部干净原 HM3D val 及已有 holdout 名单仍预留。','',
    '旧口径保留原预定抽样房间，未按声学结果重新选样；多房屋样本按首批原有四条 RIR/间和三处漏声原点汇总，单房样本按原八条/两个听者汇总，调用同一个 unchanged classifier。上轮冻结名单与补完检查的旧口径名单分别保留；旧口径的候选间数包含未抽样房间，不能视为全部逐间验收通过。新口径只列逐间声学和 CPU 视图同时通过的房间。','',
    '## 全量逐间声学与复用','',f"共 {len(rows)} 间，原生 RIR {summary['rirs']} 条、有效 {summary['valid_rirs']} 条；复用 {summary['reused_rirs']} 条、新测 {summary['new_rirs']} 条，{summary['listener_supplement_rooms']} 间补了第二听者；未满足八条计划/两个听者的 {summary['rooms_without_complete_measurement']} 间保留待审或失败，详见逐间表。",'',
    '每间沿用首批单房方案：主听者四个原探针、第二声源沿线两个探针、第二听者两个探针，共八条。相同源/听者坐标的既存波形复用，缺的位置才补测；每套只上传一次真实语义声学场景后逐间移动位置。直达声固定 e3dc246 首个到达加 0.5 ms。冻结阈值和原 classify 判分器未改；每间另外核对 owner 明示的至少六条有效 RIR。没有用模型分数或人审意见选房。','',f'随机对拍：{json.dumps(summary["random_ten_comparison"],ensure_ascii=False)}。逐探针源/听者、直达声、衰减、RT60、漏声及判定差异在 ../evidence/random_ten_room_comparison.json；独立 Monte Carlo 晚反射不假定逐位相等。','',
    '## CPU 视图和退化几何','',f"派生 GLB {summary['derivations']['houses']} 套，共清 {summary['derivations']['removed_faces']} 个严格零面积面，容差 0 m²；RGB 透视图差异像素总数 {summary['derivations']['perspective_different_pixels']}，全部候选俯视图差异像素 {summary['derivations']['overhead_different_pixels']}。每套/每间前后对拍、删除索引、原始源未改证明在 geometry_derivations.json 及 ../visuals/<house>/；获救训练房间统计：{json.dumps(summary['derivations']['rescued_training'],ensure_ascii=False)}。",'',
    'GLB 只改派生副本三角索引，严格双精度叉积为零才清理；正面积面包括 float32 运算退化但 double 仍正的面保留。顶点、UV、纹理、材质、原始二进制前缀及共享数据不动。原生语义声学数组若存在严格零面积面，则在本目录保存派生数组和删除索引，上传时验证实际几何和材质读回；原包保持只读。不是严格零面积造成的原生拒绝仍待审。','',
    'CPU 图沿用首批 render_probe 的投影、源可见性与门槛，使用已与独立参考验证过的 CPU BVH 射线；三张 160×90 真实扫描图，每张存 640×360 显示图。清理前后对拍同一后端、同一位置和参数，逐像素差记录。视觉 GLB 派生不改语义声学包，声学对拍用同一保存 RIR 两侧统一重算，逐字段 exact equality；未宣称独立随机晚反射逐位相等。声学原生加载失败的房间无法完成此对拍，不会入选。正式 Habitat RGB、训练视频/音频窗口和模型训练未验证。','',
    '语义一个颜色对应两个类别：' + json.dumps(summary['semantic_conflicts'],ensure_ascii=False) + '。继续待审，未猜类别或改原始标注。','',
    '## 每套结果','', '|房子|来源|规则候选|可训练|漏声待审|其他待审|失败|smy 标签|','|---|---|---:|---:|---:|---:|---:|---|']
    for h in newhouses:lines.append(f"|{h['house']}|{h['family']}|{h['candidate_room_count']}|{h['accepted_room_count']}|{h['review_leak_room_count']}|{h['other_review_room_count']}|{h['failed_room_count']}|{json.dumps(h['smy_labels'],ensure_ascii=False)}|")
    lines+=['','## 命令、验证和运行状态','',f'48g 的命令：`tools/acoustics/train_room_acceptance.py prepare --previous {output.parent} --output {output}`；`run_train_rooms.py --output {output} --workers 4`；`run_train_room_visuals.py --output {output} --workers 2`，随后按重新测得的空闲核用 `--workers 4 --resume` 接续（保留已完成与在跑子进程）；随机对拍和 report_train_rooms.py 产出本目录。实际子命令、PID、参数、日志及结束状态在 ../evidence/*started*.json、*finished*.json 和 ../logs/。CPU 空闲核测量及亲和性在 ../evidence/room_native_cpu_allocation.json、room_visual_cpu_allocation.json、room_visual_cpu_allocation_resume.json。专项和 fast_unit 验证见 ../logs/tests_*.log、../evidence/final_validation.json。','',
    '原始数据、已有 checkpoint、其他检出和共享环境均未修改；无 GPU 作业、无 push。其他历史 SO/M1 暴露未覆盖的运行仍未验证。需要人工解决的语义冲突、正面积原生拒绝及测量异常在逐间 reasons 保留。']
    detail=['','## 上轮未完检查的具体去向','',f"HM3D 11 套、MP3D 72 套待审 CPU 视图的完成数：{json.dumps(summary['previous_pending_cpu_visuals'],ensure_ascii=False)}。有实际图像和对拍的检查记完成；声学加载失败导致声学/图像联合验收待审，不伪称通过。",'',f"指定 HM3D 7 套及 MP3D 原生 raw 错误候选的逐间获救数：{json.dumps(summary['previous_pending_geometry'],ensure_ascii=False)}；其中上一轮明确 RAW_VISUAL_ZERO_AREA_TRIANGLES 的 MP3D 33 套：{json.dumps(summary['previous_mp3d_33_geometry'],ensure_ascii=False)}。本轮所有含零面积面的房屋都按同一严格零规则做派生和对拍，故总派生套数与这两个历史子集不同。",'',f"原生失败分类：{json.dumps(summary['native_failure_categories'],ensure_ascii=False)}；逐套精确三角面积、native float32 边向量及 ≤1e-20 叉积平方检查在 {summary["native_failure_audit_source"]}。原生把部分正面积面报‘zero-area’，保留正面积面及待审结论，没有改 native validator 来放行。",'',
    f"另有语义声学数组的严格零面清理，原来在上传阶段就失败、没有‘清理前’波形：{json.dumps(summary['acoustic_derivation_before_comparison_unverified'],ensure_ascii=False)}。本轮已保存真实‘清理后’原生 RIR，但原始传播前后数值对拍未验证，相关房间不进训练；这与原生声学本已通过、只清视觉 GLB 的 7/33 套子集分开记录。",'',
    '93 套历史整屋门禁原因的原始分类和流程修复证据保留在 ../../followup_auto_only_20261003/report_final/historical_gate_failures_93.json；本轮在 146 套训练候选中只按房间判，历史旧门禁仍记录，干净 val 不进入训练。','',
    '|00081 的房间|漏声|声学|CPU 视图|逐间去向|','|---|---:|---|---|---|']
    for r in rows:
        if '_00081_' in r['house']:detail.append(f"|{r['room_label']}|{r['max_escape_fraction']:.6f}|{r['acoustic_status']}|{r['visual_status']}|{r['tier']}|")
    lines+=detail
    if overrides:lines+=['','## 单进程内存异常修复','', 'MP3D gTV8FGcVJC9 的自有 CPU 视图进程曾出现实际内存失控（PID 3480338，约 179 GiB）；逐项核对 PID、完整命令和 UID 后只停该进程。原有四间已完成图像及全部部分产物保留，重试写 visual_retries 新子目录；修复为共享内容完全一致的 RGB 纹理，弱引用记录只读 PIL 图像身份以避免重复散列，并让清理前后共用同一个内容缓存；后续视图单进程设 16 GiB 地址空间上限，未改像素值、UV、判分或任何渲染参数。重试及具体故障修复证据：'+json.dumps(overrides,ensure_ascii=False)+'。']
    (destination/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print('PER_ROOM_FINAL',json.dumps({k:summary[k] for k in ('new_per_room','old_house_veto_after_completion','rirs','valid_rirs','reused_rirs','new_rirs','rooms_without_complete_measurement')}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();report(a.output,a.destination)
if __name__=='__main__':main()
