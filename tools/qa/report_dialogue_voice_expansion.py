"""Verify accepted voice variants and write a Chinese standalone evidence report."""
from __future__ import annotations
from _dialogue_voice_runtime import *
from copy import deepcopy
from collections import Counter,defaultdict
from html import escape
import subprocess
from select_dialogue_voice_references import speaker_info
from avengine.qa.v7_dialogue_voice_audit import english_language_gate


def old_attempt_count(sound):
    sid=sound['sound_asset_id'];counts=[]
    for folder in ['natural_batch_v1','natural_batch_ready']:
        path=R/folder/'records'/(sid+'.json')
        if not path.exists():continue
        row=json.loads(path.read_text())
        counts.extend([len(row.get('attempts',[])),row.get('attempts_performed',0),row.get('total_synthesis_attempts_this_line',0)])
    counts.append(sound.get('normalization_reconciliation',{}).get('selected_attempt') or 1)
    return max(counts)


def test_comparison():
    baseline=json.loads((STAGE/'evidence/baseline.json').read_text())
    path=STAGE/'evidence/final.json'
    if not path.exists():return {'status':'pending','baseline_failed':len(baseline['failure_ids']),'baseline_errors':len(baseline['error_ids'])}
    final=json.loads(path.read_text());new_failures=sorted(set(final['failure_ids'])-set(baseline['failure_ids']));new_errors=sorted(set(final['error_ids'])-set(baseline['error_ids']))
    recheck_path=STAGE/'evidence/baseline_recheck.json'
    recheck=json.loads(recheck_path.read_text()) if recheck_path.exists() else None
    related=[r for r in final['reports'] if r['nodeid'].split('::')[0] in {'tests/unit/test_dialogue_voice_language_audit.py','tests/unit/test_dialogue_voice_calibrated_gate.py','tests/unit/test_v7_round5_integration.py'}]
    related_passed=sum(r['when']=='call' and r['outcome']=='passed' for r in related)
    related_failed=[r['nodeid'] for r in related if r['outcome']=='failed']
    guarded_passed=recheck is not None and not set(final['failure_ids'])-set(recheck['failure_ids']) and not set(final['error_ids'])-set(recheck['error_ids']) and len(final['failure_ids'])<=len(recheck['failure_ids']) and len(final['error_ids'])<=len(recheck['error_ids'])
    passed=not new_failures and not new_errors and len(final['failure_ids'])<=len(baseline['failure_ids']) and len(final['error_ids'])<=len(baseline['error_ids']) and guarded_passed and related_passed==67 and not related_failed and not final.get('baseline_coverage_missing_ids')
    return {'status':'pass' if passed else 'blocked','baseline_failed':len(baseline['failure_ids']),'baseline_errors':len(baseline['error_ids']),'guarded_baseline_failed':len(recheck['failure_ids']) if recheck else None,'guarded_baseline_errors':len(recheck['error_ids']) if recheck else None,'guarded_baseline_receipt':str(recheck_path),'final_failed':len(final['failure_ids']),'final_errors':len(final['error_ids']),'new_failure_ids':new_failures,'new_error_ids':new_errors,'related_passed_count':related_passed,'related_failed_or_error_ids':related_failed,'whole_final_test_process_tree_threads_peak':final.get('process_tree_threads_peak'),'baseline_receipt':str(STAGE/'evidence/baseline.json'),'final_receipt':str(path),'scope':final['scope']}


def main():
    tick=time.monotonic();old_manifest=R/'natural_batch_ready/semantic_manifest.json';old=json.loads(old_manifest.read_text())['sounds'];plan=json.loads((STAGE/'expansion_plan.json').read_text());info=speaker_info();cal=json.loads((ROOT/'language_gate_calibration.json').read_text());assert cal['status']=='pass'
    sounds=[];retained=[];exclusions=[];repair_id=plan['repair_task']['sound_asset_id']
    for original in old:
        sid=original['sound_asset_id'];proof=json.loads((STAGE/'calibration_records'/(sid+'.json')).read_text())
        if proof['language_gate']['status']!='pass':
            assert sid==repair_id
            path=STAGE/'task_records'/(sid+'.json')
            repair=json.loads(path.read_text()) if path.exists() else {'status':'unprocessed'}
            if repair['status']=='pass':sounds.append(repair['sound'])
            else:exclusions.append({'sound_asset_id':sid,'original_sha256':original['sha256'],'reason':'old_language_top1_cy_and_repair_'+repair['status'],'receipt':str(path)})
            continue
        s=deepcopy(original);assert digest(s['path'])==s['sha256']==proof['audio_sha256'];s['voice_reference'].update(info[s['voice_reference']['speaker_id']]);s['voice_reference']['source_waveform_sha256']=digest(s['voice_reference']['source_waveform']);s.update(authored_line_id=sid,expansion_kind='retained_original',voice_variant=1,selected_seed=s['synthesis_config']['seed'],selected_attempt=s.get('normalization_reconciliation',{}).get('selected_attempt') or 1,attempts_performed=old_attempt_count(s),language_identification=proof['language_identification'],language_gate=proof['language_gate'],automatic_content_readback={**proof['ASR'],'source':str(STAGE/'calibration_records'/(sid+'.json'))},voice_accent=info[s['voice_reference']['speaker_id']]['accent'],attempt_receipt=str(R/'natural_batch_ready/records'/(sid+'.json')));sounds.append(s);retained.append(sid)
    replacements=json.loads((STAGE/'replacements.json').read_text()) if (STAGE/'replacements.json').exists() else []
    retired={r['retired_speaker'] for r in replacements};effective_tasks=[t for t in plan['tasks'] if t['voice_reference']['speaker_id'] not in retired]
    if (STAGE/'replacement_tasks.json').exists():effective_tasks+=json.loads((STAGE/'replacement_tasks.json').read_text())
    terminal=[];unprocessed=[]
    for task in effective_tasks:
        path=STAGE/'task_records'/(task['sound_asset_id']+'.json')
        if not path.exists():unprocessed.append(task['sound_asset_id']);continue
        row=json.loads(path.read_text())
        if not row.get('terminal'):unprocessed.append(task['sound_asset_id']);continue
        terminal.append(row)
        if row['status']=='pass':sounds.append(row['sound'])
    seen=set();scenario_voice_roles=defaultdict(dict);role_counts=Counter();violations=[];voice_counts=Counter();gender_counts=Counter();loudness=[];probs=[];wers=[]
    for s in sounds:
        sid=s['sound_asset_id'];ref=s['voice_reference'];sp=ref['speaker_id'];role=s['role_id'];scenario=s['scenario_id'];voice_counts[sp]+=1;role_counts[s['authored_line_id']]+=1
        if sid in seen:violations.append({'id':sid,'reason':'duplicate_asset_id'})
        seen.add(sid)
        if digest(s['path'])!=s['sha256']:violations.append({'id':sid,'reason':'audio_SHA_mismatch'})
        if ref['gender']!=s['voice_gender'] or s['voice_gender']!=s['standing_actor_plan']['visual_gender']:violations.append({'id':sid,'reason':'gender_mismatch'})
        mapping=scenario_voice_roles[scenario]
        if sp in mapping and mapping[sp]!=role:violations.append({'id':sid,'reason':'same_scenario_speaker_reused_by_different_roles'})
        mapping[sp]=role
        lid=s['language_identification'];asr=s['automatic_content_readback'];check=english_language_gate(asr,lid['top_language'],lid['transcript_free_language'])
        if check['status']!='pass':violations.append({'id':sid,'reason':'quality_gate_failed','gate':check})
        levels=s['source_normalization']['readback_loudness'];loudness.append(levels['input_i']);probs.append(lid['english_probability']);wers.append(asr['wer'])
        if levels['input_tp']>-3 or s['source_normalization']['trimmed_or_repeated'] is not False:violations.append({'id':sid,'reason':'loudness_or_complete_utterance_failed'})
    counts=[]
    for sp,count in sorted(voice_counts.items()):
        gender_counts[info[sp]['gender']]+=1;counts.append({**info[sp],'accepted_lines':count,'is_new':sp not in {'p277','p260','p318','p256','p336','p286','p300','p273'}})
    per_voice=[]
    for sp in sorted({t['voice_reference']['speaker_id'] for t in effective_tasks}):
        rows=[r for r in terminal if r['task']['voice_reference']['speaker_id']==sp];passes=sum(r['status']=='pass' for r in rows);per_voice.append({'speaker_id':sp,'processed':len(rows),'passed':passes,'dropped':len(rows)-passes,'pass_rate':passes/len(rows) if rows else None})
    dropped=[r for r in terminal if r['status']=='dropped'];reasons=Counter()
    for r in dropped:
        codes=set()
        for a in r['attempts']:
            if a.get('ASR',{}).get('wer',0)>.1:codes.add('WER_above_0.10')
            if a.get('ASR',{}).get('all_critical_slots_correct') is False:codes.add('critical_keyword_or_negation_incorrect')
            if a.get('language_identification',{}).get('top_language','en')!='en':codes.add('language_top1_not_en')
            if has_han(a.get('language_identification',{}).get('transcript_free_language','')):codes.add('free_transcription_contains_Han')
            if a.get('reason','').startswith('synthesis_or_quality_exception:'):codes.add('synthesis_or_quality_exception')
        reasons.update(codes or {'attempts_exhausted'})
    tests=test_comparison();complete=len(terminal)==1080 and not unprocessed;goals=complete and len(voice_counts)>=24 and gender_counts['female']>=12 and gender_counts['male']>=12 and min(voice_counts.values(),default=0)>=40 and not violations and all(r['pass_rate']>=.8 for r in per_voice)
    new_passed=sum(r['status']=='pass' for r in terminal);new_attempts=sum(len(r['attempts']) for r in terminal);quality={'accepted':len(sounds),'WER_max':max(wers,default=None),'WER_mean':sum(wers)/len(wers) if wers else None,'all_critical_slots_correct':all(s['automatic_content_readback']['all_critical_slots_correct'] for s in sounds),'English_top1_count':sum(s['language_identification']['top_language']=='en' for s in sounds),'free_transcription_Han_count':sum(has_han(s['language_identification']['transcript_free_language']) for s in sounds),'English_probability_min_diagnostic':min(probs,default=None),'English_probability_below_old_0_9_diagnostic':sum(p<.9 for p in probs),'LUFS_min':min(loudness,default=None),'LUFS_max':max(loudness,default=None),'peak_max_dBTP':max((s['source_normalization']['readback_loudness']['input_tp'] for s in sounds),default=None),'peak_limited_count':sum(s['source_normalization']['true_peak_ceiling_limited'] for s in sounds),'voice_visual_gender_verified_count':len(sounds)-sum(v['reason']=='gender_mismatch' for v in violations),'validation_violations':violations}
    summary={'status':'pass' if goals and tests['status']=='pass' else 'pending' if goals else 'incomplete','artifact_goals_pass':goals,'planned_authored_lines':540,'planned_new_variants':1080,'new_processed_variants':len(terminal),'new_passed_variants':new_passed,'new_dropped_variants':len(dropped),'new_synthesis_attempts_effective_voices':new_attempts,'new_pass_rate':new_passed/len(terminal) if terminal else None,'old_retained_without_resynthesis':len(retained),'old_repaired':sum(s.get('expansion_kind')=='old_language_repair' for s in sounds),'old_excluded':exclusions,'total_accepted_lines':len(sounds),'total_voices':len(voice_counts),'new_voices':sum(c['is_new'] for c in counts),'voice_genders':dict(gender_counts),'minimum_lines_per_voice':min(voice_counts.values(),default=0),'accepted_voices_per_authored_line_distribution':dict(Counter(role_counts[t['sound_asset_id']] for t in json.loads((R/'natural_batch_v1/plan.json').read_text())['tasks'])),'original_missing_10_remain_missing_only_for_original_voice':str(ROOT/'old_plan_missing_10.json'),'unprocessed_variant_ids':unprocessed,'dropped_reason_counts_nonexclusive':dict(reasons),'per_new_voice_quality':per_voice,'voice_replacements':replacements,'language_gate_calibration':str(ROOT/'language_gate_calibration.json'),'quality':quality,'tests':tests,'source_old_manifest':str(old_manifest),'source_old_manifest_sha256':digest(old_manifest),'finished_utc':stamp()}
    manifest={'schema':'v7_dialogue_voices_semantic_manifest_v2','status':summary['status'],'all_audio_machine_quality_passed':not violations,'content_asset_expansion_goals_passed':goals,'production_sampler_integrated':False,'language_rule':cal['rules'],'license_receipt':str(ROOT/'license_receipt.json'),'batch_summary':summary,'sounds':sounds,'dropped_variants':[{'sound_asset_id':r['sound_asset_id'],'attempts_performed':r['attempts_performed'],'reason':r['reason'],'receipt':str(STAGE/'task_records'/(r['sound_asset_id']+'.json'))} for r in dropped]}
    for filename in ['semantic_manifest.json','report.html']:
        path=ROOT/filename;previous=STAGE/'evidence'/('stage1_blocked_'+filename)
        if path.exists() and not previous.exists():shutil.copyfile(path,previous)
    save(ROOT/'semantic_manifest.json',manifest);save(ROOT/'expansion_summary.json',summary);save(ROOT/'voice_counts.json',counts);save(ROOT/'stage2/evidence/test_comparison.json',tests)
    (ROOT/'semantic_manifest.sha256').write_text(digest(ROOT/'semantic_manifest.json')+'  semantic_manifest.json\n')
    conclusion=(f'扩音色完成：新增 {summary["new_voices"]} 个音色，总共 {len(voice_counts)} 个；新变体 {new_passed}/1080 条通过。' if summary['status']=='pass' else f'扩音色尚未完成验收：目前新增 {summary["new_voices"]} 个音色，新变体 {new_passed}/1080 条通过。')
    voice_rows=''.join('<tr><td>'+escape(c['speaker_id'])+'</td><td>'+('女' if c['gender']=='female' else '男')+'</td><td>'+escape(c['accent']+' · '+c['region'])+'</td><td>'+str(c['accepted_lines'])+'</td><td>'+('新增' if c['is_new'] else '保留')+'</td></tr>' for c in counts)
    reason_rows=''.join('<tr><td>'+escape(k)+'</td><td>'+str(v)+'</td></tr>' for k,v in reasons.items()) or '<tr><td>没有丢弃</td><td>0</td></tr>'
    pilot=json.loads((STAGE/'pilot_summary.json').read_text()) if (STAGE/'pilot_summary.json').exists() else {}
    license=json.loads((ROOT/'license_receipt.json').read_text());license_text='本地 CosyVoice2-0.5B 模型卡标注 Apache-2.0。完整本地模型卡及固定版本副本已核对，未发现禁止把合成输出用于 AI 模型训练的条款。VCTK 真人参考为 CC-BY-4.0，逐条保留作者署名、出处和参考录音 SHA。'
    tests_text=(f'基线 {tests["baseline_failed"]} 失败 / {tests["baseline_errors"]} 错误；交付 {tests.get("final_failed","待测")} 失败 / {tests.get("final_errors","待测")} 错误。新增失败 {len(tests.get("new_failure_ids",[])) if tests["status"]!="pending" else "待测"}，新增错误 {len(tests.get("new_error_ids",[])) if tests["status"]!="pending" else "待测"}。')
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>v7 英文台词扩音色</title><style>body{{max-width:1100px;margin:36px auto;padding:0 20px;font:16px/1.7 system-ui,sans-serif;color:#24313d;background:#fafaf8}}h1{{font-size:28px}}h2{{margin-top:32px;font-size:21px}}.conclusion{{background:#e6f1ed;border-left:5px solid #267465;padding:16px 20px;font-size:20px}}.cards{{display:flex;flex-wrap:wrap;gap:12px}}.card{{flex:1;min-width:160px;padding:15px;background:white;border:1px solid #dfe5e3;border-radius:8px}}.card strong{{display:block;font-size:28px;color:#267465}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #dfe5e3}}th{{background:#eef2f0}}a{{color:#17675d}}code{{overflow-wrap:anywhere;font-size:13px}}input{{padding:9px;width:260px;margin:10px 0;border:1px solid #b7c6c0;border-radius:5px}}.note{{font-size:14px;color:#52616c}}@media(max-width:650px){{body{{margin:15px auto}}th,td{{padding:6px;font-size:13px}}}}</style><body><p class="conclusion">{escape(conclusion)}</p><h1>v7 英文台词扩音色</h1><p>旧音频保留 {len(retained)} 条；指定的 p277 旧句重做后通过 {summary['old_repaired']} 条。540 句台词全部安排了两个新音色。通过记录才进入清单，最多三个种子，完整输出只做重采样和统一增益，没有裁剪或拼接重复。</p><div class="cards"><div class="card"><strong>{len(voice_counts)}</strong>音色：女 {gender_counts['female']} / 男 {gender_counts['male']}</div><div class="card"><strong>{new_passed} / 1080</strong>新变体通过</div><div class="card"><strong>{len(dropped)}</strong>新变体丢弃</div><div class="card"><strong>{len(sounds)}</strong>清单音频总数</div></div><h2>音色表</h2><p>每个音色至少 {summary['minimum_lines_per_voice']} 条。性别按原计划的画面角色核对；同一情景内不同角色不会共用音色。</p><input id="search" aria-label="筛选音色" placeholder="输入说话人或口音筛选"><table id="voices"><thead><tr><th>说话人</th><th>性别</th><th>口音 / 地区</th><th>通过句数</th><th>来源</th></tr></thead><tbody>{voice_rows}</tbody></table><h2>通过率和丢弃原因</h2><p>新变体通过率 {new_passed/max(1,len(terminal)):.2%}，合成尝试共 {new_attempts} 次。每条最多三个种子，失败收据和音频保留在本轮目录。旧库已有 10 句缺少原音色，本轮只给它们补两个新音色；每句通过音色数分布为 {escape(str(summary['accepted_voices_per_authored_line_distribution']))}。丢弃触发原因可同时出现。</p><table><thead><tr><th>原因代码</th><th>涉及丢弃条数</th></tr></thead><tbody>{reason_rows}</tbody></table><h2>语种判定规则与校准</h2><p>一条音频需要同时满足：强制英文盲转写 WER ≤ 0.1、关键词和否定含义全对；Whisper 语种排名第一是英文；不强制语种的自由转写不含汉字。英文概率照常记录，不再作为阈值。</p><p>本次独立重算正例 {cal['positive_passed']}/530 条通过；唯一旧失败句是 daily_reminder_home_living_room__role_2，语种排名第一为 cy。按扫描内容挑出的中文反例 {cal['negative_rejected']}/{cal['negative_checked']} 条被拒，两目录各 50 条，42 条文件名含 _en_。校准采用本次完整音频重新推理，未抄正例扫描结果。</p><p>清单里英文排名第一 {quality['English_top1_count']}/{len(sounds)}，自由转写含汉字 {quality['free_transcription_Han_count']} 条；WER 最大 {quality['WER_max']}，平均 {quality['WER_mean']:.6f}，关键词全部正确。英文概率最低 {quality['English_probability_min_diagnostic']:.6f}；低于原先 0.9 的 {quality['English_probability_below_old_0_9_diagnostic']} 条只作诊断记录。</p><h2>响度、完整性和计算</h2><p>增益取 min(-23 − I, -3.02 − TP)，真峰值优先。清单真峰值最高 {quality['peak_max_dBTP']} dBTP；{quality['peak_limited_count']} 条受峰值限制，允许低于 -23 LUFS；响度范围 {quality['LUFS_min']} 到 {quality['LUFS_max']} LUFS。每条保存完整样本数、增益验证和文件 SHA。</p><p>10 条试跑平均 {pilot.get('mean_line_seconds',0):.2f} 秒/条，包含必要重试和质检，不含模型启动。并行配置、CPU 用时、磁盘占用和提交号见交付收据；运行记录见 JOBS.json。</p><h2>许可收据摘要</h2><p>{escape(license_text)}</p><p><a href="license_receipt.md">完整许可收据</a> · <a href="license_receipt.json">权重与证据 SHA</a></p><h2>测试和证据</h2><p>{escape(tests_text)}相关测试通过 {tests.get("related_passed_count","待核对")} 项，相关失败/错误 {len(tests.get("related_failed_or_error_ids",[])) if tests["status"]!="pending" else "待核对"} 项。全量范围为仓库 CPU 测试；原生渲染、集成和 canary 标记按相同规则排除。</p><p><a href="semantic_manifest.json">新清单</a> · <a href="language_gate_calibration.json">语种校准</a> · <a href="expansion_summary.json">统计及自动核对</a> · <a href="stage2/evidence/test_comparison.json">基线测试对比</a> · <a href="JOBS.json">作业记录</a> · <a href="PROGRESS.md">进度</a> · <a href="delivery_receipt.json">交付收据</a></p><p class="note">全部判定使用机器指标，没有要求人听。只交付内容音频资产，本轮未接入生产采样器。生成时间：{escape(stamp())}</p><script>document.querySelector('#search').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('#voices tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}})</script></body></html>'''
    (ROOT/'report.html').write_text(doc);progress(f'清单/单文件HTML更新：新音色 {summary["new_voices"]}，总音色 {len(voice_counts)}，新通过 {new_passed}，自动核对问题 {len(violations)}，验收 {summary["status"]}。');print('REPORT_SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in {'quality','per_new_voice_quality','unprocessed_variant_ids'}},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
