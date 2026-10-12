#!/usr/bin/env python3
"""Build a Chinese standalone audit page and a clearly blocked retained-source manifest."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import datetime
import hashlib
import html
import json
from pathlib import Path


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_speakers(path):
    rows={}
    for line in path.read_text().splitlines()[1:]:
        tokens=line.split()
        if len(tokens)<4:
            continue
        rows[tokens[0]]={'speaker_id':tokens[0],'gender':{'F':'female','M':'male'}[tokens[2]],'accent':tokens[3],'region':' '.join(tokens[4:])}
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--speaker-info',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    plan=json.loads(Path(args.plan).read_text())
    old=json.loads(Path(args.manifest).read_text())
    baseline=json.loads((out/'language_baseline_audit.json').read_text())
    license_receipt=json.loads((out/'license_receipt.json').read_text())
    if baseline['threshold_adoptable']:
        raise ValueError('This blocked-delivery reporter must not replace the successful synthesis workflow')
    speakers=read_speakers(Path(args.speaker_info))
    tasks={t['sound_asset_id']:t for t in plan['tasks']}
    retained=[]
    violations=[]
    by_scenario=defaultdict(lambda:defaultdict(set))
    counts=Counter()
    language_counts=Counter()
    wer=[]
    lufs=[]
    peaks=[]
    for original in old['sounds']:
        sound=deepcopy(original)
        sid=sound['sound_asset_id'];task=tasks[sid]
        lid=json.loads((out/'language_records'/(sid+'.json')).read_text())
        voice=sound['voice_reference']['speaker_id'];speaker=speakers[voice]
        source_record=Path(args.plan).parent/'records'/(sid+'.json')
        original_record=json.loads(source_record.read_text())
        final_record=Path(args.manifest).parent/'records'/(sid+'.json')
        ready_record=json.loads(final_record.read_text())
        attempts=deepcopy(original_record.get('attempts',[]))
        actual_count=ready_record.get('total_synthesis_attempts_this_line',len(attempts))
        seeds=[r['seed'] for r in attempts if 'seed' in r]
        if actual_count>len(attempts) and 'seed' in ready_record:
            seeds.append(ready_record['seed'])
        sound['voice_reference'].update(accent=speaker['accent'],region=speaker['region'])
        sound['speaker_audit']={**speaker,'source_speaker_info':args.speaker_info}
        sound['retained_existing_source']=True
        sound['language_readback']=lid
        sound['requested_language_threshold_status']=lid['status']
        sound['approved_expansion_output']=False
        sound['selected_seed']=sound['synthesis_config'].get('seed')
        sound['actual_synthesis_attempt_count']=actual_count
        sound['executed_seeds_from_prior_receipts']=seeds
        sound['attempt_count_sources']=[str(source_record),str(final_record)]
        sound['weight_license_receipt']=str(out/'license_receipt.json')
        genders={sound['voice_gender'],task['voice_gender'],task['standing_actor_plan']['visual_gender'],sound['voice_reference']['gender'],speaker['gender']}
        if len(genders)!=1:
            violations.append({'sound_asset_id':sid,'reason':'voice_visual_or_speaker_gender_mismatch','genders':sorted(genders)})
        actual_sha=digest(sound['path'])
        if actual_sha!=sound['sha256'] or actual_sha!=lid.get('audio_sha256'):
            violations.append({'sound_asset_id':sid,'reason':'retained_audio_sha_mismatch'})
        by_scenario[sound['scenario_id']][voice].add(sound['role_id'])
        proof=lid['retained_asr_recomputed_from_existing_transcription']
        if proof['status']!='pass':
            violations.append({'sound_asset_id':sid,'reason':'retained_ASR_or_critical_slot_failed'})
        wer.append(proof['wer'])
        normal=sound['source_normalization']['readback_loudness']
        lufs.append(normal['input_i']);peaks.append(normal['input_tp'])
        counts[voice]+=1;language_counts[lid['detected_language']]+=1
        retained.append(sound)
    for scenario,voices in by_scenario.items():
        for voice,roles in voices.items():
            if len(roles)>1:
                violations.append({'scenario_id':scenario,'speaker_id':voice,'role_ids':sorted(roles),'reason':'scenario_different_roles_share_voice'})
    probabilities=[s['language_readback']['english_probability'] for s in retained]
    low=[s for s in retained if s['language_readback']['status']!='pass']
    table=[{**speakers[v],'retained_lines':n,'language_at_least_0_9':sum(s['voice_reference']['speaker_id']==v and s['language_readback']['status']=='pass' for s in retained)} for v,n in sorted(counts.items())]
    female=sum(r['gender']=='female' for r in table)
    male=sum(r['gender']=='male' for r in table)
    missing_old=[t['sound_asset_id'] for t in plan['tasks'] if t['sound_asset_id'] not in {s['sound_asset_id'] for s in retained}]
    summary={'status':'blocked','reason':'existing_530_do_not_all_meet_english_probability_0_9',
             'synthesis_started':False,'pilot_ten_lines_run':False,
             'new_voice_count':0,'total_retained_voice_count':len(table),'female_voice_count':female,'male_voice_count':male,
             'planned_new_synthesis_variants':2*len(plan['tasks']),'new_synthesis_variants':0,'new_passed_variants':0,'new_dropped_variants':0,
             'retained_old_lines':len(retained),'old_plan_missing_lines':len(missing_old),
             'old_language_pass':baseline['passing_count'],'old_language_nonpass':baseline['nonpassing_count'],
             'english_threshold':0.9,'english_threshold_adopted':False,
             'english_probability_min':min(probabilities),'english_probability_max':max(probabilities),'english_probability_mean':sum(probabilities)/len(probabilities),
             'detected_language_counts':dict(language_counts),
             'retained_ASR_recomparison_pass':sum(s['language_readback']['retained_asr_recomputed_from_existing_transcription']['status']=='pass' for s in retained),
             'retained_ASR_WER_min':min(wer),'retained_ASR_WER_max':max(wer),'new_ASR_transcriptions_run':0,
             'retained_lufs_min':min(lufs),'retained_lufs_max':max(lufs),
             'retained_lufs_within_0_1_of_minus_23':sum(abs(x+23)<=.100000001 for x in lufs),
             'retained_peak_limited_below_lufs_target':sum(x < -23.100000001 for x in lufs),
             'retained_true_peak_max_dbTP':max(peaks),'retained_true_peak_at_most_minus_3':sum(p<=-3 for p in peaks),
             'retained_gender_and_scenario_voice_checks_pass':not violations,
             'metadata_violations':violations,
             'source_plan':args.plan,'source_plan_sha256':digest(args.plan),
             'old_manifest':args.manifest,'old_manifest_sha256':digest(args.manifest),
             'speaker_info':args.speaker_info,'speaker_info_sha256':digest(args.speaker_info),
             'license_receipt':str(out/'license_receipt.json'),
             'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    manifest={**old,'sounds':retained,'batch_summary':summary,'formal_usable':0,
              'status':'blocked','complete':False,'ready_for_dataset':False,
              'retained_source_snapshot_only':True,'approved_expansion_sounds':[],
              'original_batch_summary':old['batch_summary'],
              'blocked_reason':summary['reason'],'weight_license_receipt':str(out/'license_receipt.json')}
    save(out/'semantic_manifest.json',manifest)
    save(out/'audit_summary.json',summary)
    save(out/'voice_counts.json',table)
    save(out/'old_language_nonpassing.json',[{'sound_asset_id':s['sound_asset_id'],'speaker_id':s['voice_reference']['speaker_id'],'transcript':s['transcript'],'english_probability':s['language_readback']['english_probability'],'detected_language':s['language_readback']['detected_language'],'path':s['path'],'sha256':s['sha256'],'reason':s['language_readback']['reason']} for s in sorted(low,key=lambda s:s['language_readback']['english_probability'])])
    save(out/'old_plan_missing_10.json',{'source':args.plan,'sound_asset_ids':missing_old,'not_resynthesized':True})
    e=html.escape
    voice_rows=''.join('<tr><td>'+e(r['speaker_id'])+'</td><td>'+('女' if r['gender']=='female' else '男')+'</td><td>'+e(r['accent']+' / '+r['region'])+'</td><td>'+str(r['retained_lines'])+'</td><td>'+str(r['language_at_least_0_9'])+'</td></tr>' for r in table)
    low_rows=''.join('<tr><td>'+e(s['sound_asset_id'])+'</td><td>'+e(s['voice_reference']['speaker_id'])+'</td><td>'+e(s['transcript'])+'</td><td>'+f'{s["language_readback"]["english_probability"]:.6f}'+'</td><td>'+e(s['language_readback']['detected_language'])+'</td></tr>' for s in sorted(low,key=lambda s:s['language_readback']['english_probability']))
    conclusion=f'前置检查没过，本轮没有扩音色。旧 530 条里有 {len(low)} 条英文概率低于 0.9。'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v7 英文台词音色审计 · 2026-10-12</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#18212c;background:#f4f6fa;margin:0}}main{{max-width:1100px;margin:auto;padding:30px 22px}}h1{{font-size:26px}}h2{{font-size:20px;margin-top:28px}}.box{{background:white;border:1px solid #dde3eb;border-radius:12px;padding:18px;margin:16px 0}}.stop{{border-left:5px solid #b9472e}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #e2e7ef;vertical-align:top}}th{{background:#eef2f8}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}a{{color:#185caa}}.muted{{color:#536270}}ul{{padding-left:22px}}</style><main><h1>{conclusion}</h1><div class="box stop"><p>任务要求先证明旧 530 条全部能达到英文概率 0.9，再使用这个门槛。实际是 {baseline['passing_count']}/530 条达到，{len(low)} 条没有达到，所以没有做 10 句合成试跑，也没有启动后续 1080 个新音色版本。没有降低阈值，没有重做、裁剪或删除旧音频。</p><p>本轮新增音色 <b>0</b>，保留原有 <b>{len(table)}</b> 个（女 {female}、男 {male}）；新合成 <b>0</b>，新通过 <b>0</b>，新丢弃 <b>0</b>。24 个音色和每句 3 个版本的目标尚未完成。</p></div><h2>旧音色和句数</h2><div class="box scroll"><table><thead><tr><th>说话人</th><th>性别</th><th>口音 / 地区</th><th>保留句数</th><th>英文概率 ≥0.9</th></tr></thead><tbody>{voice_rows}</tbody></table><p class="muted">出处：原 530 条清单、VCTK speaker-info.txt，以及本轮 voice_counts.json。语种未过的原音频仍原样保留，不能算作满足新增门槛的数据。</p></div><h2>语种、ASR 和响度</h2><div class="box"><ul><li>新做语种识别 530 条，达到 0.9 的比例 {baseline['passing_count']/530:.2%}；英文概率最低 {min(probabilities):.6f}，最高 {max(probabilities):.6f}，平均 {sum(probabilities)/len(probabilities):.6f}。</li><li>最高概率语种分布：{e(json.dumps(dict(language_counts),ensure_ascii=False))}。英文置信度低，不等于已经证明音频说的是其他语言。</li><li>Whisper-small 每次只输入整条音频和 start-of-transcript，未给英文 token、台词、答案或上下文提示；在全部语种 token 上做 softmax。4 个常驻 worker，每个 2 个 CPU 模型线程，CUDA 禁用，nice 10。</li><li>沿用原来的盲转写，再用现有语义比较函数复核：{summary['retained_ASR_recomparison_pass']}/530 条通过，最大 WER {max(wer):.6f}。本轮没有重新盲转写，不能把旧 ASR 说成新测。</li><li>原响度收据显示：{summary['retained_lufs_within_0_1_of_minus_23']} 条在 -23 ±0.1 LUFS，{summary['retained_peak_limited_below_lufs_target']} 条受真峰值限制而更低；范围 {min(lufs):.2f} 至 {max(lufs):.2f} LUFS。{summary['retained_true_peak_at_most_minus_3']}/530 条真峰值 ≤ -3 dBTP，最大 {max(peaks):.2f} dBTP。本轮没有改旧音频，也没有重测响度。</li><li>原波形 SHA、角色性别、VCTK 性别和情景内角色音色自动核对：{'全部通过' if not violations else '发现异常，见 audit_summary.json'}。</li></ul></div><h2>未达到语种门槛的旧句子（{len(low)} 条）</h2><div class="box scroll"><table><thead><tr><th>台词 ID</th><th>音色</th><th>台词</th><th>英文概率</th><th>最高概率语种</th></tr></thead><tbody>{low_rows}</tbody></table><p>新合成尚未开始，所以没有新增丢弃。旧计划里原先丢弃的 10 条没有重做，ID 见 old_plan_missing_10.json。</p></div><h2>许可核对</h2><div class="box"><p>本地 CosyVoice2-0.5B 模型卡声明 <b>Apache-2.0</b>；保存的本地模型卡与历史 revision eec1ae6 的上游模型卡字节一致。未发现禁止把合成输出用于 AI 训练的条款，许可前置检查通过。README 的 academic purposes 说明针对上方内容/示例，没有写合成输出不得训练模型。</p><p>权重目录没有单独 LICENSE，这一点已写进收据；代码 LICENSE 仅作为补充。15 个本地模型文件的 SHA 和字节数已记录，没有声称比较过远端权重。VCTK 为 CC-BY 4.0，作者出处保留。依据：<a href="https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/blob/eec1ae6c79877dbd9379285cf8789c9e0879293d/README.md">模型卡</a>、<a href="https://www.apache.org/licenses/LICENSE-2.0">Apache 官方原文</a>和 license_receipt.json。</p></div><h2>文件和后续需要</h2><div class="box"><p>产物目录：<code>{e(str(out))}</code>（48g）。</p><ul><li><a href="semantic_manifest.json">semantic_manifest.json</a>：保留旧 530 条并补上审计字段；顶层明确 blocked、ready_for_dataset=false，没有已批准的新增数据。</li><li><a href="audit_summary.json">audit_summary.json</a>、<a href="language_baseline_audit.json">language_baseline_audit.json</a>、<a href="old_language_nonpassing.json">旧句失败列表</a>。</li><li><a href="license_receipt.md">许可收据</a>、<a href="JOBS.json">作业记录</a>、<a href="PROGRESS.md">进度</a>、<a href="delivery_receipt.json">测试、提交、CPU 与磁盘交付收据</a>。</li></ul><p>后续需要 owner 决定如何处理“旧 530 条必须全过”的条件。当前不能自行降低英文阈值、丢掉旧句，或重做只读目录里的旧版本。38 条原响度低于共同目标的事实也需要在下一份任务书中明确。</p><p class="muted">仅用机器指标，没有要求人听，没有接生产采样器，没有训练或 ACP/CCI 作业。</p></div></main></html>'''
    (out/'report.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
