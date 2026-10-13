"""Render the revised SAO readmission and speech-calibration evidence in Chinese."""
from __future__ import annotations
import argparse, collections, html, json, pathlib, time
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from tools.qa.report_sao_topup import png
from tools.qa.admit_sao_topup import load_scored, sha, norm_path
import matplotlib.pyplot as plt


def esc(value):
    return html.escape(str(value))


def summarize(output, v9_path, config_path, stage, destination):
    cal = json.loads((output / 'HUMAN_SPEECH_CALIBRATION_v11.json').read_text())
    assert cal['status'] != 'pass' and cal['threshold'] is None
    config = json.loads(config_path.read_text())
    raw_path = output / f'admission_{stage}' / 'dry_recorded_admission.json'
    raw = json.loads(raw_path.read_text())['records']
    scored = load_scored(output)
    v9 = json.loads(v9_path.read_text())['records']
    dedup_path = output / f'deduplication_{stage}.json'
    dedup = json.loads(dedup_path.read_text())
    assert len(raw) == len(scored) == 856
    assert len({r['sound_asset_id'] for r in raw}) == len(raw)
    assert len(v9) == len({r['sound_asset_id'] for r in v9}) == 1914
    assert not any(r.get('strict_v7_eligible') for r in raw)
    immutable = json.loads((output / 'evidence' / 'resume_v11_audit.json').read_text())['input_sha256']
    for path, digest in immutable.items():
        assert sha(path) == digest, path
    for r in raw:
        asr = r['whisper_transcription_recording_only']
        assert asr['role'] == 'recording_only' and asr['participates_in_admission'] is False
        assert asr['audio_sha256'] == r['sha256']
        assert 'automatic_no_speech' not in r
        assert r['human_speech_screen']['threshold'] is None
        assert r['allowed_splits'] == ['train'] and r['split'] == 'train'
        assert r['label_provenance'] == 'generated_stable_audio_open_1.0'
        assert r['license'] == 'Stability AI Community License'
        assert r['owner_acceptance'] == 'owner 2026-10-12 接受 IV(b) 风险'
        info = sf.info(r['path'])
        assert info.samplerate == 16000 and info.channels == 1
    def origin(r):
        return str(r.get('original_group') or r.get('source_original_id') or r.get('fsd50k_id') or r.get('source_origin') or r['sound_asset_id'])
    all_counts = {c:len({origin(r) for r in v9 if r['sound_class'] == c}) for c in sorted({r['sound_class'] for r in v9})}
    assert len(all_counts) == 27
    by_class = []
    for c, spec in config['classes'].items():
        rows = [r for r in raw if r['sound_class'] == c]
        scores = [r for r in scored if r['candidate']['sound_class'] == c]
        independent = [r for r in rows if not [reason for reason in r.get('current_policy_blockers',[]) if reason != 'generated_source_human_speech_calibration_failed']]
        assert all_counts[c] == spec['existing_originals']
        by_class.append(dict(sound_class=c, candidates=len(rows),
                             round1_candidates=sum(r['candidate']['round'] == 1 for r in scores),
                             round2_candidates=sum(r['candidate']['round'] == 2 for r in scores),
                             round3_candidates=0, beats_pass=sum(r['beats_pass'] for r in scores),
                             independent_prechecks_pass=len(independent), accepted=0,
                             existing_originals=all_counts[c], final_originals=all_counts[c], deficit=30-all_counts[c],
                             rejection_and_blocking_reasons=dict(collections.Counter(reason for r in rows for reason in r.get('current_policy_blockers',[])))))
    result = dict(status='blocked_human_speech_calibration_failed', candidate_count=len(raw), accepted=0,
                  by_class=by_class, all_class_original_counts=all_counts,
                  beats_pass=sum(r['beats_pass'] for r in scored),
                  independent_prechecks_pass=sum(r['independent_prechecks_pass'] for r in by_class),
                  rejection_and_blocking_reasons=dict(collections.Counter(reason for r in raw for reason in r.get('current_policy_blockers',[]))),
                  independent_screen_counts=dict(beats_below_target=sum(r['target_confidence'] < .1917129977 for r in scored),
                                                  beats_below_margin=sum(r['margin'] < .05 for r in scored),
                                                  activity_below_one_second=sum(not r['activity_pass'] for r in scored)),
                  fingerprint_actual_match_reasons=dict(collections.Counter(r['reason'] for r in dedup['records'].values() if r['status']=='blocked')),
                  calibration=str(output/'HUMAN_SPEECH_CALIBRATION_v11.json'),
                  fingerprint_validation=str(output/f'FINGERPRINT_REPAIR_VALIDATION_{stage}.json'),
                  deduplication=str(dedup_path), raw_admission=str(raw_path),
                  execution_identity=str(output/'evidence'/f'readmit_execution_source_{stage}.json'),
                  v9_v10_unchanged=True, immutable_input_sha256=immutable,
                  generated_recording_only_whisper_metadata_verified=True,
                  generated_mono_16000_metadata_verified=True,
                  target_beats_threshold=.1917129977, minimum_incompatible_margin=.05,
                  normalization_status='not_run; calibration blocks admission before normalization',
                  third_round=dict(status='not_run_human_speech_calibration_failed', classes=[], candidates=0),
                  v11_outputs=[], v11_entry_count=None, v11_sha256=None,
                  v11_cci_input_and_pull_list_status='not_created; no production v11 can pass the requested human speech check',
                  report=str(output/'REPORT_v11.html'), human_listening_used=False)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    with (output/'PROGRESS.md').open('a') as stream:
        stream.write('- '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' 全部 856 候选 v11 前置复查与仅记录 Whisper 元数据校验完成；人声校准阻塞，正式新增 0；未生成第三轮及 v11 输入。\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--summary', type=pathlib.Path, required=True)
    parser.add_argument('--v9', type=pathlib.Path)
    parser.add_argument('--config', type=pathlib.Path)
    parser.add_argument('--stage', default='v11_r12_polyphase')
    args = parser.parse_args()
    o = args.output
    if args.summary.exists():
        report = json.loads(args.summary.read_text())
    else:
        assert args.v9 and args.config
        report = summarize(o, args.v9, args.config, args.stage, args.summary)
    cal = json.loads((o / 'HUMAN_SPEECH_CALIBRATION_v11.json').read_text())
    controls = json.loads(pathlib.Path(report['fingerprint_validation']).read_text())
    scored = []
    for path in sorted(o.glob('model_scores_r*.jsonl')):
        scored.extend(json.loads(s) for s in path.read_text().splitlines() if s)
    calibration = [json.loads(s) for s in (o / 'human_speech_calibration_scores_v11.jsonl').read_text().splitlines()]
    positive = [r['score'] for r in calibration if r['polarity'] == 'positive_real_target']
    negative = [r['score'] for r in calibration if r['polarity'] == 'negative_human_speech']
    generated = [max(r['beats']['scores_by_mid'][h['mid']] for h in cal['heads']) for r in scored]
    h = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>v7 SAO v11 重新准入：校准阻塞报告</title>',
         '<style>body{max-width:1120px;margin:24px auto;padding:0 18px;font:16px/1.65 system-ui;color:#202631}h1{font-size:26px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d7dee7;padding:7px;text-align:left}th{background:#edf2f8}img{max-width:100%}code{overflow-wrap:anywhere}.warn{padding:16px;background:#fff0d6}.ok{padding:14px;background:#eaf5ef}figcaption{font-size:13px;overflow-wrap:anywhere}figure{margin:14px 0}</style><body>',
         '<h1>结论：人说话检查无法同时满足两项校准要求；v11 未发布，第三轮未生成</h1>',
         '<p class="warn">全部 856 条已有候选重新检查，Whisper 已改为仅记录。人声校准要求保留全部真实薄类录音并拦截 ≥95% 真人朗读，当前 BEATs 分数分布无法满足。未硬定阈值，未把暂缓候选写进输入，v9/v10 和旧报告均保留。</p>']
    h.append('<h2>人说话检查的校准</h2><p>取 Speech、Singing 及其 AudioSet 子类、Whispering、Hubbub/speech babble 共 '+str(len(cal['heads']))+' 个头的最大值。使用同一 BEATs AS2M cpt2、float32、5 秒块平均，末块补至至少 1 秒。正例为 v9 全部 183 条指定薄类真实录音；反例为全部 739 条 VCTK 真人朗读，无抽样挑选。</p>')
    h.append('<table><tr><th>指标</th><th>实测</th></tr>')
    fields = [('正式阈值', '未设置；校准失败'),
              ('诊断阈值（不投入准入）', f"{cal['proposed_threshold']:.10f} = 正例最大值 + 0.01"),
              ('正例通过率（诊断值下）', f"{cal['positive_pass']}/{cal['positive_count']} = 100%"),
              ('反例拦截率（诊断值下）', f"{cal['negative_block']}/{cal['negative_count']} = {100*cal['negative_block_rate']:.3f}%"),
              ('去掉余量后最高可能反例拦截率', f"19/739 = {100*cal['maximum_possible_block_rate_at_positive_max']:.3f}%"),
              ('正例最大值 / 反例第 5 百分位', f"{max(positive):.6f} / {np.quantile(negative,.05):.6f}")]
    for key, value in fields:
        h.append('<tr><td>'+esc(key)+'</td><td>'+esc(value)+'</td></tr>')
    h.append('</table><p>最高分正例是 <code>fsd_216637__pet_bird_parrot</code>，Speech=0.807925。其真实听感及是否夹杂人声未验证；按要求未把它或其他真实正例移除。仅用明确要求的 Speech、男/女/儿童 speech、Conversation、Narration、Singing 七个头，最大正例和反例拦截上限也相同，见 evidence/human_speech_calibration_diagnosis_v11.json。</p>')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.linspace(0, 1, 41)
    for values, name, color in [(positive, 'real target positives (183)', '#23866a'), (negative, 'VCTK negatives (739)', '#b45236'), (generated, 'generated candidates (856)', '#387ab0')]:
        axes[0].hist(values, bins=bins, density=True, histtype='step', linewidth=2, label=name, color=color)
    axes[0].axvline(cal['proposed_threshold'], color='black', linestyle='--', label='diagnostic only; not authorized')
    axes[0].set(xlabel='maximum human speech / singing posterior', ylabel='density', xlim=(0, 1))
    axes[0].legend(fontsize=8)
    thresholds = np.linspace(0, 1, 1001)
    axes[1].plot(thresholds, [np.mean(np.asarray(positive) <= t) for t in thresholds], label='positive retention')
    axes[1].plot(thresholds, [np.mean(np.asarray(negative) > t) for t in thresholds], label='negative rejection')
    axes[1].axhline(.95, color='#b45236', linestyle=':')
    axes[1].axvline(max(positive), color='black', linestyle='--')
    axes[1].set(xlabel='hypothetical rejection threshold', ylabel='fraction', xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    fig.tight_layout(); h.append(png(fig))
    h.append('<table><tr><th>样本</th><th>最小值</th><th>中位数</th><th>P95</th><th>最大值</th></tr>')
    for name, d in [('真实薄类正例', cal['positive_distribution']), ('真人朗读反例', cal['negative_distribution'])]:
        h.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [name]+[f"{d[k]:.6f}" for k in ('min','median','p95','max')])+'</tr>')
    h.append('</table><p>字段 <code>whisper_transcription_recording_only</code> 保留自由转写、历史判断和解码概率，且 <code>participates_in_admission=false</code>。Whisper 不参与任何准入判定。新 <code>human_speech_screen</code> 明确标记校准失败；不得把诊断值当作生效阈值。</p>')
    h.append('<h2>指纹去重修复</h2><p class="ok">阳性对照由 '+str(controls['before_detected'])+'/'+str(controls['before_count'])+'（'+f"{100*controls['before_detected']/controls['before_count']:.3f}%"+'）提升至 '+str(controls['after_detected'])+'/'+str(controls['after_count'])+'（'+f"{100*controls['after_detection_rate']:.3f}%"+'），满足 ≥99%。自身对照失败不再构成音频拒绝理由。</p>')
    h.append('<p>旧方法的 368 次漏检：365 次来自裁切改变 STFT 网格，3 次来自走秒声指纹少于 12 个。修复以弱谱峰哈希筛选，再验证完整短波形与长波形中一段的归一化互相关 ≥0.995、匹配时长 ≥1 秒；相同归一化 PCM 也算匹配。两种 16 kHz 采样相位覆盖奇数样本裁切。稀疏指纹与全部参考比较。实际匹配才拒绝，阈值对照失败不拒绝。</p>')
    h.append('<table><tr><th>变换</th><th>检出</th><th>总数</th></tr>')
    for name, counts in controls['by_transform'].items():
        h.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in (name, counts['detected'], counts['count']))+'</tr>')
    h.append('</table><p>剩余漏检见对照文件 failures。去重覆盖全部 v9 1,914 条和 856 条候选；谱峰检索仍是近似方法，检出率来自上述控制实验，未宣称逐对穷举或主观相似性全覆盖。</p>')
    h.append('<h2>逐类结果</h2><table><tr><th>类别</th><th>第一/二轮候选</th><th>BEATs 通过</th><th>其他前置指标可继续</th><th>正式通过</th><th>最终不同来源</th><th>缺口</th></tr>')
    for row in report['by_class']:
        h.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [row['sound_class'],f"{row['round1_candidates']}/{row['round2_candidates']}",row['beats_pass'],row['independent_prechecks_pass'],row['accepted'],row['final_originals'],row['deficit']])+'</tr>')
    h.append('</table><p>cat_meow 的 13 条真实裁片仍按 6 个原始来源计。正式通过均为 0，原因是未有满足校准要求的人说话门槛；“其他前置指标可继续”只包括 BEATs、有效活动、去重及亚类元数据，归一化尚未执行，不能视为最终通过。第三轮生成 0 条、涉及类别为空，因校准阻塞而未启动。</p>')
    h.append('<h2>拒绝与暂缓原因（允许重叠）</h2><table><tr><th>原因</th><th>条数</th></tr>')
    for reason, count in sorted(report['rejection_and_blocking_reasons'].items(), key=lambda kv:-kv[1]):
        h.append('<tr><td>'+esc(reason)+'</td><td>'+str(count)+'</td></tr>')
    h.append('</table><p>人说话校准失败是流程暂缓，不是判定全部音频有真人声音。目标 BEATs 门槛保持 ≥0.1917129977、不相容类间隔 ≥0.05；有效活动总时长及最长组均 ≥1 秒，未放宽。其余规则来自 thin top-up a9d11be 的原构建器。</p>')
    h.append('<h2>BEATs 分数分布</h2>')
    fig, axes = plt.subplots(4, 4, figsize=(15, 12)); axes = axes.ravel()
    for ax, row in zip(axes, report['by_class']):
        values = [r['target_confidence'] for r in scored if r['candidate']['sound_class'] == row['sound_class']]
        ax.hist(values, bins=np.linspace(0,1,26), color='#387ab0')
        ax.axvline(.1917129977, color='#b45236', linestyle='--')
        ax.set(title=row['sound_class'], xlabel='target posterior', ylabel='candidates', xlim=(0,1))
    for ax in axes[len(report['by_class']):]: ax.axis('off')
    fig.tight_layout(); h.append(png(fig))
    h.append('<h2>候选波形及频谱（未准入）</h2><p>没有新通过条目，无法贴每类 2–3 条通过示例。以下每类 2 条代表候选均明确为未准入，供查看机器波形/频谱；没有人工试听判断。</p>')
    for row in report['by_class']:
        h.append('<h3>'+esc(row['sound_class'])+' — 未准入</h3>')
        members = sorted([r for r in scored if r['candidate']['sound_class'] == row['sound_class']], key=lambda r:(-int(r['beats_pass'] and r['activity_pass']),-r['target_confidence']))[:2]
        for member in members:
            r = member['candidate']; x, sr = sf.read(r['path'], dtype='float32')
            assert sr == 16000 and x.ndim == 1
            fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            axes[0].plot(np.arange(len(x))/sr, x, linewidth=.5, color='#286d91'); axes[0].set_ylabel('amplitude')
            f, t, z = spectrogram(x, fs=sr, nperseg=512, noverlap=384)
            axes[1].pcolormesh(t, f, 10*np.log10(np.maximum(z,1e-12)), shading='auto', cmap='magma', vmin=-110, vmax=-35)
            axes[1].set(xlabel='seconds', ylabel='Hz', ylim=(0,8000)); fig.tight_layout()
            h.append('<figure>'+png(fig)+'<figcaption>未准入 '+esc(r['sound_asset_id'])+'；BEATs='+f"{member['target_confidence']:.6f}"+'；margin='+f"{member['margin']:.6f}"+'；seed='+str(r['seed'])+' / batch index='+str(r['seed_batch_index'])+'<br><code>'+esc(r['path'])+'</code></figcaption></figure>')
    h.append('<h2>许可、交付和出处</h2><p>生成音频仍标注 Stability AI Community License、label_provenance=generated_stable_audio_open_1.0 和“owner 2026-10-12 接受 IV(b) 风险”。第 IV(b) 条的输出使用限制沿用原任务说明；没有额外许可结论。所有生成来源限定 allowed_splits=["train"]，开发和测试仍仅使用真实录音。</p>')
    h.append('<p>v11 输入、CCI 镜像输入及拉取清单：未生成，等待可满足要求的人声检查方案。未提交 ACP/CCI 作业、未训练、未 push。v9/v10 校验值未变。旧报告 REPORT.html 保留。</p>')
    for key in ('raw_admission', 'fingerprint_validation', 'deduplication', 'calibration', 'execution_identity'):
        h.append('<p>'+esc(key)+'：<code>'+esc(report[key])+'</code></p>')
    h.append('<p>汇总：<code>'+esc(args.summary)+'</code>；完整最终审计：<code>'+esc(o/'FINAL_AUDIT_v11.json')+'</code>；过程记录：PROGRESS.md / JOBS.json。提交、GPU 用时及磁盘以最终审计为准。</p>')
    h.append('<small>UTC '+time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())+'；所有图为本次源文件和分数直接计算。</small></body></html>')
    destination = o / 'REPORT_v11.html'
    with destination.open('x') as stream: stream.write('\n'.join(h))
    with (o/'PROGRESS.md').open('a') as stream:
        stream.write('- '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' v11 中文单文件报告完成：校准失败、指纹修复和全候选重新检查，未人工试听。\n')
    print('REPORT_V11_DONE', str(destination), destination.stat().st_size)

if __name__ == '__main__':
    main()
