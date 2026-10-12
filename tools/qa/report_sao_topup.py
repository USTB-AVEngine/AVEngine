"""Build a standalone Chinese machine-admission report with audio plots."""
from __future__ import annotations
import argparse,base64,collections,html,io,json,pathlib,subprocess,time
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def read(p):
    return json.loads(pathlib.Path(p).read_text())

def esc(x):
    return html.escape(str(x))

def png(fig):
    b=io.BytesIO()
    fig.savefig(b,format="png",dpi=115,bbox_inches="tight")
    plt.close(fig)
    return '<img alt="machine audio plot" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=pathlib.Path,required=True)
    p.add_argument("--worktree",type=pathlib.Path,required=True)
    a=p.parse_args();o=a.output
    d=read(o/"DELIVERY.json")
    v=read(d["outputs"][0]["path"])
    new=[r for r in v["records"] if r.get("label_provenance")=="generated_stable_audio_open_1.0"]
    scored=[]
    for f in sorted(o.glob("model_scores_r*.jsonl")):
        scored.extend(json.loads(s) for s in f.read_text().splitlines() if s)
    dedup=read(o/"deduplication_final.json")
    rejects=collections.Counter(d["rejection_reasons"])
    h=['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
       '<meta name="viewport" content="width=device-width,initial-scale=1">',
       '<title>v7 Stable Audio Open 机器准入补录报告</title>',
       '<style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:16px/1.65 system-ui;color:#202631}h1{font-size:26px}h2{margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d4d9df;padding:7px;text-align:left}th{background:#eef2f6}img{max-width:100%}code{overflow-wrap:anywhere}.ok{background:#e9f5eb;padding:14px}.warn{background:#fff2d8;padding:14px}figure{margin:16px 0}figcaption{font-size:13px;overflow-wrap:anywhere}small{color:#596575}</style><body>']
    missing=d["missing_classes"];met=len(d["by_class"])-len(missing)
    h.append('<h1>结论：'+str(met)+'/13 类达到 30 个不同来源；新增 '+str(d["added_records"])+' 条训练专用音频</h1>')
    h.append('<p class="'+("warn" if missing else "ok")+'">共 '+str(d["candidate_count"])+' 个候选，'+str(d["machine_pass"])+' 条通过机器准入，v10 共 '+str(d["outputs"][0]["records"])+' 条。'+
        ("未达 30 的类别："+esc("、".join(missing))+"。已用完三倍首轮候选及这些类别最多一倍的重试预算，未降低门槛。" if missing else "所有指定薄类已达目标。")+'</p>')
    h.append('<p>全部判定来自机器指标，未做人工试听。v9 未修改。生成源仅允许 <code>allowed_splits=["train"]</code>；抽样和装配代码均阻止其进入开发集及测试集。CCI 版本和拉取清单在 48g 写好，未执行 CCI 镜像或提交 CCI/ACP 作业。</p>')
    h.append('<h2>类别结果</h2><table><tr><th>类别</th><th>原来源</th><th>首轮/重试候选</th><th>机器通过</th><th>v10 新增</th><th>合格备选</th><th>最终来源</th><th>缺口</th></tr>')
    for s in d["by_class"]:
        r1=sum(r["candidate"]["sound_class"]==s["sound_class"] and r["candidate"]["round"]==1 for r in scored)
        r2=sum(r["candidate"]["sound_class"]==s["sound_class"] and r["candidate"]["round"]==2 for r in scored)
        h.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [s["sound_class"],s["existing_originals"],f"{r1}/{r2}",s["machine_pass"],s["added_to_v10"],s["qualified_spares"],s["final_originals"],s["deficit"]])+'</tr>')
    h.append('</table><p><small>每个独立生成种子/批内位置作为一个生成来源，指纹去重后计数；cat_meow 原 13 条裁片按 6 个原来源计。合格但超过类配额的备选保留在原准入产物中，不计为拒绝。</small></p>')
    h.append('<h2>拒绝原因（允许重叠）</h2><table><tr><th>原因</th><th>数量</th></tr>')
    for name,n in rejects.most_common():
        h.append('<tr><td>'+esc(name)+'</td><td>'+str(n)+'</td></tr>')
    h.append('</table><h3>逐类屏幕读数</h3><table><tr><th>类别</th><th>目标分不足</th><th>间隔不足</th><th>Whisper 有词</th><th>活动不足</th><th>最终构建器阻断</th></tr>')
    for s in d["by_class"]:
        h.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [s["sound_class"],s["beats_below_threshold"],s["beats_margin_below_threshold"],s["whisper_lexical_rejections"],s["activity_rejections"],json.dumps(s["rejection_reasons"],ensure_ascii=False)])+'</tr>')
    h.append('</table><h2>机器检查与适用范围</h2><p>沿用 thin top-up a9d11be 的 build_recorded_admission、existing_gate_classes 和 existing_subtype_metadata_groups：BEATs AS2M cpt2 目标类 ≥0.1917129977，且比不相容 AudioSet 类高 ≥0.05。活动检测和响度归一化代码保持原实现；有效活动总时长及最长连续事件组均须 ≥1 秒，事件组内活动时长不含间隙。44.1 kHz 双声道先取通道平均，再重采样到 16 kHz 单声道，归一化沿用 -23 LUFS/-3 dBTP。</p>')
    h.append('<p>Whisper-small 自动识别语言，自由转写，无参考文本；使用 <a href="https://raw.githubusercontent.com/openai/whisper/main/whisper/transcribe.py">官方默认无语音处理</a>：p_no_speech &gt;0.6 且 avg_logprob &lt;-1 时跳过无语音片段，其余最终转写有任何 Unicode 字母或数字均拒绝。初始直接 generate 转写、标准处理所有原始尝试及概率均保留，不使用词白名单。3 条真实 VCTK 人声阳性对照均被拒绝，见 evidence/whisper_standard_pilot.json。机器转写可能误检/漏检；真实听感与人声检出准确率未验证。</p>')
    h.append('<p>指纹范围为 v9 全 '+str(dedup["reference_file_count"])+' 条及全部生成候选；使用原 thin 音频指纹（8 kHz 归一化 PCM 和谱峰星座），双向对齐，完全相同 PCM 或 ≥12 匹配哈希且对齐比例 ≥0.25 判近重复。增益/重采样/裁切阳性对照检出 '+str(dedup["positive_control_detected"])+'/'+str(dedup["positive_control_count"])+'。此方法不能证明遮盖声源或主观相似性不存在；未做新的 STARSS/DCASE/TAU 扫描。</p>')
    h.append('<p>首轮 100 步、cfg 7，按片长加 1 秒对齐缩短采样张量；第二轮改为官方 model_config 的完整 sample_size=2097152 后截取提示时长。两轮所有种子、批内位置、完整批条件、提示词和模型校验值均逐条记录；没有改模型或共享环境。</p>')
    h.append('<h2>BEATs 分布</h2>')
    fig,axes=plt.subplots(4,4,figsize=(15,12));axes=axes.ravel()
    for ax,s in zip(axes,d["by_class"]):
        values=[r["target_confidence"] for r in scored if r["candidate"]["sound_class"]==s["sound_class"]]
        ax.hist(values,bins=np.linspace(0,1,26),color="#4281b7")
        ax.axvline(.1917129977,color="#b72424",linestyle="--")
        ax.set(title=s["sound_class"],xlabel="target score",ylabel="candidates",xlim=(0,1))
    for ax in axes[len(d["by_class"]):]:ax.axis("off")
    fig.tight_layout();h.append(png(fig))
    fig,ax=plt.subplots(figsize=(12,4))
    ax.boxplot([[r["margin"] for r in scored if r["candidate"]["sound_class"]==s["sound_class"]] for s in d["by_class"]],labels=[s["sound_class"] for s in d["by_class"]],showfliers=True)
    ax.axhline(.05,color="#b72424",linestyle="--");ax.tick_params(axis="x",rotation=55);ax.set_ylabel("target minus incompatible rival");fig.tight_layout();h.append(png(fig))
    h.append('<h2>通过音频的波形及频谱</h2><p>以下只展示已写入 v10 的通过条目，每类最多 3 条。图来自归一化单声道 16 kHz 文件；没有通过条目的类别如实留空。</p>')
    for s in d["by_class"]:
        members=[r for r in new if r["sound_class"]==s["sound_class"]][:3]
        h.append('<h3>'+esc(s["sound_class"])+'</h3>')
        if not members:
            h.append('<p>本类没有通过音频，未贴示例。</p>');continue
        if len(members)<2:h.append('<p>本类仅 '+str(len(members))+' 条通过并写入 v10，无法提供 2–3 条示例。</p>')
        for r in members:
            x,sr=sf.read(r["normalized_path"],dtype="float32");assert sr==16000 and x.ndim==1
            fig,axes=plt.subplots(2,1,figsize=(12,4),sharex=True)
            axes[0].plot(np.arange(len(x))/sr,x,linewidth=.5,color="#286d91");axes[0].set_ylabel("amplitude")
            f,t,z=spectrogram(x,fs=sr,nperseg=512,noverlap=384)
            axes[1].pcolormesh(t,f,10*np.log10(np.maximum(z,1e-12)),shading="auto",cmap="magma",vmin=-110,vmax=-35)
            axes[1].set(xlabel="seconds",ylabel="Hz",ylim=(0,8000));fig.tight_layout()
            h.append('<figure>'+png(fig)+'<figcaption>'+esc(r["sound_asset_id"])+'；seed='+esc(r["generation"]["seed"])+' / batch index='+esc(r["generation"]["seed_batch_index"])+'；BEATs='+esc(r["automatic_label"][0]["confidence"])+'<br><code>'+esc(r["normalized_path"])+'</code></figcaption></figure>')
    h.append('<h2>许可</h2><p>模型及生成输出标注 Stability AI Community License，label_provenance=generated_stable_audio_open_1.0。owner 2026-10-12 接受 IV(b) 风险。第 IV(b) 条禁止将输出用于创建或改进任何基础生成式 AI 模型；这里记录所有者决定，不作许可无风险结论。仅训练集限制已经写入每条元数据并由代码执行。</p>')
    h.append('<h2>交付与出处</h2>')
    for x in d["outputs"]:
        h.append('<p><code>'+esc(x["path"])+'</code><br>'+str(x["records"])+' 条；sha256 <code>'+esc(x["sha256"])+'</code></p>')
    h.append('<p>新增文件拉取清单：<code>'+esc(o/"cci_pull_list_v10.txt")+'</code>；CCI 前缀：<code>'+esc(d["cci_mirror_prefix"])+'</code>。</p>')
    try:commit=subprocess.check_output(["git","-C",str(a.worktree),"rev-parse","HEAD"],text=True).strip()
    except subprocess.SubprocessError:commit="未验证"
    if (o/"FINAL_AUDIT.json").exists():
        audit=read(o/"FINAL_AUDIT.json")
    else:
        receipts=[read(f) for pattern in ["generation_receipt_r*.json","screen_receipt_r*.json","whisper_standard_receipt_r*.json"] for f in o.glob(pattern)]
        pilot=read(o/"evidence/whisper_standard_pilot.json")
        pilot_seconds=pilot.get("seconds",0) if isinstance(pilot,dict) else 0
        audit=dict(gpu_stage_wall_seconds=sum(r["seconds"] for r in receipts)+pilot_seconds,
          output_disk_bytes=int(subprocess.check_output(["du","-sb",str(o)],text=True).split()[0]))
    h.append('<p>代码提交：<code>'+esc(commit)+'</code>；审计：<code>'+esc(o/"FINAL_AUDIT.json")+'</code>；GPU 阶段进程墙钟 '+esc(audit.get("gpu_stage_wall_seconds","未汇总"))+' 秒（含模型加载，不是利用率积分）；报告写入前产物磁盘 '+esc(audit.get("output_disk_bytes","未汇总"))+' bytes。</p>')
    h.append('<p>统计出处：DELIVERY.json、SUMMARY_final.json、deduplication_final.json、model_scores_r*.jsonl、whisper_standard_r*.jsonl、V10_READBACK.json、各阶段 receipt；过程见 PROGRESS.md 和 JOBS.json。所有数字取自这些文件。</p><small>生成时间 UTC '+time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())+'</small></body></html>')
    destination=o/"REPORT.html"
    if destination.exists():raise FileExistsError("completed report must not be overwritten")
    destination.write_text("\n".join(h))
    with (o/"PROGRESS.md").open("a") as stream:
        stream.write("- "+time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())+" 单文件中文报告完成，内嵌通过音频波形/频谱与 BEATs 分布，未人工试听。\n")
    print("REPORT_DONE",str(destination),destination.stat().st_size)
if __name__=="__main__":main()
