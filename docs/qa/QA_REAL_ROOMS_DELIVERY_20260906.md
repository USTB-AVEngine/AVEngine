# AVEngine 统一房间 QA：交付与运行方法（2026-09-06）

实现提交：`48afe5b46f7ada95e4b72a2ad823b1eeb92091b3`。权威目录为
`48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration`，分支
`codex/multi-home-activity-integration`。本批为 **research_only**，没有正式准入、push、main 合并或 Studio 服务切换。

已交付 4 个既有房间资产上的 5 段原生 Episode：共 1260 帧、84 秒、720p/15 Hz、独立 16 kHz 双耳 WAV；
74 道有效 QA 覆盖 QA-01～24，支持 74 个 Open 与 73 个 MCQ 实例。17 个不满足条件的采样保留原因，不计入有效数。
房间来源是三套既有自建居室和原生 `apartment_0000`；本批没有把它们称为 MP3D/HM3D 扫描场景。

## 交付位置与逐项覆盖

下文的 `$R` 指仓库内 `tmp/qa_real_rooms_20260906`；
其物理位置为
`/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_real_rooms_20260906`。

| 标记 | Episode 根目录 | 活动与镜头 | 帧数 / 秒 | 有效 QA | 最终派生目录 |
|---|---|---|---:|---:|---|
| A | `walk_pair_a_v6` | 2 人，站立/行走、静态镜头、重叠发声 | 240 / 16 | 19 | `delivery_final_v1` |
| B | `walk_four_b_v2` | 4 人，行走/停留、转动镜头、自然遮挡/入画 | 300 / 20 | 21 | `delivery_final_v1` |
| C | `vocal_classes_c_v2` | 4 人站立，speech/laughter/cough/sneeze | 240 / 16 | 13 | `delivery_final_v2` |
| L | `late_direction_a_v2` | A 房的另一段双人 Episode，跟随朝向、声停后问题 | 240 / 16 | 3 | `delivery_final_v2` |
| N | `native_apartment_v6` | 原生 Apartment，2 人沿原生路径行走后停留 | 240 / 16 | 18 | `delivery_final_v1` |

- 编号、题义、输入条件、实现/采样/判分状态及 A/B/C/L/N 证据见 [统一目录与覆盖表](QA_UNIFIED_CATALOG_20260905.md)。
- 精确样本 ID、可用形式、每房 deferred 原因：`$R/qa_coverage_all_five_final_v2.json`。
- 最终合批：`$R/final_export_v1/manifest.json`；输入配方：`$R/final_export_request_v1.json`。
- 合批读回：`$R/final_delivery_readback_v1.json`；代码与运行时：`$R/delivery_code_version.json`。
- 每段的视频母版为 `capture/ue_visual_only.mp4`；带 AAC 的预览位于最终派生目录的 `preview.mp4`。
  独立无损双耳 WAV 的准确路径在该目录 `input_refs.json` 指向的音频报告中。

导出只写引用索引，不按题复制媒体。公开输入为 `qa/questions.jsonl.gz`；
标准答案、forms、证据与 deferred 记录在独立的 `qa/answers.jsonl.gz`。
`episodes.jsonl.gz` 关联计划、原生读回、视频/WAV、扩展证据；
`rooms.jsonl.gz` 集中引用房间、人物、干声与 HRTF 输入。
模型请求、原始回答、分数、类别诊断和缓存记录在 manifest 的 `batch_evidence` 中只引用一次。

## 运行

先查看 GPU/进程及请求中的 RPC 端口，再使用未存在的输出目录。示例沿用本机已验证的输入配置：

```bash
ssh 48g-jump
cd /data/jzy/tmp/wt-multi-home-activity-integration
nvidia-smi
export PYTHONPATH=src:tmp/native_python_addons_v1
QA_PY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
QA_RUN_TAG=$(date +%Y%m%dT%H%M%S)

"$QA_PY" tools/studio/run_qa_episode.py \
  --request tmp/qa_real_rooms_20260906_inputs/native_apartment_request.json \
  --output "tmp/qa_native_replay_${QA_RUN_TAG}"
```

同一入口可使用 `walk_pair_request.json`、`walk_four_request.json`、
`vocal_classes_request.json`、`late_direction_request.json`。
请求声明 room catalog、源资产 ID、sound pool、种子、QA 类型、活动/镜头及 runtime；
源与相机坐标由计划器计算。C 的最终 QA 配方是其 Episode 根目录的
`request_final_qa02.json`，只比原请求增加已评测的 QA-02。

`--plan-only` 仅返回潜在能力与候选计划，不证明原生执行成功。
完整调用依次生成计划、执行 SPEAR、读回像素/动作/发声点、渲染动态音频、出题和导出。
`qa_episode` 已接入当前分支 Studio template，但运行中的 8765 Studio 没有被切换到本分支。

仅重新出题、导出时，复用已完成的原生捕获和正确的音频报告：

```bash
"$QA_PY" tools/studio/run_qa_episode.py \
  --request tmp/qa_real_rooms_20260906_inputs/native_apartment_request.json \
  --output tmp/qa_real_rooms_20260906/native_apartment_v6 \
  --resume --derived-output "tmp/qa_native_derived_${QA_RUN_TAG}" \
  --audio-report tmp/qa_real_rooms_20260906/native_apartment_v6/delivery/audio/research_report.json
```

音频复用会核对其 plan 与原生读回引用；同为 240 帧/16 kHz 的另一段 Episode 报告也会被拒绝。
数值 RIR 已清理时，重出题仍可用 WAV/报告；重新生成音频必须选新 cache/output。

判分与合批使用现有 CLI，输出均须 fresh：

```bash
"$QA_PY" tools/qa/score_unified_questions.py \
  --questions tmp/qa_real_rooms_20260906/native_apartment_v6/delivery_final_v1/questions.json \
  --answers /path/to/model_answers.json --form open \
  --out "tmp/qa_scores_${QA_RUN_TAG}.json"

"$QA_PY" tools/dataset/export_episode_bundle.py \
  --request tmp/qa_real_rooms_20260906/final_export_request_v1.json \
  --output-root "tmp/qa_export_${QA_RUN_TAG}" --gzip-json
```

答案文件按 question_id 映射模型回答。模型的可复跑命令、实际权重/解释器/输入配置见
`$R/final_model_export_refs_v2.commands.txt` 和同名 JSON。
默认 Qwen utility 的音频输入会变成单声道，已另外运行 `dual_mono` 两个带 L/R 标签的音频输入；
没有转换为四声道。

站立/行走使用现有栅格路径与几何相机候选；原生 Apartment 复用原生路线库，并在当前加载的
UE Recast 上重查 30 段路径。路径时钟必须与请求匹配，不能把 15 Hz 路径直接冒充其他帧率。
已有静坐调用现有座点/pose bindings，新的统一入口已完成计划检查；原有 15 帧原生静坐证据复用，
未计入上述五段。静坐配方为 `tmp/qa_real_rooms_20260906_inputs/seated_plan_request.json`，
旧原生证据为 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/furnished_seated_studio_sol/studio_request_a_seated_15_v2`。
没有新增家具、坐下过渡、IK 或任意扫描家具自适应。

## 验证与限制

数据有效性与模型成绩分开记录：

- 五段原生 RGB/normal depth/target-only depth 均完成，媒体实际解码，WAV 完整读回为有限的两声道数据。
  Native Apartment 三轮捕获的位置误差最多约 `7.1e-15 cm`、朝向误差为 0，地图读回为 `apartment_0000`。
- 逐帧保存 root、动画相位、相机与绑定发声点，听者使用同一相机 rig；
  UE 厘米坐标经既有变换进入声学包，HRTF 为实际加载的 MIT KEMAR SOFA，输出次序为 left/right。
- 动态 RIR 关键帧数 A/B/C/L/N 为 16/60/80/80/80，使用既有 RIRCacheSession 与双源分片。
  N 的 160 次源/关键帧使用归并为 52 个实际 RIR jobs。源时域 raised-cosine 分区卷积、事件 tick、
  湿声尾音、线性增益与混合均留有记录；不逐源归一化掩盖增益差异。
- 全 147 个可用形式的标准答案判分通过，73 个故意错误 MCQ 均为 0 分；
  这证明生成/序列化/判分一致，不是模型准确率。
- Open 方位/时间阈值仍是研究占位值：角度 full/half 为 15°/30°，时间为 0.3s/1s；
  未做每题型、每形式、每种缺失模态的正式校准。L/QA-13 位于 -45.7577°，接近 MCQ 的 -45° 边界；
  N/QA-13 两候选同属 front，故只保留 Open，MCQ deferred。
- QA-10 当前只输出由原生 mask/depth 明确识别的人物遮挡物；静态家具身份不足时不猜。
  C/QA-23 的非语音多脉冲事件未审核分段，保持 deferred；没有把四次播放直接算成四个感知事件。
- 人物沿计划重放，不是动力学仿真。结构包围盒检查及必要的源 GLB 蒙皮姿态复核通过；
  不宣称家具接触或完整人体碰撞认证。发声点是已登记的 root 附着偏移，不宣称逐帧口型/口腔运动。

实际源/听者区域的局部几何检查共 **133 条射线，CPU 与 native 各自完成同一组控制**：

| 房间 | 成功控制 | 源到声学包对应与边界 |
|---|---:|---|
| A | 33/33 | 360/360 对应；L 坐标复核后复用 A 区域检查 |
| B | 36/36 | 185/188 对应；3 个植物叶节点未入包，未用于本组控制 |
| C | 32/32 | 503/503 对应，包含 doorway clear 控制 |
| Native Apartment | 32/32 | 118/118 有 mesh/triangle 对应；两个共享 mesh 的 sink-door 实例仍有歧义，未用于本组首命中控制 |

这些检查使用实际起点、墙体/开口语义与首次命中，不拿 self-hit 或最终撞到远墙替代防穿墙证据。
20 个 Apartment 门/抽屉名字差异已追到瞬时实例名与 mesh datablock，未据此虚报缺面。
边界边不等于漏声，少量 ray pass 不证明整屋封闭；视觉 PBR 也不证明声学材料准确。
未引入补洞或封门窗，没有适用实测 RIR，因此不宣称真实混响绝对精度。

原生运行中的 inactive animation UObject 回收问题已修复为运行期持有、结束时释放。
失败及诊断输出均保留、未纳入数据计数。Apartment v4 被误判后中断于证据压缩，其诊断更正已留档；
只计完整完成的 v6。

## 模型实测

Qwen2.5-Omni-7B：12 道题 × 5 条件，原始请求对应的结果为 **35/60**。
7 道最终题仅改变 MCQ 选项顺序，导出逐条保留原始选项、raw letter、所选语义值和 final index 映射；
不把旧字母直接套进新选项，也不宣称新排列被重新评测。

| 条件 | 正确 / 12 |
|---|---:|
| 视频 + 单声道音频 | 6 |
| 仅音频 | 8 |
| 仅视频 | 8 |
| 仅文本 | 7 |
| 视频 + 分别输入 L/R | 6 |

无弃答和解析失败，25 个错误答案保留。本小批没有证明 AV 优势或空间理解能力。
Whisper 的 QA-12 转写核对得分 1.0；QA-19 时间定位未评分。
CLAP 对原始 laughter/cough/sneeze 的类别诊断通过，成片 sneeze 窗口与 cough 相近，歧义保留。
这不是覆盖 24 类的模型认证：合批有 60 条 Qwen scored、1 条 Whisper answered、
1 条 not_scored 和 382 条 not_run；N 未做模型评测。

## 存储、耗时和缓存生命周期

以下为文件字节数，按实际文件 inode 去重；MB = 1,000,000 bytes。分母只用 74 道有效唯一题，
不使用 91 次采样、147 个形式或历史重写版本。

| 项目 | 实际 bytes | 说明 |
|---|---:|---|
| 永久媒体与证据 payload | 768,586,739 | 含无损 WAV、预览、全量 depth/ID/mask、干湿 stem、引用的原生 RGB 帧、模型请求/结果 |
| 共享场景/人物/声音/HRTF 输入 | 1,117,370,885 | 包含声明的编辑/视觉/声学资源，跨房/题去重 |
| 上述合计 | 1,885,957,624 | 永久 payload 约 10.386 MB/题；含共享输入约 25.486 MB/题 |
| 合批 JSON/JSONL gzip 索引 | 113,823 | 另外仅约 0.00154 MB/题，没有媒体复制 |
| 已清理数值 RIR，累计 | 34,468,189 | 五个完成的 cache，含 B 的已结束比较 cache |
| 可核实同时缓存峰值 | 31,129,174 | 成功 cache 同时存量 30,135,680 + 当时已存在的 A 失败 cache 993,494 |
| 当前保留数值 RIR | 645,114 | 仅 A_v2 两套失败/复用证据，未删除 |
| 当前保留 cache 元数据 | 1,710,543 | 配置、索引、空间/时钟、receipt、cleanup 等；不是数值 RIR |
| UE DDC 文件 payload | 677,110,057 | 共享生成缓存，4724 文件，单列；未当训练数据计费 |
| 追加到私有 UE Content | 895,875,103 | 既有源资产的必要运行副本：人物 91,335,034 + 原生 SPEAR Content 804,540,069 |

数值缓存跨阶段生成：native cache 在前三屋 cache 清理后才生成，累计清理量不是同时峰值。
该峰值来自持久化缓存的实际阶段盘点，不包含瞬时写入临时文件或内存。
原始几何 `rir_*.obj`、旧工作树和共享源资产没有清理。
完整生命周期见 `$R/dynamic_rir_cleanup_result_v2.json` 与 export manifest 的 cache_lifecycle。

扩展证据未省略（`omitted_extended_evidence_bytes=0`）。
未索引为永久材料的全部 RGB 调试帧、失败现场和旧派生仍保留；
本次主要输出子树盘点 3,338,208,120 bytes，输入子树 8,614,964 bytes。
它们与上表永久/缓存范围有重叠，不能相加当净新增量。
UE 安装、既有模型权重和外部原始数据集不属于本批训练 payload。
盘点见 `$R/retained_storage_snapshot_v1.json`。

| Episode | 入口实测秒数 | 计时范围 |
|---|---:|---|
| A | 430.237 | 计划 + 捕获；音频/最终出题另行完成 |
| B | 525.333 | 计划 + 捕获；音频/最终出题另行完成 |
| C | 754.574 | 计划 + 捕获；含该次资源准备/编译开销 |
| L | 450.078 | 一次入口含捕获、音频、QA、导出 |
| N | 400.197 | 一次入口含全链；调试器初始化约 46s 额外开销 |

计时直接取各 Episode 的 `episode_result.json`；不是统一硬件吞吐基准，也没有把失败尝试算进成功 Episode 耗时。
模型首次 12×5 批为 219.034s，修正其中 5 题后的重跑为 145.315s；
7 题复用原始 raw。二者属于不同阶段，不包装成一次 fresh 批耗时。

## 检查记录

最终 fast suite：**4407 passed、117 skipped、2 deselected、52 subtests passed**；
skip 多为未挂载历史/native 数据与已归档流程，不计作 native pass。
日志为 `$R/fast_suite_final_v2.log`。其后新增的音频引用一致性检查另以五套实际正确输入、
一套相同 clock 的错误 Episode 报告验证，记录在 `audio_reuse_binding_readback_v1.json`。

代码集成提交后再次读取合批 gzip 索引：4 rooms、5 episodes、74 个公开问题、91 个私有结果，
五份独立视频/WAV 引用、444 条模型状态记录均一致。
捕获发生于本任务工作树验证阶段；保留当时的 producer_version，不将后来的 Git commit 伪记为捕获时版本。
当前提交用于继续生成与复跑，不宣称所有历史音频逐字节等同于当前默认 renderer。

运行中的 Studio 健康接口仍报告
`repository_root=/data/jzy/tmp/wt-qa-v3-engine-completion`。
本任务未切换服务、推送或正式发布；研究候选与正式准入继续分开。
