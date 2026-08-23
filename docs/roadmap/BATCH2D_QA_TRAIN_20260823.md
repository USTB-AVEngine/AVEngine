# 检查点：batch2d 出题认证 + Qwen 零样本评测 + SO-7B 正式训练启动（20260823 晚）

> 接续 PILOT48_AND_AV_SMOKE_20260823.md 与 STAGE_REBUILD_20260823.md。
> 全链 research_only、episode_counted=false，正式数据集分母保持 0。

## 1. Qwen2.5-Omni 零样本三条件评测（pilot48 v2 题，189×3）

评测根：`/data/avengine_external/review/qa_v2_pilot48_eval_v1`（分数
`score_report.json`，评分脚本 `/data/jzy/tmp/score_v2_eval.py`）。
189 题三条件预测齐全，0 解析失败。

| 分组 | n | AV | V-only | A-only | AV−V | AV−A |
|---|---|---|---|---|---|---|
| 总体 | 189 | 54.5 | 53.4 | 51.3 | +1.1 | +3.2 |
| dual_required | 125 | 49.6 | 48.8 | 53.6 | **+0.8** | −4.0 |
| audio_sufficient 对照（T4 左右） | 17 | 47.1 | 41.2 | **29.4** | +5.9 | +17.6 |
| vision_dominant 对照（T9 更近） | 47 | 70.2 | 70.2 | 53.2 | +0.0 | +17.0 |

读法（写论文时的措辞边界）：
- 核心证据：dual_required 上 AV−V 仅 +0.8pp（二选一、随机线 50%）——
  零样本模型拿到音频也用不上，双模态必要性设计成立；
- T4 左右题 A-only 29.4% 低于随机，但 n=17（5/17，p≈0.07）且该管线仍有
  双耳折单声道问题，只能作方向性证据，不得下"系统性左右反转"结论；
- 视觉对照 V-only 70.2% 说明题目本身可答，排除"题出坏了"。

## 2. batch2d 批次收官（192 点全绿）

- 音频：三路并行 + b007 补渲（A 路启动时 b007 留有预拆分半成品被跳过、
  清理后漏渲）→ 192/192，root `qa_v2_batch2_audio_20260823T0700Z`。
- 出题：`tools/qa/generate_qa_v2_questions.py` → **909 题 / 192 点**
  （T2-ATTR 192、T9-CLOSER 192、T4-SIDE 189、TA-MOTION 168、T7-DURING 168；
  dual_required 528；答案字母 A/B = 444/465，bfirst 变体生效），
  输出 `qa_v2_batch2_questions_v1/questions.json`。
- 闸门 B（属性孪生翻转）机器核验 **320/320**（64 对 × 5 题型；规则：
  T2/T9 必须翻转、T4/TA/T7 必须不变；脚本 /data/jzy/tmp/check_gate_b_batch2.py）。
- 闸门 C（text_only ≤ chance，Qwen2.5-Omni 909/909 完成，GPU0，预测在
  `qa_v2_batch2_eval_v1/runs/text_only/`）：**总体 51.6%（469/909，二选一
  随机线 50%）→ 通过**。分题型：T2-ATTR 50.0、T4-SIDE 50.8、T7-DURING 50.0、
  TA-MOTION 50.0（dual_required 全部贴线）；T9-CLOSER 56.8%（109/192，
  z≈1.9、单边 p≈0.03）——对照题型、非卖点题型，记为观察项：疑似颜色先验/
  选项位置偏置，下批可做选项顺序再均衡。
- 审阅页终版：`qa_v2_batch2_review_page`（192 点含题目与成片）。
- npy 退休执行：192 点、释放 **37.08 GiB**，清单
  `qa_v2_batch2_captures_.../npy_retirement_manifest.json`（owner 政策：
  出完题只留 mp4）。

## 3. SO 训练集 `avengine_qa_v2_batch2_av_v1`

`/data/datasets/spatial-omni/avengine_qa_v2_batch2_av_v1`（组装脚本
/data/jzy/tmp/build_so_qa_v2_dataset.py，schema 对齐 avengine_temporal_av_v1）：
- train 790 / valid 119（909 题、192 点；按孪生组切分 seed=20260823，
  孪生永远与其主点同侧，防布局泄漏）；
- 音频 foaish（DCASE [W,Y,Z,X]，W=(L+R)/2、Y=(L−R)/2、Z=X=0，float32，
  16kHz 5s）；视频 = 审阅页 AV 成片（trainer 只抽帧，音轨不读）；
- **pilot48 v2 题保留为 held-out 测试集**（其上已有 Qwen 零样本基线，
  训练后 SO-7B 用同一评测管线出对比表），未混入训练；
- 溯源 `DATASET_PROVENANCE.json`（research_only、无 admission 声明）。

## 4. SO-7B 正式训练（第一轮）

- 启动 20260823 ~18:20 CST，GPU1，locate 环境，Spatial-Omni
  `avengine-av` 分支 b911a76（未推远端，待 owner review）；
- 配置：beats_lora（LoRA r16/α32）、`--with-video --time-aligned-media`
  （16 帧 588×336、时间对齐 RoPE = 论文方法点候选）、bs1×accum4、
  lr 1e-4、3 epochs、每 epoch 全量 valid 生成（119 题 EM）；
  从 SO-7B_finetuned.pt 续训（resume-model-only）；
- **无 replay**（replay+video 未兼容，owner 知情，第一轮接受）；
- 输出 `so_runs/avengine_qa_v2_batch2_av_20260823`，日志
  /data/jzy/tmp/so_qa_v2_train.log，完成标记 SO_QA_V2_TRAIN_DONE；
- 训练后动作：用 pilot48 held-out + 同一评分脚本出 SO-7B vs Qwen 对比表；
  时间对齐消融第二跑（--time-aligned-media 关）排队。

## 5. 杂项

- 全量单测复跑全绿（3111 通过/65 跳过，full_unit_20260823c.log）；
  此前一次失败为工作树临时状态假警报（音频 runner 抢修窗口）。
- 已知坑补充：范围 runner 的 skip_exists 与"清理半成品"操作要按顺序
  执行，否则会漏点（本批 b007 即此因，已补渲并核验 192/192）。
