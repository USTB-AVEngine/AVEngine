# 检查点 20260823b：pilot48 同物种批次全绿 + SO-7B 全模态训练管线打通

> 承接 STAGE_REBUILD_20260823。owner 指示"全都实现"。全链 research_only，
> qualification_claim=false，正式分母保持 0。

## 一、pilot48 批次（B2，视觉阶段）

**48/48 条链全部成功，零失败**（约 2 小时串行 GPU0）。

| 项 | 值 |
| --- | --- |
| 设计 | 32 主链 + 16 属性孪生；two-human 换色对 36（blue/green/burgundy 两两）+ two-dog 12（边牧+拉布拉多）；运动差分 24 + both_moving 8；off-screen 候选 6 |
| 输入 | `review/qa_v2_pilot48_inputs_20260823T052314Z/`（每点 spec + actor_selection + authored timeline；author 48/48 一次通过） |
| 产物 | `review/qa_v2_pilot48_captures_1ae5294_20260823T0525Z/`（每点 75 帧 rgb.npy + frame_records + receipt） |
| 核查 | `verification_report.json`：位置误差最大 **0.0 cm**、yaw 2.8e-14°、动画 4.8e-7s，48 点全 pass |
| 生成脚本 | `/data/jzy/tmp/design_pilot48.py` + `run_pilot48_captures.sh`（约束在设计侧前置——反向拟合首次实跑） |

抽帧目检：双犬点（p25）边牧行走清晰、拉布拉多在厨房岛后半遮挡（静止位）
——遮挡为真实场景现象，可见性由后续 fact 编译的 visibility pass 判定，
off-screen/遮挡类坑位正是设计所需。

**待办（音频阶段）**：同一批 timeline 走 RIR/双耳 + **原生 FOA** 渲染
（owner 已批的"每链双落"），再接间歇事件窗重混与 v2 出题认证。

## 二、SO-7B 全模态（A2–A5，Spatial-Omni `avengine-av` 分支，未推远端）

分支提交 `b911a76`（+230 行，4 项单测全绿）：

1. **trainer 视频通路**：记录级 `video_path`；ffmpeg 管道解帧（环境无
   PyAV/cv2/decord，零新依赖）；`<|video|><|audio|><|spatial|>` 前缀；
   `--with-video --video-frames 16 --video-max-pixels 200704`；bs=1 守卫；
   与 mixed replay 暂不兼容（显式报错，后续项）。
2. **时间对齐 RoPE**（`so_time_aligned_media`，`--time-aligned-media`）：
   video/audio/spatial 三流共享媒体段时间原点，落在 40ms 绝对时间格；
   spatial id 按音频跨度线性铺开，使同时刻三流 token 时间 id 相等。
   默认关闭 = 原行为逐位不变（回归单测锁定）。**这是论文方法点候选**，
   消融开关就绪。

**AV 冒烟结果**（GPU1，300 样本 ×1 epoch，23.3 分钟）：

| 指标 | 音频-only 冒烟（对照） | AV + 时间对齐 |
| --- | --- | --- |
| train_loss | 0.610 | **0.478** |
| valid_loss | 0.622 | 0.931 |
| valid EM（32 生成样本） | 0.531 | 0.438 |
| 序列长度 | ~230 | ~2330（2016 video + 122 audio + 50 spatial + 文本） |
| 显存峰值 | ~21 GB | **30.7 GB**（48GB 卡，预测 35.4GB 的安全侧） |
| 速度 | ~0.5 s/it | ~4.9 s/it（含 ffmpeg 解码） |

边界声明：仅管线验证。AV 版 EM 低于音频版符合预期——300 样本不足以让
新视频 token 与新位置几何收敛，且 v1 QA 缺陷原样存在；**任何数字无基准
含义**。数据集：`/data/datasets/spatial-omni/avengine_temporal_av_v1/`
（foaish 音频复用 + unique1000 `ue_visual_only.mp4` 配对，带 provenance）。

## 三、决策记录

- 时间对齐方案取"RoPE 时间 id 统一刻度"而非"mp4+use_audio_in_video 交
  错"：与既有 trainer 结构（分开喂）兼容、三流机制统一、消融开关单一；
  已向 owner 说明（对话 20260823）。
- ffmpeg-CLI 解码为环境现实下的无依赖选择；吞吐若成瓶颈，后续加帧缓存。

## 四、下一步队列

1. pilot48 音频阶段（RIR/双耳/原生 FOA 双落）→ 事件窗 → v2 出题 + 三闸门；
2. replay + video 兼容（防遗忘与 AV 共存）；
3. owner review `avengine-av` 分支；
4. 认证题就位后：AV 正式训练 + Qwen 三条件零样本评测 + 时间对齐消融表。
