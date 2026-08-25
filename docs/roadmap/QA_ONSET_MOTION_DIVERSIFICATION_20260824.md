# QA onset 与 motion 多样化 CPU 检查点（20260824）

## 边界与基点

本切片从 `codex/data-diversification-integration@2163763` 建立独立服务器
worktree `/data/jzy/code/AVEngine-codex-qa-onset-motion-v2`。它只完成
`DATA_DIVERSIFICATION_WORKORDER_20260823.md` §3 的 CPU 作者化与出题部分：
没有启动 UE、Habitat、RLR 或 GPU，没有生成正式数据，也没有合并 integration。
所有结果保持 `research_only`/`legacy_report_only` 口径，正式 Episode 分母仍为 0。

owner 的最新决策在本切片中优先：左右 margin、连续 60 cm、是否交换/交叉、
是否入镜都不是候选合法性 gate。它们只可作为后置难度描述。fresh 审计中有
90/192 个已作者化点低于旧 60 cm profile，但均未因此被删除。

## 实现

### 点级 AudioProgram 时序

`tools/qa/design_qa_batch.py` 现在为每个点写独立的
`audio_program.json`，并在 `spec.json` 中记录相对路径和完整 schedule 描述：

- 基础程序必须先通过既有 M6 AudioProgram 校验；
- 只在基础程序已有 Timeline/事件窗口内抽样前两个 onset；
- 第一个 onset 在基础值的前后 2 个视频帧候选内抽样，第二个在前后 4 帧
  候选内抽样；候选必须保留至少一个完整视频帧的静默间隔，并受第三个既有
  event 的起点约束；
- event 顺序、duration、source slice、endpoint/source identity、gain/fade、
  normalization、stem 策略以及第三个以后所有窗口都保持不变；
- 用点级整数 seed 做可复现伪随机选择；twin 继承原点 seed，运行时还会直接
  比较全部 event start/end，任何 schedule drift 都明确失败；
- 沿用 M6 已有 `program_content_sha256`，没有新增 hash、冻结 contract、
  baseline 或独立 gate。

### 四种 motion 与题目 gold

默认 16 点 cell 同时包含：单 source1 运动、单 source2 运动、both-moving、
both-static 和既有 off-screen 点。both-moving 与 both-static 数量相同，以免
它们各自恒为 moving/still 时引入答案偏斜。

`tools/qa/generate_qa_v2_questions.py` 现在：

- 优先消费点目录内经过 M6 校验的 materialized AudioProgram；旧输入没有声明
  点级程序时才保留 shared-base 兼容路径；
- T7 与 TA 对四种 motion 都出题，不再把 both-moving 生成后以
  `G-differential-motion` 拒绝，也不再跳过 both-static；
- first/second sample、frame、间隔、event frame window、Timeline action 序列、
  AudioProgram authority 都写入题目，保留后续难度分层所需原始线索；
- moving/still gold 从作者化 Timeline 对应 source slot 的 `action_id` 读取，
  而不是只信 `spec.motion_case`；第一声检查 onset 帧，第二声检查整个 event
  覆盖帧。若 action 不是全程 `walk` 或全程 `idle`，或与 motion_case 不一致，
  明确失败。

这里的跨文件一致性检查对应一个具体失败场景：外部 fresh 点目录可能误拷
`spec.json`、`timeline.json` 或 `audio_program.json` 中的一份，使每个 JSON
分别类型正确、每个 Git 文件和 M6 program hash 也正确，却把错误 source 的
motion 当成 gold。Git、版本号、普通单测、类型、主键/唯一约束和事务都不能
证明任意运行时目录中的三份文件彼此对应，因此在消费时读 Timeline 并核对是
必要的普通运行时校验；本切片没有为此新增 hash 或冻结 artifact contract。

## fresh CPU 实跑

### 192 点作者化与审计

- 作者化输出：`/data/jzy/tmp/qa_onset_motion_2163763_r1`
- 独立审计：`/data/jzy/tmp/qa_onset_motion_2163763_r1_audit.json`
- 状态：`pass`，但 batch 是 `legacy_report_only`，不是生产视角认证。
- 数量：192/192 authored，64 twins；所有点均为 75 帧 Timeline 和合法 M6
  materialized AudioProgram。
- motion：`s1_moving_s2_static=104`、`s1_static_s2_moving=56`、
  `both_moving=16`、`both_static=16`。
- Timeline 计划 action 与 motion_case：192/192 一致，0 failure。
- twins：64/64 的 event start/end 完全相同，同时 source1/source2 asset identity
  确实互换，0 failure。
- 旧 60 cm 描述：90 点低于 60 cm 仍保留，`candidate_legality_effect=none`。

onset 分布如下；`first_frame` 已不再恒为 4：

| 描述 | 实际结果 |
|---|---|
| first event frame | 2:42，3:50，4:36，5:32，6:32（5 档） |
| second event frame | 15:9，16:6，17:5，18:18，19:5，20:10，21:46，22:21，23:3，24:36，26:13，27:20（12 档） |
| first-to-second interval | 33 个不同 sample 间隔 |

### 出题器实跑

- 输出：`/data/jzy/tmp/qa_onset_motion_questions_2163763_r2/questions.json`
- 192/192 点均解析为 `point_materialized_audio_program`；motion gold authority
  唯一值为 `authored_timeline_action_id`。
- T2：A/B = 96/96，source1/source2 = 96/96。
- TA：moving/staying still = 96/96，source1/source2 = 96/96；四种 motion
  均有题。
- T7：moving/staying still = 96/96，source1/source2 = 96/96；四种 motion
  均有题。

这个 legacy 单视角 report-only 批次的总体 answer balance 仍为 `fail`：
T4 是 28/20，T9 是 56/59。它们没有被本切片伪装成通过；生产 strict 模式
仍会拒绝这种批次。这是后续真实多视角/点位设计端要闭环的 §3.1 项，不属于
本 CPU onset/motion 切片的完成声明。

## 测试

- 相关回归：81 passed：answer balance、design batch、extended questions、
  question generator、held-out selector、M6 AudioProgram、motion feasibility。
- 完整 fast-unit（正确环境）：
  `PYTHONPATH=src:. /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python -m pytest -q tests/unit`
  得到 `3159 passed, 118 skipped, 52 subtests passed`，耗时 326.28 s；JUnit：
  `/data/jzy/tmp/qa_onset_motion_full_unit_2163763_r2.xml`，errors=0、failures=0。
- 第一次 full-unit 使用 bootstrap env，因该环境缺 `cv2` 且未包含仓库根路径，
  collection 阶段 2 errors；原始 JUnit 保留在
  `/data/jzy/tmp/qa_onset_motion_full_unit_2163763_r1.xml`，不把错误环境尝试
  计作通过或产品回归。
- `ruff check` 和 `git diff --check` 通过。

## 尚未闭环

- 没有 UE/SPEAR 或 Habitat 实际运行/readback；Timeline action 是 CPU 作者化
  权威，不是 native actor readback。
- 没有 RLR 渲染或 WAV readback；这里只证明 materialized AudioProgram 合法，
  尚未证明点级 schedule 被真实音频 renderer 消费。
- 没有真实生产 4--8-view bank、原生像素、可见性、off-screen、T5 DoA、拒答
  或 exact-48 held-out 运行证据。
- legacy report-only 的 T4/T9 未平衡；必须在真实视角/点位设计中换点或换首发
  source 后由 strict 出题器证明 50/50±1，不能删题凑数。
- 本分支未合并、未 push，是否进入 integration 由 owner/根代理另行决定。
