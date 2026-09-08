# 绑定可行性与逐题难度画像审计（2026-09-06）

工具：`tools/qa/audit_binding_feasibility.py`；测试：`tests/test_audit_binding_feasibility.py`。状态 research_only。

## 它回答什么

一道"先用声音认出是谁，再问它怎样"的题，到底能不能答、有多难。它不看模型，只看交付产物：
一段 Episode 的归一化事实 `facts.json`、成片双耳 WAV，以及同一 delivery 目录的 `questions.json`。
所有数字都是从这些文件重新算出来的，方位公式和相机基向量与 `unified_catalog._listener_azimuth` 相同，
能导入 `avengine` 时会逐帧对账并把最大差值写进输出（`azimuth_formula_crosscheck`）。

## 运行

```bash
cd /data/jzy/tmp/wt-multi-home-activity-integration
PYTHONPATH=src /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
  tools/qa/audit_binding_feasibility.py \
  --facts <delivery>/facts.json --questions <delivery>/questions.json \
  --out tmp/<fresh_dir>/<episode>.json
```

`--wav` 缺省取 `facts.audio.path`。`--stems-dir` 缺省在成片目录及其 `audio/binaural` 子目录里找 `<actor>_*stem*.wav`
（逐源湿声，IEEE float32 WAV 也能读）。`--acoustic-package` 给 RLR 声学包 manifest 就会算直达射线；计划文件里
`resources.acoustic_package` 就是它。输出 no-clobber。阈值全部可用同名命令行参数覆盖（如 `--theta-static-deg`、
`--min-sustained-separation-s`、`--itd-min-ms`、`--itd-min-cc`、`--near-axis-deg`、`--open-angle-min-gap-deg`、
`--open-time-min-gap-s`、`--sector-margin-deg`、`--post-sound-min-displacement-deg`、`--distance-margin-m`）；
默认值全部是占位，输出的 `thresholds.calibration` 写着 `placeholder_pending_human_calibration`，等人工校准后再定。

## 输出里有什么

- `audio`：成片峰值和有效值（dBFS）、精确零采样点比例、片尾一秒有效值、左右声道相关系数。
- `events`：每个声音事件的听者相对方位（引擎帧右为正，字段 `azimuth_convention` 注明）、距离、起始可见状态，
  与最近竞争者的夹角在说话期间的起始、结束、最小、最大和变化量，说话期间说话人和竞争者是否在动，
  说话人自身方位扫过多少度，与其他事件的重叠；以及从成片实测的左右能量差（全带、0.5–1.5 kHz、2–6 kHz）、
  起点直达声窗内的双耳时差和相关系数、按人头几何算的时差预期。v2 起还有：`audible_window`（从 stem 读的可听起止）、
  `separation_over_state_window_deg`（可听窗内夹角的最小值、分位数、连续达阈时长、最近竞争者是否换人）、
  `measured_source`（量的是 stem 还是混合轨）、`mixture_contaminated`、`line_of_sight`（给了声学包就有 clear / blocked /
  partially_blocked / unmeasured）、`wet_tail_interval_from_facts`。
  `binding_feasibility` 给出两个**各自独立的三态**：`geometry_state` 和 `delivered_cue_state`，取值 candidate_pass /
  candidate_fail / unmeasured / not_applicable_no_competitor；**没有合并的 feasible**，没量到的东西永远是 unmeasured。
- `questions`：每道题的分组（绑定 13 类、条件性 3 类、音频对照 3 类、视觉对照 5 类）、四组难度画像
  （听、看、时间推理、是否需要绑定）和分歧审计：其他候选在被查属性上的取值、金标是否为多数、是否退化、
  声停后答案能否由声停前趋势外推、QA-13 金标离扇区边界多近。标记名见源码 `flags`。
- `summary`：几何状态计数、线索状态计数、直达射线计数、stem 数、污染窗数、各标记计数，以及只看视觉结构的两种常数策略
  （选多数值、选唯一少数值）在绑定组 MCQ 题上的命中数（`structural_baselines`）。

## v1（2026-09-06 下午）对 74 题批次的结果，已被下节 v2 重跑取代

输出在 `tmp/audit_binding_feasibility_20260906/{A,B,C,L,N}.json`。五段 14 个声音事件里只有原生 Apartment（N）的两个
同时满足几何分得开和成片有线索；A、L 两段两位说话人夹角 8.5°，C 段有两对只有 2.3°，B 段侧向声源被挡、成片 ILD 不到 1 dB。
成片峰值 −23 到 −36 dBFS，40% 到 84% 的采样点是精确零。逐题看，N 段 6 道题其他候选取值与金标相同，
B 段两道声停后题可由声停前趋势外推。完整解读见 owner 本机的审阅文档
`AVENGINE_PAPER_RESEARCH_REVIEW_CLAUDE_20260906.md`。

## v2（2026-09-06 晚）：按 Codex 审核第 8 节修正后重跑

Codex 在 `docs/roadmap/QA_GENERALIZED_SAMPLER_PLAN_REVIEW_CODEX_20260906.md` 第 8 节指出的七个问题都属实，v2 逐条改了：

1. 缺音频测量时不再判通过：去掉合并的 `feasible`，几何与线索各自三态，`unmeasured` 不参与任何"通过"计数。
2. 几何不再看瞬时最大夹角：用可听窗内的最小值和连续达到阈值的时长（占位 30°、1.0 秒）。
3. 线索不再用绝对 ILD/ITD 阈值：比较目标与最近竞争者按 Woodworth 球头模型算的预期时差之差（占位 0.2 毫秒），再看实测
   方向是否与预期一致；方位绝对值小于 10° 时预期近零，实测近零就算一致。正前方的声源不再因此被判"无空间信息"。
4. 有逐源 stem 时在 stem 上量起点和线索，可听窗也从 stem 读；混合轨另报，与其他事件重叠的窗打 `mixture_contaminated`。
5. 最大并发发声人数改为事件边界时间扫描、半开区间、按实体去重。
6. QA-13 和 QA-19 的分歧按题目实际可用的形式分别算（`divergence_by_form`）：Open 用数值间隙，MCQ 用该题答案域；
   只有 Open 的题不再被按扇区误标退化。
7. `line_of_sight` 由 `--acoustic-package` 的静态三角面填；坐标系不是米、Y 向上就拒绝并写原因。

另外两条更正：A 段"两人同时开口"不成立（两条 stem 可听起点 1.283 与 1.782 秒，差 0.499 秒；Codex 用另一种检测器得
1.338 与 1.804 秒）；"35/60 就是位置偏好"改为"与位置先验分不开"。

五段重跑输出在 `tmp/audit_binding_feasibility_20260906_v2/{A,B,C,L,N}.json`（占位阈值）：

- A、L：两事件几何与线索全部 candidate_fail（夹角 8.5°）；两事件互相重叠，混合轨窗都标污染，stem 上量的 ITD 分别为
  +0.047 与 −0.008 毫秒。
- C：四事件全部 candidate_fail（夹角 2.3° 与 10.7°）。
- N：两事件几何与线索全部 candidate_pass（夹角 54 到 69°，ILD 3.8 与 −6.3 dB，ITD −0.05 与 −0.37 毫秒）；QA-13 只有
  Open 形式，数值间隙 69.3°，不再标退化。
- B：只有黄衣人那句几何 candidate_pass（夹角 37.6 到 47.6°，连续 1.47 秒），但它的直达射线被静态几何挡住；−74.5° 的
  蓝衣人那句直达射线是通的，成片 ILD 却只有 1.0 dB、ITD 为 0。这推翻了我之前"被墙挡住"的猜测，需要从音频链查原因。
  结构基线：B 的 5 道绑定组 MCQ 里 3 道金标是唯一少数。
- 方位公式与 `unified_catalog` 逐帧对账五段全部差为 0。

测试 15 个（`tests/test_audit_binding_feasibility.py`），覆盖上面七条各自的已知答案合成用例。

## 边界

这些数描述的是题目的可答结构和难度，不是模型成绩，不是人工可答性证书，也不是正式准入。
