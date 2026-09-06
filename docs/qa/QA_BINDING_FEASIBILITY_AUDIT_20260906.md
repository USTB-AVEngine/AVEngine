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

`--wav` 缺省取 `facts.audio.path`。输出 no-clobber。阈值可用 `--theta-static-deg`、`--theta-motion-deg`、
`--ild-min-db`、`--itd-min-ms`、`--sector-margin-deg`、`--post-sound-min-displacement-deg`、`--distance-margin-m` 覆盖；
默认值全部是占位，输出的 `thresholds.calibration` 写着 `placeholder_pending_human_calibration`，等人工校准后再定。

## 输出里有什么

- `audio`：成片峰值和有效值（dBFS）、精确零采样点比例、片尾一秒有效值、左右声道相关系数。
- `events`：每个声音事件的听者相对方位（引擎帧右为正，字段 `azimuth_convention` 注明）、距离、起始可见状态，
  与最近竞争者的夹角在说话期间的起始、结束、最小、最大和变化量，说话期间说话人和竞争者是否在动，
  说话人自身方位扫过多少度，与其他事件的重叠；以及从成片实测的左右能量差（全带、0.5–1.5 kHz、2–6 kHz）、
  起点直达声窗内的双耳时差和相关系数、按人头几何算的时差预期。`line_of_sight` 目前留空，等射线工具填。
  `binding_feasibility` 给出"几何上分得开"（静态夹角够大，或说话期间夹角变化够大）和"成片里量得到线索"两个判断。
- `questions`：每道题的分组（绑定 13 类、条件性 3 类、音频对照 3 类、视觉对照 5 类）、四组难度画像
  （听、看、时间推理、是否需要绑定）和分歧审计：其他候选在被查属性上的取值、金标是否为多数、是否退化、
  声停后答案能否由声停前趋势外推、QA-13 金标离扇区边界多近。标记名见源码 `flags`。
- `summary`：各标记计数。

## 2026-09-06 对 74 题批次的结果

输出在 `tmp/audit_binding_feasibility_20260906/{A,B,C,L,N}.json`。五段 14 个声音事件里只有原生 Apartment（N）的两个
同时满足几何分得开和成片有线索；A、L 两段两位说话人夹角 8.5°，C 段有两对只有 2.3°，B 段侧向声源被挡、成片 ILD 不到 1 dB。
成片峰值 −23 到 −36 dBFS，40% 到 84% 的采样点是精确零。逐题看，N 段 6 道题其他候选取值与金标相同，
B 段两道声停后题可由声停前趋势外推。完整解读见 owner 本机的审阅文档
`AVENGINE_PAPER_RESEARCH_REVIEW_CLAUDE_20260906.md`。

## 边界

这些数描述的是题目的可答结构和难度，不是模型成绩，不是人工可答性证书，也不是正式准入。
