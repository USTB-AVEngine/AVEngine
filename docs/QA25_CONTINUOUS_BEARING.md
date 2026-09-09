# QA-25 连续角度题与评测

现有 QA-01～QA-24 保留。QA-25 和相关角度附问已经接入统一生成器、原生 Episode 交付和评分 CLI；模型代码、训练与模态必要性实验不属于这次实现。

## 题目定义

| 子集 | 输入与目标 | 生成条件 |
|---|---|---|
| QA-25-A | 根据声音事件的开始时间，在明确查询时刻回答发声点的方位角 | 实际声源活动、可区分的事件锚点、原生发声点和听者姿态 |
| QA-25-V | 回答指定可见实例的可见像素质心相对相机的水平角度 | 已核验且可区分的外观、真实可见质心、公开的针孔相机标定 |
| QA-25-AV | 先从可见且发声的实例建立对应，再回答它隐藏之后的声音方位 | 可见听觉锚点、查询时至少两个不同声音资产同时活动、目标隐藏、隐藏运动偏离可见阶段的静止或线性延续 |

题面时间默认使用整秒，角度答案使用整度，不显示小数。QA-25 与发声角度附问先选择整秒对应的原生帧，再核验活动与可见状态并计算答案；片尾附问保留实际末帧，以“片尾”表述，避免把末帧误写成一个不存在的整秒。时间区间默认向内取整，不把未经验证的边界扩进题目；无法表达的短区间会暂不出题。内部几何与音频证据仍保留完整精度。

角度统一要求整数数值，正前方为 0°、右侧为正、范围 [-180°,180°)。A/AV 的参考系为实际听者局部坐标，使用完整姿态，包含俯仰和翻滚；V 为相机局部像素射线。A/AV 的目标是声音发射点，V 的目标是可见像素质心，两者不混称同一物理点。

V 不使用隐藏实例的目标遮罩质心。只有全部目标像素可见时，目标遮罩质心才可作为可见质心；否则需要原生的 `visible_centroid_xy_px`，没有就暂不出题。原生交付读取本次捕获使用的视场角与实测图像尺寸，公开 `fx_px`、`cx_px`、宽和高。裁切或缩放图像时，需要相应变换标定。

AV 的孤立锚点同时检查声源活动和其他声音的混响尾声，避免把语音短暂停顿当成完全安静。生成条件只证明结构上具备候选条件；同一音色能否持续辨认、完整 AV 是否可答、A-only/V-only 是否不足及配对是否平衡，仍需感知和消融实验。输出继续使用 `research_candidate` 与 `certification.status=not_run`。

## 角度附问和兼容性

QA-03、QA-20 在事件可定位时生成显式数值角度附问。QA-24 只在片尾仍有声，或可用可见质心定位时生成；隐藏且静音时不索取隐藏位置。QA-13 原有数值 Open 直接进入连续角度统计，保留 15°/30° 评分，不重复生成附问。

原题保存在 `items`，附问保存在 `angle_followups`，各有独立问题 ID，附问通过 `parent_question_id` 关联原题。`iter_unified_items()`、公开输入导出与评分器会一起消费它们。原题的数量与配额不被附问覆盖；每个问答只评一次。默认 QA-25 的配额是 A/V/AV 各一题，`--items-per-type N` 将每个子集的配额设为 N。

公开导出使用 `question_000001` 一类的序号，避免把旧私有问题 ID 内嵌的实例、事件和帧信息交给模型。序号作用域为一份 Episode 题集，评分器自动映射回私有 ID；已有使用私有 ID 的结果仍可评分。批量合并答案时保留所属 Episode，不要重排私有题集后继续使用旧序号。只把导出的 `forms` 题面、必要标定和对应音视频交给模型。

## 使用

在 AVEngine 根目录、已配置的 Python 运行环境执行；每次选择不存在的新输出路径。

```bash
PYTHONPATH=src python tools/qa/generate_unified_questions.py \
  --input path/to/facts.json \
  --qa-ids QA-03,QA-13,QA-20,QA-24,QA-25 \
  --items-per-type 1 \
  --out tmp/angle-run/questions.json \
  --model-inputs-out tmp/angle-run/model_inputs.json
```

不指定 `--qa-ids` 时请求全部 25 类。`--no-angle-followups` 关闭附问。输入已有原生标定时保留它；也可通过 `--camera-calibration` 传入与实际媒体匹配的公开标定 JSON。

答案可以是映射，例如 `{"question_000001": "-27 degrees"}`，也可以是含 `question_id` 和 `answer` 的记录列表。

```bash
PYTHONPATH=src python tools/qa/score_unified_questions.py \
  --questions tmp/angle-run/questions.json \
  --answers path/to/model_answers.json \
  --form open --out tmp/angle-run/scores.json
```

## 读分数

- `angle_metrics.mae_deg`、`median_deg`：有合法数值答案的圆周误差均值、中位数；同时看 `parsed`、`unparsed`，不能用少答题降低误差来宣称提升。
- `accuracy_at_deg`：1°、3°、5°、10° 命中率，分母包含缺答、无效回答和拒答。这些阈值是报告维度，不是数据准入门槛。
- `by_qa`、`qa25_by_subset`：分题型与组件结果。
- `instance_binding_joint`：QA-03/QA-20 实例回答正确且角度在阈值内的联合成功率。实例答错，即使角度接近也不能通过联合指标。
- 新连续题的单行兼容 `score` 为 `1 - circular_error_deg / 180`；主结果应报告上述角度指标。QA-13 的旧分数不变。

`--form mcq` 不把只有数值 Open 的 QA-25 或角度附问算作缺答。题目文件包含私有 gold 和引擎证据，仅供评测使用；生成器自检以 gold 作为输入所得的零误差不代表模型性能。
