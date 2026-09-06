# 房间家族 QA-01～24 复测与模态依赖补充（2026-09-06）

本次针对 Apartment、Kujiale、HM3D、MP3D、自建房间分别检查保留的原生材料，
并用同一版当前 QA 代码逐段尝试完整的24项。没有重新拍摄或按成功结果删题。
实现修复提交：`605fc564a00d7e840b53d1e8da16720501c838e4`。

## 比例口径

分母为每段24项；一项至少有一个支持形式可生成才记作候选通过，Open/MCQ另列。
比例只表示所选素材在当前生成器/证据条件检查下的候选生成率，
不代表模型准确率、整类数据集成功率、完整人类可答性或双模态必要性认证。
同一房间多个Episode各算一次尝试，跨Episode类型并集另计。

| 家族 | 实际选定材料 | Episode数 | 候选通过/尝试 | 条件不足 | 候选生成率 | Open/MCQ |
|---|---|---:|---:|---:|---:|---|
| Apartment | 原生 apartment_0000 双人 | 1 | 18/24 | 6 | 75.0% | 18/24、17/24 |
| Kujiale | InteriorAgent 0020 livingroom_491 双人原生UE | 1 | 17/24 | 7 | 70.8% | 17/24、17/24 |
| HM3D | TEEsavR23oF 单扬声器诊断材料 | 1 | 0/24 | 24 | 0%* | 0/24、0/24 |
| MP3D | 17DRP5sb8fy region000、原生Habitat双Beagle | 1 | 5/24 | 19 | 20.8% | 5/24、5/24 |
| 自建 A/B/C | A、B、C 与 A 的第二段 | 4 | 75/96 | 21 | 78.1% | 75/96、75/96 |

本次最终函数执行错误为0，229个已生成形式的标准答案判分检查通过。
已保留早期适配失败及修复记录，不将适配错误混同房间物理不支持。

*HM3D 的0表示这份不完整候选没有产出有效题，不能推断HM3D房间能力为0。
其缺少主体/相机原生逐帧读回和像素实例证据，仅有单一扬声器与未审核的事件，
几何报告fail、ray not_run；独立WAV为6.65075秒，视频为5秒（MP4音轨约5.056秒）。
因此HM3D仍未完成与其他家族同等输入完整性的QA验证。

自建房间在统一24项下：A 19/24、B 21/24、C 17/24、L 18/24。
此前56/67对应原先选择的QA请求集，其中C只请求14项、L只请求5项；
本表改为全部各24项后是75/96，两种分母不能混用。
这些房间合起来覆盖24类，不代表每个房间或每个外部家族覆盖24类。

## 实际缺口

- Kujiale：原生RGB150帧，真实target-only像素证据113帧。保留真实索引，不补缺失37帧。
  Astra核对source1 burgundy在80/93/110、source2 blue在93/110的粗色；不把source2在80的边缘碎片算成外观通过。
  缺少wet-tail读回、唯一遮挡物、丰富声种和片尾149帧像素证据，相关题deferred。
- MP3D：原生Habitat，150帧/15Hz/320×240、16kHz双耳音频与逐帧actor/emitter/camera读回。
  无当前可用的实例可见性/外观审阅证据；两个Beagle的非语音事件也未审核感知分段。
  通过QA-04/05/06/16/17。QA-16/17目前是原生运动/距离真值候选，完整视觉可答性与双模态必要性未认证。
- HM3D：只是现有诊断素材的缺口审计，不能拿可播放媒体或计划替代原生执行/像素证据。
- 所有结果保持research_only/research_candidate，未作正式准入，也未在本轮重跑模型。

## 本次修复

1. 稀疏像素帧此前会因为不等于全片帧数而被整体丢弃。现在保留显式、唯一、范围合法的已知帧，
   缺失帧仍然deferred；完整旧式序列继续支持按序号索引。
2. 入画侧只用相邻帧证明；重现/遮挡转换的否定答案需要完整覆盖，
   防止把没观测到的区间解释为没有发生。
3. QA-22此前按演员清单计数，现在按实际出现集合计数，
   并将发声集合与出现集合求交。MP3D缺像素证据时，此项由原来的程序生成通过改为deferred，
   因而其比例从6/24修正为5/24。

回归：4415 passed、117 skipped、2 deselected、52 subtests passed。
日志：`tmp/qa_family_validation_20260906/fast_suite_family_fix_v1.log`。

## 哪些需要双模态

详细24项表见 [当前题面的模态依赖](QA_MODALITY_DEPENDENCY_20260906.md)。

当前按设计分为13类AV核心候选、3类条件性、3类音频对照、5类视觉对照。
A/V均默认获得同一题面；这里的双模态专指音频和视频，双耳左右声道仍是同一个音频模态。
本轮没有完成各类型×MCQ/Open的双模态必要性认证。

若只统计13类AV核心候选的生成情况，Apartment为11/13、Kujiale为8/13、
MP3D为2/13、HM3D为0/13、自建房间为43/52。
这些仍只是候选生成比例，不能称为“必须双模态样本通过率”。

原74题批次还有明显的分布限制：QA-01四个答案全为yes，QA-22四例全员发声，
QA-18四例中两例答案为无人发声。必须补充适当的负例、候选干扰和逐形式缺失模态实验，
才能声称模态不可缺少。

## 权威输出

均相对服务器 `/data/jzy/tmp/wt-multi-home-activity-integration`：

- 全家族逐项结果：`tmp/qa_family_validation_20260906/all_families_common_v1/summary.json`，各Episode有独立question JSON。
- Kujiale完整来源、原生媒体/事件映射、Astra审阅：`tmp/qa_family_validation_20260906/kujiale_reviewed_v2`。
- MP3D来源与读回：`tmp/qa_family_validation_20260906/mp3d_validation_summary_v1.json`。
- HM3D缺口：`tmp/qa_family_validation_20260906/hm3d_validation_summary_v1.json` 和 `hm3d_media_clock_readback_v1.json`。
- 模态依赖机器表：`tmp/qa_family_validation_20260906/qa_modality_dependency_v1.json`。
- 原74题分布核对：`tmp/qa_family_validation_20260906/modality_sample_distribution_v1.json`。

本次汇总在上述提交的工作树代码上运行、随后提交；没有伪称旧媒体由新提交渲染。
不修改或覆盖原74题最终导出，不推送、不切换运行中Studio。
