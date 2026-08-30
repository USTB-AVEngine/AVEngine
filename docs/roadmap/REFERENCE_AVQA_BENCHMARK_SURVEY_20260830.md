# 音视频多模态声源定位 QA Benchmark 调研报告

## 核心AVQA数据集

### 1\.1 MUSIC\-AVQA \(CVPR 2022\)

**论文**: Learning to Answer Questions in Dynamic Audio\-Visual Scenarios

**PDF**: `MUSIC-AVQA_CVPR2022.pdf`

**研究问题**:

- 如何在动态音视频场景中进行问答，需要综合多模态理解和时空推理

**数据集规模**:

- 9,288个视频（7,423真实 \+ 1,867合成）

- 45,867个QA对

- 覆盖22种乐器，分为弦乐、管乐、打击乐、键盘四类

**QA设计方法**:

**关键设计思想**:

- 大部分QA对需要同时关联音频和视觉模态

- 问题模板覆盖33种不同类型，跨越9种问答场景

- 空间定位问题强制模型建立声音与视觉位置的对应关系

**迁移到动态声源定位的QA设计**:



---

### 1\.2 AVQA \(ACM MM 2022\)

**论文**: AVQA: A Dataset for Audio\-Visual Question Answering on Videos

**PDF**: `AVQA_ACM_MM2022.pdf`

**研究问题**:

- 现有数据集要么只考虑视觉线索，要么局限于特定场景（如全景视频、音乐表演），缺乏真实生活场景的音视频问答数据集

**数据集规模**:

- 57,015个真实生活视频

- 57,335个QA对

- 总时长超过158小时

- 来源于VGGSound数据集，覆盖165个日常活动和自然声音类别

**QA设计方法（具体示例）**:

**迁移到动态声源定位的QA设计**:

---

### 1\.3 Pano\-AVQA \(ICCV 2022\)

**论文**: Pano\-AVQA: Grounded Audio\-Visual Question Answering on 360° Videos

**PDF**: `Pano-AVQA_ICCV2022.pdf`

**研究问题**:

- 360°全景视频提供了超越普通视野的音视频线索，但现有benchmark无法评估对球面空间关系和音视频关系的语义理解

**数据集规模**:

- 5\.4K个360°视频片段

- 两类问答对（带边界框标注）

**QA设计方法（具体示例）**:

**关键创新**:

- 引入球面空间嵌入\(Spherical Spatial Embeddings\)

- 问题设计考虑了全景视频的独特属性

- 所有QA都带有边界框grounding标注

**迁移到动态声源定位的QA设计**:

---

## 防止单模态作弊的Benchmark设计

### 2\.1 DAVE \(NeurIPS 2025\)

**论文**: DAVE: Diagnostic benchmark for Audio Visual Evaluation

**PDF**: `DAVE_NeurIPS2025.pdf`

**研究问题**:

- 现有benchmark存在严重的视觉偏见\(visual bias\)——答案可以仅从视觉数据推断

- 现有评估只提供聚合分数，无法区分模型在视觉理解、音频解释、音视频对齐哪个环节出错

**核心设计原则**:

> **每个问题都需要同时使用音频和视觉两种模态才能正确回答，单一模态不足以作答**
> 
> 

**QA设计方法（具体示例）**:

**评估解耦设计**:

- 将评估分解为原子子类别

- 分别测试：时间对齐能力、跨模态整合能力

- 避免聚合分数掩盖具体问题

**迁移到动态声源定位的QA设计**:

---

### 2\.2 FortisAVQA \(NeurIPS 2024 / arXiv 2024\)

**论文**: FortisAVQA and MAVEN: a Benchmark Dataset and Debiasing Framework for Robust Multimodal Reasoning

**PDF**: `FortisAVQA_2024.pdf`

**研究问题**:

- 现有AVQA方法过拟合数据集偏见，导致鲁棒性差

- 模型可能通过捷径\(shortcuts\)而非真正理解来回答问题

**QA设计方法（具体示例）**:

**第一阶段：问题改写扩展多样性**

**第二阶段：分布偏移测试**

**MAVEN去偏框架核心思想**:

- 训练时同时有单模态分支和多模态分支

- 优化目标：让多模态预测与单模态预测的分布差异最大化

- 这样模型就不能仅靠单模态"抄近路"

**迁移到动态声源定位的QA设计**:

---

### 2\.3 Unveiling Visual Biases \(arXiv 2024\)

**论文**: Unveiling Visual Biases in Audio\-Visual Localization Benchmarks

**PDF**: `VisualBiases_AVL_2024.pdf`

**研究问题**:

- 发声物体通常仅凭视觉线索就能被识别（视觉偏见）

- 这导致benchmark无法有效评估AVSL模型的真正能力

**关键发现**:

- 在VGG\-SS和Epic\-Sounding\-Object等benchmark上，纯视觉模型可以超越音视频基线

- 说明这些benchmark需要改进

**QA/评估设计（具体示例）**:

**扩展评估设计**:

**迁移到动态声源定位的QA设计**:

---

### 2\.4 AURA \(arXiv 2025\)

**论文**: AURA: A Fine\-Grained Benchmark and Decomposed Metric for Audio\-Visual Reasoning

**PDF**: `AURA_2025.pdf`

**研究问题**:

- 现有音视频benchmark只关注最终答案准确率，忽略了推理过程

- 难以区分真正理解与通过错误推理或幻觉得到的正确答案

**六大认知领域（具体QA示例）**:

**AuraScore评估指标设计**:

**迁移到动态声源定位的QA设计**:

---

## 空间声源定位相关Benchmark

### 3\.1 SpatialSoundQA / BAT \(arXiv 2024\)

**论文**: BAT: Learning to Reason about Spatial Sounds with Large Language Models

**PDF**: `SpatialSoundQA_BAT_2024.pdf`

**研究问题**:

- 首个大规模空间音频问答benchmark

- 评估LLM对3D空间声音的推理能力

**数据集规模**:

- 21,000\+模拟双耳音频片段

- 在3D环境中渲染

**QA设计方法（具体示例）**:

**空间参数设计**:

- 方位角\(Azimuth\): 0°\-360°，分为8个方向区域

- 仰角\(Elevation\): \-40°到\+40°

- 距离: 0\.5m到10m，分为近/中/远

**迁移到动态声源定位的QA设计**:

---

### 3\.2 SAVVY \(arXiv 2024\)

**论文**: SAVVY: Spatial Awareness via Audio\-Visual LLMs through Seeing and Hearing

**PDF**: `SAVVY_2024.pdf`

**研究问题**:

- 首个针对动态场景中3D空间推理的音视频benchmark

- 现有AV\-LLM主要关注静态或2D场景

**SAVVY\-Bench特点**:

- 数千个精心设计的QA对

- 测试方向性和距离关系

- **包含静态和运动物体**（与我们的任务高度相关）

**QA设计方法（具体示例）**:

**关键设计：运动物体的空间推理**:

**迁移到动态声源定位的QA设计**:

---

### 3\.3 3D Audio\-Visual Segmentation \(arXiv 2024\)

**论文**: 3D Audio\-Visual Segmentation

**PDF**: `3D-AVS_2024.pdf`

**研究问题**:

- 首次探索3D音视频分割

- 训练具身智能体利用空间音频线索进行细粒度声源定位

**数据集**: 3DAVS\-S34\-O7

- 34个场景，7类物体

- 提供3D分割掩码标注

**任务设计（具体示例）**:

**关键技术**:

- 智能体配备摄像头和双耳麦克风

- 利用双耳音频的ITD\(时间差\)和ILD\(强度差\)进行定位

- 结合深度信息进行3D定位

**迁移到动态声源定位的QA设计**:

---

### 3\.4 STARSS23 \(NeurIPS 2023\)

**论文**: STARSS23: An Audio\-Visual Dataset of Spatial Recordings of Real Scenes

**PDF**: `STARSS23_NeurIPS2023.pdf`

**研究问题**:

- 提供真实空间录音的音视频数据集

- 包含方向到达\(DOA\)和声源距离信息

**数据集特点**:

- 多通道音频\(麦克风阵列录制\)

- 同步360°视频

- 时空标注的声音事件

**标注内容（具体示例）**:

**相比STARSS22的改进**:

- 增加4小时素材

- 所有音频录音都有同步360°视频

- 标签包含声源距离信息（不仅是方向）

**迁移到动态声源定位的QA设计**:

---

## 幻觉检测与鲁棒性评估

### 4\.1 AVHBench \(ICLR 2025\)

**论文**: AVHBench: A Cross\-Modal Hallucination Benchmark for Audio\-Visual Large Language Models

**PDF**: `AVHBench_ICLR2025.pdf`

**研究问题**:

- 首个专门评估音视频LLM感知和理解能力的综合benchmark

- 音视频LLM难以辨别音频和视觉信号之间的微妙关系，导致幻觉

**数据集规模**:

- 2,136个视频

- 5,302个QnA对

- 1,106个音视频描述

**四大任务设计（具体示例）**:

**幻觉类型分析**:

**迁移到动态声源定位的QA设计**:

---

### 4\.2 AV\-Odyssey \(arXiv 2024\)

**论文**: AV\-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio\-Visual Information?

**PDF**: `AV-Odyssey_2024.pdf`

**研究问题**:

- 首个全面评估MLLM是否真正理解音视频信息的benchmark

- 发现MLLM在简单任务上也挣扎（如判断哪个声音更响、哪个音高更高）

**DeafTest设计（具体示例）**:

**数据集设计原则**:

- 每个问题都包含**文本、视觉、音频**三个组件

- 必须有效利用视觉和音频输入的线索才能推断答案

- 采用**选择题形式**，确保精确客观评估

**具体QA示例**:

**迁移到动态声源定位的QA设计**:

---

## 持续学习与长视频理解

### 5\.1 AVQACL \(CVPR 2025\)

**论文**: AVQACL: A Novel Benchmark for Audio\-Visual Question Answering Continual Learning

**PDF**: `AVQACL_CVPR2025.pdf`

**研究问题**:

- 在持续学习设置下研究细粒度场景理解和时空推理

**QA设计（具体示例）**:

**持续学习设置**:

- 模型依次学习不同类别的声音

- 测试模型是否能在学习新类别时保持对旧类别的记忆

- 120个声音类别逐步引入

**迁移到动态声源定位的QA设计**:

---

### 5\.2 TraceAV\-Bench \(arXiv 2025\)

**论文**: TraceAV\-Bench: Benchmarking Multi\-Hop Trajectory Reasoning over Long Audio\-Visual Videos

**PDF**: `TraceAV-Bench_2025.pdf`

**研究问题**:

- 首个同时需要多跳轨迹推理和评估多模态幻觉鲁棒性的benchmark

**数据集规模**:

- 578个长视频

- 总计339\.5小时

- 跨越多种类型和语言

**QA设计（具体示例）**:

**幻觉鲁棒性测试**:

**迁移到动态声源定位的QA设计**:

---

### 5\.3 OmniVideoBench \(arXiv 2024\)

**论文**: OmniVideoBench: Towards Audio\-Visual Understanding Evaluation for Omni MLLMs

**PDF**: `OmniVideoBench_2024.pdf`

**研究问题**:

- 评估全模态MLLM的协同音视频理解和推理能力

**数据集规模**:

- 1,000个高质量QA对

- 每个都有逐步推理轨迹标注

- 628个视频（几秒到30分钟）

- 8大类别，68个子类别

**质量保证流程（防止单模态可答）**:

**QA设计（具体示例）**:

**关键发现**:

- 禁用音频后，Gemini\-2\.0\-Flash性能暴跌至接近随机水平

- 说明问题设计成功——仅视觉线索不足以作答

**迁移到动态声源定位的QA设计**:



---

## QA设计方法总结与迁移建议

### 6\.1 核心设计原则

### 6\.2 防止"蒙"答案的具体策略

### 6\.3 针对动态声源定位的QA类型设计

#### 基础定位类

#### 动态追踪类

#### 多人场景类

#### 音视频对齐类

#### 反事实/鲁棒性类

### 6\.4 QA设计流程

```Plain Text
1. 确定测试能力维度（方向/距离/轨迹/多人等）
      ↓
2. 设计原始问题（开放式+选择式混合）
      ↓
3. 单模态可答性检测
   ├── 纯视觉模型能否答对？ → 能 → 删除或修改
   └── 纯音频模型能否答对？ → 能 → 删除或修改
      ↓
4. 添加负样本和干扰项
   ├── 静音样本（画面有人但没声音）
   ├── 屏幕外声源（有声音但人不在画面）
   └── 多人场景（多人在场但只有一人说话）
      ↓
5. 问题改写增强多样性
   ├── 同义改写："在哪" → "什么位置" → "哪个方向"
   └── 难度变体：方向→角度→坐标
      ↓
6. 人工验证双模态必需性
      ↓
7. 添加推理轨迹标注（可选，用于细粒度评估）
```

### 6\.5 选择题设计建议（防止随机猜对）

---

## 附录：参考文献



