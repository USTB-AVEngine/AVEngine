# 引擎能力差距审计:题型 v3 的自动化需求 vs 现有能力(20260831)

> **目的(owner 指令)**:在 pilot 工单动工前,审一遍现在的引擎是否满足
> 全案 v3 全部题目的自动化出题需求。**方法**:把全案与工单的需求逐项
> 对到仓库现有能力上,每个判定带证据路径,当场查证不靠记忆。
> **判定分级**:✅ 已有(证据在案)/ 🔧 需扩展(在现有结构上加配方或
> 参数,写明改动点与量级)/ ❌ 缺失(全新建,工单已列的标注)/
> ⚠ 结构点(涉及政策或硬合同,要 owner 知情拍板)。
> 本文档供 codex 独立复核(复核清单见第五节)。

## 0. 结论速览(R2,按 owner/codex 复核修正)

**双源五题(①⑦⑧⑨⑯)范围内无硬缺口**,可先开 pilot 工程单:所需的
全部是"新配方、新工具、参数化扩展"。前提是完成工单阶段一的五个纯新建
件(判分器、干扰项生成器、物理分类器、双声道校验器、人类作答页面)和
两个扩展(静→走单次转折轨迹、像素掩码通道按五条件接入批量线)。

**R1 的"pilot 六题无硬缺口"结论已修正**:AudioProgram 的 schema 层确实
支持 N 端点,但 **production 执行链大量硬编码 source1/source2**——
runtime binding、Apartment visual bundle、current visual capture、
RIR cache、semantic authority、full-episode validation、target-only
mask 工具七处均有二源限制。因此 **⑪以及⑫⑬⑭⑮a 不是只改挖矿器**,
需要独立的 **N-actor/N-source research 执行路径 + 4 源 canary** 作为
前置切片;⑪从首轮 pilot 移出,待 N 源路径通过后加入。

**停走轨迹拆分**:现有能力只支持全静、全走、单次延迟起步;首轮 ⑤R/⑥R
限定 **静→走(idle→walk)单次转折**;通用 walk/idle 多段切换属于中量级
timeline + 动画 QA 扩展,不入首轮。

**要 owner 知情的结构点**:①像素掩码与 npy 退休的顺序调整——**owner 已
批准,附五条件**(见第三节);②75 帧硬合同;③双源假设(已升级为上述
执行链七处,非仅挖矿器)。

**确认不够、已正确押后的**:多段式采样(⑰)、家具有向包围盒(老 T10)、
通用多段停走、多源执行路径(前置切片另立)。

## 1. 引擎核心能力盘点(逐项证据)

| 能力 | 证据 | 状态 |
| --- | --- | --- |
| **音频程序=事件表结构**:每事件带起止 tick/采样点、源端点、素材引用、增益/淡入淡出;发声时刻、事件数、次序全是数据不是代码 | `examples/dataset/current_apartment/audio_programs/qa_v2/*.json`(样例即 4 事件);字段 `start_tick/start_sample/source_endpoint_id/sound_asset_id/linear_gain/fade_samples` | ✅ |
| **源端点数**:schema 层无上限(只约束 `minItems: 1`);但 **production 执行链七处硬编码双源**:runtime binding、Apartment visual bundle、current visual capture、RIR cache、semantic authority、full-episode validation、target-only mask 工具(owner/codex 复核补充) | `schemas/m6_audio_program_v1.schema.json`;执行链各件 | ⚠ schema ✅ / 执行链双源限制 |
| **轨迹=authored timeline 逐帧位置**,任意路径可写;轨迹重组库在产 | batch2d `timeline.json`(75 帧逐帧 actor_states);`tools/dataset/recombine_source_trajectory_bank.py` | ✅ |
| **闸门 A 音频孪生**:route-swap 孪生渲染与逐题认证管线 | `tools/qa/render_axis1_twin_audio.py`、`certify_axis1_questions.py`、`src/avengine/qa/certify.py` | ✅(batch2d 实证) |
| **闸门 B 属性孪生设计器**:含孪生配额、次序变体(afirst/bfirst)、设计式速度(5 秒内 3.0–3.8 米)、**画外配额先例**("1 off-screen") | `tools/qa/design_qa_batch.py`(128 主点+64 属性孪生;闸门 B 320/320 已实证) | ✅ |
| **像素级可见性/遮挡真值**:同渲染器同相机的 modal/target-only 语义掩码,四态判定+五类 canary 全过 | `src/avengine/qa/pixel_visibility.py`;`tmp/lead_a_question_protocol_paper_ready_v3/report.md` | ✅(native 线已验) |
| **动态听者(相机运动)**:相机平移配方与运动真实感审计存在——**定性修正(owner):非正式 mechanism canary,不是正式批量准入**,批量启用前须走准入 | `tools/qa/build_strict_two_human_camera_pan_v2_candidate.py`、`audit_strict_two_human_camera_pan_motion_realism.py` | 🔧 canary 级 |
| **干扰实例移动**:配方存在——同上,**非正式 mechanism canary** | `tools/qa/build_strict_two_human_distractor_moves_v2_candidate.py` | 🔧 canary 级 |
| **受控实例资产(口径按 owner 修正)**:**犬 5 个品种 / 6 个 dog asset**(shiba_inu 2 个个体)、猫 2 种、人物 6 个,**全部 research 状态**;受控衣色 3 种;受控外观变体生成管线(FLUX/换色)在产 | `examples/runtime/source_asset_runtime_profiles.json` 计数 | ✅ research 级(第四衣色=素材需求单已发) |
| **纯文本闸门 C、三条件评测管线** | batch2d 909 题四条件实测在案 | ✅(双声道校验待建,见下) |
| **时刻精确性**:tick/采样点/帧三套坐标齐全,75 帧×5 秒映射固定 | program 字段+capture 合同 | ✅ |

## 2. 逐需求判定表

### 工单阶段一(基础设施)

| 需求 | 判定 | 改动点与证据 | 量级 |
| --- | --- | --- | --- |
| 双声道通道校验器 | ❌ 新建(工单 1.1) | 纯校验脚本,无引擎改动 | 小 |
| 发声时刻随机化、每集≥3 声 | 🔧 | 生成新 program 集(事件表数据级;现 qa_v2 程序集是固定 turn-taking,4 事件先例在) | 小 |
| 停走转折轨迹(R2 拆分,owner 定):**首轮只用静→走单次转折**(现有"单次延迟起步"能力的直接使用);**通用 walk/idle 多段切换不入首轮**——属中量级 timeline+动画 QA 扩展(动画状态切换的视觉质检要跟上) | 🔧(首轮)/押后(通用) | 首轮:设计器 motion 分支加 idle→walk 案例;⑤R/⑥R 的"没动"侧用全静点位配平答案 | 首轮小;通用中 |
| 切分隔离器(六维) | ❌ 新建(工单 1.3) | 纯清单工具;孪生同侧规则已有先例(batch2d seed 切分) | 小 |
| MCQ 干扰项生成器 | ❌ 新建(工单 1.4) | 读事实记录+2.7 规则,无引擎改动 | 中 |
| 开放判分器 | ❌ 新建(工单 1.5) | 纯判分代码 | 中 |
| 物理特征分类器探针 | ❌ 新建(工单 1.6) | 只读最终 wav,无引擎改动 | 中 |
| 错时采样器(六条件生成侧检查) | 🔧 | 组合既有件:program 事件表(锚后静默=事件排布)+timeline(位移/停走)+pixel_visibility(提问帧可观察)+条件均衡逻辑(新写);①专用角距/位移门槛是采样过滤器 | 中 |

### 逐题特有需求

| 题 | 特有需求 | 判定 | 说明 |
| --- | --- | --- | --- |
| ① 错时方位 | 锚后静默+片尾角距>2×宽带+锚后位移>满分带;Gate B=轨迹对调孪生 | 🔧 | 采样过滤器+轨迹孪生配方(timeline 数据级对调,渲染两版视觉——与属性孪生同渲染成本量级);侧向方位补桶=采样参数 |
| ② 发声时刻方位 | 同① 去错时约束 | ✅/🔧 | 同上的子集 |
| ⑦ 第 t 秒谁在叫 | 恰一只在叫子集+方位分离+素材外观独立分配 | 🔧 | program 排布+采样过滤;"发声无视觉姿态"资产事实已由 ADR-0005 制度化(pilot 复核项) |
| ⑧ 首叫时刻 | 两只首叫不同带且间隔>宽带;同干声/跨外观反平衡 | 🔧 | program 排布+素材分配矩阵(数据级) |
| ⑨ 先发声者外观 | 修复后重挖重测 | ✅ | 在产挖矿器+新 program 集 |
| ⑪ 谁发声/都不是 | 三只可见+一画外;matched-DoA 成对分布 | 🔧 **N 源前置**(R2 修正:移出首轮) | **受执行链双源硬编码限制(⚠3 七处),不是只改挖矿器**——须先建 N-actor/N-source research 执行路径并通过 4 源 canary;其余要素(画外 `offscreen_candidate` 字段预留、off-screen 配额先例、matched-DoA 采样过滤器、像素掩码接入)判定不变 |
| ⑫ 声音种类 | ≥4 发声实体多源编排;目标唯一声种 | 🔧 **N 源前置** | 同⑪走 N 源路径+4 源 canary;四声种素材数=素材侧盘点项 |
| ⑬⑭ 说话内容 | 四人四色每人一句;≤5s 装配 | 🔧 **N 源前置** | 四人=4 actor,同受执行链双源限制;第四衣色=素材项;装配规则=采样过滤器;语音素材挂 endpoint 机制✅(QS-012 native 先例,但那是双人形态) |
| ⑮a 计数 | 三实例场景+同在场数选项 | 🔧 **N 源前置** | 三实例即 3 actor,同上;选项构造=干扰项生成器规则 |
| ⑯ 两跳遮挡 | 片尾遮挡状态受控轨迹;Gate B=遮挡状态孪生 | 🔧 | pixel_visibility 四态✅(native 五 canary 含全遮/重现);"轨迹终点落在遮挡区"的受控采样=新配方(以既有 full_occlusion episode 配方为底,`build_native_full_occlusion_reappearance_episode.py` 先例);Gate B 遮挡孪生=轨迹微调重渲染 |
| ⑰ 跨段记忆 | 多段式采样 | ❌(已押后) | 75 帧合同外的新形态,维持 future extension |
| 人类作答页面 | 校准+红线实验 | ❌ 新建 | 小(网页) |

### 认证阶段

| 需求 | 判定 | 说明 |
| --- | --- | --- |
| 四输入条件渲染(全/黑帧/静音/纯文本) | ✅ | batch2d 三条件+text_only 先例;黑帧/静音是媒体处理 |
| 探针模型三层 | 🔧/❌ | SO 系与 Qwen 管线在;物理分类器新建;全部前置双声道校验 |
| 聚类 bootstrap/Holm/Monte Carlo | ❌ 新建 | 统计脚本,无引擎改动 |
| oracle 推导链重放器 | 🔧 | 事实来源(program/timeline/掩码)全为签发记录,重放器=读记录复算(新工具,证据链已齐) |

## 3. 三个要 owner 知情的结构点

1. **⚠→已批 像素掩码通道 × npy 退休政策(owner 20260831 批准,附五
   条件,执行必须全部满足)**:顺序方案 = 批量渲染加掩码输出 → 出题与
   可见性判定消费 → 之后退休。五条件:
   ① **失败不退休**——任何出题/判定/闸门环节失败的点位,原始产物保留
   待查,不得进退休流程;
   ② **正式评测与论文证据的批次保留无损 masks/depth/truth/overlay**,
   不适用退休;
   ③ **研究批退休前必须验证压缩证据可重编译**(用压缩后的留存物能重算
   出同样的判定,验证通过才许退休);
   ④ **先实测 N+1 渲染通道的峰值存储**(掩码通道叠加期的磁盘峰值),
   实测过关才批量开跑;
   ⑤ **退休工具的检查清单扩为:pixel truth、oracle 重放物、overlay、
   manifest 齐全性——不能只检查 MP4 与 questions**。
2. **⚠ 75 帧/5 秒硬合同**:`src/avengine/capture/two_human_capture.py`
   等多处断言 "exactly 75 frames"(formal 合同;review 片可任意长)。
   ⑬⑭的"四句+间隔 ≤5 秒"装配规则、⑧的时间带设计都已适配该合同;
   **任何更长片段/多段需求(⑰)= 合同级架构决策**,押后正确,不建议
   为 v3 动它。
3. **⚠ 双源假设的真实范围(R2 修正——R1 低估为"挖矿器层")**:除
   `generate_qa_v2_questions.py` 的 `ep_to_slot` 与 `design_qa_batch.py`
   外,**production 执行链七处硬编码 source1/source2**:runtime
   binding、Apartment visual bundle、current visual capture、RIR
   cache、semantic authority、full-episode validation、target-only
   mask 工具。处置(owner 定):**另立"N-actor/N-source research
   执行路径"前置切片**,以 **4 源 canary** 为验收(一个 4 源点位全链
   跑通:绑定→渲染→RIR→语义权威→验证→掩码→出题),通过后 ⑪⑫⑬⑭⑮a
   才排产;改造带回归(旧双源批次重跑一致)。首轮 pilot 收缩为双源
   五题 ①⑦⑧⑨⑯。

## 4. 总回答:引擎能力够不够?(R2 定稿)

**双源五题(①⑦⑧⑨⑯)——够,pilot 工程单可先开工**:这五题在引擎层
(渲染/声学/timeline/程序/像素真值)与执行链上均无硬缺口;差距落在
工具层(采样过滤器、新建判分/探针件)与素材层,工作量已被 pilot 工单
与素材需求单覆盖。轨迹按"静→走单次转折"限定,掩码顺序按已批五条件
执行。

**多源族(⑪⑫⑬⑭⑮a)——现在不够**:production 执行链七处双源硬编码
(第 3 节⚠3),须先完成 **N-actor/N-source research 执行路径 + 4 源
canary** 前置切片,通过后再排产。这是执行链改造,不是引擎架构改动,
但量级为中,单列切片不与首轮 pilot 混流。

**确认不够、维持押后**:多段式(⑰)、家具锚(老 T10)、片长扩展、
通用多段停走。

## 5. 复核记录(owner/codex 20260831,结论已合入本 R2)

复核发现并已修正:①"pilot 六题无硬缺口"结论过大——执行链七处双源
硬编码使⑪及多源族需独立前置切片(→⚠3、§0、§4);②停走轨迹拆分,
首轮限定静→走单次转折;③掩码顺序获批附五条件(→⚠1);④资产口径
(犬 5 品种/6 个 dog asset,全部 research)与 camera_pan/干扰移动的
"非正式 mechanism canary"定性(→§1)。

后续执行侧复核随 pilot 工单进行(4 源 canary 验收、N+1 通道存储实测、
退休工具检查清单落地)。

> 状态声明:research planning;R1 判定基于仓库 main 88e6c7a,R2 合入
> owner/codex 复核修正;本文档不改任何代码,差距项并入 pilot 工单与
> N 源前置切片执行。
