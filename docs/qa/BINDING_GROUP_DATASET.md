# Binding group dataset

## 2026-09-10：snapshot0029 最新核实里程碑

权威快照 tmp/binding_dataset_20260909_v2/completed_groups_snapshot_0029.json
当前为 status=partial_delivery、29 个 unique groups、116 个 core members，
目标仍为64组。按 room family × task family 计数如下；这里的 identity 对应
cross_event_identity，state 对应 cross_time_state，visible 对应
visible_binding，ref 对应 visual_conditioned_relation：

| 房间 | identity / state / visible / ref |
|---|---:|
| HM3D | 4 / 2 / 4 / 3 |
| MP3D | 1 / 1 / 1 / 2 |
| Kujiale | 1 / 1 / 3 / 2 |
| Apartment | 1 / 1 / 1 / 1 |
| 合计 | 7 / 5 / 9 / 8 = 29 |

完整 catalog 当前由 full_catalog_initial16_v1/catalog_index.json 与
full_catalog_expansion_batch_0001～0004/catalog_index.json 分批组成：
29 groups、116 AV/core members、1965 条 catalog items；形式计数分别为
Open 1894、MCQ 1475。最新 batch 为 batch0004，其中 QA-21 新增4条。
这些是 research_candidate/media-checked 计数；native 队列仍在推进，模型评测、
人工可答性和正式准入仍为 not_run。

状态：research_candidate；当前正在进行 integration/native generation。
本文说明分组数据结构、介质检查和评分入口，不代表 native 批次已经完成，
也不代表模型、人工校准或正式数据准入已经完成。
相机变体试验产物已排除，不作为当前 producer 或数据覆盖依据。

完整实施方案（含用户批准方案原文及后续补充约束）见
[BINDING_GROUP_IMPLEMENTATION_PLAN.md](BINDING_GROUP_IMPLEMENTATION_PLAN.md)。

## 目标和边界

绑定组把一个目标问题放进一组受控的音视频变体中，用来观察视觉实例、
声音事件和跨时间状态是否保持正确对应。每个 member 只有一个目标
Question；组内 comparisons 明确两个样本预期答案是 same 还是 different，
以及哪一种模态应保持共享。

这条路径保留统一目录中的 QA-01～QA-25。它给已有题型增加实例绑定的
变体，不删除题型，不重排 QA-25，也不把角度附问自动加入新组。一条显式
member 只计一次；QA-25 及既有 angle_followups 仍由原统一生成器和评分
入口处理。

当前四个任务族都保留，均沿用已有 QA 编号。snapshot0029 当前核实 29 个
unique、media-checked pilot groups（116 个核心成员）：visible_binding 9 组、
visual_conditioned_relation 8 组、cross_time_state 5 组、cross_event_identity
7 组，四任务×四房间矩阵为 29/64。模型、人工校准和正式准入仍未完成：

| 任务族 | 对应旧 QA | 变体含义 |
|---|---|---|
| visible_binding | QA-20 | 从可见候选中回答产生目标声音的外观 |
| visual_conditioned_relation | QA-05 | 先用视觉选定实例对，再用实测发声活动回答关系 |
| cross_event_identity | QA-20 | 跨独立声音事件追踪同一个物理实例 |
| cross_time_state | QA-13 | 预留新变体：从声音锚点追踪到后续时刻，回答真实可见质心角度 |

当前四任务×四房间 room-cell quota 如下；`missing` 表示尚未有 accepted
media-checked group：

| 任务族 | 已完成 room cells | 当前 groups | missing room cells |
|---|---|---:|---|
| visible_binding | Apartment、Kujiale、HM3D、MP3D | 9 | 无 |
| visual_conditioned_relation | Apartment、Kujiale、HM3D、MP3D | 8 | 无 |
| cross_time_state | Apartment、Kujiale、HM3D、MP3D | 5 | 无 |
| cross_event_identity | Apartment、Kujiale、HM3D、MP3D | 7 | 无 |

三种 source family 仍是全局覆盖要求：articulated_human、articulated_animal、
rigid_static_object。新增 HM3D identity group 只增加 articulated_human ×
articulated_human；它不填补 identity 任务的 nonhuman source-family 缺口。

cross_event_identity 和 cross_time_state 是本轮定义的语义变体；它们复用旧
QA 的答案评分类型，不创建新的 QA 编号。snapshot0029 中四个任务族均已覆盖
四个房间：cross_event_identity 为7组、cross_time_state 为5组；visible_binding
为9组、visual_conditioned_relation 为8组。它们共同构成29/64组的当前
research_candidate/media-checked 里程碑，仍不能代替最终64组目标。

支持的三种源类别全部保留：articulated_human、
articulated_animal、rigid_static_object。当前组合覆盖人-人、人-动物、
人-静态物、动物-动物、动物-静态物、静态物-静态物。具体样本是否能
生成，仍由 native facts、注册表和实际介质决定。

绑定问题的 required_modalities 是 audio 与 video。每个 Episode 的相机位置、
朝向和 FOV 必须在整个 Episode 内固定；当前样本采用 10 秒和配置中的
reserve tail 3 秒，但时长、FOV、tail 和其他 request 字段都由请求驱动。
组内可以用 shared_modality=audio 或 shared_modality=video 构造纯 A 或纯 V
控制关系；这些是主 AV 绑定题的诊断组成，不单独构成模态必要性结论。

当前声音输入必须来自最终事件参考峰值池，卷积后增益按请求为 0.5；旧音频
只有在实际输入未改变时才可复用。

## 当前已核实的 pilot evidence

当前权威汇总为
tmp/binding_dataset_20260909_v2/completed_groups_snapshot_0029.json，其中
group_count=29、core_sample_count=116、target_group_count=64，
status=partial_delivery。下方既有 canonical outputs 仍作为具体 pilot evidence
和历史复核入口保留；新 native expansion 的逐 slot 状态以对应的
visible_expansion_native_v1/manifest.json 和各 slot summary/audit 为准。
本节不是最终64组完成声明。

- Apartment：
  `tmp/binding_dataset_20260909_v2/visible_apartment_verified/binding_groups.json`。
  group 为 `visible_binding_apartment_v0_refresh`，world 为
  `apartment_world_0001`，room 为 `legacy_ue_apartment_0000_v1`；
  两条 shared-video 和两条 shared-audio comparison 均声明
  `answer_relation=different`，且 validation 中的 `media_check` 均为
  `pass`。
- Apartment relation：
  'tmp/binding_dataset_20260909_v2/relation_first/apartment/group_v1/assembled_current/binding_groups.json'。
  group 为 'visual_conditioned_relation_apartment_group_v1'，world 为
  'world_apartment_relation_0001'，room 为 'legacy_ue_apartment_0000_v1'；
  query 使用 blue/green、reference_time_s=0、window_s=[4, 6]；
  四成员 audio/video clocks 与 plan/native-readback/acoustic equivalence 均
  pass，四条 shared-video/shared-audio comparison 均为 different 且
  media_check=pass。
- MP3D：
  `tmp/binding_dataset_20260909_v2/relation_mp3d_group_v1/assembled_current/binding_groups.json`。
  group 为 `visual_conditioned_relation_mp3d_group_v1`，world 为
  `world_mp3d_relation_0001`，room 为 `habitat_mp3d_example_17DRP5sb8fy`；
  query 使用 blue/green、`reference_time_s=0`、`window_s=[4, 6]`；
  两条 shared-video 和两条 shared-audio comparison 均为
  `answer_relation=different`，且 `media_check` 均为 `pass`。


- HM3D：
  'tmp/binding_dataset_20260909_v2/relation_first/hm3d/group_v1/assembled_current/binding_groups.json'。
  group 为 'visual_conditioned_relation_hm3d_group_v1'，world 为
  'world_hm3d_relation_0001'，room 为 'hm3d_val_00800_TEEsavR23oF'；
  query 使用 blue/green、reference_time_s=0、window_s=[4, 6]；
  四成员的 native/export audio-video clocks 均为 10 秒、150 帧、15 FPS、
  16 kHz、双声道，四条 shared-video/shared-audio necessity comparison
  均为 different 且 media_check=pass。
- HM3D identity：
  `tmp/binding_dataset_20260909_v2/identity_first/group_v14/assembled/binding_groups.json`。
  group 为 `identity_hm3d_group_v14`，world 为
  `world_hm3d_identity_20260910_0009`，room 为
  `hm3d_val_00800_TEEsavR23oF`；这是此前13-group snapshot中的第13个 unique group，task family
  为 `cross_event_identity`。四个成员的 10 秒/150 帧/15 FPS/16 kHz 双声道
  条件、`source_context_policy=independent_states`、native/export media
  checks 和两列 shared-audio PCM equality 均通过；四条 shared-video/
  shared-audio necessity comparisons 均为 different 且 `media_check=pass`。
  `assembled_joint_v1` 是同一 group 的复核输出，不另计；模型、人工和正式
  准入仍为 `not_run`。
- Kujiale relation：
  `tmp/binding_dataset_20260909_v2/relation_first/kujiale/group_v1/assembled_blue_yellow_v2/binding_groups.json`。
  group 为 `visual_conditioned_relation_kujiale_group_v1`，world 为
  `world_kujiale_relation_0001`，room 为 `kujiale_0020_full_home_v1`；
  query 使用 blue/yellow、`reference_time_s=0`、`window_s=[4, 6]`。
  四成员均有真实 endpoint、source-activity readback 和 selected appearance
  review；native/export clocks 均为 10 秒、150 帧、15 FPS、16 kHz、双声道。
  两条 shared-video 和两条 shared-audio necessity comparison 均为
  `answer_relation=different`，`media_check=pass`。独立 native/media audit
  确认输出视频与所选 native 视频解码帧一致、输出音频与所选 native PCM 样本
  一致，且 public `model_inputs.json` 不含 private facts/actor 字段。
  旧 blue/green 失败现场、`group_spec_blue_yellow.json` 和
  `appearance_evidence_block.json` 继续保留；green 的实际 RGB+mask
  appearance review 仍为 `not_observable`，没有用 registry 标签替代 review。
- 其余 visible_binding pilot 位于
  `tmp/binding_dataset_20260909_v2/visible_first/hm3d/assembled_current/binding_groups.json`、
  `tmp/binding_dataset_20260909_v2/visible_first/kujiale/assembled_current/binding_groups.json`
  和
  `tmp/binding_dataset_20260909_v2/visible_first/mp3d/assembled_current/binding_groups.json`；
  它们与 Apartment visible-binding 组共同构成四房间 visible_binding coverage。
- slots 2–4 的 visible-binding CPU 扩展预检位于
  `tmp/binding_dataset_20260909_v2/visible_expansion_preflight_v1/manifest.json`。
  该 manifest 包含四房间各 3 个 fresh request、v0/v1 plan 和
  plan-equivalence 结果；12/12 计划通过，structural readback 也通过且没有
  与首组或同房间其他 slot 重复的 camera/assets/frame0 roots。它只记录
  `planning_status=cpu_plan_only`，`native_execution`、RLR、media capture
  均为 `not_run`，不新增 binding group/world 或 full-catalog 计数。Apartment、
  HM3D、MP3D 沿用已核实 blue+green，Kujiale 保留已核实 blue+burgundy；
  这些 human repetition slots 不覆盖 nonhuman source-specific endpoint
  geometry gap。group/world ID 和 seed 只用于后续排程标识，不能单独证明
  独立 native world。

独立的 46 retained Episode catalog regression 位于
`tmp/binding_dataset_20260909_v2/catalog_regression_v1/summary.json`，
其运行入口为同目录的 `runner.py`，fresh 请求配置为
`request_config.json`。该次运行处理 46/46 Episode；summary 中
`normalization_and_generation_pass=46`、`audio_clock_pass=46`、
`source_activity_complete=46`、`public_integer_and_no_private_labels_pass=46`，
QA-22 生成 46 项且显式可见计数域为 `[0, 1, 2, 3, 4]`。QA-18 有 45
项通过实际合法区间检查，1 项因
`speaker_appearance_review_missing` deferred。该 regression 没有产生新
介质，`model_evaluation`、`human_answerability` 和正式准入仍为
`not_run`/未声明；它使用实际 retained native readbacks 做软件再生成，
不替代 native rerender 或 binding-group admission。46 条记录中，27 条是当前 Apartment/Kujiale/HM3D/MP3D 四房间（分别 6/7/7/7）；另外 19 条是 authored_a/b/c 历史房间输入，只用于软件回归，不计作当前四房间新数据，其中两个 walk-pair 的 source_classes=unknown|unknown 也不计作 source 覆盖。

完整 QA catalog 与 core binding benchmark 分批计数。历史 full-catalog
snapshot tmp/binding_dataset_20260909_v2/full_catalog_v3/catalog_index.json
包含12个 group、12个 world、48个 AV sample、811条 catalog items，core question
48条，形式计数为 MCQ 615、Open 779；新增 HM3D identity snapshot
full_catalog_identity_hm3d_v1/catalog_index.json 另含1个 group、1个 world、
4个 AV sample、62条 catalog items，core question 4条，形式计数为 MCQ 46、
Open 62。它们是早期/分支快照，不能替代最新分批汇总。

当前完整 catalog 由
tmp/binding_dataset_20260909_v2/full_catalog_initial16_v1/catalog_index.json
和 full_catalog_expansion_batch_0001～0004/catalog_index.json 组成；按
catalog_question_count、core_question_count 和 form_counts 相加为29个
groups、29个 worlds、116个 AV/core members、1965条 catalog items、116条
core questions，形式计数为 MCQ 1475、Open 1894。batch0004 的
generated_by_qa["QA-21"] 为4，表示 QA-21 新增4条 catalog items。
completed_groups_snapshot_0029.json 与该分批 catalog 汇总相互对应，当前
status 仍为 partial_delivery，目标64组，尚有35组待完成。portable v4
仍只包含旧12-group/48-AV/811-item snapshot，不自动吸收新 catalog。模型、
人工校准和正式准入仍为 not_run。

旧 12 个 world 的 joint compensation invariance 复核记录在
`tmp/binding_dataset_20260909_v2/joint_invariance_v1/completed_groups_snapshot_0012.json`。
该记录为 `partial_delivery`，包含 12 个既有 world、48 个既有核心 sample；
其中八个 visible/relation groups 各有两条 joint audio+visual compensation
invariance comparison，实际 decoded video/PCM 与 gold validation 均通过。
这是同一批样本的联合补偿不变性复核，不是 nuisance/background invariance，
不新增 group、sample 或 world，也不改变旧 full_catalog_v3 的 12/48/811 计数；新增 HM3D identity
catalog 使用独立的 full_catalog_identity_hm3d_v1 snapshot。

state-first 的跨时刻组入口为
'tmp/binding_dataset_20260909_v2/state_first/mp3d/run_state_group_v2.py'；
当前四房间各有一个 media-checked group，结果分别位于
'tmp/binding_dataset_20260909_v2/state_first/apartment/group_v3/assembled/binding_groups.json'、
'tmp/binding_dataset_20260909_v2/state_first/hm3d/group_v1/assembled/binding_groups.json'、
'tmp/binding_dataset_20260909_v2/state_first/kujiale/group_v1/assembled/binding_groups.json'
和
'tmp/binding_dataset_20260909_v2/state_first/mp3d/group_v2/visible_centroid_readback_v1/assembled/binding_groups.json'。
它们复用保留的 native pixel masks/depth evidence 重编 fresh truth；
src/avengine/qa/pixel_visibility.py 的 _frame_truth 现在从 modal target
mask 计算 visible_centroid_xy_px，并把 target-only target_centroid_xy_px
保持为独立字段，不能用 target centroid 代替实际 visible centroid。

本轮保存策略是先保存完整、自包含的音视频、题面、真值和最小复核记录，
再考虑清理本轮可再生的大型中间产物；未完成/失败现场、共享资产、他人
进程和数据继续保留。性能与 pipeline 优化（缓存、复用、并行、读写）在
当前数据集计划完成后单独进行，不能用性能试验替代当前真实组交付。

旧 12 个 accepted groups 与 full_catalog_v3 的只读 retention preflight 见
`tmp/binding_dataset_20260909_v2/retention_preflight_v1/README.md`；
完整路径、inode 去重大小、外部引用、自包含缺口和条件性 numeric/RIR 候选见
同目录的 `report.json`。该审计不执行删除或复制；failed/incomplete、旧
camera/shared inputs 和 current-root 外部路径均保留并排除 cleanup 候选。


## 中间便携交付

当前 portable v4 仍保留旧 12-group accepted snapshot，包位于
`tmp/binding_dataset_20260909_v2/binding_delivery_v4/`；它由
`tools/qa/export_binding_delivery.py` 从
`core_dataset_v1/assembled/binding_groups.json` 与
`full_catalog_v3/catalog_index.json` 按 `(group_id, member_id)` 严格连接生成。
包内包含旧 snapshot 的 12 个 group、48 个 core member、48 条 catalog record、811 条
catalog questions（MCQ 615、Open 779）、公共媒体、私有题面和金标、48 份
copied facts，以及最小 native JSON readback。旧的 v1/v2/v3 失败或中断现场
继续保留，不作为当前交付结果。

`manifest.json` 只声明 `intermediate_delivery`，`qualification_claim=false`。
`validation/readback.json` 的独立结果为 `pass`：48/48 facts-only 题集再生，
811/811 题目结构与题面对齐，catalog MCQ 615/615、Open 779/779 私有金标
评分通过，core MCQ 32/32、Open 48/48 通过，媒体 ffprobe 引用
96 core + 96 public、130 个唯一文件通过。`input_facts_metadata_mismatch=48`
仅记录当前 normalizer 对 visibility frame key 的整数/字符串元数据表示差异；
题目 ID、forms、deferred rows、angle followups 和显示题面均逐项相等。
这些是复制包再生、映射和媒体可读性检查，不是模型评测、人工可答性、native
重渲染或正式准入证据。

活动的 facts、questions、video/audio 链接均为包内相对路径并解析在包根内；
`provenance/path_map.json` 保存普通 source-to-delivered 路径映射，
`provenance/external_dependencies.json` 列出 2359 个未复制的 UE、Habitat、
RLR、HRTF、共享声音和生成环境依赖。包不包含可移植的 UE/Habitat/RLR
运行时；外部依赖与原始绝对路径仅作为 provenance/重生成边界记录。

导出入口（目标目录必须不存在，CLI 拒绝覆盖）：

    cd /data/jzy/tmp/wt-grok-pilot46-round2
    PYTHONPATH=src \
    /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
    tools/qa/export_binding_delivery.py \
      --core-bundle tmp/binding_dataset_20260909_v2/core_dataset_v1/assembled/binding_groups.json \
      --catalog-index tmp/binding_dataset_20260909_v2/full_catalog_v3/catalog_index.json \
      --output tmp/binding_dataset_20260909_v2/binding_delivery_run01

独立复核只需把 `binding_delivery_run01` 作为输入调用
`avengine.qa.binding_delivery.validate_binding_delivery`；它不会读取 source
staging，且不会运行 UE、Habitat、RLR、模型或人工流程。

## 题目和证据要求

所有题目从 native Episode facts 生成。生成器要求已解析的声源绑定、真实
source-activity readback、双耳音频和可用的视觉观察；不从事件文件名、
缺失字段或私有 actor ID 猜答案。

visible_binding 的 gold 是目标外观值。它可以随视觉实例绑定改变，但
题面保持可比较。外观值相同不应被当成不同物理实例的身份证明。

visual_conditioned_relation 使用选定的两个视觉外观和一个整数秒参考
时刻；重叠由实际 source-activity intervals 求交，不用程序声事件的名义
时长替代。它要求至少一个可比较的候选对产生不同答案。

cross_event_identity 的预期 gold 比较事件绑定到的持久物理实例。它需要
事件锚点之间连续可观察的身份历史；声音类别相同或外观相同都不能替代
该身份定义。当前 native producer 尚未把该身份变体做成已完成数据路径。

cross_time_state 的预期查询默认是 clip_end 的实际最后帧，公开题面写片尾，
并观察完整 10 秒媒体；显式 integer query 只保留兼容入口。它先检查锚点
之后的实际静默窗口，再从查询帧的真实可见像素质心计算角度。当前四房间
state-first pilot 已完成 media readback，但最终约 64 组交付仍未完成；不能
把旧位置、隐藏几何字段或事件结束时间当作查询位置。

时间和角度的公开题面使用整数秒、整数度。内部 facts 和 gold 可以保留
完整精度；角度仍采用正前方 0、右侧为正、范围 [-180, 180) 的约定。

## 组装入口

权威工作副本是服务器上的：

    /data/jzy/tmp/wt-grok-pilot46-round2

使用当前已经落地的组装 CLI 时，在服务器工作副本执行，并为每次运行选用
不存在的输出目录：

    cd /data/jzy/tmp/wt-grok-pilot46-round2
    PYTHONPATH=src \
    /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
    tools/qa/assemble_binding_groups.py \
      --input path/to/native_group_spec.json \
      --output tmp/binding_dataset_run01 \
      --seed binding-dataset-v1

该 CLI 读取每个变体的 facts 和已选介质，生成题目、对齐组内题面，
检查比较关系，并把原始介质复制或裁切到新的输出树。源 facts、源视频、
源音频保持只读。native producer 的具体字段和实际运行用法以当前 CLI
和实际 native run 为准；当前不把计划字段写成已完成生产证据。

当前可用的 native producer 入口是
tools/dataset/prepare_binding_group_native.py，负责 visible_binding 的实际
capture/audio 输入；它接受 base request、输出根、source asset、room、
group/world、seed、RPC/GPU 和 QA ID 参数，并沿用请求中的固定相机、FOV、
clock、reserve tail 与 gain。视觉条件关系使用
tools/dataset/prepare_visual_conditioned_relation.py；当前 snapshot0029 的
29 个 unique media-checked pilot groups 已按四任务×四房间矩阵核实：
identity/state/visible/ref 分别为 HM3D 4/2/4/3、MP3D 1/1/1/2、
Kujiale 1/1/3/2、Apartment 1/1/1/1。不要用旧 camera-motion 变体补足
缺额；当前 native expansion 仍有剩余 slot 在推进。

组装输出包含：

- binding_groups.json：私有分组结果，含 group/world 身份、成员、题目
  forms、gold、evidence、介质引用、比较关系和验证状态；
- groups/：每组的私有 JSON，便于逐组复核；
- model_inputs.json：公共输入，只含不透明 sample_id、媒体引用和题面的
  model_input forms。

group_id、world_id、member_id、sample_id、媒体文件名和题目序号都是路由或
分组元数据，不应放进模型 prompt。公共输入不包含 group/variant 身份、
facts、gold、私有 actor/event/frame 字段；私有 binding_groups.json 只能
留在评分侧。

组装器要求一组至少有四个成员，并要求每个成员都获得一个共享音频且答案
改变的 necessity witness，以及一个共享视频且答案改变的 necessity
witness。split 由组继承；同一个物理 world 或同一个 native visual
episode 不得跨 train/validation/test split。

实际介质校验以内容行为为准：

- 共享音频比较解码后的 PCM 数组、采样率、形状和有限值；
- 共享视频比较视频流信息，并在文件不同的时候逐帧解码 RGB 比较；
- 共享模态不相等、声明变化但另一模态未变化，或全静音/无效媒体都会
  使组装验证失败。

这些是普通介质和关系校验，不新增 hash、冻结字节 contract 或正式 gate。

## 模型答案评分

评分只接受私有 binding_groups.json 和不透明 sample_id 到答案的映射。
每种形式单独运行；mcq 和 open 不混在同一次分数中：

    cd /data/jzy/tmp/wt-grok-pilot46-round2
    PYTHONPATH=src \
    /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
    tools/qa/score_binding_groups.py \
      --groups tmp/binding_dataset_run01/binding_groups.json \
      --answers path/to/open_answers.json \
      --form open \
      --out tmp/binding_dataset_run01/scores_open.json

MCQ 评分使用 form=mcq 和另一份答案映射；Open 评分使用 form=open。
params 可传现有统一 scorer 的显式参数。输出路径已存在时 CLI 拒绝覆盖。

完整 catalog 的 public-question 评分使用单独入口；它读取
`catalog_index.json` 中每条 record 的 `questions_path`，并使用同一 record
的 `public_question_ids` 与 `iter_unified_items` 顺序建立不透明 ID 到 private
question item 的映射：

    PYTHONPATH=src \
    /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
    tools/qa/score_binding_catalog.py \
      --catalog tmp/binding_dataset_20260909_v2/full_catalog_v3/catalog_index.json \
      --predictions path/to/predictions.json \
      --form open \
      --out tmp/binding_dataset_20260909_v2/full_catalog_v3/scores_open.json

预测文件是 list：

    [
      {"question_id": "opaque-public-id-1", "prediction": "blue"},
      {"question_id": "opaque-public-id-2", "prediction": "yes"}
    ]

`--form mcq` 与 `--form open` 分开计分。requested form 存在且题目状态有效
的所有题构成分母；缺答、解析 invalid 和 abstained 都是零分，缺少该 form
的题只记为 `unavailable_form` 并排除。duplicate、unknown 或 wrong-form ID
直接报错。输出包含整体与 `qa_metrics`，其中 `invalid` 与 `missing` 分开；
它是复用 `score_unified_item` 的 scorer 输出，不运行模型、不训练模型，也
不构成 human-answerability、modality-necessity 或正式准入证据。v3 的
GT scorer fixture 位于
`tmp/binding_dataset_20260909_v2/full_catalog_v3/scorer_fixture_gt_v2/`，
manifest 标记为 `scorer_test`；由 private gold 生成的 MCQ 615/615 和
Open 779/779 仅用于验证 public/private 映射与 CLI，不能当作模型评测。

答案文件形状是：

    {
      "sample_000173": "blue",
      "sample_000294": "yes"
    }

评分器逐 member 复用 score_unified_item。因此普通 item 结果继续遵循
现有 closed-set、count、time、transcript 和 angle 规则；公共题面只给
需要回答的内容，评分侧才读取 forms 中的 gold。

cross_time_state 的角度答案按现有角度 scorer 评分。整度是题面单位，
内部仍可记录圆周误差；角度的正确性和 group all-correct 遵循 scorer
及该组声明的 angle_tolerance_deg，不把 continuous score 的精确值误读
成新的模型指标。

## 分母和关系语义

item_metrics 是普通逐题结果。请求形式存在的 member 进入分母；missing、
invalid 和 abstained 都是零分失败。没有该形式的 member 单列为
unavailable_form，不当作缺答。

group_all_correct 只有在组内每个 member 都有请求形式时才适用。所有
member 都被正确评分且满足该题型的 full-credit 规则才算 all-correct；
任一 missing、invalid、abstained 或错误答案都会使组失败。含
unavailable_form 的组单列并从该指标分母排除。角度成员使用组声明的
angle_tolerance_deg 与 scorer 语义。

relation_metrics 针对显式 comparisons。两端都有请求形式的 comparison
进入分母，missing、invalid 和 abstained 会使关系失败并留在分母；
unavailable_form 单列并排除。关系按 scorer 的语义解析值比较：same 要求
两个解析值相同，different 要求不同。MCQ 比较选项的语义 value，不比较
A/B 位置，所以选项重排不会伪造答案变化。

关系 agreement 与 all-correct 分开。两个答案都可能错，但只要它们的
解析关系符合预期，relation agreement 仍可为真；这不能证明题目答案
正确，也不能证明模型发生了或没有发生某种 flip。输出不报告 flip-rate。

输出中的 gold_selfcheck 只核对 comparisons 的 same/different 声明是否
与私有题目 gold 一致，明确标注为 software_consistency_only。它不是模型
运行、人工可答性、A/V/AV 模态必要性或数据集准入证据。

一个显式 member 是一个计数单元。绑定组评分器不会调用统一题集的迭代器来展开
items 或 angle_followups，所以既有 QA25/角度附问不会在绑定组中被重复
计数。完整 catalog 评分器使用 catalog record 明确导出的 public ID 列表，
按 `iter_unified_items` 顺序映射 private question set；requested form 可用
的全部题进入分母，missing/invalid/abstained 均保留为零分，缺少该 form
的题不进入分母。预测中的 duplicate、unknown 或 wrong-form public ID 会被
拒绝。输出同时给 overall 与按 QA ID 的结果，并分开报告 missing 和 invalid。
若需评估附问，应把它作为明确的单独 member，并在报告中保留其
parent_question_id 关系。

## 后续模型和人工证据
主线研究问题仍是 AV 实例绑定及其模态必要性。QA-01～QA-25 是统一诊断
题型，四个 binding task family 是单独的组构造；两者都不能被一组已验证
的媒体替代，也不能把一个四成员组的结果外推成全题型或全源族覆盖。


本轮不包含模型设计、训练、模型运行或人工校准。未来若执行模型证据，
应在相同 split、相同题面预算和相同 member 集合上分别报告 full AV、
audio-only、video-only 的诊断结果，并把缺答/解析失败单独列出。纯 A/V
关系检查、group-all-correct 和普通 item accuracy 都不能单独替代模态
必要性实验。

当前状态仍是 integration/native generation in progress。snapshot0029 记录
29 个 unique groups、116 个 core members，目标64组；当前完整 catalog 汇总为
1965 条，Open 1894、MCQ 1475，QA-21 最新新增4条。本文不把这批
research_candidate/media-checked 结果写成最终64组、模型评测、人工接受或正式
dataset admission；后续 native run 完成后，继续以最新 catalog index 补充
producer 字段、输出路径、media readback 和分组覆盖结果。
