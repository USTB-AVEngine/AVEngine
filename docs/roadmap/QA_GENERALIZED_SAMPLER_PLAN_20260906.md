# 出题链泛化方案：静止相机下的条件采样与配额出题（2026-09-06，提案，待 Codex 审核）

状态：提案。作者 Claude，请 Codex 逐条审核可行性、指出与现有代码冲突的地方、补上我看不到的运行成本。
基于 `codex/multi-home-activity-integration` 分支 6adab75 的源码，具体位置写在每一节里。

## 0. 三条已经定下的决定和它们的后果

owner 2026-09-06 定了三件事，这份方案以它们为前提，不再讨论：

1. **相机只做静止。**`pan` 和 `follow_group` 这两种相机运动本阶段不用。后果是听者相对方位只随人的移动变化；
   两个说话人在耳朵里的夹角只能靠"人站哪、相机放哪、谁在什么时候说"配出来。这也把 N 段那种靠相机跟随拉开角距的路封了，
   角距必须在规划阶段主动配。
2. **出题的一切选择在合法候选里随机，不写死。**目标人物、锚定事件、查询时刻、说话顺序、发声时刻、机位，全部是从满足条件的
   候选集合里按配额抽，而不是取第一个或者按固定打分挑最好的。
3. **唯一的硬约束是声音性别与角色性别一致。**男角色配男声，女角色配女声。声音池条目已有 `gender` 字段，
   人物资产档案要补一个同名字段。现在资产和池全是男性，加女性资产时按这个字段过滤。

另外两条约束来自今天的实测，不是决定但要落实：人声片段 5 秒以内、可听语音不少于 1.5 秒；干声入池时按语音频带二次裁剪
（现在七条 VCTK 干声语音前都留着约 1 秒的 80 Hz 以下本底，幅度裁剪把它当成了信号）。

## 1. 现在的链条为什么出不了泛化的题

现状（源码）：`build_qa_episode_plan` 里路线抽样（`sample_activity_routes`）、相机选择（`select_question_camera`）、
音频调度（`schedule_audio`）是三个各自拿随机数的过程，彼此不知道对方选了什么。相机只在"请求了 QA-13 且相机模式是 follow_group"
时才给片尾角距超过 64 度的机位加分，其余情况只按"看得见几个人、离得多远"打分。音频调度按 Dirichlet 随机切时间，不看谁在走。
出题时 `generate_unified_questions` 对每类只调一次生成器，生成器返回第一个通过条件的候选；`facts["sampling"]` 钩子从来没人填。

后果（实测，74 题批次）：14 个发声事件里 11 个发生时全场没人动；A、L 两位说话人夹角 8.5 度，C 段有两对只有 2.3 度；
N 段五道核心题其他候选取值与金标相同；QA-01 四道全 yes，QA-22 四道全"出现数等于发声数"；QA-18 四道都问片长中点。

## 2. 方案总览

把"先随便生成、再检查能不能出题"改成"先抽一组条件、再生成满足条件的场景、再按配额出题"。三层各一个改动：

- **规划层**：一个带约束的拒绝采样器替代三个独立随机过程。先从配额抽一份条件画像，再依次抽人物、路线、静止机位、发声时刻，
  每一步都检查条件，不满足就换种子重抽，抽满预算就记录失败原因。
- **出题层**：每个生成器改成枚举全部合法候选，再按配额抽一个或多个；查询时刻从 request 走 `sampling` 钩子进来；
  加四项拒出条件（分歧、可绑定、可听区间、配额）。
- **产物层**：计划、事实、题目三处都记录"想要的条件"和"实际达到的条件"，审计工具只负责核对，不负责补救。

## 3. 请求格式增量（向后兼容）

现有字段都保留；旧请求不带新字段时行为不变，这样历史 Episode 仍可复跑。新字段：

```json
{
  "camera_motion": "static",
  "camera_fov_deg": 85,
  "conditions": {
    "speaker_count": {"choices": [2, 3, 4], "weights": [0.4, 0.3, 0.3]},
    "silent_actor_count": {"choices": [0, 1], "weights": [0.5, 0.5]},
    "separation_bin_deg": {"bins": [[15, 30], [30, 60], [60, 90], [90, 180]], "weights": [0.25, 0.35, 0.3, 0.1], "floor_deg": 15},
    "speech_motion": {"choices": ["speaker_moving", "competitor_moving", "all_still"], "weights": [0.35, 0.35, 0.3]},
    "speaking_source_line_of_sight": true,
    "min_gap_between_events_s": 0.5,
    "reserve_tail_s": 3.0
  },
  "sound_selection": {
    "registry": "/data/avengine_external/assets/sound_event_library_v1_20260903/sound_asset_registry_v1.json",
    "gender_match": true,
    "unique_transcripts_per_episode": true,
    "max_clip_s": 5.0,
    "min_audible_s": 1.5,
    "vad": {"band_hz": [300, 3400], "threshold_db_below_peak": 25, "highpass_hz": 80},
    "speaker_colour_counterbalance": "batch"
  },
  "audio_render": {"linear_gain": 1.0, "ambient_bed_dbfs": -60, "hrtf": "<统一的一份>", "diffraction": false},
  "qa_sampling": {"items_per_type": 2, "query_time_policy": "uniform_in_legal_window", "gold_letter_balance": "batch"}
}
```

所有数值都是占位，`floor_deg` 尤其要等人工校准。`camera_motion` 在本阶段只接受 `static`；传别的值直接报错，
不静默改成静止，免得旧请求复跑时以为自己在转相机。

## 4. 规划层：带约束的拒绝采样器

落在 `src/avengine/rooms/qa_episode.py`，替换 `build_qa_episode_plan` 里从路线到音频的那一段；`native_qa_room.py` 的
原生 Apartment 路径走同一个采样器（现在它写死"恰好两人"且不透传沉默者，一并放开）。

步骤，每步失败就回到第 1 步换种子，总预算建议 200 次，用尽后把每步的失败次数写进 `planning_result.json`：

1. **抽条件画像。**从 `conditions` 的配额里抽出：说话人数、沉默者数、目标角距箱、说话与运动的关系。
2. **抽人物与声音。**在满足人数的资产里随机选人；颜色分配由批级计数器做反平衡（同一批里每种颜色说每句话、
   每位说话人配每种颜色的次数尽量均匀）；声音按 `gender_match` 过滤后随机，一段内台词不重复；每条干声先过语音带 VAD，
   记下 `audible_start_s`、`audible_end_s`，不满足 `min_audible_s` 或超过 `max_clip_s` 的不用。
3. **抽路线。**沿用现有的 `hold_walk_hold`、`walk_hold_walk`、`stand` 三种模式和栅格寻路，但按第 1 步的 `speech_motion`
   决定哪些人要有行走区间：`speaker_moving` 要求将来的说话人在自己的发声窗内在走；`competitor_moving` 要求说话时另一个人在走
   而说话人不动；`all_still` 要求发声窗内没人动。这一步只保证"有可用的行走区间"，具体发声时刻第 5 步再定。
4. **抽静止机位。**在现有 0.55 米网格、1.55 米高、85 度视场的候选里，对每个候选算三件事：
   每位将来的说话人在自己可能的发声窗内是否在视野里且直达路径未被网格几何挡住（复用 `_projected_visible` 和 `_mesh_ray_occluded`）；
   每位说话人与离他最近的其他人在那些时刻的夹角是否落在目标箱；机位到人的距离在 1.5 到 4.5 米之间。
   满足的候选里**随机取一个**，不再按打分挑最好的。没有任何候选满足就回到第 1 步。
5. **抽发声时刻。**对每位说话人，把满足以下条件的时间窗列出来：夹角在目标箱内、有直达路径、运动关系符合第 3 步的要求、
   与其他事件的可听区间间隔不小于 `min_gap_between_events_s`、留出 `reserve_tail_s` 的片尾静默。
   在这些窗里均匀随机取起点。说话顺序随机。取不到就回第 1 步。
6. **写计划。**除了现有字段，新增 `condition_profile`（想要的）和 `achieved_conditions`（按计划几何算出来的）：
   每个事件的说话人、可听起止、与最近竞争者的夹角起止最小最大、说话人和竞争者是否在动、直达路径判定、机位编号。

两点说明。一是静止相机下三到四个人要同时满足"都在视野里"和"两两夹角至少 30 度"，在小房间里可能很难同时做到；
方案里夹角条件只对"说话人与最近竞争者"要求，而且允许非说话人在部分时刻出画，具体能达到多少要用第 7 节的 plan-only 试验量出来，
量不到再放宽箱。二是像素级可见性和最终遮挡状态在渲染前算不准，规划层只用几何射线做预判，真正的分歧条件留给出题层用像素真值判。

## 5. 出题层：枚举候选，按配额抽，四项新拒出条件

落在 `src/avengine/qa/unified_catalog.py`。

1. **枚举而不是取第一个。**每个 `_generate_qa_xx` 拆成两半：`_candidates_qa_xx(facts)` 返回全部合法候选（目标、锚定事件、查询帧的组合），
   `_emit_qa_xx(facts, candidate, seed)` 把一个候选变成题。`generate_unified_questions` 对每类抽 `items_per_type` 个候选，
   抽样用 seed 加类型加 Episode 的确定性随机数，所以同一输入永远出同样的题。
2. **查询时刻走钩子。**`_query_frame`、`_query_time` 已经会先读 `facts["sampling"]`；`qa_delivery.finalize_qa_episode`
   组装原始输入时把 request 的 `qa_sampling` 写进去。`uniform_in_legal_window` 的含义按类型定：QA-13、16、17 在
   "锚定事件可听结束加尾音"到片尾之间均匀取；QA-18 在整段里按"单人发声、无人、多人"三种时刻分层取；QA-14 在两人都可见的帧里均匀取。
3. **四项新拒出条件**，写成和现有 `_defer` 同一风格的检查：
   - `distractors_equal_gold`：绑定组的题，其他候选在被查属性上全部等于金标，或金标是候选中的多数。
   - `binding_not_feasible`：锚定事件的几何角距和说话期间角距变化都低于门槛（门槛占位，等校准）。成片实测的左右线索在渲染后才有，
     由审计工具补判，不在生成时判。
   - `query_inside_inaudible_segment`：QA-18 的查询时刻落在事件的放置区间但不在可听区间，或落在湿声尾音里却被记为"无人"。
   - `quota_exhausted`：该类型该金标值在本段或本批已达配额（比如 QA-01 的 yes 已经够了），用来配平分布。
4. **每类题面和条件的改动**，按放量审视文档第 2 节那张表执行；这里只列改动最大的几条：QA-08 题面改为"at the moment X began"；
   QA-16 英文补比较基准；QA-10 去掉目标自身颜色、帧号改秒；QA-13 加边距规则和位移门槛、统一发布约定并带字段；
   QA-17 去掉"优先找 yes"；QA-19 金标改可听起点；QA-22 四个选项同出现数；QA-03 题面改人话。
5. **金标字母配平。**生成时保留随机洗牌；导出时按类型统计金标字母分布，超出均匀值 10 个百分点的类型对部分题做确定性置换重排；
   同时给每道 MCQ 导出一份镜像或轮转排列，评测时至少跑两种排列并报告位置一致性。这一条也可以放在评测侧做，请 Codex 定放哪。
6. **产物字段。**每道题带 `cluster_id`（等于 Episode id）、`condition_profile`、`difficulty_profile`
   （听、看、时间推理、分歧四组，算法与审计工具一致）。建议把审计工具里的几何与分歧计算抽成 `src/avengine/qa/answerability.py`，
   生成器和审计工具共用一份实现，避免两边算出两个数。

## 6. 渲染与素材层

- 增益、本底噪声电平、HRTF 路径、绕射开关从 `audio_render` 读，写进渲染报告；缺省值保持现状但请求模板一律显式给出。
- 语音库从 7 句扩到注册表里的 620 条 VCTK，入池时做语音带 VAD 与 80 Hz 高通，输出的池条目带 `audible_start_s`、`audible_end_s`、`speech_duration_s`。
- 人物资产档案加 `gender`；调度器按 `gender_match` 过滤。
- 非语音类别池扩到每类至少 10 条并做感知分段审核，这一步不阻塞语音题。

## 7. 验收：先算再拍，再审

1. **单测**（合成房间，不跑 UE）：采样器输出满足条件画像；相机为静止；性别不匹配的声音被拒；可听区间正确；
   出题层在退化候选上给出 `distractors_equal_gold`；查询时刻落在合法窗内；同一 seed 两次运行逐字节一致。
2. **plan-only 可行性矩阵**（不跑 UE，几分钟一段）：三间自建房间加原生公寓，每间每个角距箱各抽 50 份条件画像，
   报告每步的成功率和失败原因直方图。这个矩阵决定每间房能供哪些箱，也告诉我们静止相机下四人场景到底能不能配出 60 度。
3. **小规模原生先导**：每间房 2 段，跑完用 `tools/qa/audit_binding_feasibility.py` 审。验收线（占位）：
   所有发声事件几何可绑定；绑定组的题里退化标记不超过两成；没有 `query_inside_inaudible_segment`；
   每类金标字母分布偏离均匀不超过 10 个百分点；成片峰值在 −6 到 −12 dBFS，精确零采样点不到 1%。
4. 先导过了再谈规模；规模由房间数定。

## 8. 分工与顺序

Codex：第 4 节采样器（`qa_episode.py`、`native_qa_room.py`、`tools/studio/run_qa_episode.py`）、第 5 节生成器改动
（`unified_catalog.py`、`unified_scoring.py` 的角度方向词规则）、第 6 节渲染参数与语音库扩池（渲染链脚本与 `tools/assets`）、
第 7 节第 1 和第 2 项。
Claude：`answerability.py` 的接口草案与阈值占位说明、审计工具增加可听区间列与批级汇总、人工校准包的题面与记录表、
plan-only 矩阵出来后的统计解读。两边不碰对方名下的文件；`answerability.py` 若由 Codex 抽出则我只提接口意见。

建议顺序：先做语音库 VAD 扩池和 `gender` 字段（不影响其他改动），同时做采样器；采样器出 plan-only 矩阵后再改生成器的候选枚举，
因为配额能设多少取决于矩阵；渲染参数化随时可做；最后跑先导。

## 9. 请 Codex 重点审这些

1. 静止相机、85 度视场、小房间、三到四人同时在视野且说话人与最近竞争者夹角不小于 30 度，几何上可行率大概多少？
   如果很低，是放宽到"只要求说话人可见"，还是缩小箱？
2. 现有栅格寻路的起点抽样以一个 hub 半径 3.1 米为限，人物间距 0.95 米起；要拉开角距是否需要改这两个数？
3. 发声时刻改成条件抽样后，`schedule_audio` 的 `overlap` 与 `repeat` 模式怎么保留？
4. 原生 Apartment 放开到三到四人，路线库和 Recast 重查够不够用？
5. 语音库 VAD 是新建一份 prepared 集还是只记偏移？两种做法对复跑历史 Episode 的影响。
6. 金标字母配平放生成侧还是评测侧？
7. RLR 打开绕射的成本和稳定性，是否值得在一段上先试。
8. `answerability.py` 抽出来之后，审计工具与生成器共用时的版本钉扎方式。
