# v7-spec 接口需求（第 2 回合）

本分支只改 QA 包、QA 配置及其测试。本清单要求 v7-spec 在自己的渲染/场景采样器检出合并实现；这里没有改它的检出、渲染器、RIR 或采样器。正式出题仍调用未修改的 unified_catalog.py，原有判分器、门槛和 any/all 判据保留。

## 1. 实测入画、完全遮挡和遮挡物身份：QA-07/09/10

接入点为 avengine.qa.v7_native_adapter.native_bundle(..., evidence)。evidence 中保留：

- pixel_visibility_truth.status = computed_modal_target_only_v1；per_instance 的键为原生 actor_id 字符串。
- per_instance[id].frames 每帧包含 frame_index、visible_pixels、target_pixels、visible_fraction、occlusion_fraction、target_bbox_xyxy_px、target_centroid_xy_px、state。帧号为原场景时钟；15 Hz；每个 10 秒窗口需实际观测 150 帧，不得由少量快照插值伪造。
- 帧状态仍使用 v6 原定义：visible_clear、visible_occluded、fully_occluded、out_of_view。target-only pass 必须保留真实相机和人物姿态，只去掉实际遮挡物。modal 中零像素不能直接推成完全遮挡；target-only 中零像素不能推成遮挡。
- camera_calibration 和 visibility resolution_hw 来自原生相机，不由题面反推。
- occluder_registry 的键是遮挡物原生实例 ID 字符串。值须有 display_label / label / category，若按颜色命名则有 appearance_value。候选对象需实际存在，不能造一个不存在的干扰物。

新增传入 occluder_evidence：

```json
{
  "frame_index_scope": "scene",
  "frame_records": [
    {
      "target_instance_id": "1",
      "frame_index": 78,
      "occluder_instance_ids": ["static_screen_17"],
      "unknown_occluder_pixels": 0
    }
  ]
}
```

target_instance_id 是 v6 _occluder_ids 实际读取的字段，不能只传 actor_id。QA 层已兼容别名后输出这个规范字段，并在截取窗口时将场景帧号平移；重复 target/frame、重复 ID、未登记 ID、有未归属遮挡像素却声称已唯一归属都会拒绝。逐像素深度或 ray/object-ID 归属应由原生后端提供，不能把窗前所有对象 ID 都列为遮挡物。

采样器应提供可实际验收的结构，而非预填答案：

1. QA-07：连续相邻帧真实 out_of_view -> visible，随后保持可用的入画时段，target-only centroid 位于 v6 的左右死区之外。默认 1280 宽时，v6 使用 max(1, width * 0.02)，此门槛不改。
2. QA-09：真实完全遮挡后再出现；“没有再出现”的样本必须给全窗口完整可见性，并满足 v6“遮挡前确实见过此人”的前提。任意一帧重新出现即是 yes，不能改成所有后续帧可见。
3. QA-10：在 v6 接受的稳定遮挡时段内逐帧有唯一、实测的 occluder ID；至少两个真实可区分的备选对象；目标与备选的颜色族、两个备选之间的颜色族不能撞 v6 的可答性条件。最终能否出题由原权威生成函数决定。

建议 owner 修改 timeline/v7_candidate_generation.py 的人物/屏风放置与轨迹安排，以及 capture/continuous_native_video.py 的原生 target-only / 深度 / 对象 ID 读回；具体实现由 owner 选择。QA 层不改变 native_geometry 或像素可见性的门槛。

## 2. 配额和四成员联选

接入点：

- binding_reservation(seed_rows, split=...) 从整个 split 的种子表计算预留数。
- select_counterfactual_group(public_by_member, private_by_member, allocation_by_member, binding_items_per_window=5, ...) 在四个实际成员中联选。
- 继续使用 v7_production_contract.question_allocation 的原 14 题轮转配额；V/A/AV 总体 30/25/45。

当前种子表 train 有 1400 个成员窗口、其中 800 个属于四成员组；dev 为 350 和 200。每四成员窗口预留 5 道真实绑定逻辑题，可得到计划总占比 5*800/(14*1400) = 20.408%（dev 同比例）。该数字仅为计划，完整实际选择通过后才可计为达标。

原生 canary 两个窗口各有 5 道共同、答案对角相同而跨模态交换改变的 AV 题，但纯视觉只有 1 道，实际需要 4 或 5 道。另一训练屋有 3 道，实际需 4 或 5。新 selector 因视觉缺额直接报错，没有补齐或改模态。

场景采样器具体需提供：

- 每窗口足够多的不同合法视觉逻辑问题；当前两屋各缺 1–4 道/窗口。增加真实可命名的人、物、进出/遮挡、视觉位置或动作安排，保留身份与连续帧读回。不能只加同题的 MCQ/改写来凑数。
- 每个四成员窗口至少 5 道共同合法 AV stem，四成员的题面/答案域相同；同一视觉成员在音频交换时视频字节完全相同，同一音频成员在视觉交换时音频字节完全相同；音频交换确实改变音频。绑定题须由四个权威答案实测筛选。
- 建议增加可绑定人物/真实播放设备，并安排作者语义脚本及角色对话，扩充 QA-12/20/26–28 的绑定覆盖；已有 5 道全部来自 QA-02，不代表其他族已经满足配额。
- 训练和开发 split 各族分别满足 question_budget_v1.json 的最低量。不能把额外资格场景当成 v7-S 种子表已完成的配额，也不能把开发集的量从训练屋外推为“已验证”。

owner 的调度器应先收集完整四成员候选，再联选、再封存。每成员各自选 14 道后贴 binding 标记不能保证总占比。此处只新增 QA 联选接口，未改调度器。

## 3. 干声映射、自动判定与物理机制

使用本轮 dry_sound_admission.json 中 strict_v7_eligible=true 的记录，并按 normalized_path 读取输入；原始文件只读。保留 original_id / source_uri / license / attribution / 自动模型和去重证据。已准入公开单声道录音可作为仿真输入；真实空间录音、评测同源来源仍禁止。

QA-21 固定 34 类域不变；35 类中的 any_audioset_class_playback 是复合占位，不是第 35 个答案。必须记录实际 content_class 和物理 emission_mechanism，然后映射至域内内容；无法对应则不出 QA-21。复合记录如果指向域外的具体 AudioSet 内容，还需显式 content_audioset_mids（与实际模型头一致）才能判类；目录占位本身不构成标签。STARSS 13 类映射中的条件映射不能当成任意类的兜底。

native evidence.events[event_id]：
sound_asset_id、sound_class、transcript（讲话）、activity_status=observed、activity_intervals_samples_global、event_unit / event_segmentation、source_qc。活动段用实测干声采样坐标、半开 [start,end)，保留左截断 anchor、同人计数、首三锚点平局规则；响度处理不能移动时钟。自动能量岛仅是 source activity，不能直接当成 QA-23 的独立事件数量。

speech_playback / music_playback 还需：
events[event_id].emission_mechanism=loudspeaker_playback；
actor_profiles[actor_id].emission_mechanism=loudspeaker_playback；
实际播放对象的 FOA 发声点（音箱驱动中心等）。
QA 层已拒绝把缺乏上述物理机制的 TTS 人物讲话静默归到播放类。live formant 讲话保留 live 机制，SPEAKERS 按实际人 ID 计数。

自动标注使用现有 BEATs 527 类头与原始 FSD50K 标签校准。细分铃声类别还要原始元数据确认；不能要求听辨。尚无可用片段的类需另提供许可清楚、评测无同源、自动标签达标的输入；不能为凑 35 类降低本轮置信证据。

## 4. 人物：实际地面、完整体型、动作时钟、MJaw 和多人

本轮 7 个人物配置的本地 MIT 许可链已按原始 FBX/纹理 SHA、Git blob、运行副本 SHA 和编译后的几何/rig 对拍闭合；4 种衣色是同一男性体型的变体，合计 3 个体型。

原生资格检查发现 R7 的导航地面与实际 stage 地面相差约 18 cm。采样器/渲染器需：

- 在每个人的原生脚底 skin 顶点下 cast_ray 到实际 stage，记录 object_id、实际地面点、signed gap；不能以 navmesh 高度等同地板。
- root 修正后重新取 native joints 与 Bip01 MJaw，再生成声音/RIR；禁止复用修正前的 emitter 点或根高替代嘴部。锚点本地 offset 保留原资产定义。
- 原 AO 没有完整人体 collision shape；空的 AO contact list 不能证明不穿模。现有 rest proxy 也未包住若干真实姿态。owner 需在自身文件中采用足够包住真实 skin/三角面的碰撞体或等价实际 geometry 检查，逐姿态/逐移动窗口检查实际家具/墙/地面、人物之间的接触。
- 本轮补了包含每个原始三角面的分骨凸包原生 Bullet 测试与 10 cm 穿地负例，但在测试位置仍未通过。因此许可通过不能等于站姿/走路正式准入。凸包是保守上界，失败不等于已证明原始皮肤真的穿模；需要更精确几何或合法放置验证。
- 新准入数值容差 15 mm 是本轮人物接触政策，不是改写 v6 真值，也不是声称 world_contact 的水平距离标准原本就是竖直脚底标准。
- Walking 源动画男性水平位移约 1.495864 m/1.066667 s（约 1.402 m/s），当前渲染轨迹/相位未对齐。需按实际移动速度选择/驱动 gait phase、支持脚接触和脚底滑动检测；idle 必须使用站姿动作，不能用静止 Walking 代替。

8 个原生实例确实已在同一个真实仿真房屋中加载，n=0…8 的干声活动/原生 MJaw-RIR 均提供了资格样本，并按 AVModel 权威重算。它们不是 8 种体型/声线，而且该探针全在画外；不得把它写成“8 人同屏、接触合格、听觉身份可辨”。
若要作同屏视觉题，owner 需另外提供有安全间距、实际 visible_pixels>0 的 6–8 人布置及上述接触证据；不能用导航点存在替代相机遮挡检查。纯音频 SPEAKERS 的物理人数规则仍是 0–8，重复活动属于同一人时只计一次。

## 5. 合并边界

第 2 回合没有必须 cherry-pick 的 owner 文件变更。此提交的 QA adapter、自动准入和联选函数可单独接入。依赖第 1 回合 23e2cdd 的 QA 文件；如 owner 已有渲染器改动，不整块覆盖第 1 回合 tools/timeline/pilot_continuous_scenes.py 的 6 行调试 hook，手工调用 make_full_questions 即可。

需要改 owner 侧文件的理由只有：实际入画/遮挡信号不能在 QA 层猜测；root/gait/MJaw/RIR 时钟属于原生渲染；场景布置和全组收集调度属于采样器。这里提交的是可调用接口和明确失败条件，实施与合并仍由 v7-spec 完成。
