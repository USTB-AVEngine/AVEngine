本轮已经接通 QA-26–28 和五类内容绑定的出题入口；合格的源台词、登记人物和完整原生窗口仍是三个分别验收的对象。仅有作者计划或静态登记不会生成“原生已准入”的题。本文件供 v7-spec 合并和接场景层，未修改其采样器、渲染、RIR、交付或封存文件。

输入和调用：

```python
from avengine.qa.v7_native_adapter import make_full_questions

qa_evidence.update(
    semantic_manifest=normalized_batch_manifest,
    native_dialogue_evidence={
        "registered_actors": actual_standing_instances,
        "placements": placements_by_actual_event_id,
        "roles": roles_by_actual_event_id,
    },
    semantic_wording_variant="standard",  # 对照重跑使用 alternate
    content_binding_bundle=actual_content_binding_bundle,
)
public, private, audit = make_full_questions(
    scene, start_s, opaque_window_id, actual_visual_readback,
    neutral_public_media, qa_evidence, collect_all=True,
)
```

这里的 manifest 使用 round5_20261004/natural_batch_ready/semantic_manifest.json；其中 sounds 只列本次自动通过的完整自然台词，dialogues 保留 180 个作者情景。role_id 是作者角色，native_actor_id 是物理实例，sound_asset_id 是音频资源；三者不可相互代替。登记和音频的 SHA-256 都要对应实际输入版本。缺少角色的情景只能安排实际通过的角色；v6 要求至少两个不同且无歧义的内容候选，不能以一段话凑一族题。

一、QA-26–28 的场景接口

1. 人物只允许正式登记的站姿。registered_actors 每行包括 native_actor_id、asset_id（正式 variant_id）、pose="stand"、body_to_world_m（GLB 的米制刚体 4×4 变换），以及 standing_pose_readback。readback 至少含 status="pass"、保留在文件中的 source、actor_id、asset_id、source_registry_sha256。原生 visual.frames 中实际 loaded asset_id 应为登记 variant_id；如引擎使用不同资源别名，额外提供 actual_loaded_asset_id 和 loaded_artifact_sha256，后者必须等于实际登记 GLB 的 SHA。计划里的资产名不能覆盖原生加载记录。

2. 不重新估计嘴。嘴坐标来自 human_roster.json 的 mouth_local_glb_m；世界坐标为 body_to_world_m × [mouth,1]。UE 使用登记 mouth_local_ue_cm；登记坐标关系为 100×(-GLB_z,GLB_x,GLB_y)，场景层仍须给出其实际世界读回。当前出题层核对每个实际帧的根部位置和发声点，误差上限为 0.002 m，不接受只有计划或单帧示意的证明。站姿窗口的实际 root/emitter 轨迹必须完整；不可把动画中不同嘴位置伪装成固定站姿。

3. scene.events 与 qa_evidence.events 延续已有接口。每个实际事件给 event_id、source_id（讲话 actor）、endpoint_id、start_tick/end_tick 和逐条 sound_asset_id、逐字 transcript。活动区间继续取测得的 source activity，使用半开区间。出题层归一化后的规范字段是 event.start_s/end_s；新增接口中的时刻使用**当前 10 秒窗口的相对秒数**，不是绝对世界时刻或 tick。源文件时长必须完整落入事件和窗口，可以在结束后补静音，不能截断末词、重放片段或加速来挤进窗口。

4. placements 以实际 event_id 为键，每行：
   status、source、event_id、actor_id；
   window_start_s、window_end_s；
   native_emitter_world_m、emitter_point_kind；
   source_sample_count、source_sample_rate_hz、source_audio_sha256；
   full_content_preserved=true、trimmed_or_repeated=false。
   sample_count/rate 指最终实际 16 kHz 完整源文件，不能填合成前未保留的 24 kHz 样本数。事件窗口边界允许一个当前场景采样点的对齐误差；实际帧的 FOA/RIR 发声位置还要独立匹配。相同音频资源再发一次也要有新的 event_id 和实际 placement；接口内部会产生只用于监督的每事件别名。

5. roles 同样以实际 event_id 为键，每行包括 status、source、actor_id、scenario_id 和 role（作者 role_id）。同一情景的不同角色必须是不同实际 actor、不同 VCTK 参考声音；同一个实际 actor 在窗口内再次讲话时沿用同一声音。性别使用正式登记的 gender 与 VCTK 元数据对应，不按名字猜测 lx 性别。

6. 电话和喇叭情景必须满足作者 required_asset_capability。正式 lx phone_call 或 megaphone_speaker 可提供对应人体和设备；普通 Rocketbox 人物必须另交 native_action_readback：status、source、actor_id、asset_kind、device_id。电话现场讲话仍从该人的登记嘴发声，电话另一端播放不冒充现场 actor。喇叭讲话使用实际 megaphone_driver，还须提供 native_driver_world_m，且实际 emitter 帧轨迹一致。登记嘴仍用于人体身份和附着关系，不用嘴坐标替代喇叭扬声口。自然声音的元数据性别匹配已核对；独立声学性别感知未验证。登记 lx 的 gender=not_annotated 时不能先配成 male/female 对话。

7. 题面互换标准：同一份实际事实重跑 standard/alternate；比较 QA ID、真值、答案域和选择题选项。QA-26 泛指“这个人”的作者题面会在出题层换成 v6 已观测的 appearance；不以作者 role、VCTK ID、资产 ID 作为公有答案。QA-27/28 分别调用原来的 v6 内容到外观、内容到方向真值。判分器、any/all、方向门槛和整数 anchor 规则均沿用权威代码。

8. 当前台词为英文，中文题面和标签沿用原作者目录。TTS 为 CosyVoice2，Whisper 不接收参考文本或答案提示。第一次合成加最多两次重试，总上限三次；失败记录和每次波形均保留。ASR 中 3:25 / 3.25 / 325 与 three twenty-five、teaspoon 与 tea spoon、tyre/tire 等固定英美拼写只做预先固定的等价归一化；不修补错误时刻或错误名词，WER≤0.10 和关键意思/否定一致均须通过。批次启动时的旧归一化与最终重算分别保留，这一步重算没有再次合成或再次 ASR；之后使用已有常数增益规则另存完整波形（−23 LUFS、true peak≤−3 dBFS），最终波形再逐句做盲 Whisper 核对，ASR SHA 指向实际归一化输入。

二、五类内容绑定接口

额外面板 ID 为 V7-CONTENT-BIND，protocol_panel="v7_content_binding"。没有改名 QA-01…28，也没有新增 QA-29 或挪用 QA-21 标签域。它可以进入已有候选选择器；正式预算是否单列由 v7-spec/owner 的发布配置决定。本轮未改正式预算和门槛。

content_binding_bundle 结构包括：
- source_registry：正式 human_roster.json；
- scene：window_s=[lo,hi]，actors、emissions 和保留的原生准入证据；
- queries：每行 query、binding_witness，可选 wording_variant；
- pairing_id/base_world_id：真实分组 ID，不能编码答案。

actors 每行含物理 actor_id、正式 asset_id、pose="stand"、唯一可见 descriptor_zh、实际 visible、登记 capability、body_to_world_m 和 standing_pose_readback。一个资源可实例化为两个不同 actor；不能用一个 actor 的两个资源槽冒充两个人。descriptor 必须能从画面实际观测区分，禁止回答隐藏 ID。

emissions 每行含 emission_id、mechanism、content_class、emitter_point_kind、measured_activity_s 半开实测活动段，以及实际位置和原生证据。actor_live 必须绑定真实 actor；loudspeaker_playback 必须给真实 device_id，actor_id 为空。乐器 subtype 与登记一致；原声吉他用 soundboard，电吉他使用真实 amplifier，手鼓提供具体 subtype。喇叭、电话与现场演员的区分同上。

五类 query：
| kind | 公有题意 | 真值 |
|---|---|---|
| same_class_actor | 同类多个候选，谁正在发声 | 指定时刻不同物理发声主体恰好一个，返回其可见描述；至少两个可见且物理可发此声的人 |
| mixed_class_actor | 拿着不同物件/乐器，谁在发声 | 同上，泛化题面不通过预先指定声类缩小候选 |
| offscreen_origin | 指定画面人物是否发出声音 | 唯一实际发声主体等于指定可见 actor 为“是”，否则“否” |
| live_or_playback | 现场人物还是音箱 | 唯一主体的真实机制；电话播放不能标为音箱 |
| first_actor | 谁先发声 | 对不同实际 actor 取左截断首锚点，唯一最早者；平局、画外答案不能硬选 |

前三/四类提供 time_s 和 public_time_s（窗口相对时刻）；first_actor 比较完整窗口。offscreen_origin 提供 subject_actor_id 和实际可见 subject_descriptor_zh。只有明确公有题面 names_sound_class_in_question=true 时 content_class 才是筛选条件；默认泛指声音时不能藏一个 private content_class 选择器改变真值。

所有类型在发布前都要求 admission_blockers 的实际 native/render/RIR/activity/visibility/physical-content/source-QC 证据和两轴反事实 witness。witness 中包含当前候选，并有真实相同视频/不同音频使答案翻转、真实相同音频/不同视频使答案翻转的世界；不能用数组替换和复制文件充当。四成员交换发声人×位置/外观应保持物理能力一致，不能让吉他资产发出长笛。题面已命名的声类若使静态画面直接排除指定人物，也不能算通过两轴反事实的绑定题。标准/互换题面共用实际真值和答案域，选项顺序独立于答案；公有 JSON 不带物理 ID、角色表或真值。

给 v7-human-roster 的剩余需求，仅以本文件转交，不直接修改其登记：
- lx 讲话资产提供正式 gender，之后才可绑定 VCTK 音色；
- 乐器/手鼓/喇叭的实际发声部位坐标、坐标系及几何附着证明；
- 同类多演员需要实际可区分的人体/外观或明确的可见位置描述；不同物理实例 ID 和各自读回不可缺；
- 非站姿仍需其完整动作验收，本轮不启用走路、坐姿、座椅。

三、QA-07、QA-09、QA-10：继续缺的原生信号

出题层已有裁窗和权威生成逻辑；v7-spec 必须采样会实际发生进/出或完全遮挡的窗口，不能只补标签。

- 同一实际相机、时间和目标的完整场景 pixel mask，以及去掉遮挡物的 target-only mask，15 Hz、10 s、150 帧；真实渲染 readback，逐帧 actor ID 和 camera pose。
- pixel_visibility_truth.status="computed_modal_target_only_v1"；per_instance[actor_id].frames 每帧 frame_index 及 visible_pixels、target_pixels、visible_fraction、occlusion_fraction、target_bbox_xyxy_px、target_centroid_xy_px、visible_centroid_xy_px、state（逐帧规范字段）；完全遮挡要求 target-only 仍有目标像素但 context 完全没有，画外和遮挡不能混用。不能把缺失帧或未知读回补为零。
- occluder_registry：逐个唯一实际 native instance ID，实际类别、几何/实例关系和保留证据；occluder_evidence.frame_records：target_instance_id、frame_index、occluder_instance_ids 显式唯一列表、unknown_occluder_pixels **显式计数**、对应 context/target-only 计数。unknown 必须测得为零才能证明唯一遮挡物；缺省不能当零。frame_index_scope 明确 scene/window，裁窗后须完整。
- 按 v6 逐帧 any/all 权威谓词判，接口不改旧的合法样本标准。进出是原生可见性随时间的变化；完全遮挡和唯一遮挡物分别满足 QA-09/10 的证据。
- 相机标定和 image width/height、像素坐标系齐全；近边界中心按零基索引+0.5。FRUSTUM 两类×八方位均匀目标、真实相机/声源反事实 pairing ID 与 SPEAKERS 的实际 actor 讲话时间表仍按 round3 接口执行。

四、配额与合并范围

纯视觉仍须增加可观测的颜色/空间关系/进出/遮挡等窗口。四成员组至少 20% 需要增加物理合法、实际双轴翻转的绑定组；增加 AV 台词源数本身不会填纯视觉缺口。采样器应安排 ≥2 个可区分的人/物，保持说话 actor 时间表，零人讲话窗仍有环境声与可见静默人物，0…8 人各层多个独立世界。以上均是 v7-spec 的场景层改动，本分支没有直接改其文件。

本轮仅修改 src/avengine/qa/、对应 unit tests 和 examples/qa/v7_admission/。QA 的新增 kwargs 是小接口；src/avengine/timeline/、dataset、渲染和声学文件本轮未改。旧提交的 timeline/v7_native_questions.py 唯一接入口若还未合并，见 round1 cherry-pick 清单；请由 v7-spec 在其正式窗口流程中接上述 qa_evidence。资产与大波形留在独立产物目录，不进 Git。各提交和依赖以交付 CHERRY_PICK_zh.md 为准。
