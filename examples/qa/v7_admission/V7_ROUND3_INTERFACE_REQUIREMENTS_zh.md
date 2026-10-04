第 3 回合接口：出题层已支持不同讲话 actor 去重、0–8 层统计、半像素 FRUSTUM、原生反事实对的成对选择及逐帧遮挡接收；下列场景信号由 v7-spec 提供。

本提交只涉及 src/avengine/qa/、QA 配置和测试。未修改渲染、RIR、采样器文件。第 2 回合提交 7e38875 的原有接口继续有效；本轮接口是在它之上的增量。

**1. 讲话 actor 与槽位分离**

scene.actor_speech_schedule.schema 必须为 v7_actor_speech_schedule_v1，activity_complete=true 只可在全部现场讲话干声活动已自动测量并保留收据后写入。缺少记录不能推导为 0 人。

actors 每项必须有 actor_uid（父世界内稳定，反事实成员共享）、native_actor_id（正整数、只作私有监督）、native_endpoint_id（实际 MJaw）、video_track_id（实际视频实例）、native_readback。保留 native_object_id / semantic_id 对应表以供核对。不同声音槽、事件、TTS 资产、可见人数、方位均不能替代 actor。

event_bindings 每项必须有 event_id、source_slot_id、endpoint_id、class_id（STARSS 0/1）、actor_uid、native_actor_id、emission_mechanism、activity_observed=true、activity_receipt、sound_asset_id。scene.events 保留半开 start_tick/end_tick，100 ms 活动帧来自实际干声样本的任意相交规则；左截断 anchor、同人去重、首三锚点平局沿用 v0.2。每项必须与真正发声的原生嘴部实例一致；同人多槽映射到同一嘴部轨迹。播放语音不能标成现场讲话。

native_bundle 在形成 v6 facts 前调用 canonical_speech_scene，把槽位映射到物理 actor；generate_questions 同样校验。speaker_state_sidecar 输出每个登记 actor 的 10 Hz is_speaking 和 event→video actor 对应，只进入训练监督旁路，不进入公开题面、选项、模型部署输入。场景层还须提供实际说话/嘴部动作的同步证据，不能只给教师标签而画面始终静止。

必须按 train/dev 分别生产 0,1,2,3,4,5,6,7,8，各层至少 2 个独立父世界。计数窗/相机变体/四成员变体不是独立世界。附加问法、open/MCQ 不增加题量。轮流、重叠、画外、同方位不同人、同人多段/多资产、运动中再次发声都必须保留。0 人窗保留环境声和可见不讲话的人。

**2. FRUSTUM 坐标、分层与反事实对**

camera_calibration 使用 fx,fy,cx,cy 和 coordinate_convention。zero_based_pixel_centers 时先加 (0.5,0.5) 转成像素边界；pixel_boundaries 时不再加。范围严格为 [0,W)×[0,H)，后方不在画内，无 FRUSTUM 死区。必须保留 native render_camera.projection_matrix、sensor node world matrix、实际 MJaw 点和 NDC→像素边界坐标。QA-04 的 5° 死区、QA-25 的 15°/30° 判分不变。

v02_panel_coverage_v1.json 是额外工程目标，原有 v7_question_budget_v1.json 最低量不变。SPEAKERS train 每个整数 16、dev 每个整数 3；FRUSTUM 两类 × 世界水平八方位，train 每格 8、dev 每格 2。世界坐标八向与相机相对八向分开报告：85° 水平视场无法让画内类覆盖相机相对全部八向，不能伪造这种交叉配额。另需在各类可行的相机相对方位范围内均匀采样，含左/右及上/下边界。

pair_id、world_id、split、source_actor_uid、dry_audio_identity、activity_identity、question_stem_id 都是私有字段。每对恰有 2 个实际原生媒体成员，canonical_answer 必须一内一外。只改相机朝向时 source_trajectory_identity、listener_position_identity、world_foa_identity 保持一致，视频必须重渲染，FOA 转到新相机基。只改声源位置时 actor/干声/活动/相机不变，必须有实际 rir_recomputed_receipt。不得把镜像、换题面、修改真值当反事实。

validate_counterfactual_pair 校验收据与不变量；select_balanced_panels 先收集候选、按独立世界轮转并成对取舍，不能先逐成员截断。调用时传 split，禁止拿训练世界填开发配额。只要有配额或世界数缺口，返回 blocked，不能宣称平衡面板交付通过。

**3. QA-07/09/10 的原生逐帧信号**

每个 10 秒窗提供完整 150 帧 modal 和 target-only 实例 mask。DenseNativeVisibility 流式调用未修改的 v6 pixel_visibility._frame_truth；同一原生渲染上下文、同一相机、同一姿态/骨架/皮肤、同一帧时钟，只移除其他遮挡物的绘制。不能另开渲染器再由矩阵重建姿态，不能 OR/modal 补 target-only，不能用投影框或稀疏抽帧伪装实际遮挡。

每帧必须有 target_instance_id、frame_index、明确 occluder_instance_ids 与 unknown_occluder_pixels。semantic ID→原生物体 instance ID 的注册表必须唯一。阶段/墙面未注册像素保持未知；未知像素不认证唯一物体。QA-10 至少 2 个真的、颜色族不同且与目标不撞色的可见物体备选，标签由原生 RGB+mask 自动验收。

QA-07 需要真实 out_of_view→visible 入画及稳定的侧向像素区间；生产采样器补固定相机的人物走入/走出。QA-09 需要真的完全遮挡与重新出现，并保留遮挡前可见证据；否定题须覆盖完整窗口。QA-10 需要整个可公布遮挡区间的唯一实例 ID，不能拿物体存在于场景中当遮挡证据。

动态相机可以通过 evidence.camera_frame_readbacks 提供每帧整数 frame_index、实际 position_m/yaw_deg。适配器要求 150 帧连续、有序且数值有限。原生 camera_calibration 和像素真值随同交付；这只是事实接口，未改 v6 真值/门槛。

**4. 人物、干声和四成员配额的剩余接口**

人物：原生阶段地面射线与 navmesh 高度分别保留；按实际 deformed sole 求 root 修正，全部动作采样核对脚底、穿模/悬空、步态位移/阶段时钟、全体变形皮肤的碰撞覆盖。AO 的空 contact list 不能证明人体无碰撞。刚体皮肤包络必须明确其视觉原点/COM，自动 COM 重心偏移要单独核对。FOA 每帧源点用实际 MJaw，不用 root+固定高度。原始 walk 动作到 MIT 上游的对应关系仍须补原始 motion ID、许可文本/路径、本地派生链。

干声：现有 35 类清单中的 15 类为空，需要许可允许的原始单声道候选及原始 ID/元数据。允许公开录制音效作为仿真输入，排除 STARSS/DCASE/TAU 和评测同源片段，自动模型、活动边界、响度及去重收据全部保留。复合 any_audioset_class_playback 必须给真实内容类与播放机制，不能增加第 35 个 QA-21 答案。新语音可用有许可库或作者 TTS；自动判不能改成请人听。

四成员：每个绑定父世界多放可区分的人/物，至少提供每窗 5 条在四份真正不同原生媒体中同题干、答案变化的绑定逻辑题；先整组收集再保留。按原有 v7-S 种子，800 个 quartet 成员窗×5/19600=20.408% 是接口规划，当前种子窗实测仍为 10/72=13.89%，不能把规划写成已达标。纯视觉每窗至少 4–5 条可答候选，补真实入画、遮挡、属性/数量/位置布置；不能以音频题填视觉缺口。

本轮 CPU 原生验证与报告放在 /data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round3_20261004/。新增验证窗只作能力/真值验收，不混入原始 v7-S 种子量或开发量；正式交付仍由 v7-spec 完成。
