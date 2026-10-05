# 给 v7-spec / v7-human-roster 的第 4 回合接口需求

本代理只改出题逻辑、准入配置和验证测试；renderer、RIR、场景采样器与其他检出保持只读。以下是需要 owner 接入的具体字段，不能把接口单测说成原生场景完成。

## 内容绑定 scene（私有监督旁路）

content_binding_scene：
- window_s=[start,end)，实际采样时钟；world_id/base_world_id/split。
- actors[]：actor_id=稳定不同物理 actor；descriptor_zh=原生可区分外观；visible=本题时刻 native modal 可见；capability={asset_kind,instrument_subtype,roster_admission_source}。
- emissions[]：emission_id、mechanism(actor_live/loudspeaker_playback)、content_class、actor_id（仅 live）或 device_id（仅 playback）、emitter_point_kind、instrument_subtype、measured_activity_s（实际源活动半开区间，非计划表）、dry_asset_id/原始 ID/许可准入、actual_emitter_world_trajectory、source_waveform_readback。
- native_readback：video/audio/pixel_actor_identity/measured_source_activity/asset_license_and_capability/physical_emitter_attachment/audible_emitter_uniqueness 的 pass+保留证据路径。先后题另给 source_order_listener_order_agree，实际源端与 listener per-source W 活动起始顺序一致。
- query：kind、time_s（已有观测帧/采样坐标，不插值猜运动）、public_time_s=time_s-window_s[0]（题面显示的相对秒数）、content_class（私有设计目标；只有题面明确命名类别时才允许筛选活动）、subject_actor_id+subject_descriptor_zh（仅画外来源题），names_sound_class_in_question 明确记录。
- public 只放中性 question_id、题面、完整选项和媒体；所有 actor ID、source ID、activity、capability、pairing_id 与答案只在 teacher sidecar。

v7_content_binding.py 的 content_truth 只计算真值；admission_blockers 要求原生及跨模态反例回执。新题型是独立 V7-CONTENT-BIND，不改 QA-01…QA-28 名称/域/评分器。

## 四成员 pairing

group_id/pairing_id/base_world_id；active_choice∈{0,1}、appearance_permutation∈{0,1}，整组四个状态。
每状态保留 实际 video、FOA、worldFOA/source W、native actor/source 附着、活动、干声 ID、许可、相机 P/pose；每对提供同 V 换 A 与同 A 换 V 的媒体摘要、实际独立真值和同题/同选项域证明。
人物换位重算原生 RIR；交换衣色时验证几何、声学材质、源点和波形不变。只给配对 ID 不足以认证反事实。
binding_witness 的每个轴需 native_pair={scenes:[私有scene0,scene1],queries:[query0,query1],media:[{video_path,audio_path},{video_path,audio_path}],answer_domains:[domain0,domain1]}；出题层重新计算两答案并读取实际媒体，核对相应相同/不同 SHA 与同题/同域，不能只填 status:pass；实际音频须解码为完整四通道窗口，视频须能解码。每轴的一侧必须包含当前候选的完整 scene+query，不能借用其他世界的反事实回执；native source 回执路径必须实际存在。
不得构造拿吉他发长笛声。辅助 offscreen/机制题若没有两模态必要性反例，不计绑定配额。

## 自然讲话与作者台词

输入 examples/qa/v7_admission/semantic_daily_v2.json；不要调用 FormantSpeech/eSpeak 作为内容题 fallback。
角色的 meaning_answer/transcript 与 roster actor 一一绑定；角色之间意思不同。自然合成句不得裁切或循环。保留 CosyVoice2 code/model revision、模型许可声明与原文、VCTK 原始 speaker/utterance/mic/transcript/CC BY4 归因、实际 voice gender。本机完整 VCTK 本身不是评测素材；旧 pool 的 103 条 eval、8 个 eval speaker 仍隔离，参考计划仅 16 个 train speaker。

semantic_manifest.sounds[] 的新增必需字段由 natural_speech_blockers 检查：
speech_policy=v7_natural_authored_speech_v1；
synthesis_readback.engine={name:CosyVoice2,natural_neural_tts:true,model_id,model_license,code_revision}；
voice_reference={dataset:VCTK-0.92,license_spdx:CC-BY-4.0,license_chain_status:pass,speaker_id,utterance_id,source_waveform,source_transcript,gender,split:train}；
voice_gender；native_placement_readback.visual_gender；
automatic_content_readback={status:pass,model_id,source,reference_text,wer≤0.1,all_critical_slots_correct:true,human_listening_used:false,audio_sha256}；
原有完整合成波形/时钟、native placement event_id/actor_id/full_content_preserved 与 dialogue_role_evidence 字段仍需保留。ASR 不许只检查 Speech 标签；关键意思槽错误必须拒绝。ASR 回执必须绑定实际完整波形 SHA，不能给旧文本复用新音频。固定缩写展开属于文本规范化，原始转写、原始 WER 与规范化 WER 都保留。
540 条计划台词每条最多13个英语词；实际时长仍须逐条测量。场景层必须容纳2–4角色的整句，不得裁切、循环或加速凑10秒；无法容纳就缩小该原生验证实例或返回 duration_fit 阻塞。多角色重叠需要对渲染后的分源/混合语音做自动内容核对，不能只沿用干声 ASR。

## 第 3 回合仍需接入

1. SPEAKERS：完整 actor_speech_schedule/actor_uid→native_actor_id→pixel_instance_id→source-slot/event-bindings。0–8 每层 train/dev 各至少 2 个独立父世界；同人多段、轮流/重叠、画外、同方向不同人、移动再说话；0 人窗保留环境声+原生可见静默人物。ID 只当私有训练监督。同一 actor 在一个父世界内保持同一 voice_identity_id（VCTK speaker 或同一自然克隆参考），多段多文件用同一说话人的不同句子；默认同一世界的不同 actor 使用不同 reference speaker，至少8人场景须有8个独立可辨 voice identity，不能把换文件当换人；跨独立世界随机重新绑定兼容性别的声音和身体，不让音色固定对应某个衣色/底模。phone remote playback 不算本地 actor；megaphone live 仍是同一讲话 actor。
2. FRUSTUM：各 split 画内/画外都足量，按世界八方位均匀布置并记录可行相机相对格。pair_id+counterfactual axis、同源/同时间/同 actor/实际媒体不变量；改变相机方向旋转 FOA并重渲染，改变源位置重新 RIR。像素零基中心加 0.5，边界 [0,W)×[0,H)，近四边界样本。不得按真实 prior 分配。
3. QA-07/09/10：每目标每帧同一 native 上下文、同相机姿态、同 sensor pass 的 modal mask 与 target-only mask；严格 modal⊆target-only。全区间逐帧唯一 occluder ID、unknown pixels=0、相同相机/窗口完整证据和自动颜色区分备选；真实入画、完全遮挡再出现。此前独立重渲染的 masks 和 SceneNode 无 children 的尝试失败，不能改 v6 any/all 或门槛救样本。
4. 配额：四成员窗至少 5 条经严格绑定认证的逻辑题；每 member window 至少 4–5 条纯视觉合法候选，至少多个可辨人物/物品/颜色/空间关系布置。实际来源/许可通过后再规划多源类别。原 seed 10 窗 V/A/AV=14/206/169，目标140题需42/35/63，纯 V 缺28；四成员10/72=13.89%，距离≥20%仍差6.11个百分点。新能力探针不能补进原 seed 外推。
5. 人物物理：native MJaw/source world trajectory、skin/foot/toe 地面与完整接触、不穿模/悬空、步态位移/时钟与完整 motion 许可；8 实例/人数容量不等于人物动作已准入。
6. QA-21 固定34类不变；新增真实乐器/鸟/咳嗽等在域外不强行补未知/domestic。STARSS 乐器映射9、鼓掌2、笑4、脚步6、门7、敲门12；咳嗽、鸟无精确13类对应，面板跳过。干声“音乐内容”不自动说明播放，机制由原生 emitter 决定。

## 给 v7-human-roster（不改其检出）

10 类 lx 资产：可用 runtime、许可链及依赖/输出条款、实际 asset_kind 和 instrument_subtype、visual_gender、local mouth 与真实 emitter 点、native 附着/接触/站姿证据、instance semantic ID 和自动外观回读。资产文件存在与标准化完成不能代替准入；ISNet 分割 checkpoint/revision/许可待补。
Rocketbox：角色可换色区域与禁止改色的职业服装、同身体不同 instance 支持、站姿/走路许可与接触、声源点。role count 2–4 不以不同 TTS 文件数当 actor。
宠物鸟：最终 parrot/canary/具体品种 asset ID、口/喙声源点、许可与 native 能力。德国牧羊犬沿用 dog_bark，不根据声音猜犬种。
megaphone 两点明确区分：actor mouth 是身份/讲话监督，driver 是实际扩音 FOA 发声点；phone 同理区分本地 speech 与远端 playback。

本轮已只读接入 v7-human-roster 的 registrations/human_roster.json：30个 Rocketbox 底模/110小类有MIT副本对应和双端静态资格；完整episode仍未运行。LX10的视觉性别是 not_annotated，ISNet/源提交链未闭合，乐器/手掌/扩音器driver点未提供，不能用mouth统一替代这些声源点。同类多人严格绑定还需要至少两种原生可区分外观；同一LX资产同姿态同衣色仅复制两份并以左/右作答案，不能自动计严格绑定。

v7_spec 需在 source-input 接口选择 DRY_SOUND_ADMISSION.json 中 strict_v7_eligible=true 的实际 normalization.path / normalized_path，核对 SHA/原始ID/许可/当前源排除回执；不能继续调用 FormantSpeech 构造普通讲话或 eSpeak 内容题。声音文件在一个世界内绑定到 actor_speech_schedule，而非文件数量。manifest的 human_live_natural_tts 映射为已有 schedule 枚举 human_live_tts，同时保留自然模型与ASR完整回执；不得靠改真值枚举去放回旧formant输入。自然语音整句归一化后，对实际最终波形重新做ASR或提供可复算的内容等价链，不能复用不同SHA的ASR回执。完整窗口须通过实际监听端自动内容核对。

给 roster 的新增明确缺口：为 LX10 提供有来源依据的 visual_gender（尤其电话/喇叭），完整ISNet checkpoint/revision/许可及源提交回执；各实际非嘴部 emitter 点必须按原生几何回读，手鼓给具体子类型。当前40/120双端静态存在，不等于这些内容绑定接口已满足。调用 v7_roster_intake.intake_roster 可只读核对、输出静态候选，bind_physical_actors 允许同一variant多实例，但每个actor_uid/native_actor_id须唯一；完整episode标记仍为false。
