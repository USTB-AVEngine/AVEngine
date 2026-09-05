# 多房间、当前 QA 与个性化时空记忆：源码和论文对照

2026-09-05。服务器集成分支 codex/multi-home-activity-integration；精修代码已合入 24c828a。本文是当前证据与后续研究设计，不是模型能力达标声明，也不新增题型注册、冻结 contract、baseline 或 gate。

当前结论：12 类稳定 QuestionSpec 都有实现、单测和保留的原生通过样例；新三屋实际完成的是 QS-002、QS-003、QS-012 的 research-only 数据联通。跨片段身份、个性化偏好更新、历史时空查询目前未实现。四人静坐 16 秒可检验出题和音视频绑定链路，不能据此证明长期记忆或真实社交行为覆盖。

**全部 12 类：下表中文例题是保留的原生样例的自然语言转述，并非声称每种都已在新房间生成。**

| 编号 | 当前能力 / type | 例题 → 答案 | 新三屋 |
|---|---|---|---|
| QS-001 | 外貌→是否发声 / appearance_to_speaking | 那只彭布罗克威尔士柯基发过声吗？→ 是 | 尚未重出 |
| QS-002 | 声音→外貌 / sound_to_appearance | 发出那声猫叫的猫是什么品种？→ 英国短毛猫 | 已跑通 |
| QS-003 | 谁先发声 / who_spoke_first | 谁最先发声？→ 边境牧羊犬 | 已跑通 |
| QS-004 | 发声者左右 / speaker_side | 第 0 帧的猫叫来自听者哪侧？→ 左侧 | 尚未重出 |
| QS-005 | 声音重叠 / overlapping_speech | 猫叫与狗的低吼/叫声重叠过吗？→ 是 | 尚未重出 |
| QS-006 | 发声期间运动 / speaking_while_moving | 发出猫叫时，那只猫在移动吗？→ 是 | 尚未重出 |
| QS-007 | 画外进入 / offscreen_to_onscreen | 目标 source1 从画面哪侧进入？→ 左侧 | 尚未重出 |
| QS-008 | 发声时遮挡状态 / occlusion_while_speaking | 第 0 帧发出猫叫的个体是什么可见状态？→ 清晰可见 | 尚未重出 |
| QS-009 | 遮挡后重现 / reappeared_after_occlusion | 目标完全被遮挡后重新出现过吗？→ 否 | 尚未重出 |
| QS-010 | 遮挡物身份 / occluder_identity | 第 45 帧什么挡住了目标？→ Round_Table_Chair_01 | 尚未重出 |
| QS-011 | 部分遮挡转完全可见 / became_clear_after_partial_occlusion | 目标从部分遮挡变为清晰可见过吗？→ 否 | 尚未重出 |
| QS-012 | 外貌→说话内容 / appearance_to_spoken_content | 受控外貌标签为 male 的人物说了什么？→ It's eleven o'clock. | 已跑通 |

例题中的 source1 和资产名用于对照保留的 evaluator 记录，不建议直接作为面向模型的自然语言描述。实际问句要给出可观测且唯一的指代。单纯拥有 instance_id 不意味着模型或人能识别目标。

稳定类型注册位于 src/avengine/qa/question_spec.py:43-168。四人 adapter 中 card13/card14 对应 QS-012/QS-002，不是新增 QS-013/QS-014。三屋通过意味着原生事实、音频和答案生成可联通；并没有运行待评测模型来得到其正确率。

**本轮三屋真正生成的三道题。**

以下来自 polished_v3_final 的三套正确姿态 SPEAR/UE 捕获及配套 RLR 音频，不是尚在验收中的 v6 精修资产：

- 谁先发声？→ 蓝衣人物，seated_rocketbox_male_adult_01_blue_v1。
- 蓝色上衣的人说了什么？→ Ask her to bring these things with her from the store.
- 说出上一句的人上衣是什么颜色？→ 蓝色。

三套房间当前使用同一批四个人、同一套台词与顺序，所以这三题的答案相同；变化的是空间布局、原生像素、遮挡与双耳混响。这可以检查跨房间运行，并不构成人物、台词和行为的独立多样性评测。四句都完整进入音轨，另外三人对应的不同台词也已保留在事件绑定中；当前 adapter 选出的这三题围绕蓝衣人物。

衣色边界：四人 adapter 复用了稳定 QuestionSpec 的通用 coat_value，把受控人物衣色写进 coat_profile.value 和 Fact attributes.coat_value。当前 voice binding 没有显式 color 字段，因此 actor_color 从资产 ID 的 _blue/_pink/_green/_white_v1 后缀获得作者标签（tools/qa/adapt_four_speaker_research_qa.py:94-101,289-345,518-529）。原生像素证据确认的是人物可见/遮挡，不会自动证明上衣区域足够可见、像素确实为该颜色或模型识别正确；正式衣色能力评测仍需真实图像审查。资产属性作为合成真值是合理的，不能将可见性通过进一步冒称细粒度视觉识别已验证。

**17 张设计卡与当前实现的关系。**

下表完整列出 docs/roadmap/QA_TYPE_DESIGN_V3_20260830.md 的题卡。其历史模型分数和“在产”描述不自动等于本轮已重测；本轮只确认上面的 12 类源码和保留产物。

| 卡 | 题义与例题（设计例，不是本轮新增产物） | 实现关系 |
|---|---|---|
| 1 | 发出第二声的狗，片尾相对听者在哪边？→ 左 | 新错时题义，设计稿 |
| 2 | 发出第二声时，声音在哪边？→ 左 | 即时空间定位的音频对照，设计稿 |
| 3 | 第一声来自左边还是右边？→ 左 | 复用 QS-004；已有旧线产物 |
| 4R | 第 2 秒谁离你更近？→ 黄毛狗 | 给定时刻的视觉对照，旧挖矿器需按新定义重测 |
| 5 / 5R | 叫声期间靠近还是远离？/ 叫停后片尾比发声时更近吗？→ 更近 | 原形态音频运动对照；错时版待实现 |
| 6 / 6R | 第二声期间在动吗？/ 第二声结束后动了吗？→ 是 | 前者复用 QS-006；错时版待实现 |
| 7 | 第 3 秒哪只在叫？→ 黄毛狗 | QS-001 的时刻锚定扩展，设计稿 |
| 8 | 黑白毛狗第一次叫在第几秒？→ 2.4 秒 | 新数值时刻题义，设计稿 |
| 9 | 先叫的是黑白毛还是黄毛？→ 黑白毛 | QS-003 选择目标 + QS-002 外观；有旧线实现，需新条件重测 |
| 10 | 发声时是否在动？→ 是 | QS-006 复用；与卡 6 原形态重复 |
| 11 | 画面中哪只狗在叫，还是都不是？→ 都不是，声源在画外 | 复用 QS-001/QS-008 事实；成对批量采样仍属设计 |
| 12 | 深色狗发出什么声音？→ 低吼 | 反向扩展，多源编排候选 |
| 13 | 蓝衣人说了什么？→ 某条唯一台词 | QS-012；本轮四人 research adapter 已联通 |
| 14 | 说某句的人穿什么？→ 某种衣色 | QS-002 反向绑定；本轮 research adapter 已联通；“没人说过”负例是音频对照 |
| 15a / 15b | 在场几只、几只叫过？/ 一共几声？→ 3 只、2 只 / 4 声 | 数量组合扩展 / 音频计数对照；非稳定 QS 新编号 |
| 16 | 最先叫的目标，片尾是什么遮挡状态？→ 完全遮挡 | QS-003 + QS-008 组合；链式采样与过程判分待完成 |
| 17 | 第一段沙发边叫过的猫，第二段在哪？→ 餐桌旁 | 跨段记忆明确延后，当前未实现 |

卡 15a 的组合题不自动要求每一步严格跨模态；卡 4R/5/6/10 等单模态可答形态保留作对照。不能用题卡数量替代已实现类型数，也不能把素材、事实字段存在当作相应采样器和模型评测已完成。

**两篇论文已核对的启发。**

Hear you are 使用单张 360 度图像和 10 秒双耳音频，在 Matterport3D/SoundSpaces 2.0 内构造场景，每场景一个声源；Table 1 的 9 个模板分属空间对应、相对位置、空间与语义消歧、可见性。多个同类物体和语义不匹配的声源可减少“听到狗叫就选狗”的捷径。它没有验证多人物跨片段个性化记忆。Table 4 里部分可见目标的相对位置题，仅视觉也表现很强，不能因为输入有两种模态就宣布二者都必要。[论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Ryu_Hear_you_are_Teaching_LLMs_Spatial_Reasoning_with_Vision_and_CVPR_2026_paper.pdf)

VoxParadox 用 2,000 道、10 类题制造台词与实际发声特征冲突，覆盖音高、响度、语速、音域、语调、说话人匹配/计数以及年龄、性别、情绪标签。主要借鉴点是同时测声学真值准确率和追随误导台词的比例。它是副语言感知评测，不是空间记忆评测。附录 Table 3 中部分任务的人类真值判断也并不完美，因此 TTS 身份/风格元数据不能代替目标音轨的可辨认性检查。[论文](https://arxiv.org/pdf/2605.27772)

**建议新增的研究问题：以下全部是提案，不是当前已实现 QA。**

| 能力 | 可落地生活场景与例题 | 如何让证据可检验 |
|---|---|---|
| 同类目标的空间消歧 | 餐桌旁几人都静坐；“刚才说话的是左边戴眼镜的人，还是右边戴眼镜的人？” | 同类干扰物、不同位置；跨故事打乱声音与衣着绑定；当前片段可辨认 |
| 细粒度音频→视觉 | 几个人轮流说同一句；“说得最快的人上衣是什么花纹？”→ 细条纹 | 同台词控制，实际音轨语速与可见花纹证据；花纹/眼镜等新受控资产尚待制作 |
| 细粒度视觉→音频 | “戴红色手环的人第二次说话，比第一次音高更高还是更低？”→ 更高 | 姿态不必复杂；可見手环、事件锚和可测 F0；先验证音轨经过混响后仍可分 |
| 语义与声学冲突 | 声音用下降语调说“我的语调正在上扬”；“句尾音调怎样变化？”→ 下降 | 同一语义的不同声学版本；同时记录错误台词选项跟随率 |
| 短时跨模态追踪 | 厨房的人说完话后静默走到客厅；“刚才提醒开饭的人最后在哪？”→ 客厅 | AVEngine 规划人物与听者；查询窗口内不再次报出答案，不靠脚步声单独泄漏 |
| 跨片段身份关联 | 第 1 段看到并听到人物，第 2 段只听新句子；“这是谁？”→ 前段靠窗的人 | 故事内稳定身份/声音，故事间重排绑定；不同台词、跨房间混响；不是 ID 字段查询冒充重识别 |
| 个人偏好与更新 | 早餐明确说喜欢窗边；午后说今天改坐靠墙；“按他最后表达的偏好，应指向哪张椅子？”→ 靠墙椅 | 显式偏好陈述 + 时间顺序 + 当前椅子语义；不从外貌推测喜好；不需实际拉椅子动作 |
| 历史时空与证据来源 | 上午在书桌旁说“杯子留在厨房”，稍后在餐桌旁纠正为餐厅；“他最后在哪里修改了杯子的位置说明？”→ 餐桌旁 | 同时记人、时间、位置、陈述对象和更新；不把“他说的位置”混为物体真实位置 |
| 观测不足时的记忆回答 | 只看到某人离开视野，无后续观测；“他现在确定在哪？”→ 无法确定，最后在厨房门口 | 分开世界真值与模型可见历史；被挡住、画外本身不判房间失败 |

建议先做音高/语速/语调/匿名说话人等可控声学属性，以及真实像素可辨的衣纹/眼镜/饰物。情绪可作为感知语气扩展，不把一个合成风格标签当成真实内心状态。响度题要明确问“听者收到的响度”还是“说话人的发声强度”；距离、房间吸收和混响会改变前者，不能用 dry gain 直接给两种题相同答案。

个性化的最低要求是绑定到特定人的经历或明确偏好，并能在新情境使用和更新；仅固定身份不构成 personalization。跨段检索不自动等于长期记忆，需分别报告跨度、间隔、干扰事件数和跨房间程度。若故事只有几分钟，应称多片段/中短程记忆，不直接写长期。

**建议的最小工程接入。**

沿用 AVEngine 的 Episode、镜头/听者规划、声音事件、像素事实和 QuestionSpec。增加普通故事配置，串联多个 Episode，并用普通稳定实体主键维持故事内身份；事件记发生时间/地点，证据记实际可见或可听区间，显式偏好记生效/被纠正时间。先定义每个问答的观察截止时间，确保不会读到未来事件。真值生成可以读取引擎记录；被评模型只能得到规定的音视频历史和问题。

房屋适配分层：所有房间可复用人体姿态与已有行走；只有已经具有可靠家具语义/坐面/朝向的资产参与指椅子、静坐、家具参照题。自制房间可完整控制这些信息，扫描房间若没有对应标注就跳过该类编排。生产相机、人物/听者路径和问题采样统一由 AVEngine 负责。房间作者只交付几何和语义，作者近景只用于质量审查。

**怎样判断够不够验证论文主张。**

1. 工程正确性：已有源码测试、时钟/坐标/音视频 readback 和原生样例检验了数据链路。当前通过的例题不是模型成绩。
2. 可答性：从真实 UE 像素和最终双耳音轨验证目标细节、声学特征和时间锚可观察；内部真值精确不保证人能答。遮挡与画外按题目判断，不要求每帧所有人完整可见。
3. 模态贡献：沿用现有缺失模态评测，分别查看视觉、音频、完整音视频以及去空间线索条件。单模态对照题仍有价值，不把所有题硬称 AV 必要。
4. 记忆贡献：做当前片段相同但历史不同、答案不同的成对故事；比较完整历史、删去关键历史、打乱历史、人物历史互换和过期偏好。删去历史后只剩无法确定时，评测应允许正确表达不足。
5. 泛化：按人物/声源录音/场景划分，故事内保持身份一致、故事间重排；不能把同一个 dry utterance 换三个房间后分到训练和测试来假称泛化。
6. 报告边界：逐题族、逐时间跨度和真实/合成来源分别报数与模型结果。保持已有正式准入机制，当前三屋与新研究提案均不自动获得正式准入。

不把上面的评测条件包装成新的发布 gate，也不默认新增 baseline 文件或 hash 锁。它们针对具体混淆：只读台词、固定衣声映射、只看当前画面、使用未来信息和按房间混响猜身份。普通类型、主键和单测可保护数据结构，但是否存在感知捷径必须看实际输入及对照实验。

**源码审计原始记录。**

# QA catalog source audit

Authority: server /data/jzy/tmp/wt-multi-home-activity-integration at 4d65583. Stable QuestionSpec catalog: 12 types (QS-001..QS-012). Card13/card14 in the four-speaker adapter map to QS-012/QS-002; they do not create QS-013/QS-014.

| QS | Catalog type | Native example and answer | Facts / modalities | Status | Code |
|---|---|---|---|---|---|
| QS-001 | appearance_to_speaking / 外貌→是否发声 | 外貌属性“breed_id=pembroke_welsh_corgi”的个体是否发过声？ -> **是** | instances.attributes, sound_events; video, audio | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:728 |
| QS-002 | sound_to_appearance / 声音/内容→外貌 | 发出受控声音“clothov2_cat_meowing_v1”的个体，其 breed_id 是什么？ -> **british_shorthair** | event_sound_bindings, sound_events, instances.attributes; audio, video | implemented, unit tested, official native pass; three-room pass, research_only | src/avengine/qa/question_spec.py:827 |
| QS-003 | who_spoke_first / 谁先发声 | 谁先发声？ -> **Border Collie** | sound_events.start_tick, instances; audio | implemented, unit tested, official native pass; three-room pass, research_only | src/avengine/qa/question_spec.py:844 |
| QS-004 | speaker_side / 发声者左右 | 第 0 帧发出“clothov2_cat_meowing_v1”的个体在 Listener 左侧还是右侧？ -> **左侧** | active event, per-frame DOA; binaural_audio | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:867 |
| QS-005 | overlapping_speech / 重叠发声 | 受控声音“clothov2_cat_meowing_v1”与“clothov2_growling_barking_dog_v1”是否发生过重叠？ -> **是** | two event half-open windows; audio | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:901 |
| QS-006 | speaking_while_moving / 发声时是否运动 | 发出受控声音“clothov2_cat_meowing_v1”时，发声个体是否在运动？ -> **是** | event frames, per-frame moving; audio, video | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:924 |
| QS-007 | offscreen_to_onscreen / 画外→入画 | 受控实例“source1”从画面左侧还是右侧进入？ -> **左侧** | pixel states, entry centroid; video, pixel_visibility | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:963 |
| QS-008 | occlusion_while_speaking / 发声时遮挡状态 | 第 0 帧发出“clothov2_cat_meowing_v1”的个体处于什么遮挡状态？ -> **清晰可见** | active event, selected-frame pixel state; audio, video, pixel_visibility | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:1050 |
| QS-009 | reappeared_after_occlusion / 遮挡后重新出现 | 受控实例“source1”是否在完全遮挡后重新出现？ -> **否** | ordered pixel states in one Episode; video, pixel_visibility | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:1079 |
| QS-010 | occluder_identity / 遮挡者身份 | 第 45 帧是谁遮挡了受控实例“source1”？ -> **Round_Table_Chair_01** | native occluder IDs, static object registry; video, pixel_instance_visibility | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:1126 |
| QS-011 | became_clear_after_partial_occlusion / 部分遮挡→完全可见 | 受控实例“source1”是否从部分遮挡变为完全可见？ -> **否** | adjacent pixel states in one Episode; video, pixel_visibility | implemented, unit tested, official native pass; not materialized in three rooms | src/avengine/qa/question_spec.py:1100 |
| QS-012 | appearance_to_spoken_content / 外貌→说了什么 | 外貌属性“sex_or_gender_label=male”的个体说了什么？ -> **It's eleven o'clock.** | appearance, statement/transcript/language registry binding; video, audio | implemented, unit tested, official native pass; three-room pass, research_only | src/avengine/qa/question_spec.py:760 |

Live current-code validation of retained paper_ready_v3: 6 Episodes, 2,230 candidate cases, 12/12 minimum pass, paper-balance pass, five visual-canary pass. This is protocol execution evidence, not formal admission or paper-scale readiness.

## Capability boundary

- **persistent_identity: unsupported.** No cross-clip/cross-room identity mapping or query; instance_id and source_slot_id are Episode-local.
- **personal_preference_habit_update: unsupported.** No preference, habit, personalized profile, or update state in scoped QA source/config/schema.
- **voice_timbre_reidentification: unsupported.** Explicit sound/event bindings identify speakers; no voiceprint or timbre re-identification.
- **fine_grained_visual_attributes: partial.** Only six controlled registry fields: breed_id, size, body_build, life_stage, coat_value, sex_or_gender_label; no pixel-derived fine-grained recognition.
- **object_permanence: within_episode_only.** QS-009 detects full-occlusion then visibility inside one Episode; no cross-clip persistence.
- **historical_spatiotemporal_memory: unsupported.** Current queries consume one Episode Fact table. Same-clip frame/event order, including one 16-second clip, is not long-term memory.

Single-frame state and within-Episode frame/event order support current or short-horizon temporal QA. The 16-second three-room clips are not long-term memory.

## QA-v3 card boundary

The v3 document is research planning (docs/roadmap/QA_TYPE_DESIGN_V3_20260830.md:13-15). It lists 17 cards across main candidates, controls, combinations, extensions, and future work (:291-477). Several reuse QS facts; numeric bearing/time/distance, counting, delayed-reference variants, and cross-segment card17 are proposed meanings. Card17 is explicitly deferred and currently unanswerable without stable identity and multi-segment sampling (:471-476). Do not count these as 17 implemented QuestionSpec types.

## Verification sufficiency

Sufficient: catalog/protocol consistency, deterministic evaluator and rejection paths, per-Episode native Facts/registries/readbacks/pixel evidence, retained delivery validation, and actual three-room PCM/pixel binding for three types.

Insufficient for new themes: cross-Episode/room identity, preference or habit updates, voiceprint/timbre re-identification, pixel-grounded fine-grained attributes beyond six registry values, cross-segment object permanence, historical-query storage/evaluator, human/baseline evidence, and formal admission.
