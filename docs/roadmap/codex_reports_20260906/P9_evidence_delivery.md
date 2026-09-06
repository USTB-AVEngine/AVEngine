# P9 evidence and delivery closure

## 1. 修改文件

- `src/avengine/rooms/qa_evidence.py`
  - `derive_actor_occluders` 只读取 EvidenceContract 的 modal/target-only 掩膜和 pixel truth；兼容 `depth_derived_modal_semantic`/`modal`，支持稀疏显式帧，不补未知帧。
  - `build_pixel_appearance_review` 读取真实 PNG 或 `rgb.npy`，在 native instance mask 内比较登记的 `top_color`、`coat_profile.value`、`finish`；不从 `asset_id`/`display_label` 推断像素外观。
  - 人使用 top color，动物使用 coat，设备使用 finish；review 记录保留实际 RGB frame refs 和登记来源。
- `src/avengine/rooms/qa_delivery.py`
  - `finalize_qa_episode` 从 P1 NeutralReadback、EvidenceContract 文件和 P6 audio report 组装同一 raw bundle；UE 使用真实 `frame_readbacks.json`，Habitat 使用真实 `frame_records.json`，不伪造读取路径。
  - 支持传入 plan root 或 capture root；Habitat 无共同计划时从实际 case manifest/capture receipt 建立最小中立计划。
  - finalizer 先验证证据契约与实际双耳 WAV，再调用既有 `normalize_episode_bundle` 和 `generate_unified_questions`；未修改 `normalize_episode_bundle`。
  - 对 P6 IEEE float32 WAVE 使用项目实际 readback 校验并注入 facts 的真实音频路径；旧 UE 输入保留其实际无损双声道 WAVE。缺少 native video 时，若有完整 RGB/PNG capture 则先生成 derived visual master，再独立 mux 音频；没有原生视频或完整 RGB 时才将 preview/export 标记为 `not_run`。
  - 最高层 controller 在没有预供 audio report 时，UE 走现有 `build_audio_command`，Habitat 从共同计划显式物化 AudioProgram 并调用 `python -m avengine.cli m5 render-current-mp3d-dynamic-audio`；不依赖临时拼接脚本。
  - Habitat endpoint 绑定只接受 plan/materialized tracks 或 P1 neutral_readback.entity_identities 的 source_endpoint_id；common plan 缺少顶层 endpoint 时从 neutral identity 补齐，缺少权威绑定则失败，不猜测 {actor_id}_mouth。
- `tests/unit/test_p9_evidence_delivery.py`
  - 新增 sparse mask、动物/设备 RGB 外观、UE/Habitat 同一 raw bundle 和 derived video/mux 路径测试。
- `docs/roadmap/codex_reports_20260906/P9_evidence_delivery.md`
  - 本报告。

实现提交为本报告首次加入分支的提交，可用 git log -1 -- 本文件定位；父 Agent 已审阅真实产物后统一提交。未 push/merge，未修改 normalize_episode_bundle、Claude 审计器或其测试。

## 2. 验证

- 权威工作树：`/data/jzy/tmp/wt-multi-home-activity-integration`。
- 项目 Python：`/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`，`PYTHONPATH=src:tmp/native_python_addons_v1`。
- 最终相关集合 tests/unit/test_p9_evidence_delivery.py、test_qa_native_occluder_evidence.py、tests/test_question_driven_rooms.py、test_p6_audio_unification.py：**28 passed、0 failed、0 skipped，15.54秒**；日志 tmp/p9_finalize_20260907/parent_final_tests_v3.log。此前完整21项通过；父Agent增加human palette和6项coat语义回归后，最终未deselect任何项。
- `py_compile`（`qa_evidence.py`、`qa_delivery.py` 和 P9 测试）：通过。
- 两个最终输出的 `evidence_contract_validation.json` 均为 `status=pass`、`formal_certification=false`；A 为 240 帧，MP3D 为 30 帧。
- ffprobe 核对 A `preview.mp4` 为 240 帧/15 Hz/16.0 s 视频和 2 声道/16 kHz/16.0 s 音频；MP3D `visual_rgb.mp4` 为 30 帧/15 Hz/2.0 s 视频，`preview.mp4` 为同样视频并带 2 声道/16 kHz/2.0 s 音频。
- MP3D P6 mixture 使用项目 `read_float32_wav(..., verify_sidecar=True)` 读取为实际 float32 双耳输出；A 使用其原始输入的实际无损双声道 WAVE，二者 facts 的 sample count/clock 均一致。
- P5 common plan 无 audio report 的自动路径已实跑：生成的 AudioProgram 事件 endpoint 为 source2_emitter、source1_emitter，receipt 为 status=research、2 events、80 keyframes；没有使用类别名推断。

## 3. 原生收口产物

MP3D 使用 P4 的真实 Habitat capture `tmp/p4_beagle_speaker_parent_registry_only_v1` 与 P6 corrected `tmp/p6_audio_20260907/mp3d_unified_v3/research_report.json`：

- facts：`tmp/p9_finalize_20260907/mp3d_final_v7/facts.json`
- questions：`tmp/p9_finalize_20260907/mp3d_final_v7/questions.json`
- contract validation：`tmp/p9_finalize_20260907/mp3d_final_v7/evidence_contract_validation.json`
- appearance review：`tmp/p9_finalize_20260907/mp3d_final_v7/appearance_review.json`
- actor occluders：`tmp/p9_finalize_20260907/mp3d_final_v7/actor_occluders.json`
- visual：`tmp/p9_finalize_20260907/mp3d_final_v7/visual_rgb.mp4`
- preview/export：`tmp/p9_finalize_20260907/mp3d_final_v7/preview.mp4` 和 `tmp/p9_finalize_20260907/mp3d_final_v7/export/manifest.json`，均为 pass。
- `visual_rgb.mp4` 由 P4 实际 `rgb.npy` 按 capture clock 编码，`preview.mp4` 再独立 mux P6 双耳音频；视频 encode 和 mux 均使用 fresh derived 输出，没有覆盖 native capture 或 P6 WAV。
- beagle 登记外观为 `standard_tricolor`，实际 native mask 内 30 个帧的检查均为 `not_observable`（`warm_brown=0`，不足以证明三色 coat）；音箱登记 `black_ash`，30 个帧均以实际 mask RGB 检查为 `reviewed/pass`。两条记录均带 registry source 和实际 RGB 数据。
- questions：24 类请求，**7 valid、17 deferred**；deferred 原因保留现有证据/事件边界，包含 beagle 外观不可观测、无入口转场、无唯一遮挡身份、无合法 post-sound window 或未审事件分段等。

A 同输入回归使用旧 delivery `tmp/qa_real_rooms_20260906/walk_pair_a_v6/delivery_final_v1/input_refs.json` 指定的原始 plan/capture/audio/appearance；没有使用 P5 新 A：

- facts：`tmp/p9_finalize_20260907/a_same_input_v5/facts.json`
- questions：`tmp/p9_finalize_20260907/a_same_input_v5/questions.json`
- contract validation：`tmp/p9_finalize_20260907/a_same_input_v5/evidence_contract_validation.json`
- appearance review：沿用 input_refs 指定的真实 `walk_pair_a_v6/evidence/appearance_review.json`
- preview/export：`tmp/p9_finalize_20260907/a_same_input_v5/preview.mp4` 和 `tmp/p9_finalize_20260907/a_same_input_v5/export/manifest.json`，均为 pass。
- 240 帧、15 Hz、256000 samples；缺少 P1 sidecar 时从实际 UE `frame_readbacks.json` 写入 fresh derived `neutral_readback.json`，不修改原 capture。
- questions：24 类请求，**18 valid、6 deferred**；这是父 Agent 在已提交的 P8 055c054 上重跑的结果；问题数取决于该段合法候选与查询设置，facts 语义对照保持稳定。
- 与旧 `delivery_final_v1/facts.json` 的 time、source1/source2 root_positions、emitter_positions、moving、listener positions/basis、event actor/time/sound semantics、pixel visibility 逐项对照一致；`tmp/p9_finalize_20260907/a_same_input_v5/semantic_comparison.json` 中上述 checks 全为 `true`。
- 旧/新已知表示差异仅为 neutral source/yaw 元数据、registry species enrichment 和 P6 source activity 字段；这些不改变所对照的时钟、位置、事件或可见性语义。

P4 HM3D animal/speaker 实跑收口使用同一 capture tmp/p4_hm3d_animal_speaker_20260907_v5/capture 与 corrected v2 receipt：

- finalizer derived 输出：tmp/p9_finalize_20260907/p4_hm3d_final_p9_v1/facts.json、questions.json、evidence_contract_validation.json、visual_rgb.mp4、preview.mp4 和 export/manifest.json。
- EvidenceContract、RGB video encode、独立 audio mux、preview/export 均为 pass；240 帧、15 Hz、256000 samples、2 声道/16 kHz/16.0 s。
- 两个 actor 的真实 RGB mask review 均为 reviewed/pass：beagle 的 standard_tricolor 与 speaker 的 black_ash；音频事件绑定为 source1_emitter/source2_emitter。
- 输入 receipt 为 tmp/p4_hm3d_animal_speaker_20260907_v5/audio_endpoint_bound/research_report_metadata_corrected_v2.json；项目 unified receipt validator 和实际 float32 双耳 readback 均通过。该 receipt 的 sources.position_authority 为 P1 NeutralReadback entities[].emitter，listener.authority 为 P1 NeutralReadback.camera[0]，input_neutral_readback_path 指向 P4 capture 的 neutral_readback.json。

### P5 common-plan 自动入口实证

复用已有 P5 common plan/capture tmp/p5_sampler_20260906_v1/mp3d_shared_plan_v5，不重捕获且不传入 audio_report：

- 自动生成的 AudioProgram：tmp/p9_finalize_20260907/mp3d_auto_p5_common_v2/audio_program.json；P1 endpoint 来源为 capture 的 neutral_readback.json#/entity_identities，实际事件绑定为 source2_emitter、source1_emitter，没有 source1_mouth/source2_mouth 猜测。
- 自动 CLI receipt：tmp/p9_finalize_20260907/mp3d_auto_p5_common_v2/audio/research_receipt.json，status=research、2 个事件、80 个 keyframes；sources.position_authority 为 P1 NeutralReadback entities[].emitter，listener.authority 为 P1 NeutralReadback camera[0]，input_neutral_readback_path 指向同一 P1 文件。
- 随后的 finalizer derived 输出：tmp/p9_finalize_20260907/mp3d_auto_p5_common_v5；EvidenceContract、derived RGB video、独立 mux、preview/export 均为 pass，facts/questions 为 240 帧/15 Hz/256000 samples，15 valid、9 deferred。
- finalizer 的 contract_bundle 保留 P5 capture 的真实 frame_records.json 和 neutral_readback.json，音频路径指向上述自动 CLI 生成的真实 binaural mixture。

### 父 Agent 的酷家乐与跨类别颜色检查

父 Agent 使用正式P3酷家乐包/P5 sampler完成 tmp/matrix_kujiale_humans_20260907_v1：当前native map、实测floor、静止85°相机、两个人形、240帧/16秒，RGB中两人的上衣清楚可见，前景植物遮住蓝衣角色的部分腿部。原生capture与P6实际双耳音频均完成；最终delivery_v2/{facts.json,questions.json,appearance_review.json,preview.mp4,export/manifest.json}通过P9，12 valid、12 deferred，视频/音轨均16秒/2ch。父 Agent 查看实际首帧，未记作人工验收。

该实跑发现新增通用palette把裸露手臂的warm_brown当作上衣竞争色，导致蓝衣2763像素与皮肤2910像素竞争并误判不可观测。人类上衣现在复用原inspect_coarse_top_color的颜色域、512像素与1.6倍优势规则，没有降低旧人类门槛。新delivery_v2中两人均按实际mask RGB reviewed；旧delivery_v1保留诊断。

动物按登记的standard_black_white、standard_red_white、standard_red、standard_yellow、standard_blue和standard_ruddy等coat语义比较颜色组成；British Shorthair的standard_blue为灰蓝毛色，不直接套人类的高饱和蓝衣区间。未知取值明确记registered_appearance_value_classifier_not_implemented / interface_not_implemented。新色彩阈值均标placeholder_coarse_color_only；finish只作粗颜色核查，不认证木纹、纹理或精细phenotype。

实际补充：tmp/p12_native_capture_human_british_20260907_v2/native_capture_readable_v1/parent_registered_color_review_v1.json 直接用该capture neutral里观测到的asset/slot identity和正式registry作外观声明，读取真实rgb.npy与mask。Blue human和British standard_blue均在3个所查帧reviewed：首帧蓝衣968像素（旧512门槛），猫灰蓝成分366/434像素。父 Agent查看了按正确RGB顺序导出的原生contact sheet；猫仍触及下边界，不宣称完整体型或人工接触认证。

### 20260907 酷家乐动物/设备自动收口

`finalize` 自动音频现在依据 RoomPackage 的实际 renderer 分派；
`plan_coordinates=renderer_neutral` 仅是共用坐标声明，不能当作 Habitat。
中立计划缺 renderer 会明确失败。`build_audio_command` 新增可选
`capture_root`、`plan_root`，由 finalizer 传入实际选中的契约与计划目录，
并读取其中真实 NeutralReadback，避免把重捕获重新指向旧失败 capture。
P6 同时修复了 explicit neutral + 旧 cache 的输入适配，格式不变。

真实新路径：`tmp/matrix_kujiale_beagle_speaker_20260907_v2/`，当前酷家乐包、
measured floor、新建可逆 stage、240帧/15Hz/85°静止相机。原 stage 缺书架
音箱 Content，复制 task stage 并只补入已 NativeLoadObject 验证的 speaker
subtree；未改共享 Content 或 map。首次 clone 缺外部 `../plugins`，启动
报 `Unable to load plugin 'SpCore'.`，补原有插件引用后新 `capture_retry_v3`
通过。该 capture 包含真实 beagle/音箱根和 emitter、完整像素证据。

最终 `delivery_v3/` 不传 audio_report，自动P6双声道、P9合同/preview/export
均 pass；音画均16秒，8 valid/16 deferred，原始24类分母保留。父查看实际
0/120/239帧：音箱可见；beagle靠画面下右边缘，故像素毛色检查仍
not_observable，未当作完整外观验收。两次自动音频失败分别为
`ValueError: Habitat actor 'source1' has no authoritative source endpoint (plan, materialized track, or neutral entity_identities)`（错误renderer路由）
和 `frame readbacks must contain camera and emitters`（neutral/cache适配），
旧 delivery_v1/v2 保留，最后使用正确源目录完整运行。

相关35单测通过，实际PCM/契约核对在 `delivery_v3/parent_validation.json`。
P4原点修复后的最新HM3D/MP3D收口另见
`tmp/p4_rigid_origin_captures_20260907_v1/{hm3d,mp3d}/delivery_v1`，旧COM居中
素材保留诊断；经实际camera/clock/emitter逐值相等后复用原P6音频。

## 4. 未完成与边界

- P4 MP3D 原始 capture 没有 native video master，但提供了完整真实 `rgb.npy`；P9 已按真实 clock 生成 derived `visual_rgb.mp4` 并独立 mux 音频，因此当前 MP3D preview/export 已通过。native video master 本身仍不存在。
- MP3D actor occluder 记录保留 native mask 交集的 unresolved 计数；未识别静态几何不会被命名成家具或声源。
- 未运行人工试听、模型评估或正式 dataset admission；结果保持 `research_only`、`formal_admission=false`。

## 5. 后续接口

- `build_pixel_appearance_review(capture_root, plan, *, frame_stride=15, asset_registry=None)`：登记外观值来自 `realized_attributes` 或 source asset registry；像素证据只来自真实 RGB 与 native instance mask。
- `finalize_qa_episode(episode_root, derived_root, *, repository, request=None, audio_report=None, appearance_review=None)`：`episode_root` 可为 UE plan root/capture root 或 Habitat capture root；`audio_report` 对 P6 可直接传 corrected unified report。返回 facts/questions、contract validation、preview/export 状态和实际输入路径。
- raw bundle 使用统一字段：`plan`、`actors`、`frame_readbacks`、`pixel_visibility_truth`、`audio_program`、`audio_readback`、`research_report`、`appearance_review`、`occluder_evidence`、真实 source paths。坐标和时钟以 P1 NeutralReadback 为权威。
- 视频辅助接口从实际 `rgb.npy` 或完整 PNG 序列复用 rawvideo encoder；先生成无音频 visual master，再用 `-map 0:v:0 -map 1:a:0` 独立 mux，不使用 `-shortest`，并以 ffprobe 帧数/帧率/音频声道和采样率复核。

## 6. 总前提冲突与 owner 决策

- 未改变房间/资产、声学、正式 admission 或 Claude 审计器边界。
- P4 的 Habitat 30 帧 capture 可用于真实 MP3D facts/questions；带视频的 P9 derived preview/export 已由真实 RGB capture 提供，但仍保持 research-only，不构成正式数据集准入。
