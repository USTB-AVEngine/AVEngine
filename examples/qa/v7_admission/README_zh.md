已接入全部 QA-01…QA-28 的权威候选收集，但只有证据满足原定义时才出题；当前默认连续生成器仍须显式传入 qa_evidence 才会调用此接口。原生验证和资产准入结论在任务外部报告中，不等同于正式发布。

`src/avengine/qa/v7_families.py` 直接调用未修改的 unified_catalog 和 semantic_questions。缺候选、真实干扰项、完整活动或遮挡证据时保留 deferred。QA-21 单独调用权威已有的固定域配置模式，公开域为原 34 类；其他族不改变事实策略、判据或判分器。`any_audioset_class_playback` 只保留为旧池复合占位，必须记录底层内容和扬声器机制，不能成为 QA-21 答案。

`v7_native_adapter.make_full_questions` 接受实际 native_visual_readback、native FOA 和私有 qa_evidence。证据必须包括：

- appearance_review.actors：人工/显式视觉复核的 field、value、label_zh、label_en、frame_refs 和 source。
- pixel_visibility_truth：真实 modal/target-only 同相机、同姿态四态读回；只有 visible_pixels 不够。
- events：逐事件真实干声 activity_intervals_samples_global、activity_status=observed、出处。不得用排程长度代替活动长度，不将现场人声改名为 speech_playback。
- foa_readback：实际十秒四通道 ACN_WYZX_SN3D_camera 文件以及保留的方向验证；事实中 channel_count 仍为 4，v6_stereo_receipt=false。
- post-sound 题另须逐事件实际湿声读回的 wet_intervals_global_s。缺失时拒绝相应题。
- semantic_manifest：项目作者脚本、完整合成波形读回、匹配转录、native_placement_readback 和 dialogue_role_evidence。不能将旧试点 trim/repeat 音频作为完整语义证据。
- protocol_scene：由真实干声活动得到的 100ms 半开标注和原生嘴部轨迹，不用未观测的移动插值补目标。
- 可选 question_episode_id：四成员组共用的问法随机化命名空间；native_window_id 仍保留实际成员窗口。只控制问题问法与选项顺序，不替代物理身份或真值。
- 可选 question_seed、family_counts：供权威问法与题族轮转；十四题和模态分配继续由 v7_production_contract 控制，缺额抛错。

公开 question_id、logical_question_id 和媒体路径必须使用与答案无关的中性标识；带 speakers_3 等答案含义的诊断目录不能直接成为模型输入。本次最终验证产物单独建立中性媒体别名，原始诊断目录仅作私有证据。

`v7_protocol.py` 单列 STARSS23_v0.2 面板，不替换 v6 QA-23 的事件定义。覆盖人数 0…8（溢出仅开放题）、同人计数、半开段、左截断、前三个锚点及全体平局检查、5° 死区、QA-25 原 15°/30° 评分信息。几何画内外不使用遮挡。QA-21 只有音乐播放和水龙头的严格映射，CLASS13 的公开域单列十三类。wording_variant=standard/alternate 是与答案无关的问法对照。

书面 v0.2 协议要求 QA-25 查询“锚点后”第一个整数秒，即 floor(anchor)+1。参考 AVModel 的 ceil(anchor) 在整数锚点处不同；这里保持书面协议，记录差异，不修改 AVModel 或旧评分器。调用方需要据此对齐训练侧。

准入工具示例（输入路径由调用者显式提供，输出必须是新目录）：

```bash
python -m avengine.qa.v7_asset_admission --pool POOL_JSON --official-metadata-zip FSD_METADATA_ZIP --mapping CLASS_MAPPING_JSON --humans NATIVE_ASSETS_JSON --output NEW_OUTPUT --normalize-research
python -m avengine.qa.v7_semantic_assets --script examples/qa/v7_admission/semantic_dialogues_v1.json --output NEW_SEMANTIC_OUTPUT
```

准入工具只读源数据。旧录音的逐 clip 许可与严格仿真监督资格分别统计；CC0/CC-BY 候选不自动变成仿真主线素材。研究归一化采用共同 -23 LUFS 目标及 -3 dBFS 真峰上限的常数增益，保留时长和切分；峰值受限或无法测得综合响度时单列。正式计数维持 0，直到许可证、来源、感知识别和原生资产/动作准入均完成。

人物 M2 输出用实际 Bip01 MJaw 的 muzzle 骨骼锚点读回，不用历史 UE 固定身高偏移代替。上游 MIT、已有运行能力、接触/动作合格、完整本地许可链分别记录；颜色变体和多人实例数均不等于身体身份数。


完全相同的公开题面在同一窗口仅计一次，保留其 MCQ/开放形式，不用多个采样帧补齐配额；同题面真值冲突会拒绝。去重后的纯视觉候选不足时，十四题模态分配抛错，交付只能保留候选池，不能宣称配额合格。
