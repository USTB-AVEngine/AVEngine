# P8 候选采样、题面、逐形式拒出与判分

## 1. 修改文件与提交

修改 src/avengine/qa/unified_catalog.py、unified_scoring.py、tests/unit/test_qa_unified_catalog.py、test_qa_unified_scoring.py 与本报告。实现提交为本报告首次加入分支的提交，可用 git log -1 -- 本文件定位。共用 structural_baselines 从 P5 已提交的 avengine.qa.answerability 复用；没有修改 Claude 审计器或测试。

24 类均提供 _candidates_qa_01…24 / _emit_qa_01…24，先枚举轻量候选再均匀无放回抽取。QA03/22/23/24 语义固定为一个候选；QA09/11 是整个片段内的存在性问题，每个合法目标只贡献一个候选，不按遮挡持续帧数增加权重。QA21 已要求目标全片声类唯一，所以按目标枚举。发声停止后的候选严格绑定抽中的事件/查询帧，不在 emit 中转向另一个更容易的事件。候选不足不复制问题。

ID 包含目标、事件与查询/观察窗；coverage_by_qa 存列表，分别报告题型覆盖、有效题数和未满足配额。执行 uniform_in_legal_window，保留 frame0，显式非法帧报错。QA14 枚举合法目标对和查询帧；QA05 枚举真实事件对。公开 model_input 保持题面/选项白名单，画像、身份、几何、分歧和金标留在私有字段。

反事实在题面实际使用的同一锚点/查询窗比较所有实体，包含无自身声音事件的沉默竞争者；不能拿各角色各自的发声窗口充当同窗对照。Open 和 MCQ 使用各自真实答案域；缺失值不当不同。distractors_equal_gold 按所有可用竞争者同值拒出，QA01负例与集合/计数等语义例外保留。多数/唯一少数只记结构统计，不作为拒出原因；QA13也清理了旧的“任一竞争者同带就拒出”旁路。

QA13 三个视野内角带、目标出画延期、5°边界占位及Open数值形式分开；QA22选项数等于合法域；QA17写查询终点；QA10排除自身后单选项仅MCQ延期；QA08写起点；QA16写比较基准；QA07写选定入画帧。QA12对同一目标的多次讲话写明第几次，并分别报告台词归属和WER；不以全局发声时间提示替代身份关联。QA18只按真实 source activity 判断当前发声者并避开湿尾边界；已知全静音的 none 分支不要求虚构外观review，active target仍需要实际外观证据。

判分修正：负角度与一致的方向词可接受，矛盾方向拒绝；中文否定先于单字方向别名匹配。

## 2. 测试

项目 Python /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python，PYTHONPATH=src:tmp/native_python_addons_v1。

最终 **59 passed, 0 failed, 0 skipped，2.37秒**（catalog 52、scoring 7），日志 tmp/p8_parent_latest_tests_20260907_v7.log。覆盖多候选/不足/ID、同seed、frame0/非法窗、Open/MCQ分开、负方向词/中文否定、正确和错误答案、QA13三带/多数值/缺失值、QA22合法K、QA17终点/不换事件、QA10单选项、真实活动与尾音、同窗反事实含沉默实体、整片目标权重、多次讲话题面和none查询。

父 Agent 修正过程中保留了失败日志：v1 的旧QA17测试在一个无关较早窗口制造竞争者运动，按正确同窗规则已不成立；测试改为在真正合法查询窗内提供运动对照。新增测试的 deepcopy 名称导入遗漏在 v2/v3 报 NameError，已修复。v5 的测试曾假定第一个轻量候选可让 base generator 换到另一个事件；现改为验证实际能发出自身的候选，不放宽生产行为。最终无失败。

## 3. 实际样例、复现与亲自核对

最终输出根目录：tmp/p8_parent_final_20260907_v3/。每个子目录有 input_facts.json、questions.json、questions_repeat.json；总表 summary.json。原五段使用各自最新 delivery_final_v1/v2 的真实事实文件，同 seed p8-20260907 两次逐字节一致、无重复ID，每段保留24类覆盖记录。

| 段 | 尝试题型 | 有效题数/有效题型 | 未满足配额/延期 |
|---|---:|---:|---:|
| late_direction_a_v2 | 24 | 17 | 7 |
| native_apartment_v6 | 24 | 10 | 14 |
| vocal_classes_c_v2 | 24 | 13 | 11 |
| walk_four_b_v2 | 24 | 16 | 8 |
| walk_pair_a_v6 | 24 | 17 | 7 |

原五段的有效题型并集为21类，原始延期原因完整保留：

- late_direction_a_v2: QA-01=distractors_equal_gold; QA-07=no_entry_transition; QA-09=no_reappearance_transition; QA-15=no_distance_trend_during_event; QA-16=no_distance_change_after_event; QA-18=missing_source_activity_readback; QA-21=sound_class_option_domain_too_small
- native_apartment_v6: QA-01=distractors_equal_gold; QA-06=distractors_equal_gold; QA-07=no_entry_transition; QA-08=distractors_equal_gold; QA-09=no_reappearance_transition; QA-10=missing_occluder_identity; QA-11=distractors_equal_gold; QA-14=no_valid_distance_query; QA-15=distractors_equal_gold; QA-16=no_distance_change_after_event; QA-17=distractors_equal_gold; QA-18=missing_source_activity_readback; QA-21=sound_class_option_domain_too_small; QA-24=distractors_equal_gold
- vocal_classes_c_v2: QA-01=distractors_equal_gold; QA-06=distractors_equal_gold; QA-07=no_entry_transition; QA-09=no_reappearance_transition; QA-11=distractors_equal_gold; QA-12=transcript_option_domain_too_small; QA-15=no_distance_trend_during_event; QA-16=no_distance_change_after_event; QA-17=distractors_equal_gold; QA-18=missing_source_activity_readback; QA-23=event_segmentation_not_reviewed
- walk_four_b_v2: QA-01=distractors_equal_gold; QA-06=distractors_equal_gold; QA-09=distractors_equal_gold; QA-10=missing_occluder_identity; QA-11=distractors_equal_gold; QA-18=missing_source_activity_readback; QA-21=sound_class_option_domain_too_small; QA-24=distractors_equal_gold
- walk_pair_a_v6: QA-01=distractors_equal_gold; QA-06=distractors_equal_gold; QA-07=no_entry_transition; QA-09=no_reappearance_transition; QA-15=no_distance_trend_during_event; QA-18=missing_source_activity_readback; QA-21=sound_class_option_domain_too_small

另外三个明确标注来源的研究补充使有效样例并集达到 QA01–24（24类），没有把延期项当有效项：

- supplement_qa01：真实 P2 single_active 的混音/stem和同一 retained walk_pair_a_v6 视觉。source1_mouth_stem.wav 的PCM逐字节全零，source2非零；得到 source1=no 与source2=yes 两题。源证明在 tmp/p8_supplement_20260907_v1/qa01_p2_single_active/。
- supplement_qa09：从真实20秒 walk_four_b_v2 捕获生成新的17秒/255帧诊断前缀，保留全部四实体与全部四个完整语音事件（最后源事件16.5193125秒结束）。源文件、原Episode时钟/offset不变；不使用255帧后的可见性或外观审阅。原 source2 在前缀终点仍未重现，source3/source4 已重现；仅对前缀内有真实外观review的 source3/source4 出正例。原目录 tmp/p8_qa09_native_prefix_20260907_v1/；最终 native_prefix_preview_complete_audio_v2.mp4 实测255帧/15fps/17秒、两声道音轨17秒，lossless WAV为原PCM前272000 samples逐字节相同。初版preview的音轨提前结束只保留诊断，没有采用。
- supplement_qa18：同一 P5 a_plan_v3/capture_retry_v2 的 actual actors/emitters/camera 与 P6 a_unified_p5_v3_r15/research_report.json。父 Agent 把 raw bundle 三条native流逐值比对到源 normal_pass_readbacks.json，全部相等；实际P6 WAV为256000×2、16kHz，第一秒精确全零，frame0在活动与湿尾外，得到none。交叉核对 tmp/p8_parent_final_20260907_v1/qa18_actual_readback_crosscheck.json；该样例的P8证据可独立验证，P9收口集成另报。

我核对了最终问题、原始输入/活动区间、反事实字段、ID、配额和公开字段。上述是源/数值/生成链验证，未执行人工可答性认证、模型评测或模态必要性认证。

## 4. 未完成与证据边界

原五段本身仍只有21类有效题，并不因补充样例而变成24类全有效：QA01原全发声导致结构退化，QA09原全片触发实体全部重现，QA18原事实缺真实source activity。补充来源单独列出，不改原五段或把它们的分母缩小。前缀是已有原生观测的派生研究视图，不是新capture，也不是46段批次。其原source2在前缀没有外观review，因此不能作为被问的蓝衣目标；没有借用290帧review。

早期把旧 walk_pair_a_v6 facts 与另一段 P5/P6 音频叠加的QA18 overlay已标 invalid_overlay_cross_episode，排除出有效并集；最终用匹配native输入替代。P7 prepared 的人工记录、P9全流程、P10批级结构基线和正式评测仍由对应任务完成；单题通过不是可绑定/模态必要性认证。

## 5. Claude 接口

generate_unified_questions(raw_or_facts, qa_ids=None, seed=..., items_per_type=1) 返回 items、coverage_by_qa[qa_id]列表、candidate_counts、coverage_summary、unmet_quota_by_qa和deferred。每种形式各自有 form_status；只评估实际存在的 forms/model_input。绑定几何失败使用 binding_geometry_candidate_failed，查询源活动和湿尾原因分开。设备为运动目标记 not_applicable_by_definition，参照/竞争者仍参与。

source_activity_intervals_samples 是 episode sample clock 的半开区间，缺失不假定正在发声；wet-tail是独立读回。结构统计按题型×形式×实际K汇总，不能把多排列或同一语义题计成新样本。问题和生成输入均位于上述最终目录。

## 6. Owner 决策

没有修改阈值来填配额、没有按模型结果改选项、没有删资产或题型。本项没有新增owner决策。59项测试和24类样例只证明生成器实现与可追溯样例链；原五段的21类边界、P7人工试听缺口及正式admission要求继续保留。
