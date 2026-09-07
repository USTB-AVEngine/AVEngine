# P10 批清单、覆盖、评测排列与后台执行

## 1. 文件与提交

权威树：48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration，
分支 codex/multi-home-activity-integration。本报告随P10实现提交；
提交后会补充干净提交上的整合验证记录。没有修改Claude审计器或其测试。

- src/avengine/qa/batch_manifest.py：运行前分配资产、登记外观、声音身份和固定画像；请求逐集独立，失败配额左连接保留；Episode/视觉变体/房间/路线/声音身份的连通分组切分。
- src/avengine/qa/batch_sound_pool.py：读取现有P7派生集与已登记事件PCM，按配置中的物种/对象语义关联。复用现有sound_events检测器，仅生成元数据，不裁剪、归一化或改写PCM。
- src/avengine/qa/batch_coverage.py：五态完整分母、实际目标/参照/全局问题范围、联合轴汇总、逐形式结构统计和声音来源。
- src/avengine/qa/evaluation_permutations.py：评测前循环排列、私有金标映射、公开输入白名单、位置与一致性统计。
- src/avengine/qa/batch_delivery.py：实际交付核对、现有审计器调用、原生条件实测、音频电平/活动窗、截图及整批自动汇总。
- src/avengine/rooms/conditioned_sampler.py：可选preallocated_sound_asset_ids_by_actor在选片时实际生效；不带该字段的既有请求路径保留。
- tools/dataset/build_qa_batch_manifest.py、build_qa_batch_coverage.py、build_qa_evaluation_permutations.py、run_qa_batch.py：对应CLI。
- tools/dataset/run_qwen_content_controls.py：消费预先生成的排列，支持2–6选项，保存语义ID、排列ID、实际题面及完整模型模板输入；旧原始回答和未排列映射保留。
- examples/dataset/qa_pilot_46_20260907.json：46格配置，四家族各6组合、A/B/C各6组合、四家族各1个单活跃源。
- 五个test_qa_batch_*.py及test_qa_evaluation_permutations.py、TOOL_INDEX与当前执行记录。

颜色直接取登记的top_color、coat_profile.value、finish/body_color，未知保持未知。
声音按原始speaker/file身份分配，再在其合法片段中选择；运行过程中不改身份或画像，
也不使用共享配额计数器。GPU和RPC端口是执行资源，允许启动前依据实际资源调整。

## 2. 验证与修复

已完成的分项验证：

- 父整合相关套件65 passed / 0 failed（2.57s）：
  tmp/p10_parent_related_tests_20260907_v3.log。
- 最后声音来源/五态回归12 passed / 0 failed：
  tmp/p10_coverage_tests_parent_20260907_v1.log。
- 最后固定身份预算与后台执行回归24 passed / 0 failed：
  tmp/p10_batch_budget_runner_tests_20260907_v1.log。
- tool index 1 passed / 0 failed：
  tmp/p10_tool_index_tests_20260907_v1.log。
- 分项有重叠，不把这些计数相加当唯一测试总数。完整提交上的唯一计数另记。

关键修复与原始失败边界：

- 最早的coverage读取遇到allowed_event_classes为null；已按真实可空字段处理。
- 统计核对纠正了“所有题型都用外观值作结构基线”、缺少声音元数据就把设备台词题
  记成不适用、缺证据与规则延期混记的问题。现在消费P8已经由共用
  answerability.structural_baselines生成的逐形式结构；缺结构明确unmeasured。
- 源码接线检查发现后台wrapper会预创建controller输出目录，而真实控制器要求
  fresh输出。已改为wrapper attempt_01内的新episode子目录，并用会拒绝既有输出
  的fake controller回归，不把“进程能启动”当验收。
- 一个子验证入口未设置项目PYTHONPATH，报
  ModuleNotFoundError: No module named 'soundfile'；
  使用PYTHONPATH=src:tmp/native_python_addons_v1后通过，没有安装或修改全局依赖。
- 44/46固定请求在声音重放检查中找到合法选片；两条repeat设备配对报
  clip_budget_or_transcript_candidates_exhausted。
  进一步按原片长算出的最小排程分别13.4080625s、13.42s，超过保留尾段后的13s。
  实现现在提前保留这两个确定的失败配额，不替换已分配身份，不重复启动必败作业。

## 3. 实际产物与核对

首批五段覆盖的最终输入：
tmp/p10_coverage_20260907_v1/input_manifest_with_sound_origins_v1.json。
对应P8根目录tmp/p8_parent_final_20260907_v3，仍为73道有效语义题、47条延期，
合计120个题型请求；五段有效题型并集21类，没有混入三个补充样例改变分母。

最终覆盖：
tmp/p10_coverage_20260907_v1/coverage_v10/
（coverage.json、coverage.csv、structural_baselines.json、provenance.json）。

详细分母59资产×7房间×24题型=9912；联合QA×声源类别×家族×room_id为504格。
详细资产目标行当前为produced46、deferred_by_rule96、not_applicable_by_definition1400、
interface_not_implemented1197、evidence_missing_or_unsampled7173。23个全局问题
另有明确scope，参与类别进入联合轴；语义question_id去重，不把参与资产行当新题。
9个挂墙/吊顶资产的接口缺口保留在正交interface_gap字段中（1512个资产/房间/QA行），
其中自身运动题仍按题义不适用。声音原始路径、身份与上下文声音分别记录；
vocal_classes_c_v2缺少可定位的源音频报告，原始声音来源保持unmeasured。

评测前排列：
tmp/p10_eval_permutations_20260907_v2/
（public_permutations.json、private_gold_mappings.json、permutation_report.json）。
70个可运行MCQ、202个排列，覆盖K=2/3/4/5；另3道仅Open可用。
K=6由单测验证。公开输入不含truth/profile/evidence/gold映射；未运行模型，
一致性为unmeasured。一个排列或重复的同一排列不计为一致性证据。

分组切分：
tmp/p10_grouped_splits_20260907_v1/。
原五段中4段因房间/原始声音身份相连，只能保持在同一组；C段缺声音来源，未分配。
实际示例为train4/eval0/unassigned1。46条预分配也相连为一个组，预览为train46/eval0；
这不是已运行数据，也不构成独立训练/测试集，工具没有为了80/20比例拆开连通组。

46条预分配检查：
tmp/p10_pilot46_manifest_20260907_v2/；
tmp/p10_pilot46_preallocation_replay_20260907_v1.json；
tmp/p10_pilot46_repeat_budget_diagnostic_20260907_v1.json。
最终后台输入会在P10提交后的干净代码上重新生成。495条可用声音元数据来自
296条P7语音及199条现有非语音事件；原始登记分母与1175条事件排除原因保留。
原PCM未改。动物和设备检测阈值仍是既有研究占位参数。

真实收口验证：
tmp/p10_existing_delivery_review_20260907_v2/，
tmp/p10_existing_batch_summary_probe_20260907_v1/。
读取已有酷家乐16秒原生片：实际PCM为256000×2、16kHz，峰值0.101615034
（-19.86084dBFS）；已有审计器运行完成，实际条件、起点/查询帧截图、
9912格覆盖、分组及待试听记录均生成。此为保留片段上的集成验证，没有重拍，
不计入46段。亲自查看提取的起点帧：音箱在左、Beagle在右下，犬足部位于画面下缘并有裁切；保留原片的外观证据边界。本项没有代填人工试听。

## 4. 未完成与证据边界

- P10实现和准备不等于46段已交付；后台批次另有进度、失败和实际产物。
- 首批设备遵守P4/P11“先放地上”，候选为50个落地资产；全部59个资产仍在覆盖分母。
  初版预分配误含挂墙/吊顶对象，v1保留为启动前诊断；画像类别/角距配额未改。
- A房和酷家乐各一条device+device repeat固定身份超出排程预算；这两格保留失败与缺额。
- 实测条件逐字段报告。facts没有原生body-proxy点时，其LOS仍unmeasured；
  不用计划条件补成achieved，也不据此宣称人工可答性或模态必要性。
- P7已由owner明确接受；5条最终成片的人工抽听是后续批次记录，尚未伪填。
- 无正式数据集准入、规模扩量或模型评测声明。

## 5. Claude/执行接口

- 每个request可直接交给tools/studio/run_qa_episode.py，画像/资产/声音身份已固定。
- 后台入口：
  PYTHONPATH=src:tmp/native_python_addons_v1 <project-python>
  tools/dataset/run_qa_batch.py --manifest <batch_manifest.json>
  --output <fresh-output> --max-parallel 4。
- wrapper保存request.json、attempt_01/stdout.log、stderr.log、outcome.json；
  原生controller输出位于attempt_01/episode/{plan,capture,delivery}。
- 总progress.json为原子快照，events.jsonl为追加事件；outcomes.json保存全部结果。
  后台结束自动生成summary/的覆盖、失败格、音频/活动、切分和5条待试听记录。
- 原始请求未满足的配额保留在batch_outcomes.json；重试须使用同一逻辑Episode ID、
  新输出根和--episode-id，不覆盖旧attempt或改画像补数。
- 审计器只调用既有CLI，不改其文件；阈值保持placeholder。
- permutation_id和semantic question_id分开，私有映射用于评分与一致性回算，
  不作为新语义问题，也不进入模型公开输入。

## 6. Owner裁定

owner已于2026-09-07接受P7当前处理结果，记录见
tmp/p7_prepared_audio_v3/owner_acceptance_20260907T020953Z.json。
随后明确授权：完成P10、推送当前分支，再后台运行已授权46段并提供日志和预计耗时。
本次据此推送当前codex分支；main合并、Studio切换和超出46段扩量仍不在范围内。
