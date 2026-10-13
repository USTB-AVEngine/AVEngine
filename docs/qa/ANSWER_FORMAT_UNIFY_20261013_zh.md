本分支修好以后出题的题面与训练答案格式，原始真值、判分器、采样和门槛保持不变。

整数角度统一为 `-180` 到 `179` 的纯整数文本，禁止度号和正号；保留原有取整、包裹与完整精度。选择题有 1–26 个选项时用 A–Z，并写 `Answer with the option letter only.`；超过 26 个时用 1–N，并写 `Answer with the option number only.`。真实 QA-21 的 34 类属于数字分支；选项顺序和正确索引保持不变。八方向开放答案只用 front、front-left、left、back-left、back、back-right、right、front-right。QA-22 训练答案用实体数、空格、发声实体数，如 `2 1`；原始数组、规范真值、证据和 ID 保留。判分器没有修改，没有模型输出的字母换数字映射。其他时间、时间段、状态、类别、外观与转录沿用原有语义和序列化。

基线提交：`6d8f3d760cfd30d9c8d4dcf9302dae3376d49f0f`；本地分支：`codex/answer-format-unify-20261013`。仅在独立 worktree 修改，未推送、未合并、未切换生产检出。

从本分支的独立检出运行 `PYTHONPATH=src python tools/qa/generate_unified_questions.py --input FACTS.json --qa-ids QA-13,QA-25 --seed ORIGINAL_SEED --out NEW_DIR/questions.json`。`_question_item` 生成统一题面和训练字段；v7_validation 的独立重放仅将 MCQ 的期望 ID 改为同一 26 选项边界规则，canonical 真值与数值检查保留。原始 `truth.label` 是历史证据显示标签，不能拿它当训练 teacher。训练导出调用新的 `training_answer`，不能再直接把判分模块的旧 `canonical_teacher_answer` 当新 teacher。v7 的 `v7_families.export_items`、协议题、continuous_scene、native/content-binding 及最终 `collect_native_pool` 共用 `normalize_export`，开放原始 answer 保持 JSON 类型，训练读取 teacher_answer。

封存第一阶段的复制工具是 `PYTHONPATH=src python tools/qa/normalize_retained_v7_questions.py --manifest SEALED_MANIFEST.json --output NEW_COPY_DIR --scorer EXISTING_SCORER.py --source-commit ORIGINAL_COMMIT`。目录必须新建；复制工具复用原始事实和种子，生成训练 views、原始路径/提交号、逐题比较和媒体源定位文件。以后装训练数据读取副本的 training_views.jsonl；public JSON 的相对媒体路径按 MEDIA_SOURCE.json 定位，不能在副本里假造新媒体。phase 2 检出和运行作业本轮没有切换；生产方在 worker 封存后的停顿点再审核移植。

本仓库新增格式单测 9 项通过；真实旧/新出题对拍 5598 个视图通过，各入口另存 200 个题目形式的小批审计。26/27/34 选项边界、所有选项正确索引、角度无度号、QA-22 数组保留和 `2 1` / `[2, 1]` 同解析同得分均有测试。旧/新 fast-unit 都是 463 passed、36 skipped（历史事实不在独立检出）；定向 QA 回归旧/新都是 164 passed。840 道阶段一副本逐题同 ID、同原始真值、同得分且媒体路径可解析。统一出题对拍包含重复种子候选，2026 个视图对应 1042 个去重题目形式，不能当成 2026 个独立事实。

测试原始输出随提交附在 [ANSWER_FORMAT_TEST_OUTPUT_20261013.txt](ANSWER_FORMAT_TEST_OUTPUT_20261013.txt)；完整命令、输入出处、对拍逐行结果、小批产物和 diff 见任务产物目录 implementation_v1 下的 reports、logs、smoke、diffs。执行时关闭 CUDA、CPU 数值库单线程；所有 NEW_DIR/NEW_PACKAGE 都须使用新目录。
