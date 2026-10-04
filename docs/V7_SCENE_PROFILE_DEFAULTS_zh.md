已将后续v7默认种子采样接入总体画像，并保留独立0–8人覆盖；第8回合160正式alpha窗口/2240题及冻结输入保持原版本。新画像本轮仅生成请求与验证配置，原生渲染与动作准入尚未完成，不计新增正式题目。

改动均在48g指定worktree `/data/jzy/code/AVEngine-v7-spec-acoustics-20261003`、分支`claude/v7-spec-acoustics-20261003`：新增`src/avengine/timeline/v7_scene_profile_defaults.json`/`v7_scene_profile.py`，将`tools/timeline/prepare_v7_quota_seed_tables.py`默认改为`--scene-profile aggregate-profile`，保留显式`balanced-coverage`；`generate_v7_scenes.py --plan-only`识别新请求，实际候选入口拒绝原生准入未完成时回退旧两人/alpha；补JSON包数据、测试、工具索引和`V7_SPEC_zh.md`第19节。未修改qa目录、判分器、真值、验收门槛、原始数据和旧产物。只运行CPU（验证与测试各单线程），无CCI/GPU作业，不push。

来源为48g `/data/jzy/tmp/claude_hardbank_20260923/real_scene_profile_20261004/`的`v7_recommended_config.json`、`V7_RECOMMENDATIONS_zh.md`、`target_scene_profile.json`及`selection_audit.json`。审计记录72段统计录音（60视频、12仅音频），评测面板78段、留出房间18段均整段排除，交集0；SAVVY媒体0，仅公开文档。这里只读既有总体边际及审计，不打开真实媒体、不读取或复刻逐段布局/轨迹/帧/关键点/活动边界、不用波形拟合RIR/材质/RT60、不按真实评测成绩调参。评测集合变更须重做排除审计，当前来源校验见`evidence/source_audit.json`。

默认分布与抽样对照（百分比；“实测”均指新生成的请求，不是原生视频）：

| 项目 | 总体画像 | 默认画像支路 | 请求抽样结果 |
|---|---|---|---|
| 水平360°带可见人数0 | 0.52% | 0.52% | 0.59% |
| 水平360°带可见人数1 | 6.28% | 6.28% | 6.57% |
| 水平360°带可见人数2 | 23.77% | 23.77% | 24.27% |
| 水平360°带可见人数3 | 51.99% | 51.99% | 51.52% |
| 水平360°带可见人数4 | 17.44% | 17.44% | 17.03% |
| 已解析姿势：坐 | 58.76% | 58.76% | 59.24% |
| 已解析姿势：站 | 39.00% | 39.00% | 38.54% |
| 已解析姿势：走 | 1.25% | 1.25% | 1.29% |
| 已解析姿势：蹲 | 0.99% | 0.99% | 0.93% |
| 姿势未知支路 | 60.57% | 同质量独立随机化 | 60.36% |
| 同时讲话人数0，100ms | 41.39% | 同画像边际 | 41.36% |
| 同时讲话人数1，100ms | 54.06% | 同画像边际 | 54.11% |
| 同时讲话人数2，100ms | 4.51% | 同画像边际 | 4.50% |
| 同时讲话人数3，100ms | 0.05% | 同画像边际 | 0.04% |
| 讲话距离中位 | 1.870m | 1.870m | 1.873m |
| 画外（默认仅水平几何） | 75.94%全视锥几何 | 均匀方位×85°=76.39% | 76.28%水平几何 |

姿勢“已解析”占39.4263%，未知60.5737%按建议坐/站/走/蹲各25%独立随机化。因此全体默认姿势并非坐58.76%/站39.00%/走1.25%，混合后坐/站/走/蹲约38.31%/30.52%/15.64%/15.53%；它是明确的未知支路工程先验，不能声称来自真实总体。动作没有验收时只保留请求并拒绝执行，不用站姿冒充。群组、手势、衣色是未验证代理，仅弱先验/覆盖提示（最大权重0.1，不做硬拟合）；现有独立appearance种子与准入调色板保持独立。

人数代理是水平360°带检测数，不是85°裁剪内人数或真实占用人数。100ms同时讲话人数与10秒窗不同讲话actor数分开；画像支路最少3个潜在讲话身份属于工程设计，并非实测独立人数分布。说话距离用聚合分箱及分位数构造单调逆CDF，再独立抽样；原生嘴部距离与speech-actor-seconds权重仍须后续验收。方位继续均匀[-180°,180°)，HFOV85°，不按76%画外再加权；完整上下界、遮挡与FOA仍须由原生证据同步判定。房屋/材质/RIR/噪声未按此画像拟合。

默认90%画像父世界＋10%覆盖父世界，覆盖人数0–8确定性轮转；400训练父保留40覆盖父，0/1/2/3各5父、4/5/6/7/8各4父；100开发父保留10覆盖父，0为2父，其余各1父。可见人数0–8也覆盖，0讲话世界留可见静默人，并由原非讲话类配额要求环境声。1万条完整种子请求：7000训练、1500开发、1500测试；完整版训练700覆盖父。混合全体可见人数3实抽47.49%，画像支路51.52%，两者需单列。原生未验收余下房屋仍`awaiting_admitted_house`，不得派发。

产物在本轮新目录 `/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round9_scene_profile_20261004/`：`profile_seed_tables_v1/`含小集400/100及完整版表，训练/开发前缀逐字节一致；scene_id/scene_seed/各目的种子/房屋划分/成员/切窗均与输入基表相同；输入只接受原v3或平衡v4，不静默覆盖v5诊断或冻结alpha配置。完整对照`profile_default_comparison_v2.csv`，分支/频数/距离/方位及前缀证明`evidence/profile_sampler_validation_v1.json`、`evidence/profile_comparison_v2.json`。旧v1对照表姿勢汇总包含覆盖父，新v2明确限制画像父范围，旧诊断文件保留。

48g命令：`python tools/timeline/prepare_v7_seed_tables.py --houses <round8/houses_alpha_registered.json> --output <round9/base_seed_tables_v1>`；`python tools/timeline/prepare_v7_quota_seed_tables.py --previous <round9/base_seed_tables_v1> --output <round9/profile_seed_tables_v1>`（默认profile，无额外开关）；`python tools/timeline/generate_v7_scenes.py ... --limit 400/100 --plan-only`（500/500）；`python -m pytest -p no:cacheprovider -m fast_unit`。具体路径/脚本/时间/测试结果见`logs/`与`HANDOFF.json`。配置代码仅单机CPU验收，没有新画像原生画面/动作/RIR或正式出题验证；完整版仍未可生产，第8回合站姿alpha继续用44235e5和既有种子。

提交号、完成时间（新加坡）与所有本轮作业结束核对见`HANDOFF.json`，由owner将新默认接口及待动作准入事项转给CCI生成线。
