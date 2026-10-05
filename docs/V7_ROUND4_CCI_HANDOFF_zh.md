# v7 第四回合 CCI 交接：候选与诊断版本

**正式生产不可运行；不要启动v7-S或完整版批量生成。** 本提交没有合格的移动RIR策略。CCI环境/分片入口由cci-datagen线负责，本线仅在48g改和测，没有登录CCI。最终提交号及代码包记录在本轮产物`delivery/COMMIT.txt`、`delivery/source.bundle`。代码只同步，不改公共环境；产品应写owner指定AFS目录，由调度器确保单个父场景只有一个writer，4小时重启后按已封存阶段恢复。

## 输入与兼容边界

48g检出：`/data/jzy/code/AVEngine-v7-spec-acoustics-20261003`，分支`claude/v7-spec-acoustics-20261003`。证据根A：`/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round4_energy_scenarios_20261004/`。旧输入只读根O：同级`round3_production_20261003/`。

最终新请求表：`A/seeds_quota_v4_release_r3/v7_S_train.jsonl`（400父场景）、`v7_S_dev.jsonl`（100）；完整版`v7_full_train_seed_plan.jsonl`（7000）、`v7_full_dev_seed_plan.jsonl`（1500）、`v7_full_test_seed_plan.jsonl`（1500）。外schema为`v7-production-inputs-20261004-v4`，内quota revision v4。S train/dev分别与full前400/100行字节一致。仅500行已分配历史通过的物理屋/房间，其余9500行房屋仍null，禁止调度。`validation.json`是请求配额审计，不是媒体/正式验收。`.partial`不是调度输入，禁止扫描整个目录当种子表。

房屋、旧runtime、候选资产输入分别为`O/houses_admitted.json`、`O/simulation_inputs_v1.json`、`O/assets_native_canary.json`。旧runtime为v6实际false/order0；true/order10仅为本轮对照，未取得house或移动资格，不能静默改默认值。旧候选资产仍为research，不是已许可正式准入。

`tools/timeline/generate_v7_scenes.py --plan-only`已用最终请求表核验train400/dev100，只返回`requested_only_native_not_admitted`，不生成目录或媒体。正式模式在启动worker前拒绝；v4实际resolver亦明确拒绝降为v3两人路径。具体阻断：移动能量四臂17/9/17/9过线；完整像素流第53帧失败；完整28族、30/25/45模态和≥20%严格绑定未齐；13/34类、35类干声及人物/运动/许可正式准入未齐；新v4布局、声学、噪声及干声配额真实resolver未准入；新增房屋或true10 profile需重新验收。不要以`--candidate-only`绕过这些条件发布训练集。

## 产物与断点

沿用`docs/V7_PRODUCTION_INTERFACE_zh.md`的父场景/成员/窗口结构：`parents/<scene_id>/rir_cache/`及同一parent下分阶段attempt与completed指针，成员下10秒RGB MP4、四通道FOA WAV、同步证据、公开question和私有答案/角色证据。OA及MCQ共享一个logical_question_id。窗口公共地址不得编码答案；本轮36窗110题的中性诊断副本不计正式量，也没有每窗12–15道合格题。

parent和visual recipe新增材质/visibility版本；旧seal遇到版本不匹配将拒绝并保留原文件。不得人工改recipe使旧输出冒充新验收。RIR缓存只复用完全相同几何、材质、profile、listener/source及runtime identity。SQLite进度仅用于本地单writer；跨AFS多writer锁未验证，P19应按scene_id独占分片。

本轮没有删除中间产物，失败attempt、参考RIR及诊断波形全部保留。首轮生产不能清理未封存或失败attempt；未来只有已完整发布且独立读回的阶段，才按原接口的`--compact-completed`契约处理临时逐帧文件。保留最终MP4/FOA、QA、同步/材质证据、输入recipe、阶段指针、cache身份及验收记录。AFS上的删除行为与重启恢复尚未由本线验证。

## 合并清单与分工

原始材质清单`d790e8c8c0bfdf96aeed7158d97aac12045ea896`、`486799f3d57caaae44ab692bf8cef41a396defb6`已合入；不取参考快照b6fa5e7。题族正式清单`23e2cddf004f22848bef2e0908eeb7b81013db2b`已合入，不取其未提交后续。仅该线提交手工拥有qa源码，跨分工改动为可选`timeline/v7_native_questions.py`证据接口。本线渲染/RIR/采样器/交付改动见Git diff；未改v6真值或任何评分器/原门槛。正式合格题数仍0。
