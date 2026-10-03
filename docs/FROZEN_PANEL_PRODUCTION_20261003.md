12 套测试房 × 60 组的 CPU 生产输入已经准备好；正式生成仍等待生成器、修复资产和 CCI 渲染环境三项验收，当前没有启动 CCI 或 GPU 作业。

本轮数据在 48g 的 `/data/jzy/tmp/claude_hardbank_20260923/test_panel_pilot_20261003/followup_05/`，代码从 `a152503` 独立检出，房屋推荐固定为 `e3dc2460b3444d0a7aa5b7c7be3c5d5193153bed`。以下是三项交付后的操作说明，不代表已经执行。

## 输入及版本参数

`inputs/preflight_config.json`、`inputs/frozen_seed_plan.json`、`preflight/<house>/prepared.json` 是 720 组的构造输入。每组保存实际地板/姿态、相机基、配额、冻结种子、完整语音选择及 CPU FOA 预检摘要。实际生产的 attempt 0 读取这一姿态，不重新抽样；整组质量拒绝后的 attempt N 使用 `frozen_scene_seed + N*100003`，记录拒绝原因，并避开 20 个 pilot 父场景及全部预留姿态。拒绝仅依据现有媒体/几何审核。CPU预检的房间回退顺序固定，不按模型表现或声学指标重新排序；正式质量重试在当前通过CPU筛选的房间重抽整组姿态。

先复制 `inputs/production_versions.pending.json` 为一个新文件，填入下表。没有 `accepted` 与真实验收文件哈希会直接阻止 worker，不能把 pending 改成 accepted 来绕过验收。

| 参数 | 由哪项交付提供 | 核验 |
|---|---|---|
| `generator.repo/revision/entrypoint/entrypoint_sha256` | v7-spec 正式生成器验收 | Git HEAD 精确匹配、跟踪文件无修改、入口文件 SHA256 |
| `generator.stage_contract` | 正式生成器与此面板适配入口 | 必须实现 `frozen_panel_stages_v1`：`prepare/capture/audio --config --index`，capture 另收 `--gpu`；必须消费 `prepared_geometry_root`/冻结种子/整组 retry 参数 |
| `assets.revision/registry/registry_sha256` | asset-materials | 修复版注册表身份；不能使用原玻璃反光版作为正式资产 |
| `assets.template_plan/template_plan_sha256/actor_assets` | 同一修复资产交付 | 模板里的两位 actor 的 asset_id 与 asset_revision 精确匹配修复版；模板明确指定一对同几何、不同服装资产 |
| `render_environment.revision/runtime/hostnames/machine` | CCI EGL/P19 验收 | machine=CCI，hostnames 为验收的实际主机名，runtime 四个路径来自该环境 |
| 三项 `acceptance_report/acceptance_report_sha256` | 对应交付 | 保留各自验收文件及 SHA256；版本验证读取、核对，不改写 |
| 三项 `files_sha256` | 对应交付 | 填资产 GLB/配置及需固定的运行时锁、材质/声学包等实际依赖文件哈希 |

生成器入口须含本提交的冻结姿态接线，或提供等价适配。新生成器的实际 CLI/原生数据契约现在**未验证**；若与这份入口不兼容，需要在自己的集成检出补适配，再通过下述 20 组对拍。这里没有替别人定版或修改其检出。

CCI 还要显式填写复制后的基础配置路径：`speech` 及每条 dry clip 的 `path`、`hrtf`、`template_dataset`、`frozen_seed_plan`、每房 `acceptance_result/acoustic_package_manifest/room_manifest`，以及这些元数据内部的 scan/navmesh/semantic/annotation/声学网格路径。只在新的测试输入目录修改复制的路径，保留原文件 SHA256 的迁移记录；地板、姿态、种子、配额不得改动。`inputs/houses.48g.json` 和 `inputs/scope.checked_rooms.json` 已导出为现有配置构建器的格式。CCI 数据与声学包的实际路径等待 cci-datagen 交付，本轮未登录 CCI、未搬运原始数据。

配置构建器可检查已迁移的本机路径和已验收房间：

```bash
"$PANEL_PYTHON" tools/qa/prepare_frozen_panel_config.py \
  --base-config "$PANEL_INPUT/base.cci.json" \
  --houses "$PANEL_INPUT/houses.cci.json" \
  --scope "$PANEL_INPUT/scope.checked_rooms.json" \
  --panel-id frozen_test_v2 --split test --groups-per-house 60 --expected-houses 12 \
  --seed 2026100300 --pilot-verification "$PANEL_INPUT/retained_final_after_prune.json" \
  --output "$PANEL_INPUT/base.formal.cci.json"
```

`base.cci.json` 的基础 seed 应保留 pilot 的旧值以满足构建器的新 seed 检查；最终每组 seed 实际来自冻结表，不用全局 seed 重排。基础配置应保留 training/development 排除表、推荐 commit 和冻结 seed 文件字段。版本绑定后会保存每房 prepared.json 哈希，修改后 worker 拒绝消费。

## 先对拍两套房

`inputs/canary_replay/` 已保存原来实际通过的 20 个姿态及真实 seed，包含质量重试后的姿态；旧试跑本来有重试，直接重跑初始 seed 不等于对拍。该目录不含评分答案；新 canary 仍为 pilot，不得收入最终面板。复制后更新 `canary_base_config.json` 的本机路径和 `prepared_geometry_root`，保持十组/房、原姿态和原 seed。正式排除表在正式配置里保留，canary 的排除表为空，仅用于重放这 20 个 pilot。

在验收的 CCI 环境中，设置 `PANEL_CODE`、`PANEL_PYTHON`、`PANEL_INPUT`、`PANEL_TEST_ROOT`、`PANEL_GPU` 为实际交付路径和明确授权的物理 GPU。`PANEL_TEST_ROOT` 必须是独立测试目录。不能使用训练/开发数据目录；每 worker 单独 output，顺序使用一张 GPU，CPU RLR 8 线程。首次 min_free_vram_mb 至少 12288，后续不低于实测峰值 +4096 MiB，遵守 CCI 与训练同卡验收的具体资源约束。

```bash
cd "$PANEL_CODE"
"$PANEL_PYTHON" tools/qa/bind_frozen_panel_versions.py \
  --config "$PANEL_INPUT/canary_replay/canary_base_config.json" \
  --versions "$PANEL_INPUT/production_versions.accepted.json" \
  --prepared-root "$PANEL_INPUT/canary_replay" \
  --output-parent "$PANEL_TEST_ROOT" --output "$PANEL_INPUT/canary.accepted.json"
"$PANEL_PYTHON" tools/qa/run_frozen_panel_shard.py \
  --config "$PANEL_INPUT/canary.accepted.json" --output "$PANEL_TEST_ROOT/canary_new_version" \
  --shard-index 0 --shard-count 1 --gpu "$PANEL_GPU"
"$PANEL_PYTHON" tools/qa/score_frozen_panel.py seal \
  --root "$PANEL_TEST_ROOT/canary_new_version" --expected-groups 20
CUDA_VISIBLE_DEVICES='' "$PANEL_PYTHON" tools/qa/verify_retained_frozen_panel.py \
  --root "$PANEL_TEST_ROOT/canary_new_version" --output "$PANEL_TEST_ROOT/canary_retained_verify.json"
CUDA_VISIBLE_DEVICES='' "$PANEL_PYTHON" tools/qa/compare_frozen_panel_canary.py \
  --before "$PANEL_TEST_ROOT/pilot_baseline" --after "$PANEL_TEST_ROOT/canary_new_version" \
  --before-proof "$PANEL_INPUT/retained_final_after_prune.json" \
  --after-proof "$PANEL_TEST_ROOT/canary_retained_verify.json" \
  --output "$PANEL_TEST_ROOT/canary_version_comparison.json"
CUDA_VISIBLE_DEVICES='' "$PANEL_PYTHON" tools/qa/compact_frozen_panel.py measure \
  --root "$PANEL_TEST_ROOT/canary_new_version" --output "$PANEL_TEST_ROOT/canary_sizes.json"
```

`pilot_baseline` 是原20组 public、scoring_private、retention、inputs/config.json 的保留包；复制它不会失效原承诺，留存验证会把旧绝对 native/dependency 路径解析至自己还原的副本。评分私有目录必须继续私有，只让原 score_frozen_panel.py 读答案。基线 config 的旧外部路径供 resolver 识别，不应重写已封存真值/证书/承诺。

对拍要求 20/20 原三层验证与封存审计通过、80 个成员、每配额5、父场景20个独立、`inconsistent_episode_relations=[]`、M1/M2/跨模态键数2/2/4。比较实际 camera/feet/basis/计划mouth、房间、配额、固定题面、150帧/15fps/10秒及完整 FOA 布局/长度。每个新四元组内部应同的视频/音频必须哈希一致、目标证据变化；这些仍由原验证器判断。

跨版本像素哈希变化是材质修复预期；完整 PCM 哈希/RMS差、原生几何 hash、FOA角/C50/目标台词比均报告诊断，不能用跨版本 PCM 不同来改原 QA 门槛。若原生人物几何或 emitter 确实改变，按新版本重新做 CPU 地板/摆位/FOA 预检，不能沿用旧预检结论。保留现有10°、C50≥0dB、目标比≥12dB、可见服装像素/颜色比等门槛，整组失败就剔除。需抽看新视频/听台词确认反光修复与联合可解性，同时重测体积、生产时间、显存峰值。没有实际交付及这一轮对拍就不能开产720。

本轮只运行了比较器的旧版自比较 CPU 冒烟，不是新资产/新生成器/CCI 对拍验收。

## 正式分片与逐组清理

绑定正式配置时必须附这套版本的20组对拍、完整留存验证及体积报告；旧版自比较不能解除开产阻塞：

```bash
"$PANEL_PYTHON" tools/qa/bind_frozen_panel_versions.py \
  --config "$PANEL_INPUT/base.formal.cci.json" \
  --versions "$PANEL_INPUT/production_versions.accepted.json" \
  --prepared-root "$PANEL_INPUT/preflight" \
  --output-parent "$PANEL_TEST_ROOT/frozen_test_v2" \
  --canary-comparison "$PANEL_TEST_ROOT/canary_version_comparison.json" \
  --canary-verification "$PANEL_TEST_ROOT/canary_retained_verify.json" \
  --canary-sizes "$PANEL_TEST_ROOT/canary_sizes.json" \
  --output "$PANEL_INPUT/formal.accepted.json"
"$PANEL_PYTHON" tools/qa/run_frozen_panel_shard.py \
  --config "$PANEL_INPUT/formal.accepted.json" \
  --output "$PANEL_TEST_ROOT/frozen_test_v2/shard_0" \
  --shard-index 0 --shard-count 4 --gpu "$PANEL_GPU"
```

仅在 CCI 明确分配四张卡/四个资源槽之后，分别用 shard-index=0/1/2/3 启动，GPU 和 output 各不相同。每组 global index `i % 4` 分片，每片180组；若只分配一张卡就 shard-count=1，顺序720组。本轮未启动这些命令、未创建队列作业。要在 zsh 服务器写循环/组合逻辑，先存为 bash 脚本再用 bash 执行。

worker 对每个认证组执行：原 native 验证 → 最小见证与每组承诺 → 仅留存文件重验 → 核对原文件 inode/大小/hash后删除该认证父场景中间件 → 清理成功 attempt 的五个日志及重复配置，仅留 hash、执行摘要与拒绝原因。失败组、其他检出、原始数据和 checkpoint 不清理。之后才处理下一组，因此不积累545GiB原数组；每 worker 通常需额外一组原生数组加一组还原缓冲，按当前试跑约1.5–2GiB估算，最终版本峰值未验证。质量失败原型另占空间，累积时应诊断，不能扩大清理范围。

中断恢复用原命令加 `--resume`，同一份配置和 output。已做留存的组直接从保留文件验证，不重新生成已删除的原数组。分片末尾冻结 retention/COMMITMENTS.sha256；评分脚本再各自 seal（180组或该分片实际indices数），然后逐片执行 verify_retained。运行目录内 run_state.json/PROGRESS_zh.md/日志和 PID 必须保留；只可停止核对过 PID 与命令的本任务进程。

多分片合并用新包装器，它调用原 scorer，随后合并最小见证，不修改原判分器或真值：

```bash
"$PANEL_PYTHON" tools/qa/assemble_retained_frozen_panel.py \
  --root "$PANEL_TEST_ROOT/frozen_test_v2/sealed_panel" --expected-groups 720 \
  --shard-root "$PANEL_TEST_ROOT/frozen_test_v2/shard_0" \
  --shard-root "$PANEL_TEST_ROOT/frozen_test_v2/shard_1" \
  --shard-root "$PANEL_TEST_ROOT/frozen_test_v2/shard_2" \
  --shard-root "$PANEL_TEST_ROOT/frozen_test_v2/shard_3"
CUDA_VISIBLE_DEVICES='' "$PANEL_PYTHON" tools/qa/verify_retained_frozen_panel.py \
  --root "$PANEL_TEST_ROOT/frozen_test_v2/sealed_panel" \
  --output "$PANEL_TEST_ROOT/frozen_test_v2/final_retained_verify.json"
```

发布仅 public 媒体和题面；答案/映射/证书/最小验证见证/承诺留在私有评分包。共享blob与媒体用硬链接去重，跨文件系统复制需另计副本。全量CCI生成、多分片原生合并与新版本性能均**未运行/未验证**；先按上文20组验收。
