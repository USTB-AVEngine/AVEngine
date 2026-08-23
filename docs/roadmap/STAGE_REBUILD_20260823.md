# 检查点 20260823：全 actor stage 重建 + 双换色人冒烟 + SO-7B 训练管线冒烟

> Owner 指示（20260823）："不要降级，调整优化引擎本身"。本检查点记录由此
> 完成的引擎工程、全部失败尝试与最终验证。全链 research_only，
> qualification_claim=false，正式分母保持 0。

## 一、成果一览

| 项 | 结果 |
| --- | --- |
| 注册表 | 3 个受控换色男性变体注册（`rocketbox_human_male_adult_01_top_{burgundy,green,blue}_research_v1`，revision `20260823_v9`，commit `a84760e`） |
| 新工具① | `tools/ue/build_minimal_closure_report.py`（闭包报告生成器，commit `1a00bad` + 动画种子修复 `76b04b3`） |
| 新工具② | `tools/ue/assemble_package_stage.py`（package stage 组装器，commit `1ae5294` + SpContent/SDK 校验修复，本 commit） |
| 授权快照 | `/data/avengine_external/ue-assets/actor_content_registry_v9_20260823T033709Z/`（14 actor × 28 内容目录，0.38 GiB，带 SNAPSHOT_PROVENANCE） |
| 依赖图 | metadata 编辑器 stage 重扫 `analysis_asset_registry_20260823T034622Z/dependency_graph.json` |
| 闭包报告 | 同目录 `minimal_closure_report_20260823T0510Z.json`（variant `registry_v9_full_actors`：472 内容包，44 种子，全部唯一映射） |
| **生产 stage** | `/data/avengine_external/ue-package-stages/apartment_0000_1ae5294_20260823T0512Z/`，BuildCookRun exit 0，pak 含 SpContent 29 项 + controlled_material 117 项 |
| 冒烟捕获 | `review/apartment_two_controlled_humans_smoke_1ae5294_20260823T0555Z/`：75/75 帧，相机与双 actor 位姿误差 0.0，动画误差 ≤1.1e-16s，**蓝衣/绿衣两人同框视觉正确**（owner 目检 pending） |
| SO-7B 冒烟训练 | `Spatial-Omni/so_runs/avengine_temporal_smoke_20260823/`：beats_lora 续训 `SO-7B_finetuned.pt`，400 样本 1 epoch 213.6s，train_loss 0.610 / valid_loss 0.622 / valid_EM 0.531，GPU1 峰值 ~21.3 GiB |
| 训练数据集 | `/data/datasets/spatial-omni/avengine_temporal_foaish_v1/`（200 间歇 wav mid/side 伪 FOA 化 + temporal QA 2589/282，按集切分，带 DATASET_PROVENANCE） |

现 stage 覆盖全部 14 个注册 actor（此前仅 beagle + 原版男性）——两犬、
两猫、两换色人配对全部可执行，v2 同物种批次的 stage 前置条件解除。

## 二、失败记录（按序，全部保留在盘）

1. cook#1（stage `…1a00bad_20260823T0405Z`）exit 8：未设
   `AVENGINE_SPEAR_{BOOST,RPCLIB,YAML_CPP}_ROOT`（SpModuleRules 要求，
   prefixes 在 `/data/avengine_external/ue-sdk/`）。已在组装工具中加前置
   校验。**该 stage 废弃**，保留为失败证据。
2. 捕获#1：闭包缺 `Standing_Idle` 包——动画按名播放不是 Blueprint 硬依
   赖，必须显式做种子（旧报告即如此）。工具已修（`76b04b3`）。
3. 捕获#2：shell 链断裂导致 cwd 错误（操作失误，非引擎问题）。
4. 捕获#3："stage uproject must enable SpContent"——源码切片 uproject
   仅经 AdditionalPluginDirectories 引用插件；组装工具现自动注入显式
   enable 条目（与 0820 留存 stage 一致）。
5. 捕获#4：`avengine_spear_ext` 缺失——宿主扩展经 `PYTHONPATH=
   /data/avengine_external/spear-host-sdk/avengine-spear-ext-cp312-…` 注入
   （studio runner 同法）。
6. 目检虚惊：rgb.npy 是 **BGR**（handoff 已警告过），直接当 RGB 存图会
   出现"蓝皮肤 + 橙衬衫"假象；出片必须 `[:,:,::-1]` 或 `--channel-order
   bgr`。lead-b 变体片源对证后确认资产颜色正确。

## 三、复跑要点（下次照抄）

```
# 闭包（在 lead-a 根目录）
$PY tools/ue/build_minimal_closure_report.py \
  --dependency-graph <analysis>/dependency_graph.json \
  --source-asset-registry examples/runtime/source_asset_runtime_profiles.json \
  --variant-name <name> --source-root <актор快照> --source-root <0820房间快照> \
  --output <fresh.json>
# 组装 + cook（需三个 SDK env）
AVENGINE_SPEAR_BOOST_ROOT=/data/avengine_external/ue-sdk/boost-1.90.0-ue55-r2 \
AVENGINE_SPEAR_RPCLIB_ROOT=/data/avengine_external/ue-sdk/rpclib-d1bb5b4 \
AVENGINE_SPEAR_YAML_CPP_ROOT=/data/avengine_external/ue-sdk/yaml-cpp-edadfec \
$PY tools/ue/assemble_package_stage.py --closure-report … --variant-name … \
  --stage-root <fresh> --source-commit <sha> --run-buildcookrun
# 捕获（宿主扩展经 PYTHONPATH）
PYTHONPATH=/data/avengine_external/spear-host-sdk/avengine-spear-ext-cp312-8a36d4d-20260821T0030Z \
avengine m5 capture-current-apartment-visual … --graphics-adapter 0
# 依赖图重扫（metadata stage，UE 5.5 无头）
AVENGINE_ANALYSIS_OUTPUT=<dir>/dependency_graph.json /data/UE_5.5/…/UnrealEditor \
  <metadata>/SpearSim/SpearSim.uproject -RenderOffscreen -graphicsadapter=0 \
  -NoAssetRegistryCacheWrite -AbsLog=… -UserDir=<dir>/user -unattended -nop4 \
  -nosplash -NoSound -run=pythonscript -script=<dir>/asset_registry_dependency_export.py
```

## 四、SO-7B 训练冒烟的边界声明

目的仅为**管线验证**（数据转换→加载→beats_lora 反传→生成评测全通）。
v1 QA 的设计缺陷（DATA_EVAL_20260823）原样存在，valid_EM 0.531 无任何
基准含义；视频模态尚未进 trainer（AV 版 trainer 改造为待办，需 owner
对 Spatial-Omni 仓库改动点头）。数据集与训练产物均标注
research_pipeline_smoke_only。

## 五、后续（更新自 QA_V2_TRIAL_20260823）

1. owner 目检冒烟帧（蓝/绿衣）→ 通过后 Studio `apartment_end_to_end`
   模板切至新 stage/closure；
2. 32+16 点同物种批次待 owner 批准开跑（现在 two humans / two dogs /
   two cats 都已 stage 就绪）；
3. 引擎工作②（约束驱动场景规划器）设计稿；
4. AV 版 SO-7B trainer 改造方案与 owner 对齐。
