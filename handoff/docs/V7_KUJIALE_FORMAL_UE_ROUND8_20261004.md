# 酷家乐正式 UE alpha 接线与房间配额（2026-10-04）

原生物理准入仍为 18 套、29 间；本轮不启动 116/464 父场景量产。共用正式静态录音 alpha 已在 0022 R0 完成四成员组、8 个窗口、112 道逻辑题。完整 v7 的动作及全部类别配额尚未实现；不能把本轮测速副本计为独立训练世界。

本任务唯一代码检出是 `/data/jzy/code/AVEngine-codex-v7-kujiale-20261003`，产物根是 `/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round8`。原始 InteriorAgent、共享 Content、其他检出、公共环境只读；不 push，不向 owner 发送页面。

## 接线与版本

共用代码快照提交 `07e485969e132ed6eb4c7d3599214c9a2365da59`，共用 driver 基线提交 `977104cbeaccb650cb5375664aba8c3c3e79d07c`。peer 检出为 `/data/jzy/code/AVEngine-v7-spec-acoustics-20261003`，peer HEAD `a39ba8f4693be26e3f3b817b7fac449b19185e7e`，其正式 alpha 文件当时仍在工作区；逐文件 SHA 是实际版本依据，见 `evidence/common_interface_commit_v1.json`、`common_driver_commit_v1.json`、`common_member_mix_hook_v2.json`。

`house.input.visual_backend=ue_spear` 选择 `NativeUEAlphaVideo`；其余出题/选题/反事实审计/许可/封存仍调用共用 alpha 管线。UE 使用隔离 SpearSim 项目、原始 MIC、生产 EV3/4000K 光照、2560×1440 渲染后降为1280×720，15fps、85°水平 FOV。450 帧 RGB、模态深度、每个人物 target-only 深度及实际 K2 相机/根节点/动画时钟读回，构成30秒世界；每成员输出2个10秒窗口，每窗150帧、16kHz/160000样本的ACN WYZX SN3D相机坐标FOA，mono逐样本等于W。CPU PathFinder只读导航，不用Habitat渲染酷家乐画面。

当前正式 profile 是 `v7-S-alpha-recorded-static-native-14-v1`，实际两条录音语音、四个静止人物（两活跃、两静默）。源准入目录为同一19类86条加600条VCTK、同一7人许可目录；试跑实际用其4种原始上衣外观与VCTK男性录音子集。每窗14逻辑题、实际14条open记录；当前共用alpha未输出MCQ，不能翻倍计数，四成员组每窗5个绑定项；排除QA-26/27/28、完整动作和完整v5场景配额，`full_v7_generation_ready=false`。不采用合成formant/TTS填补录音。

冻结整屋声学指标文件保持f9db982内容（SHA `a85e178d23381653a212ee97ea5eaaac41e2843d4a9a20af124daa8ed87eda96`），既有QA/判分代码不改。上游移动RIR快照依赖新的first-arrival helper，本任务保留旧冻结指标，尚未验证移动路径；当前静态alpha不调用该路径。

换正式版本时依次核对：

1. `v7_alpha_generation/sampling/audio`、`v7_full_question_delivery`及`qa/v7_*`逐文件SHA，保留同一候选题/模态配额/反事实选择，不用更换判分器或改真值解决缺题；源码版本变更用新stage recipe或输出目录。
2. `source_qa_evidence(...,window_start_s=...)`保留真实源时钟，窗口只含半句时不附全文；录音及人许可目录须重新核SHA，原始UE MIC不跟随Habitat PBR派生包。
3. `alpha_row(...,force_quartet=False)`保留房间配额表的单成员/四成员混合；`--layout`只能绑定同一房子的同一准入房间，禁止继承0022试跑布局。
4. UE的实际相机读回、每点原生floor reference、原2cm脚底/胶囊、0.5mm根节点与1e-4秒动画一致性必须保留。未来动作、非语音类及可适配卧室的正式人物数量需要主管线接通后重新验收，不能沿用静态alpha的通过标记。

peer已有相同共用模块时，只需cherry-pick后续UE接线/工具提交；先提交它自己的alpha文件再合入UE driver delta，避免未提交同名文件冲突。UE运行目录和config、19类源目录、7人许可源都需在48g按实际部署绑定；不把隔离UE项目移到共享Content或CCI P19。

## 房间配额与启动

`room_quota_v4/v7_S_train.jsonl`：29×4=116父场景，93单成员+23完整四成员=185成员、370窗、5180逻辑题；完整版29×16=464父场景，371单成员+93四成员=743成员、1486窗、20804逻辑题。1.6期望值为185.6/742.4，185/743是当前共同选择器允许的最近整数混合提案。S种子、成员和题号索引是完整版的同一子集。上限按房间绑定，无酷家乐dev/test。

`layouts_all_cpu_v1`只搜索原3个已测UE相机，20/29间找到四人体布局；`layouts_frozen_rules_cpu_v2`用原房间多边形、>=0.25m相机净空、原源距离/夹角和原胶囊要求扩大CPU相机搜索后为23/29。0021R1、0026R1、0030R1/R2、0042R1、0067R1仍未找到当前四人体布局。这6间保留物理准入，但不能据此启动当前四人体alpha；新增相机的3间也待原生逐场景验证。CPU预检结果不增加正式窗口计数。

示例仅说明接口，本轮未执行量产命令：

```bash
PYTHONPATH=src:. /data/jzy/envs/room-selection-merged-20261003/bin/python \
  tools/rooms/pilot_kujiale_alpha.py \
  --house /OWN_OUT/ue_round7/delivery_v1/generation_inputs/kujiale_0022_original_v1/house.json \
  --seed-row /OWN_OUT/ue_round8/SELECTED_CAPPED_SEED.json \
  --layout /OWN_OUT/ue_round8/layouts_frozen_rules_cpu_v2/kujiale_0022_R0.json \
  --inputs /OWN_OUT/ue_round8/reference/runtime_inputs_ue_alpha.json \
  --assets /OWN_OUT/ue_round8/reference/assets_alpha_static.json \
  --source-admission /OWN_OUT/ue_round8/reference/SOURCE_ADMISSION_alpha.json \
  --human-admission /OWN_OUT/ue_round8/reference/HUMAN_ASSET_ADMISSION.json \
  --output /OWN_OUT/ue_round8/UNIQUE_GENERATION_DIRECTORY
```

实际GPU启动必须由`v7_kujiale_ue_guard.py --request OWN_REQUEST.json`包裹，不直接执行示例。request保留隔离项目、所选GPU、实测峰值+4GiB、CPU affinity、预算根和中央监控快照。新导演出包只含窗口媒体、公共题、私有答案、原生证据和许可文本，questions/answers逐文件与共用封存字节相同；未提交临时兄弟文件不进入交付。

## 磁盘和传输

旧150GB迁移预留已撤销；本任务以80GiB空闲（85.899GB）加2GB停机缓冲保护用户至少80GB要求，新增分配块<=40GB，原总任务<=80GB。`evidence/disk_gpu_curve.jsonl`记录`statvfs`空闲与唯一inode分配块；硬链接不重复计算物理占用。`job_allocated_footprints_v1.jsonl`每10秒测各作业实际分配块。显存用`nvidia-smi -q -x`统计精确PID/创建时刻/argv子树的G/C+G图形进程；不是compute-apps。

每个通过的父场景：封存→原子发布`__portable_v2`（禁止覆盖）→核题/答SHA→删自己的中间件，保留3帧/外观及静态接触见证。然后`prepare_kujiale_cci_batch.py`冻结文件清单，用指定`/home/neos/jzy/tools/cci_pull_from_48g.sh`按批拉到CCI mirror。共享helper不改；本地私有ssh包装器把其复杂远程正文交给bash执行。CCI逐文件核SHA/大小后，`reclaim_kujiale_cci_batch.py`才删除这批48g派生包及对应已压缩源封存文件。错误SHA、新添未列文件、软链接和跨范围路径都拒绝回收。

首套70文件70137642字节已在CCI核SHA：pull29.070472秒，总传输核验41.402210秒，有效载荷2.412676MB/s，包含JSON压缩收益；不等于网络裸速。仍可用用户给定1.7MB/s保守估传输，再用完整并行测速与真实单成员/四成员包体积更新两档预测。


## 上游v2与本轮实测边界

本轮完成测速的运行是初始共用快照的`v7-S-alpha-recorded-static-native-14-v1`。收尾时peer已经提交`44235e58eed56ebc7161738561e96be574447fb9`：v2新增`parent_appearance_assets(assets,row)`，按appearance种子排列外观且保持endpoint槽；并把单成员的counterfactual状态写为`not_applicable_declared_single`。既有QA/答案/题族选择代码相同。这两项应由上游提供，UE delta不覆盖或移除它们。

cherry-pick时只选UE接线/资源/配额提交，不要把07e4859和977104c两个旧共用快照覆盖到已含v2的peer。合并必须保留peer的profile=v2、appearance排列调用和单成员N/A状态。隔离索引的三方补丁应用及源码解析证据见本任务`ue_round8/evidence/peer_v2_patch_compatibility.json`（该证据只证明补丁/源码兼容，不能代替原生GPU执行）。当前低磁盘没有跑v2新的UE原型，所以v2原生资格化仍未验证。

v1测速的单成员窗口没有执行反事实比较，本轮运行v1旧模板的通用`counterfactual_answers=pass`字段不能解释为执行过比较。报告与指标将其明确读为不适用；已有封存媒体/题/答不改。四成员正式原型则实际执行过共同反事实检查。

已测2/4/8实例为6.471/12.702/22.003成员场景每小时；12实例因共享盘启动预留不足而未运行。当前S预算185成员、约8.41h、约3.73GB；完整版743成员、约33.77h、约14.95GB，使用当前静态alpha的单成员保守折算，不能保证完整动作/全19类管线相同吞吐。现有封存派生纹理414文件SHA前后验证硬链接去重约4.0GB；原始资产没动。

收尾已恢复单成员N/A状态表达（与当前共用driver一致），不改QA/答案或已封存副本。历史运行标志的解释见报告；新的运行必须按其自己的源码身份重新封存。
