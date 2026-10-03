# v7 第三回合生成接口与CCI交接

代码只在48g `/data/jzy/code/AVEngine-v7-spec-acoustics-20261003`修改、测试；分支`claude/v7-spec-acoustics-20261003`。新产物根：`/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round3_production_20261003/`。前两回合目录只读。

**状态：生产候选，正式生成被阻止。** 原生RGB五窗检查已通过；移动RIR能量在实际骨骼轨迹80次中点原生对照中63次失败；全部28题族和完整M2运动/资产准入仍未通过。`--candidate-only`只生成研究候选。调用正式模式在模拟器启动前失败；不能让P19把候选计入v7-S/论文正式量。CCI环境、同步及分片执行属于`20261003-205913_complex_cci-datagen`线，本线没有登录CCI或起作业。

## 输入

- `houses_admitted.json`：物理house、split、已知暴露、明确声学pass结果、只读原始visual/nav输入、既有声学包路径、checked_rooms及摆放见证。位置仅在已测room_scope内采样；86套自动候选不是86套RLR正式pass。
- `seeds_release_v3/v7_S_train.jsonl`400行、`v7_S_dev.jsonl`100行；完整版train/dev/test种子规划7000/1500/1500行。`v7-production-inputs-20261003-v3`：父场景ID/uint64种子、分开的trajectory/camera/appearance/pose/audio/event_schedule/questions/window随机流、房屋/房间、30秒、候选切点100ms ticks；首窗4.0–5.4秒、次窗14.0–19.8秒，0.2秒网格，保留至少4秒前史、发布窗数、成员、逻辑题数及全成员modal配额地址。
- `simulation_inputs_v1.json`：v6原生RLR配置、现有runtime-prefix/Magnum/RLR路径、零半径、16kHz/4秒IR、材料/编译包来源；CCI线按已有同步资产建立显式路径映射，只改变挂载路径，不改变物理参数。
- `assets_native_canary.json`：已注册M2人物包/请求、endpoint、发声嘴部anchor及外观。当前两个人物仅用于research候选；不是正式资产清单，不能冒充8种正式人物。

代码读取每条seed，不从完成顺序重新随机种子。相同parent同物理屋和split，组成员共用物理因素；大小数据集的400个训练前缀逐字段相同。完整版未分配行拒绝调度。`seeds/`、`seeds_v2/`与`seeds_frozen_v3/`为诊断草案，保留；CCI只取`seeds_release_v3/`及validation。

## 命令接口

```bash
python tools/timeline/generate_v7_scenes.py \
  --houses INPUT/houses_admitted.json --seeds INPUT/seeds_release_v3/v7_S_train.jsonl \
  --runtime-inputs INPUT/simulation_inputs_v1.json --assets INPUT/assets_native_canary.json \
  --output TASK_OUTPUT --offset 0 --limit 1 --workers 1 --plan-only
```

先检查全部500个已分配S输入。当前正式模式去掉`--plan-only`会拒绝三个明确阻塞项；要做小量资格样本，明确加`--candidate-only`并配置下面的软件CPU环境。候选场景的受控模板仅两个人、讲话、合法线段上的连续随机行走幅度/停步、100ms事件扰动及独立相机/声线/服装分配，不能作为完整60屋正式多样性实现。

必须`CUDA_VISIBLE_DEVICES=''`、OMP/OpenBLAS/MKL各1线程，48g所有本任务工作进程总量≤64。CPU RGB使用`LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe EGL_PLATFORM=surfaceless`及Mesa ICD。用 `bash tools/capture/build_cpu_egl.sh TASK_PRIVATE_BUILD` 构建任务自有`.so`，`LD_PRELOAD`该软件设备选择器；不覆盖系统/共享运行库。CCI线需私有runtime内提供Mesa ICD/软件DRI及EGL依赖，未在CCI验证。本地临时、Matplotlib、Numba、XDG缓存全部指向任务目录，关闭Mesa shader缓存。实际渲染器读回swrast并拒绝NVIDIA OpenGL驱动。

## 输出与续跑

```text
TASK_OUTPUT/
  shard_OFFSET_COUNT_TIMESTAMP.json         # 本次完成回执，candidate/正式计数分开
  parents/PARENT_ID/
    input_recipe.json                     # 完整输入，重复运行改变输入则拒绝
    visual_v0_completed.json              # 完成指针，只在全文件齐备后发布
    visual_v0_attempt_UUID/                # 连续30秒视频+骨骼嘴部/可见像素证据
    visual_v1_completed.json              # 反事实第二外观（若有）
    dry_completed.json                    # 独立CPU进程一次合成，所有成员共享干声
    dry_attempt_UUID/
    audio_MEMBER_completed.json
    audio_MEMBER_attempt_UUID/
      scene_foa_full_tail.wav              # 从0秒起，保留完整4秒卷积尾
      native_RLR_setup.json
      audio_evidence.json                 # 配置、原生IR/缓存/CPU及墙钟计时
    rir_cache/                            # exact-pose .npy + SQLite完成记录+identity
    windows_MEMBER_completed.json
    windows_MEMBER_attempt_UUID/WINDOW_ID/
      video.mp4                           # 原生H264，10秒，1280×720，150帧
      audio_foa.wav                       # 160000样本，4通道ACN W/Y/Z/X SN3D camera
      audio_mono.wav                      # 同一W通道，无二次归一化
      questions.json                     # 12–15个逻辑题，公共题面，不含私有答案
      private_answers.json               # 答案和实际问题证据地址，训练loader不可读
      evidence.json                      # 窗口、活动、native读回与准入状态，私有
```

中断前未完成的stage留在独立attempt，重启只建新的attempt；已完成stage校验相同输入及必需文件后跳过。exact-pose IR逐条提交，因此长场景在CCI四小时重启后能复用已完成的IR。完成指针不覆盖，丢失必需文件和recipe漂移直接报错；不是“发现目录就算成功”。本线除单元测试外，曾核对命令、父子关系和产物根后中断自有原生worker，保留27条已完成IR及visual stage，随后恢复完成两窗并验证封存重放；固定v3四成员场景也完成封存重放。证据：第三回合 `evidence/native_recovery_result.json` 和 `evidence/frozen_v3_canary_readback.json`。真实CCI重启/AFS跨机锁语义未验证。

每个parent由CCI分片调度器保证只有一个writer；AFS上的SQLite/hardlink一致性和跨节点重复任务去重须由CCI线验收，本工具不擅自清锁、杀旧job或接管未知writer。不在4小时边界反复覆盖媒体。无stage完成标记的半截视频不进入数据集。

## 可释放的中间产物

只针对本任务新建产物，以下是后续生产的删除许可边界，本回合仅在自有seeded-canary和固定v3四成员父场景验证过封存清理，未执行批量GC：

- 原生RGB/depth/semantic逐帧流直接送编码器，仅留选定验收样本，不将密集30秒数组落盘。
- 某member全部窗口及私有证据完成并读回后，其连续30秒H264、全尾FOA可释放。`--compact-completed`先硬链接完整窗口到新的delivery_attempt，保留全场景原生姿态/可见性及RLR设置的私有provenance，再原子发布delivery_completed，最后才删除本parent自有的visual/audio/dry/窗口中间attempt和rir_cache。续跑先验证封存交付，不再要求已释放中间文件。研究失败样本默认保留；必须显式加该选项。
- 未完成/落败attempt和发布后的`.partial`硬链接，仅在本parent所有成员完成、单writer且封存指针有效后可清理。封存工具只删除该parent的已知stage前缀和rir_cache，拒绝符号链接；禁止扫描AFS共享根或删别人的缓存。
- 当前RIR缓存按parent私有，所有成员完成并封存后可释放；活跃parent保留断点。未来跨parent只读共享缓存要由调度器建立独立保留规则，本工具不删除外部缓存。原始干声、扫描、已有checkpoint和前两回合产物一律不删。

## 发布审计

`audit_v7_question_budget.py`从实际窗口公共/私有题目计逻辑题、30/25/45预算、逐QA族最低量及≥20%已验证绑定题；成组审计需实际成员媒体和私有答案，预定group不计数。未达标返回非零状态。移动能量资格文件、RGB验证、干声/人物准入与此题目审计须全部通过，当前没有可正式发布的组合。

两个native30秒样本切6窗；另有4成员研究组、恢复样本和固定v3四成员样本，合计20个候选验收窗口、280道不同逻辑题，每窗14道。固定v3四成员场景为8窗112题，50个绑定逻辑题有实际答案改变的对照；这不是500场景正式完成量。所有实际测量、失败插值文件、5窗原生图和计时在第三回合产物目录。原始试点、修正动作时钟后的样本都保留，用后者作为现行native验收依据。题族最低量是数据集级验收，不要求每个窗口塞入全部28族。

干声在每个父场景用一个全新CPU进程生成并缓存，所有反事实成员复用同一波形；发声关联交换只交换两个dry bus。无关外观变体须先确认原生嘴部轨迹完全相同，然后硬链接相同音频，不重新调用TTS造成隐性声学变化。干声中间数组在封存后可随自有parent的中间attempt清理。公共question_id/logical_question_id用于索引和去重，模型消息只含题面、公开选项及媒体，不输入种子、成员因素地址或私有证据。


## 冻结交接文件

第三回合产物根的`CCI_HANDOFF.json`记录冻结提交、输入哈希、正式就绪false、阻塞项、候选命令和最终证据地址；`source/`保存该提交的代码快照及本提交补丁，供CCI线同步。`REPORT_zh.md`为一页中文实测报告。现行seed目录只用`seeds_release_v3/`；不得用早期诊断草案覆盖它，也不得给完整版未分配行自动轮换12套旧房。

v7-S全部500行可先通过`--plan-only --limit 400`（train）及`--plan-only --limit 100`（dev）检查；此检查不启动模拟器，不证明已测全部房屋/人物摆放。正式调用目前会在创建输出和启动模拟器前拒绝。P19仅可把此提交用于接口接线和显式研究候选；能量、题族、干声/人物准入及CCI本地资格全部通过前，不发布v7-S或论文正式数据。
