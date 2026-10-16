# 性能优化实施指导（20260823，给协作代理的工作说明）

> 对象：基于 `main@bfe272e` 只读审计报告（六个优先级）的实施者。
> 审计方向整体正确；本文逐条给出采纳结论、项目侧约束与验收要求。
> 结论先行：**最值得做的是"资源流水线 + memmap"**，与审计一致。

## 0. 铁律与禁区（先读，违反即返工）

- 全链 research_only、episode_counted=false；正式数据集分母保持 0；
- 不新增 hash gate / frozen contract / 新准入闸门（性能优化尤其不许
  顺手加"content hash 一致才复用"之类的永久 gate——见 P6）；
- fresh/no-clobber：任何优化不得覆盖历史输出；
- blender_custom 禁用、Skokloster 排除，维持现状；
- canary 注册表（examples/m6/registries/*）不要动：字节序排序 +
  content hash 敏感，动了会挂 test_m6_registry；
- **不要触碰正在跑的东西**：GPU1 的 SO-7B 训练、locate/vsm2 conda 环境、
  `/data/datasets/spatial-omni/*`、`/data/avengine_external/review/qa_v2_*`
  评测与批次产物目录。用卡先 `nvidia-smi`（GPU3 长期被他人占用）；
- 工作流：按仓库 README"开始使用"走 fork + 任务分支，每切片独立提交，
  合并前跑全量单测 `pytest tests/unit -q`（约 5 分钟，当前基线
  3111 passed / 65 skipped，见 /data/jzy/tmp/full_unit_20260823c.log）。

## 1. 逐条采纳结论

### P5-审阅页 index O(N²) —— 已完成，勿重做
`build_batch_review_page.py` 已在 `d604636` 重写：每 10 个 clip debounce
一次 points.json、结束终写；同时页面改为卡片平铺+严格懒加载。此项从
清单划掉。

### P5-其余三项 —— 采纳，低风险，先做
- **log tail**：从文件尾按块反读，只解码最后 N 行。验收：对一个 >1GB
  日志请求 tail 的耗时应为常数级；行为与旧实现输出一致（含 UTF-8 断字
  处理）。
- **QA author 进程内批处理**：可以做，但保持每点独立 fresh 输出目录、
  独立失败记录、独立 receipt；失败点不得影响后续点。不许把 192 点合并
  成单目录单清单。
- **AudioProgram 进程内按 program_id 缓存**：采纳。只做进程内 dict，
  不要新持久缓存文件。

### P3-memmap 直写 —— 采纳，收益真实，注意三点
- `.npy.partial` + `open_memmap` + 完成校验后 atomic rename 的方案正确；
  现有 reader 只认 `rgb.npy`，partial 天然不会被误读——保持这个命名，
  不要写成 `rgb.npy.tmp` 以外还能被 glob 撞上的名字；
- Apartment / MP3D / M5.1 三条路径**分三个提交**，各自带对照测试：
  同输入下与旧实现逐像素一致（RGB/depth/semantic/矩阵全部 allclose
  或 array_equal）；
- 失败路径验收：中途 kill 后目录里只允许留 `.partial`，不得留半个
  `rgb.npy`。
- 与 npy 退休政策的关系见 §3，不冲突。

### P1-GPU/CPU 流水线 —— 采纳，预计收益最大，约束补充
- lane 划分正确（GPU 捕获 lane / CPU+RLR+FFmpeg lane）；
- owner 决议：**视觉渲染最多 2 张 GPU**；默认 1 个 GPU lane，配置显式
  给卡；GPU3 不可用（他人长期占用）；
- 音频侧并行安全性已实证：RIR cache 写入为 tmp+rename 原子写，
  batch2d 已三路并行跑完 192 点零冲突。CPU lane 并发度可到 6
  （owner 决议"音频 CPU 放开、下批 6 路"）；
- 每 lane 独占 RPC 端口与输出目录的要求正确；
- 验收：同一批次流水线跑 vs 串行跑，产物集合与逐点 receipt 等价，
  wall time 有量化对比记录。

### P2-UE 进程复用 —— 即 owner 已批的"SpearSim 常驻批模式"
这不是新提案：owner 20260823 已拍板批准，排在训练启动之后实施。
审计给的验证清单全部采纳，追加：
- 复用 session 下每集的 research_receipt 必须独立完整，闭包与 stage
  不变才允许复用同一进程；
- A/B 对照验收：同一点分别用"独立进程"与"复用 session 第 N 集"各跑
  一次，要求 pose/时间线 JSON 严格一致、帧数/分辨率/收据字段全等；
  RGB 因 UE 渲染非决定性不必逐字节，但需通过现有质量检查且抽帧目检；
- 失败即弃整个 session、不污染后续集——正确，照做；
- 与 P1 分开提交验证（审计自己的建议，赞同）。

### P4-孪生 RIR 复用 —— 先测量，再决定做不做
入口**已经接了 M6x RIR cache**（batch2d 三路 runner 共享同一 cache，
原子写）。孪生与主点房间/listener/轨迹全同 → RIR key 应当已命中。
所以：
1. 先量化 batch2d 实跑的 cache 命中率（cache 目录统计或加一行日志）；
2. 若确认已命中，此项只剩"每集重复创建 RLR context + 上传房间"
  （~6 秒/集），优先级降到最末；
3. 若确认未命中，先查 key 构成再动工——**禁止**通过调大
  `rir_stride_frames` 省时间（改变声学采样语义，审计也这么说，一致）。

### P6-一次 cook/package —— 现状即如此，不动
同意"不是当前瓶颈"。补充一条铁律呼应：DDC 复用、sibling staging 原子
发布都可以做，但 content hash 不得变成新的永久 gate。

## 2. 实施顺序（对审计顺序的唯一修改：去掉已完成项）

1. log tail 反读 + AudioProgram 进程内缓存（一起一个切片也行）；
2. 捕获数组 memmap（三路径三提交）；
3. QA author 进程内批处理；
4. Studio GPU/CPU lane 解耦（默认单 GPU lane）；
5. SpearSim session 复用（A/B 对照验收）；
6. P4 先测命中率，再定；
7. 显式多 GPU lane（≤2，最后评估）。

## 3. 一处必须对齐 owner 决议的表述

审计"明确不能采用"清单里的"只保存 MP4 并提前删除 RGB truth"：
**npy 退休是 owner 已拍板的政策，不是要修的 bug**。时序是
捕获 → 出题 → 闸门核验 → 审阅成片核验（75 帧 ffprobe）→ 带清单退休
（`tools/qa/finalize_batch_visuals.py`，manifest 含 sha256）。
"不提前删"指的是出题与核验完成**之前**；之后按政策退休。请勿"修复"
或回退该工具。它与 P3 memmap 完全正交：memmap 优化的是捕获期内存
峰值，退休政策管的是批次收官后的磁盘。

## 4. 现成可复用的事实（省得重新踩坑）

- rgb.npy 是 BGR（读时 `[:,:,::-1]` 或 `--channel-order bgr`）；
- pkill -f 会匹配自己的 ssh 命令行（用脚本文件内 pgrep）；
- SpearSim 复跑要点（SDK env 三件套 + spear-ext PYTHONPATH）见
  docs/roadmap/STAGE_REBUILD_20260823.md；
- 范围 runner 的 skip_exists 与"清理半成品"要按顺序执行，否则漏点
  （batch2d b007 教训，见 BATCH2D_QA_TRAIN_20260823.md §5）。
