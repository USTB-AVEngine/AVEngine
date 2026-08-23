# 数据多样化工单：多房间 × 多视角 × 均衡 held-out（20260823，给协作代理）

> 背景：当前 QA v2 批次全部固定在 apartment_0000 的一个角落、一个相机朝向
> （(-70,65) yaw −145，为与 M1 听者权威做 1e-6 互证而锁定）。owner 要求
> 后续数据集覆盖多房间、多视角。本工单给出实施设计、约束与验收标准。
> 阅读前置：PERF_OPT_GUIDANCE_20260823.md §0 铁律禁区（同样适用）。

## 0. 为什么相机是"锁定"的（先理解再动）

相机锁定不是偷懒：动态音频的 listener 位姿必须与视觉 M1 捕获请求里的
相机严格一致（渲染链内 1e-6 互证），所以"换视角"=作者化一个新的 M1
capture request + 该视角自己的 RIR 缓存（listener 变了，RIR 键全变）。
多视角的本质是"多个经过认证的 (相机位姿, 标定, 走廊子集) 三元组"，
不是在渲染时转相机。

## 1. 阶段一：同房间多视角（成本最低，先做这个）

### 1.1 视角库（viewpoint bank）
- 新建 `examples/qa_v2/viewpoint_bank_apartment_v1.json`：每条含
  viewpoint_id、相机位置/朝向（UE cm 与 yaw）、覆盖描述。目标 4–8 个
  视角：覆盖客厅另一侧、厨房反打、窗侧逆光、低位/高位各至少一个。
- 候选视角先在 Studio 里人工摆（Studio 已有相机第一视角预览），
  导出位姿进 bank——不要凭空写数字。

### 1.2 每视角认证（fail-closed，缺一不可）
1. **可行域覆盖**：视锥内可行走面积 ≥ 阈值（用现有 navmesh/障碍快照
   做视锥求交，Studio 的草稿校验代码可复用）；
2. **侧向标定**：现有叉积公式 c=(fx·dy−fy·dx)/|d|（c>0=右）对任意
   yaw 通用，但每个视角必须重做"渲染帧人工核验"：渲 2–3 帧已知位置
   的角色，比对 c 值与画面左右。核验帧和结论存
   `examples/qa_v2/viewpoint_calibration/<viewpoint_id>.md`；
3. **4 点位 mini-pilot**：每个新视角先跑 4 个点（视觉+音频+出题+
   闸门 B/C），审阅页人工过目后才准进量产。

### 1.3 走廊库按视角过滤
走廊段是地面弦（世界坐标，与相机无关），复用现有
straight_corridor_bank_v1.json，新增按视角过滤器：段两端+中点在该
视角视锥内、且锚帧侧向判定有效（不贴画面中线：|c| ≥ 阈值，避免
pilot48 式弱线索题）。产物：每视角一个走廊子集清单。

### 1.4 工具改造点（都是加参数，不是重写）
- `tools/qa/design_qa_batch.py`：+`--viewpoint-bank` +`--viewpoint-id`
  （或 per-point 轮转分配）；锚帧侧向闸门读该视角标定；spec.json 记录
  viewpoint_id；
- `tools/m7/render_current_apartment_dynamic_audio.py`：listener 位姿
  从 M1 request 读（现状如此），每视角一个 M1 request 模板
  `examples/m6x/fixed_apartment/m1_capture_request_<viewpoint_id>_720p.json`；
- RIR 缓存：键含 listener，天然按视角隔离，无需改；但每视角第一批会
  全量算 RIR（慢），建议每视角先跑 mini-pilot 暖缓存；
- 出题器：题面不变（题目本来就不提相机）；答案键均衡见 §3。

## 2. 阶段二：多房间

### 2.1 kujiale（UE/SPEAR，生产链与 apartment 同构）
每个新房间需要：M3 声学包 + UE stage（用已入仓的
`tools/ue/build_minimal_closure_report.py` + `assemble_package_stage.py`
组装，SDK env 三件套见 STAGE_REBUILD_20260823.md）+ 端点注册表条目 +
可行域权威。走廊来源变化：kujiale 没有 v1 episode 可挖，改从 navmesh
生成候选直线弦 + 碰撞半径闸门（min_separation ≥60cm 照旧）+ 速度带
仍读 motion_profiles_v1.json。每房间入口 4 点 mini-pilot 同 §1.2。

### 2.2 mp3d（habitat-sim 路径）
视觉走 habitat 渲染（现有 m5 current_mp3d 工具链，two-beagle 已验证），
音频同 RLR。批量化=把 `design_qa_batch.py` 的点位规划接到现有
mp3d route author（seed 驱动）上。注意 mp3d 相机/听者约定与 UE 系
不同（坐标系换算已封装在现有工具里，别自己重写换算）。

### 2.3 房间铁律
blender_custom 禁用、Skokloster 排除不变；新房间/新资产一律先过
owner 审（资产注册表新增条目要 owner 点头）；正式分母保持 0。

## 3. 答案键均衡（今晚的教训，硬性要求）

pilot48 的侧向题 gold 12:5 不均衡 + 弱线索，导致"低于随机"假象
（模型固定偏答即可显著偏离 50%）。此后每批：
- 出题器对每个二选一题型**按批强制答案 50/50±1**（不满足就在设计端
  换点位/换首发声者，不是删题）；
- 侧向题增加 |c| 下限（锚帧时刻声源不得贴画面中线）；
- 闸门报告里输出每题型答案分布，进检查点文档。

## 4. 标准 held-out（搭量产的车）

首个多视角量产批同时产 **48 个专用测试点**（不进训练）：视角/配对/
运动状态分层抽样、答案键均衡、全部闸门 + 人工审阅页过目。产出后
pilot48 降级为"弱线索探针"（附加分析用，不再当主评测）。

## 5. 可一并移交的模型侧工程（原本在我清单里）

1. **trainer 多视频批处理**：`train_so_qa.py` 视频 collator 现有
   bs=1 闸（`with_video currently requires batch_size 1`）。扩展成
   多视频批（processor 支持视频列表；per-sample fps 列表、
   video_second_per_grid 已是张量）。显存参考：bs=1 峰值 30.7GB/48GB。
   验收：bs=2 与 bs=1×accum2 的 loss 曲线统计一致（同 seed 前 50 step
   逐步对比）；
2. **双卡 DDP 启动**：trainer 已有分布式支持（torchrun +
   shard_dataset_for_rank），补一个 2 卡启动脚本 + 冒烟（100 step），
   注意 GPU2/3 常被他人占用，只用显式指定的卡；
3. **replay+video 兼容**：trainer 里是显式 NotImplementedError，
   实现后解锁"防遗忘混训"；
4. **原生 FOA 输出**：`render_current_apartment_dynamic_audio.py`
   现在只落双耳；给 RLR 传播加 ambisonic FOA 双落（W/Y/Z/X 4ch wav），
   训练侧就能扔掉 foaish 的 mid/side hack，拿到真前后/俯仰线索。
   这是模型效果上限最大的一项，但要先与 owner 确认 RLR SDK 的
   ambisonic 输出面（RLRAudioPropagationPkg runtime-b 是否暴露）。

## 6. 边界（这些留给 lead-a 会话，不要动）

训练/评测/消融的执行与分析、论文叙事、检查点文档、owner 决策的
传达。正在跑的：GPU0 消融训练（so_qa_v2_ablate.log）勿动；
`/data/datasets/spatial-omni/*`、`qa_v2_*` 产物目录勿写。

## 7. 实施顺序建议

1. §1 视角库 + 标定协议 + 4 视角 mini-pilot（先只做 apartment）；
2. §3 出题器答案均衡（小改，先行合入，量产前必须就位）；
3. §4 均衡 held-out 48 点（搭第一个多视角量产批）；
4. §5.1/5.2 trainer 批处理与 DDP（与 1–3 并行，不同人可同时做）；
5. §2.1 kujiale 首房间端到端（一个房间打通再横向复制）;
6. §2.2 mp3d、§5.3/5.4 视资源排。
