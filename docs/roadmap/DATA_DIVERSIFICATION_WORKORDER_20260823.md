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

## 3. 出题设计硬性修正（20260824 错题分析追加，均为硬性要求）

错题分析（SO7B_TRAIN_ABLATION_20260824.md §4）确认：时序题 T7/TA 模型
100% 常量作答，根因是数据结构退化。除 §3.1 均衡外追加两条：
- **发声时刻随机化**：first_frame 不得恒定（当前恒为 4），首/次发声
  的 onset 在可行窗口内抽样，且两声间隔也要有分布；
- **T7/TA 必须覆盖 both_moving 与双静止点位**：只有"恰好一人动"的
  批次里，"任何运动检测"或常量作答的捷径不会被数据惩罚。

### 3.1 答案键均衡（原有要求）

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

## 6. 运行时安全边界（只关乎不踩踏，不限制范围）

正在跑的：GPU0 消融训练（so_qa_v2_ablate.log，标记
SO_QA_V2_ABLATE_DONE 落地前 GPU0 勿用）勿动；GPU2/3 常被他人占用，
用卡前 nvidia-smi 确认；`/data/datasets/spatial-omni/avengine_*`、
`/data/avengine_external/review/qa_v2_*` 既有产物目录只读；新产物
一律 fresh 目录。

## 7. 实施方式：放开做（owner 20260823 拍板）

**不设顺序、不设范围上限**：§1–§5 加 §8 的所有条目你自己排优先级、
自己定并行度，多房间/模型工程/FOA 可以同时展开，敢挑大的。保留的
只有两类硬约束：
1. §0 引用的铁律禁区 + §6 的运行时安全边界；
2. **自证要求（唯一不可谈判项）**：
   - 凡产出视觉内容（新视角、新房间、新题型的锚帧）——**必须自己看
     渲染帧**：每个切片交付时附帧图 montage 或审阅页链接，并在文档里
     写"我看过 X 帧，确认了 Y"（参照 20260823 侧向核验：p02/p06/b007
     三帧对着 gold 逐一比对的做法）；
   - 凡改渲染/音频/出题管线——必须有机器核验（逐像素/逐字节对照、
     闸门 B/C、答案分布统计、全量单测 3111 基线不回退）；
   - 凡改训练器——必须有等价性冒烟（同 seed 前 N step loss 对照）；
   - 不接受"应该没问题/逻辑上是对的"作为完成态；没有证据的切片
     一律视为未完成。

## 8. 追加的开放挑战项（原 lead-a 待办，全部移交，自由发挥）

- **T5 数值 DoA 题型**：出"声源方位角"数值题，评分带用 owner 已拍板
  的占位（±15°/±30°；时刻类 ±0.3s/±1.0s），gold 从 UE 捕获读回算，
  进出题器成为第 6 个题型；
- **拒答配额**：每批 10–15% 的题设计成"给定模态不可答"（如纯音频问
  颜色），gold=拒答选项——训练模型知道"什么时候不该答"；
- **off-screen 题型启用**：设计端已有 offscreen_candidate 标记，把
  "画外声源"做成正式题型（声源出画期间发声，只有音频能答）；
- **SpearSim 常驻批模式**（指导书 P2，owner 已批）+ **GPU/CPU 流水线**
  （P1）——量产吞吐的两大头；
- **原生 FOA 双落**（§5.4）——模型上限最大的一项；
- **trainer 工程**（§5.1–5.3：多视频批、双卡 DDP、replay+video）；
- **timeline author 路点分段**：让声源走多段折线（现在只有单段直线），
  解锁"绕过障碍物走""先近后远"类时序题；
- 上述任何一项做完，在 docs/roadmap/ 落检查点文档（含自证证据）+
  独立提交，格式参照 20260823 系列。

## 9. lead-a 会话保留项（其余全放）

训练/评测/消融的执行与分析、错题分析、论文叙事与对比表、owner 决策
传达。运行时安全边界见 §6。
