# 检查点：SO-7B 首轮训练 + 时间对齐消融 + 错题定位（20260824 凌晨）

> 接续 BATCH2D_QA_TRAIN_20260823.md。全链 research_only，正式分母 0。
> 训练数据 avengine_qa_v2_batch2_av_v1（790/119），held-out = pilot48
> v2 题（194，与 Qwen 零样本基线同题；对比按 189 题交集）。

## 1. 两个训练 run（配置仅差一个开关）

| run | RoPE | best | valid EM 曲线（ep1/2/3） |
|---|---|---|---|
| avengine_qa_v2_batch2_av_20260823 | **时间对齐** | ep3 (loss 0.245) | 55.5 / 64.7 / 56.3 |
| avengine_qa_v2_batch2_av_ablate_notime_20260823 | legacy | ep3 (loss 0.217) | 51.3 / 65.5 / **73.1** |

其余同：beats_lora r16、16 帧 588×336、foaish、bs1×accum4、lr 1e-4、
3 epochs、无 replay、从 SO-7B_finetuned.pt 续训。GPU1/GPU0 各一跑。

## 2. held-out 主表（pilot48 189 题交集，Qwen2.5-Omni 零样本对照）

| 分组 | n | **legacy-e3** | aligned-e3 | Qwen AV | Qwen V | Qwen A |
|---|---|---|---|---|---|---|
| **总体** | 189 | **70.9** | 67.2 | 54.5 | 53.4 | 51.3 |
| T2-ATTR（跨模态指代） | 47 | **80.9** | 63.8 | 57.4 | 51.1 | 59.6 |
| T4-SIDE（纯音频左右） | 17 | 64.7 | **70.6** | 47.1 | 41.2 | 29.4 |
| T7-DURING | 39 | 41.0 | 41.0 | 48.7 | 53.8 | 59.0 |
| T9-CLOSER | 47 | **97.9** | 97.9 | 70.2 | 70.2 | 53.2 |
| TA-MOTION | 39 | 59.0 | 59.0 | 41.0 | 41.0 | 41.0 |

**当前最佳模型 = legacy RoPE epoch_003：总体 70.9%，超 Qwen AV +16.4pp**
（checkpoint：ablate run 的 epoch_003_trainable.pt / best_trainable.pt）。

## 3. 消融结论（如实）

909 题规模下**时间对齐 RoPE 未带来收益**：总体 legacy 高 3.7pp（主要
来自 T2-ATTR +17.1），T4 aligned 略高（12/17 vs 11/17，n=17 属噪声），
而假说中最应受益的时序题 T7/TA 两种 RoPE **逐题完全相同**（见 §4，两
者都常量作答，压根没进入"用时序"阶段）。当前证据不支持把时间对齐
作为已验证方法点写论文；处置：保留双跑记录，**在 3k 点规模复测**后
再定（若大规模下时序绑定开始涌现，对齐才可能显差异）。

## 4. 错题定位（held-out e3 + batch2d valid 交叉验证）

- **T7/TA 完全未学会**：两类题模型 100% 输出 "moving"（held-out 39/39
  全 moving；batch2d valid 44/44 全 moving）。其 41%/59% 纯为 gold 构成
  的假象。失败本质 = **发声次序 → 人物 → 该时刻状态**的时序绑定未
  涌现，这正是跨模态指代 QA 的核心难点（论文动机的直接证据）。
- **T4 "低于随机"疑云已解**：aligned-e2 与 Qwen-A 都是 17 题全答
  "right"（固定偏答 × gold 12L:5R 不均衡 → 29.4% 假象）；e3 两跑都
  真正判向（11–12/17）。**gold 本身无误**：人工核验 pilot48 p02/p06 与
  batch2d b007 三帧，首发声者画面位置与 gold 一致（帧图
  /data/jzy/tmp/{p02,p06,b007}_f4.jpg）。
- 学习梯度清晰：T9（视觉距离）≈ 满分 → T4（空间侧向）70% → T2（跨模
  态指代）63.8→80.9 → T7/TA（时序绑定）0 学习。

## 5. 下批数据的针对性修正（已并入工单）

1. 发声时刻随机化（当前 first_frame 恒为 4，too regular，模型无从学
   "第二次发声在何时"）；2. T7/TA 必须覆盖 both_moving 与双静止点位
   （打破"恰好一人动"的退化结构，否则常量/捷径作答无法被数据惩罚）；
3. 二选一答案键按批强制均衡（已是硬性规定）；4. 均衡 held-out 48 点。

## 6. 产物索引

- 评测：`so_runs/*/bench_pilot48_heldout/epoch_00{2,3}/{predictions.jsonl,result.json,compare_vs_qwen.json}`
- 对比脚本：/data/jzy/tmp/compare_so_vs_qwen.py（含通用 EM 计分器
  score_generic_em.py）
- bench AV 支持：Spatial-Omni 分支 avengine-av（已镜像为 AVEngine 仓库
  分支 spatial-omni-avengine-av，含全部 12 commit）
- 训练/消融日志：/data/jzy/tmp/so_qa_v2_{train,ablate}.log
