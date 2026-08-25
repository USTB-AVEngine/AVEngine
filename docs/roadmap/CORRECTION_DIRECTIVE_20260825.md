# 纠偏指令（20260825）：优先级重排 + 三项立即修正

> 对象：承接 `DATA_DIVERSIFICATION_WORKORDER_20260823.md` 与
> `DATA_DIVERSIFICATION_HANDOFF_20260825.md` 的执行方。
> 本文**覆盖**上述两份里与之冲突的优先级安排；未冲突部分继续有效。
> 依据：owner 决定 + `WORLD_CONTACT_BASELINE_20260825.md` 的实测证据。

## 0. 为什么要纠偏（一句话）

已完成的工作是真实且有证据的，但**最大的一笔投入押在了资产最贫瘠的房间**：
MP3D 的 160 点全是两只外观相同的比格，**造不出我们卖点最强的跨模态属性指代题**
（held-out 80.9% 的那一类），而资产最齐全、生产链最成熟的公寓反而停在
`BLOCKED.json`。同时三项低成本的吞吐/质量修正被跳过，正在持续产生浪费。

## 1. 立即停止：MP3D audio160 与其 FOA waiter

**决定（owner）**：停。这批音频服务的 480 道题只有 TA/T7/T5，暂时用不上，
不值得再占 9 小时机器时间。

执行要求：

1. **只停这两个进程**，用 `ps -p` 确认后再 kill，**禁止 `pkill -f`**（会误杀自己的 ssh）：
   - 父 runner：`run_mp3d_audio160_after_target_ef26330.py`
   - FOA waiter：`run_current_dual_foa_after_audio160_9196814.py`（它在等一个永远不会到来的 receipt）
   - 先停 waiter，再停 runner 及其当前子进程。
2. **一个字节都不要删**：已完成的点（停止时约 40/160，每点 dry/stem/mixture/receipt 齐全）
   是合法证据，保留原目录。
3. 在产物根目录写一份 `PARTIAL_STOP.json`：停止时间、已完成点数、停止原因
   （owner 决定，优先级调整）、以及"已完成点仍然有效"的明确声明。
   **不要**写成 `failure.json`——这不是失败。
4. 停止后解冻两个此前被冻结的 worktree
   （`AVEngine-codex-diversification-integration`、`AVEngine-codex-foa-dual`）。

## 2. 立即开启并落盘 RIR 缓存

**问题**：当前每点只输出 dry/stem/mixture（2.6 MB），**RIR 算完即扔**；
audio160 明确禁用了持久缓存。仓库里 `src/avengine/m6x/rir_cache.py` 是现成的，
缓存键为 `["source_position_m", "listener_position_m", "listener_orientation_wxyz"]`。

要求：

1. 此后所有音频批次**默认启用持久 RIR 缓存**；
2. **把 RIR 作为一等产物落盘**（参照旧管线格式 `rir/{samples.npy, lengths.npy, metadata.json}`），
   并写进该点的 receipt；
3. 明确复用边界，写进文档：同房间 + 同听者位姿 + 同声源位置 → 可复用；
   换机位/换轨迹/换房间 → 必须重算。**跨点不省，点内极省**；
4. 收益说明（这是要做的理由）：闸门 A 的路线互换孪生、拒答变体、发声时刻随机化、
   同画面换声音——**本质都是"同画面不同音频"，有 RIR 就只剩卷积+混音（秒级）**。
   没有缓存则每套变体都要重烧完整 RIR。

## 3. 公寓拉回第一优先

现状：`apartment_viewpoint_review_round4_finalized_r1/BLOCKED.json`，
卡在"需要 4 个保留代理视角、只有 3 个"。**这个门槛本身要放宽**——
它把整条最成熟的生产链堵死了，代价与收益不成比例。

执行顺序（每步都要有产物，不是计划）：

1. 从已有的 target-agnostic 相机目录里**自主选 4–8 个明显不同的位姿**
   （不经 Studio 摆位，符合 owner 既定口径）；
2. 每个视角建独立 M1 capture request；
3. 在**真实 UE/SPEAR** stage 上跑 normal + target-only 捕获与 readback；
4. 每视角独立 RIR/音频（**启用缓存**）；
5. 每视角一个**四点 mini-pilot**：视觉 + 音频 + 出题 + 人工逐帧看片
   （曝光、构图、遮挡、身份、动画）；
6. 通过后才进量产。

公寓的资产优势必须用起来：**3 个换色人 + 比格 + 已注册的生成动物**，
这是唯一能产出 T2-ATTR（跨模态属性指代）与 T9-CLOSER 的房间。

## 4. MP3D 这批的正确定位

不作废、但**明确标注为"难度/泛化子集"**：单一 actor 类型（Beagle×Beagle）、
仅 TA/T7/T5 三类题、无属性指代题。文档与后续报表中不得把它当作主集的等价补充。
恢复该批的前提是：先给 MP3D 补上第二种可用 actor（见 §5）。

## 5. 资产双腿（新增要求）

**事实更正**：人物**已经有** Habitat 那条腿——`src/avengine/m5_1/human_runtime.py`
（把 Rocketbox 人物编译进 Habitat 蒙皮运行时）、`tools/m5_1/capture_two_human_mp3d.py`
（两人 MP3D 捕获），产物实证在
`tmp/m5_1/human_runtime_api_probe/{visual.glb, walking_actions.npz, ...}`。
Codex 写的 MP3D 生成器是 `bounded two-Beagle materializer`（硬校验拒绝混用资产），
**属于自我设限，不是引擎限制**。

要求：

1. MP3D 要扩 actor 类型时，**先接已有的 human runtime 路径**，不要重造；
2. 今后新资产**默认产两条腿**：UE（uasset/蓝图 + 进 stage）与
   Habitat（GLB + URDF + 关节映射 + 烘焙动作 npz + 触地相位 + 碰撞代理 + 发声锚点）；
   只做一条腿要显式说明理由。

## 6. 动画质量与步速（新增，优先级高）

依据 `WORLD_CONTACT_BASELINE_20260825.md` 的实测：

- 比格 walk 动画**隐含步速 ≈0.33 m/s**，人物 Walking **≈1.384 m/s**，而两者共用同一条
  配置带 0.60–0.78 m/s（batch2d 实测 0.72–0.77）——**同一条带把两个物种推向相反的错误**：
  人在原地踏步（滑步比 46%），狗在前向滑行（滑步比 56%）；
- 渲染器步频固定 25 帧/周期、**与速度无关**（`current_apartment_visual.py:359`），
  于是每秒滑步约 0.42 m；
- 现行接触相位标注在 walk 上出现 **5/25 帧腾空**，不符合 walk 步态。

要求：

1. 把七项判据实现为 `tools/m2/audit_animation_quality.py`（支撑面倾角、爪高差、
   滑步、穿地、步态模式、隐含步速、接触相位一致性），纳入资产验收与回归；
2. **每个资产记录其动画隐含步速**（写进 runtime profile 或 motion_profiles），
   来源必须是动画反推，不是房间几何；
3. **采用步频缩放**（推荐，且是唯一可行解）：`period = round(base_period × implied / actual)`。
   不能只改速度配置——房间对角线 /5s ≈ 0.79 m/s 是硬上限，人物的 1.38 m/s 走不下；
   缩放后人物 period 16→29 帧、犬类 25→11 帧，travel 速度不变而滑步消除；
4. **验收门槛按 owner 口径放宽**：悬空 ≤3 cm、支撑面倾角 ≤3° 视为通过（现状均通过，
   不必追求完美）；**只把"滑步比 ≤20%"作为必须修的硬项**（现状 46–56%）；
5. **在扩到 3000 点之前完成**；已产出的 batch2d/pilot48 不作废（滑步不改变答案），
   但要在论文数据说明里如实记录。

## 7. 文档节流

51 份 roadmap 检查点 / 56 个代码文件的比例过高。此后**每条工作流一份滚动文档**
（apartment / kujiale / mp3d / trainer / throughput / assets），追加更新而不是每片新建。

## 8. 优先级总表（覆盖旧安排）

| 序 | 事项 | 状态 |
|---|---|---|
| 1 | 停 audio160 + FOA waiter，保留证据 | 待执行 |
| 2 | 开启并落盘 RIR 缓存 | 待执行 |
| 3 | 公寓 4–8 视角 → 四点 mini-pilot → 量产 | 第一优先 |
| 4 | 动画质量判据 + 速度带修正 | 与 3 并行 |
| 5 | 资产双腿（新资产默认；MP3D 接 human runtime） | 与 3 并行 |
| 6 | Kujiale 原生闭环 | 3 完成后 |
| 7 | MP3D 恢复（需先有第二种 actor） | 延后 |
| 8 | trainer 收尾（multi-video 证据合入、长跑）、吞吐三项 | 按资源穿插 |
| 9 | 双卡 DDP | owner 指令：延后 |

铁律不变：全链 `research_only=true`、`episode_counted=false`、正式分母 0；
fresh/no-clobber；不新增无理由的 hash/gate；进 main 由 owner 决定。
