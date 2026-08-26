# 工单：挂装面感知的姿态检查 + 八个资产重做

2026-08-26。给 Codex。前置：`b368c43`（静态源首轮）已完成并核查通过，见 §0。

**修订 1（2026-08-26，接受 Codex 的只读审计）。** 他找出三处范围错误，全部成立，
我核过了；核的过程里又发现第四处，也是我的。四处都已就地改正，改动列在 §0.1，
原文没有留在下面自相矛盾。**执行前请读 §0.1。**

这份工单有三件事：**A** 修检查工具（我的活,交给你）、**B** 重做七个资产、**C** 回答一个数据疑问。
A 必须先做完，因为在 A 之前那份"该重做"的名单里混着我的误判。

---

## 0. 先说清楚：你上一轮的报告我核查过了，没有问题

逐条核过：28 个形态资产、面数 59,921–60,000、28/28 `research` + `authorized=false`、
`placement` 四件套（`attachment_surface` / `facing` / `footprint_bbox` /
`rir_cache_recompute_required`）、`454984c` 是 HEAD 祖先、`b368c43` 是核心提交、
工作树干净。`facing` 是真描述不是模板话。

**回归数字也对得上。** 我在自己的 HEAD 上跑出 `1 failed, 3255 passed, 65 skipped,
71 subtests`，那 1 个失败是 `test_tool_index_is_current`，**是我弄坏的** —— 我加了十来个
工具却只手改了 `docs/TOOL_INDEX.md` 一行没跑生成器。跑完 `tools/build_tool_index.py`
之后 `tests/unit` 回到 `3198 passed, 65 skipped, 71 subtests`，和你报的**逐位一致**。
你留的那个测试抓住了我，谢谢。

倾斜也没有被藏：每个资产的 `acceptance.resting_pose_verdict` 和
`base_normal_tilt_deg` 都写着，包括 `leaning` 的那些。

### 0.1 修订：四处范围错误

| # | 原文 | 正确 | 谁发现 |
|---|---|---|---|
| 1 | "全部 **44** 个已发布资产" | **40 个**。44 条 `asset.json` 里只有 40 个有 `finalized.glb`；缺的四个是 `cat/`×3 和 `dog/`×1，它们带 `animated.glb` + `prepared.glb`，是**另一条流水线**（绑骨动画，+Z 朝上，没有 emitter 锚点）。**不要为了凑 44 把动物塞进这个工具。** | Codex |
| 2 | `attachment_surface` 缺失"**10** 个" | **22 个**。那 10 个只是本轮的便携/台面件；另外 **12 个是我的旧 audio_playback 资产**，同样没声明。所以"按落地假设"这条路径会覆盖 22 个，不是 10 个。 | Codex |
| 3 | Task B 只列了七个 | **八个**。漏了 `audio_playback/television/flat_panel_16_9_two_splayed_feet`，**29.61°**，是全部资产里最歪的一个 —— 而且是**我的**资产。漏掉自己那个最差的、只列别人的，标准不一致。 | Codex |
| 4 | — | `acceptance.base_normal_tilt_deg` **在 40 个资产里只有 28 个存在**。Codex 那 28 个走发布器，写了 `resting_pose_verdict` + `base_normal_tilt_deg` + `secondary_*`；我那 12 个是用 `--apply` 事后补的，**只写了 `resting_pose_verdict`**。照 `acceptance.base_normal_tilt_deg` 读会在 12 个资产上 KeyError。**Task A 顺手统一这两条路径的输出字段。** | 我，核查上面三条时 |

---

## 1. Task A：让姿态检查读挂装面

### 现状与问题

`tools/assets/measure_static_resting_pose.py` 量的是**底面法线偏离正上方多少度**，
判据是"底面大致贴着地面、不歪"，宽松分档：`level ≤ 3°`、`acceptable ≤ 8°`、
`leaning > 8°`。它对**落地件**是对的。

对壁挂件和吊顶件它量的是错的平面。壁挂空调的"底面"是贴墙的背板，不是地面；
烟感的安装面朝上贴天花板。所以现在这三个被判 `leaning` 的壁挂件——

| 资产 | 判定 | 实际 |
|---|---|---|
| `climate_control/air_conditioner/wall_split_white` | 9.79° leaning | **判据用错了地方** |
| `kitchen_appliance/microwave_oven/over_range_silver` | 10.18° leaning | 同上 |
| `plumbing_fixture/floor_drain/exposed_bottle_trap_silver` | 13.56° leaning | 同上（另见 §3） |

——**不能当缺陷读**。反过来说，现在也没有任何检查能发现"壁挂空调没贴平墙"，
真缺陷会被"反正壁挂件都这样"淹掉。两个方向都要修。

### 要做的

读 `asset.json` 的 `placement.attachment_surface`，按面分别量：

- **`floor`**：不变。底面法线对 +Y。现有实现和分档都是校准过的，**不要改**。
- **`wall`**：安装面是**背板**，它的法线应当**水平**。量背板法线与水平面的夹角。
  背板的选法参照现有底面的选法：取沿"背"方向最外侧那一层切片里近平面的面，
  法线统一朝外。"背"由 `acceptance.front_axis`（现有资产是 `positive-x`）反推。
- **`ceiling`**：安装面是**顶面**，法线应当对 +Y（朝上顶住天花板）。
  等价于把落地件的逻辑上下翻转。
- **缺失**（现有 **22 个**：本轮 10 个便携/台面件 + 12 个旧 audio_playback）：
  按 `floor` 处理，但在输出里**明确记下"挂装面未声明，按落地假设"**，
  不要静默当成落地。

### 必须先验一件事，别直接写

**壁挂件的网格到底有没有一个可识别的平背板？** 我没验过。如果某个资产的背面是弧面
或者破碎的，那"背板法线"就不存在，这时候**要报"无法测量"，不要编一个数出来**。
先对那 8 个 `wall` 资产 + 1 个 `ceiling` 资产量一遍背/顶面的面积占比和平面度，
把结果写进执行记录。占比太小或平面度太差的，输出 `no_mounting_plane_found` 并说明。

**采纳 Codex 提出的区分，它比我原来的划法更准：`wall` 不是一类东西。**
"贴墙平装"（壁挂空调、门铃、烟感、壁挂电话）有平背板；
"接到墙上的管件"（瓶式存水弯）**没有平背板**，它是通过一段管子连到墙。
后者本来就该走 `no_mounting_plane_found`，那不是失败，是正确答案。
执行记录里把这两类分开写。**现阶段不因此新增 gate、也不冻结 contract。**

### 不要做的

- **不要改 `level ≤ 3°` / `acceptable ≤ 8°` 这两个分档。** 它们是按"人眼在家具尺度上
  什么时候看得出来"定的，而且 owner 明确要求宽松、不钻牛角尖。
- **不要为了让某个资产通过而放宽任何判据。** 判据变了就不是同一个检查了。
- **不要改落地件那条路径的逻辑。** 它现在的两个特性是被单元测试逼出来的，别退回去：
  按**面的最低顶点**选底面（按质心选会让倾斜 26° 的柜子底面质心升出切片，
  于是报"没找到底面"而不是"歪"——**静默漏放**）；选面时**对绕序不敏感**
  （选近垂直的面不论朝向，再统一翻向下——信任绕序会让法线翻转的网格同样静默通过）。

### 单元测试

照 `tests/unit/test_assets_static_resting_pose.py` 的现有写法加（它合成倾斜盒子 GLB，
不需要任何数据集）。至少覆盖：

1. 壁挂件背板垂直 → 判 level；背板倾斜 10° → 判 leaning。
2. 吊顶件顶面水平 → level；倾斜 → leaning。
3. **同一个网格，声明 `wall` 和声明 `floor` 得到不同判定** —— 这条最重要，
   它证明挂装面真的被读进去了，而不是参数被接受了却没生效。
4. 无可识别安装面 → `no_mounting_plane_found`，不是一个假数字。
5. 挂装面缺失 → 按落地处理，且输出里标着"assumed"。

### 交付

1. 改完的工具 + 单元测试，`tests/unit` 全绿。
2. 在 **40 个适用的刚性静态资产**上重跑（动物那 4 个不属于这个工具），
   给出按挂装面分组的判定表。
3. `acceptance` 字段在两条发布路径上统一（见 §0.1 第 4 条）。
4. **更新后的"该重做"名单** —— 这是 Task B 的输入。
5. 工具有增删就跑 `python tools/build_tool_index.py`（我刚在这上面栽过）。

复现我的测量：

```bash
/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
  tools/assets/measure_static_resting_pose.py \
  --asset-root /data/avengine_external/assets/sound_source_assets_v1 \
  --report /tmp/resting_pose.json
```

---

## 2. Task B：重做真正歪的资产（八个）

**Task A 做完再动。** 名单以 A 的输出为准，下面是我按现有（落地判据）测出来的，
其中壁挂那三个大概率会被 A 洗掉。

**落地件（`attachment_surface: floor`），我认为是真缺陷：**

| 资产 | 底面倾斜 |
|---|---|
| `plumbing_fixture/bathtub/built_in_alcove_white` | **18.52°** |
| `plumbing_fixture/sink_with_tap/counter_vanity_white` | **13.88°** |
| `plumbing_fixture/sink_with_tap/pedestal_basin_white` | **12.77°** |
| `plumbing_fixture/toilet/elevated_tank_exposed_pipe_white` | **10.56°** |

**未声明挂装面的台面/便携件，同样该重做**（这些东西摆在台面上，10° 以上看得出来）：

| 资产 | 底面倾斜 | 归属 |
|---|---|---|
| `audio_playback/television/flat_panel_16_9_two_splayed_feet` | **29.61°** | **我的**，原文漏了，最歪的一个 |
| `household_clock/alarm_clock/digital_cube_black` | **12.90°** | 本轮 |
| `climate_control/air_conditioner/portable_floor_white` | **11.67°** | 本轮 |
| `kitchen_appliance/blender/jug_blender_black` | **10.69°** | 本轮 |

那台电视按 §0.1 第 3 条补上。它是 16:9 平板带两只外张脚，**唯一可能的豁免理由**是
"脚是设计成外张的、底面法线本来就不朝正上"。如果量完确认是这个原因，那就在执行记录里
写明豁免依据；否则一样重做。**不要因为它是我的资产就跳过。**

**一个重要的判断依据：同族的兄弟件是正的** —— 独立浴缸 0.43°、外露冲水管马桶 1.83°、
地漏 1.21°、双铃闹钟 2.24°、中央底座电视 1.60°。
**所以流水线能做对，这八个是个别生成失败，不是系统性问题。**
先直接重跑；重跑仍不合格的才走方法修订协议（`provenance.json`、
`this_is_not_a_seed_retry`、禁止重放哈希那一套）。

**验收：** A 版工具测出的 `resting_pose.verdict != "leaning"`。
`acceptable`（3–8°）算过 —— owner 明确不要严格。

**不要做的：** 不要动 `formal_dataset_registration_authorized`，仍然全部 `false`、
正式数据分母仍然是 0。

---

## 3. Task C：一个数据疑问

`plumbing_fixture/floor_drain/exposed_bottle_trap_silver` 的
`attachment_surface` 声明为 **`wall`**，而同族的 `floor_drain_silver` 是 **`floor`**。
它的 `facing` 写"侧出口朝向管道墙"——水槽下的存水弯确实靠墙，所以可能是有意的。

**Codex 的判断我接受：更像是有意的，别改成 `floor`。** `facing` 明确写着侧出口朝向
管道墙，水槽下的存水弯确实是连到墙的。但**"连到墙"不等于"有平背板的壁挂物"** ——
这个区分见 §1 的补充，它决定新工具怎么量:这个资产应当走
`no_mounting_plane_found`，而不是被当成壁挂平装件去量一个不存在的背板。

要做的只剩一件：在执行记录里写清"贴墙平装"和"接墙管件"的判据，
让下一个人不用重新推。**不新增 gate，不冻结 contract。**

---

## 4. 背景：为什么 owner 要宽松

两条明确指示，别往回拧：

- **"不要过于严格，3d 的这些东西其实都很不稳定…底面大概贴着地面就行，不歪，
  稍微浮空、稍微歪一点点都能接受，别钻牛角尖。"**
- **"允许穿模，不要弄太严格。"**（那条是对路径说的，但同一个取向）

所以：`acceptable` 是通过；**接触面积只记录、不参与判定**（弧底的音箱就算完全水平
接触斑也很小，拿面积当闸门是钻牛角尖）；只有"人一眼看得出来歪了"才算不合格。
