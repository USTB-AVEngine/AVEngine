# 移动声源与 JAEGER 对齐：现状、实测数字、以及我推翻过的结论

2026-08-26。这份文件是给接手的人的，不是进度汇报。凡是数字都是实测的，凡是我先前说错
又被数据推翻的，都写在 §5，因为那一节比其他任何一节都更能省下别人的时间。

场景：HM3D val。声学渲染：SoundSpaces 的 habitat 音频传感器。视觉渲染：AVEngine 自己
装好的 habitat prefix（owner 硬性要求，不用 Blender）。

---

## 1. 现在能做什么

一条链，每一环把下一环需要的东西写进产物，不靠人记，也不靠调用者发明参数。

```
tools/routes/compile_hm3d_dynamic_source_bank.py   navmesh 上找合法路线
tools/audio/make_source_orbit_bank.py              绕听者的环绕轨迹（听环绕用）
        ↓
tools/scene/choose_listener_pose.py                听者位姿候选（深度定朝向）
        ↓
tools/audio/render_moving_source.py --layout ambisonics
                                                   逐帧 FOA + 逐帧方向校验
                                                   声学试听候选，写回 accepted_index
        ↓
tools/visual/render_moving_source_video.py         读 accepted 那条出画面
        ↓
tools/audio/render_moving_source.py --layout binaural
                                                   读同一条出双耳（HRTF）
```

**FOA 和双耳是同一个渲染器的两种输出布局**，不是两个工具。曾经写成两份，结果两份时变
卷积各自漂移（一份长出扫窗、一份长出左右符号检查，互相没有），已合并。

辅助工具：

| 工具 | 干什么 |
|---|---|
| `tools/audio/measure_semantic_surface_area.py` | 按**表面积**统计 HM3D 语义类别 |
| `tools/audio/build_hm3d_material_map.py` | 生成 HM3D 的类别→声学材质映射 |
| `tools/audio/calibrate_surface_materials.py` | 拿混响时间反查某个面该用哪个材质 |
| `tools/audio/compare_material_acoustics.py` | 开/关材质的声学对比 |
| `tools/audio/hm3d_semantic_download.sh` | 下 HM3D 语义 v0.2（凭据走 curl -K，不进命令行） |

---

## 2. JAEGER 到底怎么做的

**论文 §3.1 写了方法，代码库里没有生成代码。** 上游 `liuzhan22/JAEGER` 只有训练和推理，
`data/data_tools/` 下唯一的文件是把**已渲好的 RIR** 和 LibriSpeech 干声做 `fftconvolve`。

方法：SoundSpaces 2.0 双向路径追踪，HM3D 有语义标注的子集，场景划分 130/15/36。
采样规则：声源与接收器**同一房间**、接收器可通行、声源在 **1–4 m** 内、**离障碍物 0.5 m**、
只保留**测地距离 < 2× 欧氏距离**的配对、插入音箱要在 1920×1080 语义帧里占 **≥500 px**。

### 2.1 我们已经逐位复现了它的 RIR

在 `00800-TEEsavR23oF` 上，用它 metadata 自己给的听者位姿和声源坐标：

```python
backend.scene_id = ".../TEEsavR23oF.glb"      # 普通 glb，不是 basis
backend.load_semantic_mesh = False
spec.enableMaterials = False
spec.channelLayout.type = Ambisonics; channelCount = 4
spec.acousticsConfig.sampleRate      = 16000
spec.acousticsConfig.directSHOrder   = 1
spec.acousticsConfig.indirectSHOrder = 1
spec.acousticsConfig.indirect        = False    # 关键
```

**20 个 task 里 19 个逐位完全相同**（`max |Δ| = 0.000e+00`，相关系数 `1.00000`）。
第 20 个（`task1_00006`）IR 长度和声源位置都对但相关系数 0，像样点级时间偏移，**未解释**。

`render_moving_source.py --direct-only` 就是这个开关，所以**基线和扩展出自同一份代码**。

### 2.2 它发布的 IR 里没有混响

`simulation_ds/test` 全部 **2790 个 IR、36 个场景**，起始后 5 ms 之外的能量占比：
中位 **4.97e-13**、最大 1.78e-12、**0/2790 超过 1%**。单条细看：直达波峰值 0.1614，
**紧邻下一个样点 1.0e-3（一个样点跌 44 dB）**，能量 100.000% 在 1 ms 内。

**发布的 RIR 实际上只有直达波。** 一个歧义必须记下：把 `enableMaterials=True` 打开而
语义网格上传失败时，输出和 `indirect=False` **数值完全一样**，所以无法区分它是故意关的
还是上传失败。产物一样：没有房间声学。

### 2.3 覆盖范围与通道约定

1000 个 single_source 全量：方位角**严格在 ±40° 内**、俯仰角 ±25° 内、距离
**0.86–4.12 m**。声源永远在相机视野里、永远很近。

- **通道顺序和我们一致，不需要任何转换。** 400 个样本拟合 `x=ch3, y=ch1, z=ch2`、
  三符号全正，中位 0.00°、400/400 在 5° 内；次优排列中位 27.8°。
- 角度标签：`azimuth = atan2(-x, -z)`、`elevation = asin(y)`，中位和最大误差都是 0.000°。
  前方 −Z、上方 +Y 与 habitat 一致，但**方位角朝 habitat 的左侧为正** ——
  `source_label=Right` 的 132 个样本中位 **−20.11°**。按 +X=右 读会每个角反号。

### 2.4 它有的、我们以为没有的

它用 **Hunyuan3D-1.0 生成了 120 个落地音箱**（96/12/12）插进场景，Task C 回归 3D 框、
Task D/E 把人声匹配到画面里某一个音箱。**"多个外观不同的音箱"它有。**

我们真正多的是两样：**移动声源**（它一个都没有，两种任务类型全静态），以及**多种形态**
的受控资产（它 120 个全是落地塔）。

---

## 3. 声学设置：哪些真的重要

### 3.1 材质必须开，而且天花板是最大的那个杠杆

开材质要**三样同时到位**，缺一样都静默失败：语义 scene dataset config、
`load_semantic_mesh=True`、显式 `setAudioMaterialsJSON()`（`enableMaterials` 只是说
"去找一个"）。而且 config 必须是**顶点色版本**：

```bash
# HM3D-Semantics v0.2 声明 has_semantic_textures: true，而这个 habitat 只懂顶点色，
# 于是报 "Mesh vertices were NULL"、几何不上传、输出退化成直达波单独。
# v0.2 的 .semantic.glb 同时带 COLOR_0，所以改这一个布尔值就够了。
python - <<'PY'
import json
H = "/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/val"
d = json.load(open(f"{H}/hm3d_annotated_val_basis.scene_dataset_config.json"))
d["stages"]["default_attributes"]["has_semantic_textures"] = False
json.dump(d, open(f"{H}/hm3d_vertexcolour_val_basis.scene_dataset_config.json", "w"), indent=2)
PY
```

**天花板的映射是这整件事里最大的单点改动。** `ceiling` 原本指向 `Acoustic Tile`
（商用吸声吊顶，中频 α **0.667**），而天花板占 HM3D 标注面积的 **17%**：

| `ceiling` 指向 | 中频 α | T60 | 判定 |
|---|---|---|---|
| Acoustic Tile（原） | 0.667 | **0.237 s** | 对住宅太死一半 |
| **Gypsum Board（现）** | 0.053 | **0.518 s** | 住宅区间内 |
| Plaster on Concrete Block | 0.057 | 0.523 s | 区间内 |
| Concrete, Rough | 0.060 | 0.557 s | 区间内 |

衰减直线度 0.996–0.998，所以外推有意义。参考区间"住宅 0.3–0.6 s"是**文献值，不是这些
扫描房间的实测** —— 我们没有它们的实测 RIR，只能说哪个赋值把扫描房子放进真实房子占的
区间里。地板同样校准过：`floor → Carpet`（中频 α 0.25）给 **0.518 s，本来就是好的，
不用改**。`Wood Floor` 会到 0.62 s（太活）。

### 3.2 类别映射：值得做，但杠杆比我说的小

按**面积**排（不是按个数），36 个标注场景、820 个类别、42341 m²：

| | 之前 | 之后 |
|---|---|---|
| 标签数 | 64 | **123** |
| 面积覆盖率 | 78.8% | **92.7%** |
| 匹配类别（00800 / 00802） | 26 / 30 | 46 / 50 |
| T20 变化 | — | 再短 4–6 ms（约 7%） |

**只有 7%**，因为墙 30% + 天花 17% + 地板 12% = 59% 的面积本来就映射好了，新增的 14 个
百分点落在家具门窗上。规则：只加标签，不碰任何吸收/散射/透射数值；每条映射记下它跟随
哪个标签；映射到面积报告里不存在的类别是**硬错误**。

顺带修了配置里三个让标签失效的缺陷：`piperefrigerator`（两个类别名被粘成一个，
`pipe` 和 `refrigerator` 双双失效，合计 0.35% 面积）、`floor` 重复两次、
`carpet` 无标签而最吸声的地毯材质不可达。

### 3.3 分析窗：0.25 ms，而这个数字改了三次

| 版本 | 依据 | 为什么错 |
|---|---|---|
| 12 ms | 在 JAEGER 数据上读 0.000° | 那批数据**没混响**，任何窗都是 0 |
| 2 ms | skokloster 石头大厅扫窗 | 不是住宅 |
| 1 ms | 住宅 + 默认材质 | 默认材质**高估混响 2 倍**，早期反射分布也不对 |
| **0.25 ms** | 住宅 + 校准后的真材质 | 目前站得住 |

校准后的房间（T20 中位 117.4 ms）上扫窗：

| 窗 | 中位误差 | 5° 内 |
|---|---|---|
| **0.25 ms** | **0.01°** | **85/120** |
| 0.5 ms | 1.48° | 71/120 |
| 1 ms | 3.89° | 61/120 |
| 2 ms | 14.44° | 36/120 |

**机理**：污染短窗的不是晚期混响，是**早期反射**。所以"混响小了窗能放长"这个直觉是错的
—— 材质让混响变短的同时改变了哪些早期反射落进窗内。

**别信这个常量。** `--window-sweep` 零成本（响应已经渲好），在你自己的房间读那张表。
另外 0.25 ms 在 16 kHz 下**只有 4 个样点**，它成立是因为直达波近乎脉冲，换成起振软的
声源就不成立。

### 3.4 双耳

引擎自带 `Binaural` 布局 + `setListenerHRTF()`，HRTF 在
`/data/avengine_external/rlr-sdk/hrtf/`（带 LICENSE 和 PROVENANCE）。

**头的朝向必须传向量，不能传角度**，而且必须和画面相机一致 —— FOA 通道在世界坐标，
方向校验不在乎朝向；双耳的左右**完全由头朝哪定**。

渲之前先证左右符号，而且**在场景最开阔处证，不在这段片子的听者位置证** ——
那个检查验的是渲染器/HRTF/通道顺序，是配置属性。在靠墙的听者处做会失败：右探针在耳高
被挡而 navmesh 说地面能走，两耳都低 11 dB，看起来**和左右通道接反一模一样**。

---

## 4. 遮挡不是缺陷，是标签问题

JAEGER **故意避开**遮挡（论文原话 "to reduce degenerate cases caused by fully occluded
direct paths"），所以它的声源永远在 ±40°、0.86–4.12 m 内。

我们的路线会被挡。真正的问题**不是被挡，是标签**：如果一帧标着"方位角 30°"而声强矢量
指向 70° 开外，那是拿假标签训模型。三条路：

1. 扔掉 —— JAEGER 的做法，题目变简单
2. **留着，标成"被遮挡 / 方向不可恢复"** —— 推荐，那是它的数据根本测不了的能力
3. 留着并保留几何标签 —— 要求模型猜，不建议

**这是数据集设计决定，不是待修的 bug。** 逐帧报告里每帧都带 `error_deg`，闸门有牙齿：
六条路线里三条 15/15 通过（最大 0.27°），三条 0/15（中位 44/44/70°，有一帧 171° ——
声强矢量指向声源反方向，因为唯一到达的能量是墙的反射）。

---

## 5. 我说错又被数据推翻的（读这一节最省时间）

| 我说过 | 实际 | 怎么被抓到的 |
|---|---|---|
| AVEngine 的 habitat 运行时没编译，用不了 | 装好的 prefix 有 5 个，渲染正常 | owner 质疑后去查，`import habitat_sim` 是错的钥匙 |
| 视觉端崩了 | 是我喂了 `.basis.glb` 而 Magnum 没有 BasisImporter，纹理全失败留空句柄→**段错误退出码 139，不抛异常** | 换非压缩 glb 就通 |
| `absorption[1]` 是 1 kHz 的吸收系数 | 是**最低频点（通常 125 Hz）** —— 害我说 Gypsum Board "0.29 太吸声"（中频 0.053）、floor 的 Carpet "0.010 硬反射"（中频 0.25） | 打全谱才发现 |
| 材质标签里没有 `floor`、没有 `ceiling` | 两个都有 | 读截断的输出读漏了 |
| 类别覆盖是保真度上最实的缺口 | 按面积已覆盖 78.8%，补到 92.7% 只动 T20 约 7%。**真正的杠杆是天花板指向哪个材质（0.237 → 0.518 s）** | 按面积量而不是按个数 |
| 移动声源"0/64800 采样点脱离 navmesh，全合法" | `is_navigable` 默认 **0.5 m 垂直容差**，在房子里能匹配到楼上楼下。收紧到 5 cm 后某层 **2225/2400 不合格** | 收紧容差 |
| 连续 47 个采样落在栅格外，"太长了不可能是量化误差" | 正好相反 —— 最短路径**长时间贴边界走**，长连续段就是量化在边界上该有的样子。格子 5→1 cm，比例 5.67%→1.38%，每点都在一格之内 | 三个候选逐个测 |
| SoundSpaces 和 JAEGER 的 FOA 通道顺序不同，要加符号翻转 | **同一个排列**，记法冲突而已。照那句话加转换会把所有方向搞反 | 拿它自己的数据拟合 |
| JAEGER 的方法从公开材料无法回答 | 论文 §3.1 写得很清楚 | owner 逼问后去读 PDF |
| 我们比它多"视觉上可区分的多个声源" | 它有 120 个 Hunyuan3D 落地音箱 + 匹配任务 | 读论文 Table 1 |
| 我那个下载脚本"凭据绝不出现在进程列表" | habitat-sim 的 downloader 把 `--user id:secret` **拼进 curl 命令行**，`ps` 可见 | 读它源码 |
| 相机对着第 0 帧声源就能拍到 | 那个点**埋在墙里**，8 秒对着一面墙，字幕还报 "source in frame" | 看图 |
| 传向量就解决了朝向问题 | 解决了歧义，没解决**责任** —— 还是得有人替每个房间决定；而且位置由音频选、朝向由视频选，一件事拆在两处 | owner 指出 |

另外两个**我自己的检查抓住我自己**的例子，说明失败闭合的价值：

- 双耳左右符号检查连拦我三次：先是头部右向量 Z 符号写错，再是"yaw"在两处约定不同
  （视频的循环用 `(sin,0,-cos)` 参数化、`look_at` 内部算 `atan2(-x,-z)`，**同一个数字
  命名相差 60° 的两个朝向**），最后是探针放在墙里。
- 单元测试抓出选底面的两个静默漏放：按面质心选，倾斜 26° 的柜子底面质心升出切片，
  报"没找到底面"而不是"歪"；以及信任网格绕序，法线翻转的网格同样静默通过。

还有一个**静默失败**值得单独记：HM3D 实例颜色在 `COLOR_0` 里存**线性**值，标注文本列
**sRGB** 十六进制。直接哈希线性字节，第一个场景 **395018 个面匹配到 0 个**，而工具只会
平静地报"全部无标注"，不报错。

---

## 5.12 路线合法性：测量，不当闸门

**遮挡和合法性是两件事，我一度混成一件。** 能不能听见是听者和几何决定的，
遮挡是**我们要的样本**（JAEGER 故意避开它，所以它的题目更简单）；
能不能走通是另一回事，那个要查。owner 的指示："**允许穿模，不要弄太严格。**"

`tools/routes/verify_route_legality.py` 只查路径、不谈可听性，而且**报告不闸门**：
擦墙角几厘米可以接受，只有"采样脱离 navmesh"或"净空低于 `--hard-floor-m`（默认 5 cm）"
才算真突破 —— 那是穿墙,不是贴墙。

| | navmesh 路线 | 环绕轨迹 |
|---|---|---|
| 可用 | **16/16** | **0/3** |
| 另外满足完整 0.190 m | 4/16 | 0/3 |
| 最差净空 | 0.151–0.393 m,中位 0.169 | **0.0** |

环绕轨迹 0/3 是**正确的标签**：它是听音演示，故意穿墙，不该以"路线"的名义进数据集。

**那几厘米的原因不是切角，我先猜错了。** 加了"把采样吸回可行走面"的修正后
**一个数字都没变**，因为采样从来没脱离 navmesh。真因是 **Recast 按整体素腐蚀**：
0.20 m 半径请求的最小实现内缩是 **0.102 m @ 0.05 体素、0.158 @ 0.02、0.182 @ 0.01**。
体素尺寸现在是参数，默认 0.02。规划余量参数存在但默认 0：0.03 能把净空从 0.170 提到
0.197，但可通行面积从 40.2 掉到 36.7 m² 并且连通性变差 —— 按 owner 的取向不值得默认付。

## 5.13 静态资产首轮的核查（Codex 交付 + 我的评估 + 他的反审计）

Codex 交付 28 个静态形态资产（`b368c43`）。我逐条核查：资产数、面数 59,921–60,000、
28/28 `research` + `authorized=false`、`placement` 四件套、HEAD 血缘、工作树干净 ——
**全部成立**，`facing` 是真描述不是模板话。倾斜也没被藏，每个 `acceptance` 里都写着
`resting_pose_verdict`。

回归：我的 HEAD 上是 `1 failed, 3255 passed`，**那 1 个失败是我弄坏的**
（加了十来个工具却只手改 `TOOL_INDEX.md` 一行没跑生成器）。跑完生成器回到
`3198 passed, 65 skipped, 71 subtests`，和他报的**逐位一致**。他留的那个测试抓住了我。

**我给他写的工单里有四处范围错误，三处是他只读审计找出来的**，见
`WORK_ORDER_RESTING_POSE_20260826.md` §0.1。最该记的一条：我列了别人八个最差资产里的
七个，**唯独漏掉自己那个最歪的**（29.61° 的双脚电视）。那不是标准。

他还提了一个比我准的区分：**`wall` 不是一类东西** —— "贴墙平装"有平背板，
"接墙管件"（瓶式存水弯）没有，后者应当返回 `no_mounting_plane_found`，
**那是正确答案不是失败**。

---

## 6. 还没做的

### 交给 Codex 的（工单 `WORK_ORDER_RESTING_POSE_20260826.md`，已修订）

1. **Task A**：让 `measure_static_resting_pose.py` 读 `placement.attachment_surface`，
   分别处理 floor / wall / ceiling / 未声明，并统一两条发布路径的 `acceptance` 字段。
   40 个适用资产，22 个未声明 surface。
2. **Task B**：重做八个真歪的（含我那台 29.61° 的电视）。
3. **Task C**：写清"贴墙平装"和"接墙管件"的判据。

### 我自己的

4. **HM3D train split 语义**正在后台下（145 个标注场景）。到手之后要验：
   我那张材质映射是**只按 val 的 36 个场景**建的，92.7% 的覆盖率在 train 上还成不成立。
5. **把音频搬进 AVEngine 自己的声学运行时** —— 现在输入齐了（语义 + 校准过的材质映射
   + 实测 T60 目标）。这条同时消掉"视觉在 AVEngine、音频在 ss2"的跨环境割裂，
   而位姿文件之所以要存在，一半原因就是这个割裂。
6. **成规模产数据**：整条链跑遍多场景多楼层，带逐帧遮挡标注。工具已经验过了，
   这是把工具变成数据的那一步。缺的决定只有 §4 那个标签策略。
7. **JAEGER `task1_00006`** 为什么不逐位相同（长度和位置都对，相关系数 0，
   像样点级偏移）。19/20 已足够定论，这条是好奇心。

### 归档，不必再动

8. **遮挡帧的标签策略**（§4 三选一）：owner 已明确"**不排斥遮挡**"，所以不是筛掉，
   是逐帧标注 —— 逐帧报告已经带 `error_deg`，实现上已经在了，剩下的只是选哪套标签名。
9. **听者试听只查路线中点**：按上面那条，这不再是缺陷。位姿文件按排名试听候选，
   录用哪个写回文件；路线后段被挡是**要保留的样本**。
10. **可行域栅格 vs navmesh 的 1%–14% 分歧**：已解释（格子中心量化），
    现在只是"路线贴墙多近"的诊断指标，不是闸门。
11. **动物资产做移动声源**：猫狗缺发声锚点、缺坐标系声明、+Z 朝上。
    这是资产流水线的活，判断"动物的发声锚点在哪"更接近 Codex 的领域。

---

## 7. 关键路径速查

```
场景        /data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/val/
语义 config  .../val/hm3d_vertexcolour_val_basis.scene_dataset_config.json
材质映射     /data/avengine_external/assets/hm3d_material_config.json
视觉 prefix  /data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z
magnum      /data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages
RLR SDK     /data/avengine_external/rlr-sdk/RLRAudioPropagationPkg
HRTF        /data/avengine_external/rlr-sdk/hrtf/mit_kemar_normal_pinna_16k_v2_sofar_20260820T2200Z
声学环境     conda activate ss2（habitat-sim 0.2.2 + RLR）
视觉环境     /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python，PYTHONPATH=src
JAEGER 数据  /data/datasets/JAEGER/simulation_ds/test/  （36 个场景，2790 个 IR）
```

两个会咬人的陷阱，再说一遍：**视觉渲染不要喂 `.basis.glb`（段错误，不抛异常）**；
**在未加载 navmesh 的 PathFinder 上查询会段错误**。
