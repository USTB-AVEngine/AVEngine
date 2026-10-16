# 交接 20260826b：JAEGER 对齐 + FOA 链路（给压缩后的自己）

> 前一份交接是 `HANDOFF_20260826_STATIC_AND_COAT.md`（静态资产），本份接在它后面。
> 静态资产的结论在 `STATIC_REFERENCE_RUN_20260826.md`。这份记的是**声学侧**。

## 0. 状态快照

| 项 | 值 |
|---|---|
| 分支 | `cc-static-sound-sources`，最新提交 `d33c938`，已推 |
| 已发布资产 | 16 个（4 动物 + 12 静态），`/data/avengine_external/assets/sound_source_assets_v1/` |
| 静态资产朝向 | 12 个里 11 个已校平，沙岩智能音箱度量被拒故保持歪着（记录里写明） |
| HM3D 下载 | **进行中**，`/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/val/`，已 7.2 G，三个 uid 里第三个在跑 |
| Codex | 在同一分支并行做家电/固定件；镜像目录里会出现他们的 profile 修订，别当成自己的 |

**owner 已拍板**：①静态物必须水平（已实现并全量重跑）；②毛色要在 FLUX 生图时用 prompt 指定，不是后处理；③塔形不再追三饰面，转做多音箱配置；④HM3D 要下、数据集放公共目录。

## 1. 声学链路：四个真实发现（都已实测）

### 1.1 12 ms 分析窗在真实房间里静默偏差约 18°（最重要）

团队现成脚本（`render_test.py` / `render_real.py`）用**检测到 onset 之后 12 ms** 的窗算强度矢量。
空箱子 `testroom` 里 12 ms 内没有别的东西到达，所以测出 0.000°，这个窗长从没被质疑。
真实房间的早期反射 **3–6 ms 就到**。实测扫窗（skokloster-castle，4 个声源）：

| 窗口 | 0.5 ms | 1 ms | 2 ms | 3 ms | 6 ms | 12 ms |
|---|---|---|---|---|---|---|
| 误差 | 0.0° | 0.0–0.5° | 0.0–1.2° | 0–2.7° | 15.5–17.9° | 17.3–19.9° |

**用 1 ms，不是 2 ms。** 这条修正了本文档原来的结论。

skokloster-castle 是一座空旷的石头大厅，0.5–2 ms 都测出 0.0°，所以 2 ms 当时看着安全。
**在有家具的住宅里不安全。** 2026-08-26 在 HM3D `00804-BHXhpBwSMLh` 上，对一条移动声源
路线的 25 个逐帧响应扫窗：

| 窗口 | 0.25 ms | 0.5 ms | 1 ms | 2 ms | 4 ms | 12 ms |
|---|---|---|---|---|---|---|
| 中位 | 0.00° | 0.00° | 0.00° | 0.12° | 4.06° | 16.46° |
| 最大 | 0.00° | 0.00° | 0.81° | **6.16°** | 24.81° | 26.94° |
| 5° 内 | 25/25 | 25/25 | **25/25** | 21/25 | 13/25 | 0/25 |

而且 2 ms 的污染**随距离增长**：4.6 m 处读 4.8°，1.1 m 处读 0.0°。直达波随距离变弱，
反射不变弱。所以"远处的声源测不准"看起来像遮挡，实际是窗太长。

`tools/audio/render_moving_source.py --window-sweep` 可以在任意路线上复现这张表 ——
**不要信这个数字，在你自己的场景上扫一遍**，因为它显然依赖房间。
**这条要同步给 binding_data 的 owner** —— Phase 1 那 2790 个 IR 是用 `audio_utils.py`
（STFT 域、另一条路径）验的，不一定受影响，但 `render_real.py` 这条肯定受。

### 1.2 `render_real.py` 的真值公式是错的

```python
gaz = atan2(rel[1], rel[0]);  gel = asin(rel[2]/rr)   # 这是 z-up 的算法
```
Habitat 是 **y-up**。它自己的输出把一个水平方向标成"俯仰 +90°"，最大误差 173.86°，
而末尾那句 `=> renderer geometrically correct` 是**无条件打印**的。别信那句。

### 1.3 通道常量 `CH_X,CH_Y,CH_Z = 3,1,2` 是对的（我一度怀疑错了）

我猜"ambisonic 是 z-up 所以要换轴"——**错的**。用 8 个**非轴对齐**方向重新拟合，
最差 0.000°，符号全正、无需换轴。之前的验证之所以可疑，是因为它只用轴对齐方向，
那种情况下多个排列并列、拟合返回碰巧先看到的那个（`render_test.py` 的输出里
同一次运行对不同声源给出了 `chs(1,3,2)` 和 `chs(3,1,2)` 两种答案，就是简并的证据）。

**修正（2026-08-26）：两者其实是同一个排列，不存在通道顺序差异。**

本文档原来写"SoundSpaces = `[W,Y,Z,X]`，JAEGER 发布数据 = `[W,Z,X,Y]`，两者不同"，
依据是 `binding_data/audio_utils.py` 第 35 行。那是**记法冲突，不是数据差异**：

- `audio_utils.py` 按 **front/left/up** 给通道命名（`CH_Z,CH_X,CH_Y = 1,2,3`），
  然后 `front=-Ix, left=-Iy, up=+Iz`。那个"符号翻转"是 x/y/z → front/left/up 的换算，
  不是换轴。
- 换算回 habitat 的 x/y/z：`x=ch3, y=ch1, z=ch2`，**三个符号全正**。

拿 JAEGER 自己发布的数据实测过了(`simulation_ds/test` 的 400 个 task1，
用 `position_local` 当真值，1 ms 窗)：`order (3,1,2) signs (+,+,+)`
**中位 0.00°、p90 0.00°、400/400 在 5° 内**；次优的排列中位 27.8°，毫无歧义。

**所以读 JAEGER 的 FOA 不需要任何转换**，和我们渲染出来的用同一组常量。
反过来说，照原来那句话去加一层转换，会把方向搞错。

JAEGER 的角度标签约定（同一批数据实测，中位和最大误差都是 0.000°）：
`azimuth = atan2(-x, -z)`、`elevation = asin(y)`，x/y/z 是听者局部 habitat 坐标。
即**前方 -Z、上方 +Y 都和 habitat 一致，但方位角朝 habitat 的左手方向为正** ——
听者右侧的声源在它的标签里是**负**方位角（`source_label=Right` 的 132 个样本方位角
中位 -20.11°，`Left` 的 104 个中位 +19.83°）。按 +X=右 去读会每个角都反号。

### 1.4 `cast_ray` 在这个构建里不测静态场景

只测刚体，开不开 physics 都返回 0 命中 —— 拿它做视线遮挡检查是**空转**。
改用 navmesh：`geodesic / straight <= 1.08`。

## 2. 关于"某个场景不能用"——这个说法是错的

我一度说 `apartment_1` 不适合声学仿真，**那是拿 12 ms 那把坏尺子量的**。
用 2 ms 重测：4 个方向里 3 个降到 0.1–5.6°，只有一个在**所有窗长下**都错 103°（那个 5.6° 现在看应该用 1 ms 再测一次）
——那是真正被遮挡的摆位（声源在家具里/墙后）。

**正确结论：不存在"这个场景不能用"，只存在"这个摆位不能用"。**
所以逐摆位的 DoA 校验是那道闸门，HM3D 同样适用。

## 3. 环境事实

- **ss2 的 habitat-sim 相机渲染在这台机器上崩**（`GL::Renderer::Error::InvalidValue` + core dump），
  试过 `gpu_device_id` / `MAGNUM_DEVICE` / `EGL_PLATFORM` / 强制 NVIDIA vendor 全无效。**音频完好。**
- **Blender 的 EEVEE 渲染一直正常** → GPU/GL 栈没问题，坏的是 habitat 的 EGL 路径。
- **AVEngine 自己有 habitat 视觉端**（`src/avengine/rooms/habitat_capture.py`），
  但 `/data/jzy/code/habitat-sim-AVEngine/` **只有源码没有编译产物**，
  `AVENGINE_HABITAT_RUNTIME_ROOT` / `_PREFIX` 也没设，所以那条路现在跑不起来。
  **想要 habitat 出视觉，就得把那个运行时装上**；否则视觉走 Blender。
- 只有 `ss2` 环境有可用的 `habitat_sim`（0.2.2，SoundSpaces 2.0 钉死这个版本）。

## 4. 脚本位置（都在 `/tmp/`，要提交进仓库）

| 脚本 | 作用 |
|---|---|
| `/tmp/plan_from_navmesh.py` | navmesh 采样出接收点+音箱摆位，带测地/直线比值的视线检查 |
| `/tmp/insert_speakers_and_render_foa.py` | 按摆位渲 FOA，逐声源验 DoA，输出 `scene_report.json` |
| `/tmp/fit_foa_axes.py` | 用通用方向拟合 FOA→世界轴映射（用来推翻/确认常量） |
| `/tmp/hm3d_download.sh` | 从 `~/.hm3d_token` 读凭证下载，带三次重试 |
| 已进仓库 | `tools/blender/render_instance_diversity_check.py`（房间尺度看饰面分不分得开） |

**token 在 `~/.hm3d_token`（600 权限，两行：id / secret）。我不把它写进命令行。**
owner 说过可以全权代做，但这条我保留 —— 凭证不从对话里复制进命令。

## 5. 已验证跑通的东西

`skokloster-castle` 里放 4 个外观不同的音箱，每个当声源渲 FOA：

| 资产 | 距离 | DoA 误差 |
|---|---|---|
| 书架箱 胡桃 | 4.88 m | 0.04° |
| 书架箱 白 | 1.99 m | 0.00° |
| 智能音箱 炭灰 | 4.01 m | 0.95° |
| 回音壁 黑 | 2.01 m | 1.46° |

声源放在资产记录的**发声锚点偏移**上（低音单元/栅格中心），不是物体原点。

## 6. 下一步（按顺序）

1. **下载完成后**：在 `sound-spaces/data/scene_datasets/` 建软链指向
   `/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d`，核对那 36 个场景
   （`00800-TEEsavR23oF` … `00894-HY1NcmCgn3n`，全在 val 范围）都在。
2. **把 §4 的脚本提交进仓库**（现在还在 `/tmp`，压缩后会找不到）。
3. **在 HM3D 场景上跑通同一条链**，逐摆位 DoA 校验。
4. **视觉**：要么装 AVEngine 的 habitat 运行时，要么用 Blender + 同一份 placement 文件。
5. **把 §1 的四条发现同步给 binding_data 的 owner**（12 ms 窗那条尤其）。

## 5.5 JAEGER 的声学到底怎么做的（2026-08-26 读码+读数据）

**代码不发布渲染器，但论文写了方法。** 公开仓库里没有一个文件引用 habitat_sim /
soundspaces / AudioSensor（我们的 `binding_data/` 不算），`data/data_tools/` 下只有一个
`conv_ir_speaker_foa.py` 做 `fftconvolve`。方法在论文 §3.1（arXiv 2602.18527）：

> *SoundSpaces 2.0* renders multi-channel spatial audio by simulating room impulse
> responses (RIRs) over realistic 3D meshes with material-dependent acoustics. It
> employs bidirectional path tracing (Cao et al., 2016) to model direct sound and
> higher-order effects, including reflections, transmission, diffraction, and air
> absorption.

形式上就是 `A_c^(r)(t) = (R_c(·; s, r, θ) * A^(s))(t)`，`c ∈ {0,1,2,3}`。
所以它不"拟合"任何东西 —— 是在 HM3D 的三角网上做几何声学仿真，房间的声音特征来自
**网格几何 + 由 HM3D 语义类别决定的材质**。视觉用 Habitat-Sim，场景取自
**HM3D 有语义标注的子集**，按场景划分 **130/15/36**（train/val/test）。

它的采样规则（论文原文，值得对齐）：
- 声源和接收器**必须在同一个房间**，"to reduce degenerate cases caused by fully occluded direct paths"
- 先采一个可通行的接收器位姿，再在 **1–4 m** 内采声源，**离障碍物留 0.5 m**
- 只保留 **测地距离 < 2 × 欧氏距离** 的配对
- 插入的音箱要在 1920×1080 的语义帧里占 **≥500 px**，否则丢弃

## 5.6 但发布出来的 IR 里没有混响（全量实测）

**这是这一串里最要紧的发现，而且和论文文字对不上。**

拿 `simulation_ds/test` 全部 **2790 个 IR、36 个场景**统计"起始后 5 ms 之外的能量占比"：

| | |
|---|---|
| 中位 | **4.97e-13** |
| p99 | 1.48e-12 |
| 最大 | 1.78e-12 |
| 超过 1% 的 IR | **0 / 2790** |
| W 通道非零样点数 | 中位 281（IR 长度约 2100） |

单条细看（`00800-TEEsavR23oF/task1_00003`，距离 1.84 m）：直达波在样点 81，峰值 0.1614，
**紧邻下一个样点是 1.0e-3 —— 一个样点跌 44 dB**；能量的 **100.000%** 在起始后 1 ms 内。

**发布的 RIR 实际上只有直达波。** 不管他们渲的时候用了什么配置，产物里没有反射。

同几何对撞（用我们的 ss2 音频传感器，在它 metadata 给的 `agent_pos` / `position_world` 上渲）：

| | 它的 | 我们的 |
|---|---|---|
| 起始样点 | 54 / 169 / 74 / 81 / 48 / 99 | **6 个全部完全相同** |
| 反推采样率 | ≈16 kHz | 16 kHz |
| IR 长度 | ~2100 样点（131 ms） | ~26300（1.65 s） |
| Schroeder T20 | **0.0 ms** | 160–214 ms |
| 直达/混响能量比 | **+111 ~ +118 dB** | −9 ~ −13 dB |
| DoA 恢复 | 0.00° | 0.00–0.27° |

起始样点六个全中，说明**引擎、采样率、坐标约定都和我们一致**；差别全在混响上。

**三个后果，直接影响我们怎么用它：**

1. **§1.1 那个 12 ms 窗的坑，根源在这里。** `binding_data/audio_utils.py` 写着
   "VERIFIED == metadata.DoA to 0.0000 deg" —— 那是拿**无混响**的 IR 验的，任何窗长都会
   给 0.00°。所以 12 ms 在 JAEGER 数据上看着完美，一到真实渲染就偏 18°。
   **不要再用 JAEGER 的数据去校准任何窗长。**
2. 论文把 Neural IV 的动机写成"robust directional cues even under reverberation"，
   但我们手上这个 test split 里**没有混响**。（只有 test，train 未验证。）
3. **我们的数字和它的不可直接比较。** 我们渲的是带 160–214 ms 衰减的 IR，
   定位本来就更难。要比就得在同一条件下比。

**我们自己也有一个反向的缺口：** 它用 HM3D 语义子集 + 材质相关声学；我们盘上
**没有 HM3D 语义数据**（只有 `.glb` / `.basis.glb` / `.basis.navmesh`），而且
`tools/audio/` 三个工具全都 `enableMaterials = False`。要做材质相关声学，
得先拿到 HM3D 的语义版本。



产物形态（`SpatialSceneQA/hm3d_foa_av_v2`，test split 实测）：

| | |
|---|---|
| 每个 task | `rgb.png` + `depth.png` + `metadata.json` + IR，**单帧，不是视频** |
| task1 `single_source` | 1000 个，一个 `ir.npy`，形状 `(4, 2129)` float64 |
| task2 `dual_source` | 895 个，`ir_male.npy` + `ir_female.npy`，卷积后**直接相加**，不做电平配比 |
| 场景内人数 | 1–3 个（`num_speakers`），其中一个/两个是发声源 |
| 标注 | `visual_objects` 用 **HM3D 语义实例**（`instance_id` + `category`，如 wall/picture），每个都带 DoA |

**全部是静态的。没有移动声源的任务类型。** 一个 task = 一个静止听者 + 一个静止声源
+ 一帧图。

**修正一条我说错的话：** 我曾说"我们比它多了视觉上可区分的多个声源"，**这是错的**。
论文 §3.1 写明它用 **Hunyuan3D-1.0 生成了 120 个落地音箱模型**（96/12/12 分给
train/val/test）插进场景，Task C 就是回归插入音箱的 3D 框，Task D/E 就是把一段人声
匹配到画面里的某一个音箱。那正是"多个外观不同的音箱"这个配置。

我们真正多的是两样：**移动声源**（它一个都没有），以及**多种形态**的声源资产 ——
它 120 个全是落地塔，我们是书架箱 / 智能音箱 / 回音壁 / 落地塔 / 电视，
每个带受控属性（form factor × material × finish）、发声锚点和 provenance。

**覆盖范围很窄**（1000 个 single_source 全量统计）：

- 方位角 **严格在 ±40° 内**（min -39.97、max +39.92，100%）
- 俯仰角 ±25° 内
- 距离 **0.86–4.12 m，中位 1.85 m**

也就是说**声源永远在相机视野里、永远很近**。它从不问"身后的声音"、"隔着墙的声音"、
"房间对面的声音"。

顺带一个有意思的对照：我们移动声源那批 episode 里，验证干净的距离是 1.1–4.6 m，
全军覆没的是 4.2–7.2 m —— **验证得干净的那一段基本就是 JAEGER 的距离带**。
在有家具的住宅里超过约 4 m，直达波定位本来就难。

## 5.7 我们已经逐位复现了 JAEGER 的 RIR（2026-08-26）

上游 GitHub（`liuzhan22/JAEGER`）确认只有训练和推理代码，没有生成/仿真代码 ——
不是我们本地 checkout 不全。但**配置可以反推出来，而且我们已经复现了**。

在 `00800-TEEsavR23oF` 上，用它 metadata 给的 `agent_pos` / `agent_rot_quat` /
`audio_source.position_world`，配置如下渲染：

```python
backend.scene_id = ".../TEEsavR23oF.glb"      # 普通 glb，不是 basis
backend.load_semantic_mesh = False
spec.enableMaterials = False
spec.channelLayout.type = Ambisonics; channelCount = 4
spec.acousticsConfig.sampleRate     = 16000
spec.acousticsConfig.directSHOrder  = 1
spec.acousticsConfig.indirectSHOrder = 1
spec.acousticsConfig.indirect       = False     # 这一行是关键
```

**20 个 task 里 19 个和它发布的 `ir.npy` 逐位完全相同**
（`max |Δ| = 0.000e+00`，相关系数 `1.00000`）。
唯一不符的 `task1_00006`：IR 长度完全相同（2177 = 2177），声源位置也确认是 metadata
标的那个，但相关系数 0 —— 像一个样点级的时间偏移。**未解释，不编解释。**

**一个必须写下来的歧义：** 把 `enableMaterials = True` 打开、而语义网格上传失败时，
输出和 `indirect = False` **数值完全一样**（task1_00000 两条路都是 2102 样点、
D/R = 116.39 dB）。所以**无法区分它是"故意关了 indirect"还是"开了材质但上传失败"**。
两种情况下产物都一样：没有房间声学。

## 5.8 HM3D 语义：v0.2 已下载，但 0.2.2 用不了它

- **必须 v0.2。** v0.1 只标注 JAEGER 那 36 个场景里的 20 个。
- v0.2 的 annots 是 **`.tar`**，不是 `.tar.gz`（v0.1 才是 gz）；照 gz 拼会 404。
  0.2.2 的 downloader 不知道 v0.2，正确 uid 定义在
  `habitat-sim-AVEngine/src_python/habitat_sim/utils/datasets_download.py`。
- 下完实测：**有语义的场景正好 36 个，和 JAEGER 的 36 个测试场景 0 缺失 0 多余。**
  也就是 **JAEGER 的 test split 就是 HM3D-Semantics v0.2 的 val split。**
- 落盘位置：`versioned_data/hm3d-1.0/hm3d/val/<scene>/<id>.semantic.glb` + `.semantic.txt`，
  两个 annotated dataset config 在 `hm3d/` 层，val 专用那个还要**拷一份进 `val/`**
  （它用相对路径，habitat 按 config 自身所在目录解析）。

**但材质目前跑不通。** config 里写着 `has_semantic_textures: True` —— v0.2 是**纹理式
语义**，而 habitat-sim 0.2.2 只懂顶点色式（v0.1），这正是它的 downloader 只列 v0.1 的原因。
实测：`.semantic.txt` 的 130 个类别读到了，其中 **26 个匹配上材质标签**，但语义网格传给
RLR 时报 `Mesh vertices were NULL` + `The active scene does not contain semantic
annotations`，于是整个几何没上传，输出退化成直达波单独。

材质库是 `sound-spaces/data/mp3d_material_config.json`：24 种材质、64 个标签，
按语义标签名匹配，通过 `sensor.setAudioMaterialsJSON()` 显式设入
（`enableMaterials` 只是说"去找一个"）。标签是给 MP3D 写的，
这个场景 130 个类别里只有 26 个能对上。

## 5.9 材质已经跑通了，改一个布尔值（2026-08-26）

先说走不通的那条：`habitat_sim-0.3.3` 那个 egg **不能用**。它 `import` 就挂在
`AttributeError: UP` —— bindings 是对着更新的 magnum 编的，而 ss2 的 magnum
有 `Vector3.y_axis` 却没有 `UP`，egg 自己也不带 magnum。不重编 magnum 这条路是死的。

**真正的口子在语义 glb 本身。** 直接解 v0.2 的 `.semantic.glb` 看它的顶点属性：

```
mesh attributes: ['COLOR_0', 'NORMAL', 'POSITION', 'TEXCOORD_0']
images: 41   materials: 41   textures: 41
```

**它同时带顶点色和语义纹理**，不是纯纹理的。habitat-sim 0.2.2 懂顶点色，
只是 config 里 `has_semantic_textures: True` 把它推上了它走不通的那条路。

所以修法是**把那个布尔值改掉**：

```bash
# 从 annotated 配置生成一份顶点色版本
python - <<'PY'
import json
H = "/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/val"
d = json.load(open(f"{H}/hm3d_annotated_val_basis.scene_dataset_config.json"))
d["stages"]["default_attributes"]["has_semantic_textures"] = False
json.dump(d, open(f"{H}/hm3d_vertexcolour_val_basis.scene_dataset_config.json", "w"), indent=2)
PY
```

然后 `--dataset-config` 指向 `hm3d_vertexcolour_val_basis...json`，
`load_semantic_mesh = True`、`enableMaterials = True`、
`sensor.setAudioMaterialsJSON(mp3d_material_config.json)`。
`Mesh vertices were NULL` 消失，日志开始逐类别报
`Material for category 'toy' was not found. Using default material instead.`
—— 查表真的在跑。

**材质的效果很大**（`tools/audio/compare_material_acoustics.py`，声源/听者几何取自
JAEGER 自己的 task）：

| 场景 | 语义类别 | 匹配到材质 | T20 关材质 | **T20 开材质** | D/R 关 | D/R 开 |
|---|---|---|---|---|---|---|
| 00800-TEEsavR23oF | 130 | 26 | 191.2 ms | **75.5 ms** | −11.53 dB | −8.66 dB |
| 00802-wcojb4TFT35 | 206 | 30 | 205.6 ms | **75.1 ms** | −10.99 dB | −7.16 dB |

DoA 两边都是 0.00°，方向不受影响。

**方向值得注意：开材质之后混响变短了，不是变长。** 默认材质吸收系数 0.1（相当反射），
一视同仁贴到所有面上；真实材质里有地毯、窗帘、软家具，吸收高得多。所以
**默认材质渲染是高估混响的，大约 2.5 倍**。

**这条直接影响我们之前所有的声学数字：** §6 移动声源那批渲的是 160–214 ms 衰减，
用的是默认材质，**太混了**。要重跑。

覆盖率仍是个缺口：26/130 和 30/206 的类别能对上材质，其余落回 Default。
`mp3d_material_config.json` 的 64 个标签是给 MP3D 写的，HM3D 的词表细得多
（`bath mat`、`electrical controller`、`tissue box`…）。要更准就得给 HM3D 补一张映射表。

## 5.10 用材质重跑移动声源：混响腰斩，而安全窗还要更短

同一条路线、**同一个钉死的听者**（`--listener`，这个参数是为此加的 —— 听者试听本身要
渲 IR，材质一变通过的候选就变，两次跑连听者都不一样，A/B 就废了）：

| | 关材质 | **开材质** |
|---|---|---|
| T20 中位 | 171.6 ms（160.1–211.9） | **81.0 ms（75.3–92.8）** |
| DoA @ 0.25 ms | 0.00° | 0.00° |
| DoA @ 0.5 ms | 0.00° | 0.00° |
| **DoA @ 1 ms** | 0.63° | **3.39°** |
| DoA @ 2 ms | 2.16° | 3.89° |

**混响少了一半，安全窗却要更短。** 四条路线全部开材质：

| 路线 | T20 中位 | 0.25 ms | 0.5 ms | 1 ms | 2 ms |
|---|---|---|---|---|---|
| y+0.163 ep0 | 96.1 ms | 0.00 | 0.00 | 0.38 | 3.26 |
| y+0.163 ep1 | 60.1 ms | 0.01 | **4.98** | **7.25** | 8.61 |
| y+3.163 ep0 | 80.8 ms | 0.05 | 0.26 | 0.31 | 0.42 |
| y+3.163 ep1 | 84.5 ms | 0.00 | 0.02 | **6.06** | 11.18 |

**只有 0.25 ms 四条都干净。** 默认值已改成 0.25 ms。

**这个数字改了三次，每次都是因为测试条件更接近真实：**

| 版本 | 依据 | 为什么错 |
|---|---|---|
| 12 ms | JAEGER 发布数据上读 0.000° | 那批数据**没有混响**，任何窗都是 0 |
| 2 ms | skokloster-castle 扫窗 | 空旷石头大厅，不是住宅 |
| 1 ms | 有家具住宅 + 默认材质 | 默认材质**高估混响 2 倍**，且早期反射分布不对 |
| **0.25 ms** | 住宅 + HM3D v0.2 真实材质 | 目前四条路线成立 |

**机理**：污染 1 ms 窗的不是晚期混响，是**早期反射**。材质只降低晚期能量，却改变了
哪些早期反射落进窗内。所以"混响变小 ⇒ 窗可以放长"这个直觉是错的。

**别信这个常量。** 用 `--window-sweep`（响应已经渲好了，扫窗零成本）在你自己的房间里
读那张表。另外 0.25 ms 在 16 kHz 下**只有 4 个样点** —— 它成立是因为直达波近乎脉冲，
换成起振软的声源就不成立了。

（注意上面那些 `within 5 deg` 的计数 8/15、10/15、13/15：那是**路线级遮挡**，不是窗的
问题 —— 听者试听只检查了路线中点，路线其余部分可能被墙挡住。中位数看窗，最大值看遮挡。）

## 5.11 一个渲染器，两种输出布局（2026-08-26）

**FOA 和 HRTF 不是两个任务，是同一次渲染的两种输出布局。** 我一度写成了两个脚本，
结果树里有两份时变卷积循环，而且**已经开始各自漂移** —— FOA 那份长出了扫窗，
双耳那份长出了左右符号检查，互相都没有。现在合成一个
`tools/audio/render_moving_source.py --layout {ambisonics,binaural}`，
凡是与布局无关的部分只写一遍。

| 布局 | 通道 | 用途 |
|---|---|---|
| `ambisonics` | 4，世界坐标轴 ACN `[W,Y,Z,X]` | 携带完整场；逐帧方向校验必须用它（声强矢量要方向通道） |
| `binaural` | 2，`[left, right]`，经显式 HRTF | 耳机可直听，前后不塌 |

各自的校验也各归各位:ambisonics 扫分析窗;binaural 在**开阔处**证明左右符号。

**串起来的链条**(每一环把下一环需要的东西写进报告，不靠人记):

```
轨迹库 → ambisonics 渲染（定听者、验 DoA、写 render_report.json）
       → 视频（选朝向，把 camera_aim_world 向量写进 video_manifest.json）
       → binaural（--from-report 复用听者，--video-manifest 读朝向）
```

**双耳的朝向必须传向量，不能传角度。** 传角度试过一次，正好撞出这个检查存在的意义:
视频挑朝向的循环里 yaw 参数化的是 `(sin, 0, -cos)`，而消费它的 look-at 内部算的是
`atan2(-x, -z)` —— **同一个数字命名了相差 60° 的两个朝向**。向量不会被误读，
而且头部自身的轴是从旋转矩阵里读出来的，不再由角度反推。

**泛化上清掉的东西:** 材质库路径不再有机器专用默认值（配 `--dataset-config` 时必须显式给
`--materials-json`，因为 `enableMaterials` 只是叫传感器"去找一个"）；采样率成为参数；
资产目录、HRTF、场景、朝向全部由参数进来。三个工具里已无 `/data/jzy` 之类的硬编码。

`--direct-only` 一并留在这个工具里 —— 那是复现 JAEGER 发布 RIR 的开关（见 §5.7）。

## 6. 移动声源（2026-08-26 追加）

`tools/routes/compile_hm3d_dynamic_source_bank.py` 找路线，
`tools/audio/render_moving_source.py` 渲声音。逐帧一个冲激响应，干声跟逐帧响应
做时变卷积 + 重叠相加，所以混响不会每帧重启。

**听者必须靠渲染来选，不能靠距离。** 在距离带里随机取 navmesh 点，约一半时候取到被墙
隔开的位置：同一个场景六条 episode，三条每帧都在 0.27° 内，三条**每帧都失败**，中位
44°、44°、70°，有一帧读到 171°（声强矢量指向声源反方向，因为唯一到达的能量是墙的反射）。
测地距离预测不了这个，navmesh 不知道耳朵高度上站着什么。

所以工具先在路线中点渲一发响应试听者（一两秒），不合格就换。加上这道预筛之后，同样六条
episode **全部 15/15 通过**，最大误差 0.07–0.57°，只花了 1–3 个候选。

## 7. 还开着的决定

- **毛色轴**：owner 已定方向（prompt 指定而非后处理），仓里现成的路是
  `SPEAR/docs/generated_animal_real_reference_coat_workflow.md`
  （渲四视图 → FLUX.2 带 seed 编辑 → 投影回原 UV，不重绑骨）。**还没动手。**
- **`instance_axes.rigid_static_object` 里混进了固定属性**（form_factor / material），
  是 Codex 的代码，读起来像它们在实例间会变。没动，等协调。
